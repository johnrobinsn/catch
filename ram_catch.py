'''
RAM (c) John Robinson 2022
'''

import argparse
import numpy as np
import random

import cv2

from torch import optim
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

from torch.utils.tensorboard import SummaryWriter

from ram import *
from ram_visualize import *

from catch import Catch

def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='RAM Catch')

parser.add_argument('--render',action='store_true',default=False,
                    help='render the environment')
parser.add_argument('--catch-grid-size',type=int,default=24,
                    help='grid size for catch game')
parser.add_argument('--seed',type=int,default=42,
                    help="seed used for random initialization")
parser.add_argument('--demo',action='store_true',default=False,
                    help='demo catch from best checkpoint')


args = parser.parse_args()

def set_seed(seed=543):
  '''REPRODUCIBILITY.'''
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  random.seed(seed)
  # When running on the CuDNN backend, two further options must be set
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter(comment='RAM Catch')

def adjust_learning_rate(optimizer, epoch, lr):
  lr = max(lr * (0.95 ** epoch), 1e-7)
  optimizer.param_groups[0]['lr'] = lr
  optimizer.param_groups[1]['lr'] = lr
  return lr

# hyperparameters
image_size = 24
scale = 3
glimpse_size = 6

n_batches = 350
batch_size = 64

lr = 0.0001
std = 0.25

# Need to specialize some of the modules for catch
# Using an LSTM for recurrent core
# We don't have a supervised label for action.  
# Pure reinforcement learning based Loss for both Action and Location networks.

class CatchCore(nn.Module):
  def __init__(self):
    super(CatchCore, self).__init__()
    self.core = nn.LSTMCell(input_size = 256, hidden_size = 256)

  def initialize(self, B):
    self.cell = torch.zeros(B, 256).to(device)

  def forward(self, g):
    output = torch.zeros(self.cell.shape[0], 256).to(device)
    output, self.cell = self.core(g, (output, self.cell))
    return output

class CatchAction(nn.Module):
  def __init__(self):
    super(CatchAction, self).__init__()
    self.fc = nn.Linear(256,3)

  def forward(self,h,greedy):
    a_out = self.fc(h)
    a_prob = torch.softmax(a_out,1)
    pi = Categorical(a_prob)
    if greedy:
      a = pi.sample() # draw best
    else:
      a = torch.randint(0,3,(h.shape[0],)).to(device) # draw randomly
    logpi = pi.log_prob(a)
    return logpi, a-1

class CatchModel(nn.Module):
  def __init__(self,image_size,channel,glimpse_size,scale,std):
    super(CatchModel, self).__init__()
    self.glimps = Glimpse(image_size,channel,glimpse_size,scale)
    self.core   = CatchCore()
    self.location = Location(std)
    self.action = CatchAction()

  def initialize(self, B):
    self.core.initialize(B)
    self.l = (torch.rand((B,2))*2-1).to(device)

  def forward(self,x,greedy=True):
    g = self.glimps(x,self.l)
    state = self.core(g)
    logpi_l,self.l = self.location(state)
    logpi_a,a = self.action(state,greedy)
    return a,logpi_a,logpi_l


logstep = 0

class CatchLoss(nn.Module):
  def __init__(self, gamma):
    super(CatchLoss, self).__init__()
    self.baseline = nn.Parameter(torch.zeros(1,1).to(device), requires_grad = True)
    # learn weights to blend composite loss function 
    self.wa = nn.Parameter(torch.ones(1,1).to(device),requires_grad=True)
    self.wl = nn.Parameter(torch.ones(1,1).to(device),requires_grad=True)
    self.wb = nn.Parameter(torch.ones(1,1).to(device),requires_grad=True)

    self.gamma = gamma
    self.notinit = True

  def initialize(self, B):
    self.logpi_l = []
    self.logpi_a = []

  def forward(self,reward,logpi_a,logpi_l,done):
    global logstep
    self.logpi_l += [logpi_l]
    self.logpi_a += [logpi_a]
    if done:
      if self.notinit:
        self.baseline.data = reward.mean()
        self.notinit = False
      R = reward
      a_loss, l_loss, b_loss = 0, 0, 0
      R_b = (R-self.baseline.detach())
      for logpi_l, logpi_a in zip(reversed(self.logpi_l), reversed(self.logpi_a)):
        a_loss += - (logpi_a * R_b).sum()
        l_loss += - (logpi_l.sum(-1) * R_b).sum()
        R_b = self.gamma * R_b
      b_loss = ((self.baseline - R)**2).sum()

      combined_loss = a_loss*torch.exp(-self.wa)+self.wa+l_loss*torch.exp(-self.wl)+self.wl+b_loss*torch.exp(-self.wb)+self.wb
      writer.add_scalar('a_loss',a_loss,logstep)
      writer.add_scalar('l_loss',l_loss,logstep)
      writer.add_scalar('b_loss',b_loss,logstep)
      writer.add_scalar('loss',a_loss+l_loss+b_loss,logstep)
      writer.add_scalar('baseline',self.baseline.item(),logstep)
      writer.add_scalar('R',R.sum(),logstep)
      writer.add_scalar('R_mean',R.mean(),logstep)
      writer.add_scalar('wa',self.wa.item(),logstep)
      writer.add_scalar('wl',self.wl.item(),logstep)
      writer.add_scalar('wb',self.wb.item(),logstep)
      writer.add_scalar('combined_loss',combined_loss,logstep)
      logstep = logstep+1

      return a_loss,l_loss,b_loss,combined_loss
    else:
      return 0,0,0,0

# epsilon-greedy(ish)
# periodically sample random action to avoid getting stuck in a local minima
eps_max = 0.9
eps_min = 0.01
eps_epochs = 20 # epsilon greedy hyperparam
eps = lambda x: max(eps_min, eps_max - x*(eps_max-eps_min)/eps_epochs)

title = 'Catch!'

def train():
  epoch = 0
  model = CatchModel(image_size=image_size,channel=1,glimpse_size=glimpse_size,
          scale=scale,std=std).to(device)
  loss_fn = CatchLoss(gamma=0.9).to(device)
  optimizer = optim.Adam([{'params': model.parameters(), 'lr': lr}, {
                        'params': loss_fn.parameters(), 'lr': lr}])
  env = Catch(batch_size=batch_size, device=device)

  while True:
    epoch += 1
    #current_lr = adjust_learning_rate(optimizer, epoch, lr)
    current_lr = lr
    model.train()
    train_aloss, train_lloss, train_bloss, train_reward = 0, 0, 0, 0
    for batch_idx in range(n_batches):
      optimizer.zero_grad()
      model.initialize(batch_size)
      loss_fn.initialize(batch_size)
      done = 0
      saccades = []
      while(not done):
        data = env.getframe() # get a frame from the game

        # roll the dice to see if we should explore (non-greedy)
        greedy = random.random() >= eps(epoch)
        action, logpi_a, logpi_l = model(data,greedy)

        done, reward = env.step(action)
        aloss, lloss, bloss, combined_loss = loss_fn(reward, logpi_a, logpi_l, done)

        if args.render:
          ndata = data.cpu().numpy()[0].squeeze()
          ndata = np.stack((ndata,)*3,axis=-1) # opencv likes images to have 3 colors
          cv2.imshow('Catch!',ndata) # just render the first game in the batch
          cv2.waitKey(1)

      combined_loss.backward()
      optimizer.step()

      train_aloss += aloss.item()
      train_lloss += lloss.item()
      train_bloss += bloss.item()
      train_reward += reward.sum().item()

    avg_train_aloss = train_aloss / (n_batches*batch_size)
    avg_train_lloss = train_lloss / (n_batches*batch_size)
    avg_train_bloss = train_bloss / (n_batches*batch_size)
    avg_train_reward = train_reward * 100 / (n_batches*batch_size)
    print(f'=> Epoch: {epoch} Average loss: a {avg_train_aloss:.4f} l {avg_train_lloss:.4f} b {avg_train_bloss:.4f} Reward: {avg_train_reward:.1f} lr: {current_lr:.7f}')
    writer.add_scalar('reward',train_reward*100/(n_batches*batch_size),epoch)
    writer.add_scalar('eps:',eps(epoch),epoch)
    

    torch.save([model.state_dict(), loss_fn.state_dict(), optimizer.state_dict()],f'./chkpt/catch_{epoch}.pth')

def demo():
  model = CatchModel(image_size=image_size,channel=1,glimpse_size=glimpse_size,
          scale=scale,std=std).to(device)
  loss_fn = CatchLoss(gamma=1).to(device)
  optimizer = optim.Adam([{'params': model.parameters(), 'lr': lr}, {
                        'params': loss_fn.parameters(), 'lr': lr}])
  env = Catch(batch_size=1, device=device)

  # load the pretrained model for demonstration purposes
  m,l,o = torch.load('./chkpt/best_catch.pth',map_location=torch.device(device))
  model.load_state_dict(m)
  while True: # demos are forever
    model.eval()
    model.initialize(1)
    loss_fn.initialize(1)
    done = 0
    saccades = []
    while(not done):
      data = env.getframe()

      action, logpi_a, logpi_l = model(data,True) # demoing; so always take the best action
      done, reward = env.step(action)
      saccades.append(model.l[0].cpu().detach().numpy())
      status = ''
      status_color = (0,0,0)
      if reward > 0:
        status = 'Caught'
        status_color = (0,255,0)
        print(status)
      elif reward < 0:
        status = 'Missed'
        status_color = (255,0,0)
        print(status)
      cv2.imshow('Recurrent Visual Attention', visualize(title,data,saccades,glimpse_size,scale,status,status_color,zoom=8))
      cv2.waitKey(1 if reward == 0 else 1000)

if __name__ == "__main__":
  title = f'Catch!'
  print(title)

  if not args.demo:
    print('Training...')
    train()
  else:
    print('Demoing...')
    demo()
