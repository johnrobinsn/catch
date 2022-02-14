'''
RAM (c) John Robinson 2022
'''

import argparse
import numpy as np

from torch import optim
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.tensorboard import SummaryWriter

from ram import *
from ram_visualize import *

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
parser.add_argument('--demo', action='store_true', default=False,
                    help='demo catch from best checkpoint')

parser.add_argument('--dataset', type=str, default='centered',
  help='dataset to use "centered" | "translated" | "cluttered"')

args = parser.parse_args()

# Writer will output to ./runs/ directory by default
writer = SummaryWriter(comment=f'MNIST {args.dataset}')

batch_size = 128

def adjust_learning_rate(optimizer, epoch, lr, decay_rate):
    lr = lr * (decay_rate ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class RecurrentAttention:
  def __init__(self,T,lr,scale,decay,image_size,glimpse_size,dataset):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 0} if self.device.type=='cuda' else {}
    self.dataset = dataset

    self.train_loader = torch.utils.data.DataLoader(torch.load('./prepped_mnist/{}_train.dat'.format(self.dataset)),
                                              batch_size=batch_size, shuffle=True, **kwargs)
    self.test_loader = torch.utils.data.DataLoader(torch.load('./prepped_mnist/{}_test.dat'.format(self.dataset)),
                                              batch_size=batch_size, shuffle=True, **kwargs)

    self.T = T
    self.lr = lr
    self.std = 0.25
    self.scale = scale
    self.decay = decay
    self.image_size = image_size
    self.glimpse_size = glimpse_size
    self.model = Model(im_sz=self.image_size,channel=1,glimpse_size=self.glimpse_size,scale=self.scale,std = self.std).to(self.device)
    self.loss_fn = Loss(T=self.T,gamma=1,device=self.device).to(self.device)
    self.optimizer = optim.Adam(list(self.model.parameters())+list(self.loss_fn.parameters()), lr=self.lr)
    self.epoch = 0

  def load(self,fn):
    # load checkpoint.
    m,l,o = torch.load(f'./chkpt/{fn}',map_location=torch.device(self.device))
    self.model.load_state_dict(m)
    self.loss_fn.load_state_dict(l)
    self.optimizer.load_state_dict(o)

  def load_epoch(self,epoch):
    # load checkpoint.  Skip this if you want to train from scratch
    self.epoch=epoch
    self.load(f'{self.dataset}_{epoch}.pth')

  def save(self):
    torch.save([self.model.state_dict(),self.loss_fn.state_dict(),self.optimizer.state_dict()],'./chkpt/{}_'.format(self.dataset)+str(self.epoch)+'.pth')

  def train(self, num_epochs):
    # train.  Skip this if you just want to use a pretrained model from above
    for self.epoch in range(self.epoch+1,self.epoch+num_epochs+1):
      '''
      Training
      '''
      adjust_learning_rate(self.optimizer, self.epoch, self.lr, self.decay)
      self.model.train()
      train_aloss, train_lloss, train_bloss, train_reward = 0, 0, 0, 0
      for batch_idx, (data, label) in enumerate(self.train_loader):
        data = data.to(self.device)
        label = label.to(self.device)
        self.optimizer.zero_grad()
        self.model.initialize(data.size(0), self.device)
        self.loss_fn.initialize(data.size(0))
        for _ in range(self.T):
          action,logpi = self.model(data)
          aloss,lloss,bloss,reward = self.loss_fn(action,label,logpi)
        loss = aloss+lloss+bloss
        loss.backward()
        self.optimizer.step()
        train_aloss += aloss.item()
        train_lloss += lloss.item()
        train_bloss += bloss.item()
        train_reward += reward.item()

      avg_train_aloss = train_aloss / len(self.train_loader.dataset)
      avg_train_lloss = train_lloss / len(self.train_loader.dataset)
      avg_train_bloss = train_bloss / len(self.train_loader.dataset)
      avg_train_reward = train_reward * 100 / len(self.train_loader.dataset)

      print(f'Train({self.dataset})> Epoch: {self.epoch} Average loss: a {avg_train_aloss:.4f} l {avg_train_lloss:.4f} b {avg_train_bloss:.4f} Reward: {avg_train_reward:.1f}')

      self.save() # save the model

      '''
      See how we're doing against the validation set
      '''
      self.model.eval()
      test_aloss, test_lloss, test_bloss, test_reward = 0, 0, 0, 0
      for batch_idx, (data, label) in enumerate(self.test_loader):
        data = data.to(self.device)
        label = label.to(self.device)
        self.model.initialize(data.size(0), self.device)
        self.loss_fn.initialize(data.size(0))
        for _ in range(self.T):
            action,logpi = self.model(data)
            aloss, lloss, bloss, reward = self.loss_fn(action, label, logpi)
        loss = aloss+lloss+bloss
        test_aloss += aloss.item()
        test_lloss += lloss.item()
        test_bloss += bloss.item()
        test_reward += reward.item()

      avg_test_aloss = test_aloss / len(self.test_loader.dataset)
      avg_test_lloss = test_lloss / len(self.test_loader.dataset)
      avg_test_bloss = test_bloss / len(self.test_loader.dataset)
      avg_test_reward = test_reward * 100 / len(self.test_loader.dataset)

      print(f'Test({self.dataset})> Epoch: {self.epoch} Average loss: a {avg_test_aloss:.4f} l {avg_test_lloss:.4f} b {avg_test_bloss:.4f} Reward: {avg_test_reward:.1f}')

      writer.add_scalar('avg_test_aloss',avg_test_aloss,self.epoch)
      writer.add_scalar('avg_test_lloss',avg_test_lloss,self.epoch)
      writer.add_scalar('avg_test_bloss',avg_test_bloss,self.epoch)
      writer.add_scalar('avg_test_reward',avg_test_reward,self.epoch)

  # returns a single random mnist sample image (grayscale) and it's associated label
  def getRandomSample(self):
    sample = next(iter(self.test_loader))
    return sample[0][0][0],sample[1][0].item()

  def eval(self,image,num=None):
    if not num:
      num = self.T
    self.model.eval()
    #test_aloss, test_lloss, test_bloss, test_reward = 0, 0, 0, 0
    data = torch.unsqueeze(torch.unsqueeze(image,0),0) # put the single image into a batch of one
    data = data.to(self.device)
    self.model.initialize(1, self.device)
    final_action = None
    saccades = [self.model.l[0].cpu().detach().numpy()]
    #print('ll:', self.model.l)
    for _ in range(num-1):
      action,logpi = self.model(data)
      #final_action = np.asscalar(action.argmax().cpu().detach().numpy())
      final_action = action.argmax().cpu().detach().numpy().item()
      #saccades.append(logpi[0].cpu().detach().numpy())
      #print('logpi:',logpi)
      saccades.append(self.model.l[0].cpu().detach().numpy())
    return final_action,saccades

  def demo(self):
    self.model.eval()
    self.load(f'best_{self.dataset}.pth')
    while True: # run forever
      for batch_idx, (data, label) in enumerate(self.test_loader):
        saccades = []
        data = data.to(self.device)[0].unsqueeze(0)
        label = label.to(self.device)[0].unsqueeze(0)
        self.model.initialize(data.size(0), self.device)
        self.loss_fn.initialize(data.size(0))
        vdata = (data - torch.min(data))/ (torch.max(data)-torch.min(data))
        for _ in range(self.T):
          action,logpi = self.model(data)
          final_action = action.argmax().cpu().detach().numpy().item()
          saccades.append(self.model.l[0].cpu().detach().numpy())
          vdata = (data - torch.min(data))/ (torch.max(data)-torch.min(data))
          cv2.imshow('Recurrent Visual Attention', visualize(title,vdata,saccades,self.glimpse_size,scale,''))
          cv2.waitKey(200)
        if final_action == label:
          status = f'Correct: {final_action}'
          status_color=(0,255,0)
        else:
          status = f'Predicted {final_action} was {label[0]}'
          status_color=(255,0,0)
        print(status)
        cv2.imshow('Recurrent Visual Attention', visualize(title,vdata,saccades,self.glimpse_size,scale,status,status_color))
        cv2.waitKey(1000)


if __name__ == "__main__":
  title = f'MNIST {args.dataset} dataset'
  print(title)

  # Configure hyperparameters for different datasets
  if args.dataset == 'centered':
    T = 7
    lr = 0.001
    decay = 0.95                    # learning rate decay
    im_sz = 28                      # input image size
    scale = 1                       # number of times to scale glimpses
    glimpse_size = 8               # glimpse size
    num_epochs = 100
  elif args.dataset == 'translated':
    T = 4
    lr = 0.0001
    decay = 0.975
    im_sz = 60
    scale = 3
    glimpse_size = 12
    num_epochs = 200
  elif args.dataset == 'cluttered':
    T = 4
    lr = 0.0001
    decay = 0.975
    im_sz = 60
    scale = 3
    glimpse_size = 12
    num_epochs = 200
  else:
    print('Unsupported dataset:', args.dataset)
    parser.print_help()
    exit()

  ra = RecurrentAttention(T,lr,scale,decay,im_sz,glimpse_size,args.dataset)
  if args.demo:
    print('Running Demo...')
    ra.demo()
  else:
    print('Training...')
    ra.train(num_epochs)

