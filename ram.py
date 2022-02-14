'''
RAM (c) John Robinson 2022
'''

import math
import numpy as np

import torch
import torch.utils.data
from torch import nn , optim
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class Retina(nn.Module):
  def __init__(self,image_size,width,scale):
    super(Retina, self).__init__()
    self.hw = int(width/2) # half width
    self.scale = int(scale)
    self.image_size = image_size

  def extract_patch_in_batch(self,x,l,scale):
    l = (self.image_size*(l+1)/2).type('torch.IntTensor')
    low = l
    high = l + 2*(2**(scale-1))*self.hw
    patch = []
    for b in range(x.size(0)):
      patch += [x[b:b+1,:,low[b,0]:high[b,0],low[b,1]:high[b,1]]]
    return torch.cat(patch,0)

  def forward(self,x,l,view=False):
    B,C,H,W = x.size()
    padsz = (2**(self.scale-1))*self.hw
    x_pad = F.pad(x,(padsz,padsz,padsz,padsz),"constant" if view else "replicate")
    patch = self.extract_patch_in_batch(x_pad,l,self.scale)

    #out = [F.interpolate(patch, size=2*self.hw, mode='bilinear', align_corners = True)]
    out = [F.max_pool2d(patch, kernel_size=2**(self.scale-1))]
    cntr = int(patch.size(2)/2)
    halfsz = cntr
    for s in range(self.scale-1,0,-1):
      halfsz = int(halfsz/2)
      #out += [F.interpolate(patch[:,:,cntr-halfsz:cntr+halfsz,cntr-halfsz:cntr+halfsz], size=2*self.hw, mode='bilinear', align_corners = True)]
      out += [F.max_pool2d(patch[:,:,cntr-halfsz:cntr+halfsz,cntr-halfsz:cntr+halfsz], kernel_size=2**(s-1))]
    out = torch.cat(out,1)
    return out

class Glimpse(nn.Module):
  def __init__(self,image_size,channel,glimpse_size,scale):
    super(Glimpse,self).__init__()
    self.image_size = image_size
    self.ro    = Retina(image_size,glimpse_size,scale)
    self.fc_ro = nn.Linear(scale*(glimpse_size**2)*channel,128)
    self.fc_lc = nn.Linear(2, 128)
    self.fc_hg = nn.Linear(128,256)
    self.fc_hl = nn.Linear(128,256)

  def forward(self,x,l):
    ro = self.ro(x,l).view(x.size(0),-1)
    hg = F.relu(self.fc_ro(ro))
    hl = F.relu(self.fc_lc(l))
    g  = F.relu(self.fc_hg(hg)+self.fc_hl(hl))
    return g

class Location(nn.Module):
  def __init__(self,std):
    super(Location,self).__init__()
    self.std = std
    self.fc = nn.Linear(256,2)

  def forward(self,h):
    l_mu = self.fc(h)
    pi = Normal(l_mu,self.std)
    l = pi.sample()
    logpi = pi.log_prob(l)
    l = torch.tanh(l)
    return logpi,l

class Core(nn.Module):
  def __init__(self):
    super(Core, self).__init__()
    self.fc_h = nn.Linear(256,256)
    self.fc_g = nn.Linear(256,256)

  def forward(self, h, g):
    return F.relu(self.fc_h(h) + self.fc_g(g))

class Action(nn.Module):
  def __init__(self):
    super(Action, self).__init__()
    self.fc = nn.Linear(256,10)

  def forward(self, h):
    return self.fc(h)  # Do not apply softmax as loss function will take care of it

class Model(nn.Module):
  def __init__(self,image_size,channel,glimpse_size,scale,std):
    super(Model, self).__init__()
    self.glimpse = Glimpse(image_size,channel,glimpse_size,scale)
    self.core   = Core()
    self.location = Location(std)
    self.action = Action()

  def initialize(self,B,device):
    self.state = torch.zeros(B,256).to(device)
    self.l = (torch.rand((B,2))*2-1).to(device)

  def forward(self,x):
    g = self.glimpse(x,self.l)
    self.state = self.core(self.state,g)
    logpi_l, self.l = self.location(self.state)
    a = self.action(self.state)
    return a,logpi_l

class Loss(nn.Module):
  def __init__(self,T,gamma,device):
    super(Loss, self).__init__()
    self.baseline = nn.Parameter(0.1*torch.ones(1,1).to(device),requires_grad=True)
    self.gamma = gamma
    self.T = T

  def initialize(self,B):
    self.t = 0
    self.logpi_l = []

  def compute_reward(self,recon_a,a):
    return (torch.argmax(recon_a.detach(),1)==a).float()

  def forward(self,recon_a,a,logpi_l):
    self.t += 1
    self.logpi_l += [logpi_l]
    if self.t==self.T:                                    # final glimpse
      R = self.compute_reward(recon_a,a)
      a_loss = F.cross_entropy(recon_a,a,reduction='sum') # self-supervised class using cross-entropy
      l_loss = 0
      R_b = (R - self.baseline.detach())
      for logpi in reversed(self.logpi_l):
        l_loss += - (logpi.sum(-1) * R_b).sum()
        R_b = self.gamma * R_b
      b_loss = ((self.baseline - R)**2).sum()

      return a_loss,l_loss,b_loss,R.sum()
    else:
      return None,None,None,None

