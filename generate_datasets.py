# Generate preprocessed image/label datasets

from pathlib import Path
from random import shuffle
from termios import VMIN

import numpy as np
#from matplotlib import pyplot as plt

import torch
from torchvision import datasets,transforms


Path("./prepped_mnist").mkdir(parents=True, exist_ok=True)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vmin = 0
vmax = 0

def makeCentered():
  global vmin,vmax
  print('Downloading MNIST Dataset')
  train = datasets.MNIST('./mnist', train=True, download=True,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),]))
  test = datasets.MNIST('./mnist', train=False, download=True,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),]))
  print('train data len: {}'.format(len(train)))
  print('test data len: {}'.format(len(test)))

  # serialize dataset tensor to disk
  print('Creating Centered MNIST Dataset')
  torch.save(train,'./prepped_mnist/centered_train.dat')
  torch.save(test,'./prepped_mnist/centered_test.dat')

  # summary stats
  t = torch.cat([x for (x,y) in train])
  vmin = torch.min(t)
  vmax = torch.max(t)
  print('vmin:',vmin)
  print('vmax:',vmax)

# Assumes centered dataset is on disk
def makeTranslated():
  print('Generating Translated MNIST Dataset')
  im_sz = 60

  def translate_img(x,to_sz):
    C,H,W = x.size()
    x_t = -torch.ones(C,to_sz,to_sz) # background of MNIST is mapped to -1
    torch.fill_(x_t,vmin)
    loch = np.random.randint(0,33)
    locw = np.random.randint(0,33)
    x_t[:,loch:loch+H,locw:locw+W] = x
    return x_t

  train = torch.load('./prepped_mnist/centered_train.dat')
  test = torch.load('./prepped_mnist/centered_test.dat')

  new_train = []
  for i in range(len(train)):
    new_train.append( (translate_img(train[i][0],im_sz),train[i][1]) )

  new_test = []
  for i in range(len(test)):
    new_test.append( (translate_img(test[i][0],im_sz),test[i][1]) )

  torch.save(new_train,'./prepped_mnist/translated_train.dat')
  torch.save(new_test,'./prepped_mnist/translated_test.dat')

# Assumes centered dataset is on disk
def makeCluttered():
  print('Generating Cluttered MNIST Dataset')
  num_clutter = 4
  clutter_sz = 8
  im_sz = 60

  def clutter_img(x, N_clutter, clutter_sz, to_sz):
    C,H,W = x.size()
    clutter_patches = []
    ind = H-clutter_sz+1

    for _ in range(N_clutter):
      [r,c] = np.random.randint(0,ind,2)
      clutter_patches += [x[:,r:r+clutter_sz,c:c+clutter_sz]]
    shuffle(clutter_patches)
    x_t = -torch.ones(C,to_sz,to_sz) # background of MNIST is mapped to -1
    torch.fill_(x_t,vmin)

    ind = to_sz-H+1
    ind_ = to_sz-clutter_sz+1

    [loch, locw] = np.random.randint(0,ind,2)
    x_t[:,loch:loch+H,locw:locw+W] = x
    for _ in range(N_clutter):
      [r,c] = np.random.randint(0,ind_,2)
      x_t[:,r:r+clutter_sz,c:c+clutter_sz] = torch.max(x_t[:,r:r+clutter_sz,c:c+clutter_sz], clutter_patches.pop())

    return x_t

  train = torch.load('./prepped_mnist/centered_train.dat')
  test = torch.load('./prepped_mnist/centered_test.dat')

  new_train = []
  for i in range(len(train)):
    new_train.append( (clutter_img(train[i][0],num_clutter,clutter_sz,im_sz),train[i][1]) )

  new_test = []
  for i in range(len(test)):
    new_test.append( (clutter_img(test[i][0],num_clutter,clutter_sz,im_sz),test[i][1]) )

  torch.save(new_train,'./prepped_mnist/cluttered_train.dat')
  torch.save(new_test,'./prepped_mnist/cluttered_test.dat')

if __name__ == "__main__":
  makeCentered()
  makeTranslated()
  makeCluttered()
  print('Done.')

