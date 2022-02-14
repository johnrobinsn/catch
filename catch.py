# Simple 'toy' catch game
import torch

class Catch():
  def __init__(self, grid_size=24, batch_size=128, device=torch.device('cpu')):
    self.grid_size = grid_size
    self.batch_size = batch_size
    self.device = device

    self.ball_x = None
    self.ball_y = None
    self.angle = None

    self.paddle_y = None

    self.reset()

  def reset(self):
    self.ball_x   = torch.randint(0,self.grid_size,(self.batch_size,)).to(self.device)
    self.ball_y   = 0
    self.paddle_y = torch.randint(0,self.grid_size-1,(self.batch_size,)).to(self.device)
    self.angle    = torch.deg2rad(torch.rand(self.batch_size)*90 + 45).to(self.device)
    self.dx       = (torch.cos(self.angle)/torch.cos(torch.deg2rad(torch.tensor([45]).to(self.device))))

  def getframe(self):
    f = torch.zeros(self.batch_size, 1, self.grid_size, self.grid_size).to(self.device)
    ball_x = self.ball_x.clamp(0,self.grid_size-1).long()
    for i in range(self.batch_size):
      f[i,:,self.ball_y,ball_x[i]] = 1
      f[i,:,-1,self.paddle_y[i]:self.paddle_y[i]+2] = 1
    return f

  def step(self,action):
    self.ball_y += 1
    self.ball_x = self.ball_x + self.dx

    out = ((self.ball_x<0) | (self.ball_x>=self.grid_size))
    self.dx = self.dx * torch.index_select(torch.tensor([1,-1]).to(self.device), 0, out.long())

    self.paddle_y = (self.paddle_y + action.long()).clamp(0,self.grid_size-2)

    done = self.ball_y == self.grid_size-1
    reward = torch.zeros(self.batch_size,).to(self.device)
    if done:
      ball_x = self.ball_x.clamp(0,self.grid_size-1).long()
      reward = ((ball_x >= self.paddle_y) & (ball_x <= self.paddle_y+1)).float()
      reward = reward * 2 - 1
      self.reset()

    return done, reward

