'''
RAM (c) John Robinson 2022
'''

import numpy as np
import cv2
import PIL

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import torch

from ram import Retina

pltfont = font_manager.FontProperties(family='sans-serif', weight='normal')
fontfile = font_manager.findfont(pltfont)
font = ImageFont.truetype(fontfile, 25)
fontsm = ImageFont.truetype(fontfile, 15)

# I apologize for the mess that lies therein
def visualize(title,x1,saccades,glimpse_size,scale,status,status_color=(0,0,0),zoom=4):
  current = saccades[-1]

  tl = torch.tensor(np.array([saccades[-1]]))

  x = np.uint8(x1.cpu().numpy()[0][0]*255.)
  orig_size = x.shape[0]

  r = Retina(orig_size,glimpse_size,scale)
  glimpses = r.forward(x1,tl,True)[0].cpu().detach().numpy()

  colors=((0,0,255),(0,255,0),(255,0,0)) # max scale of 3

  x2 = np.stack((x,)*3,axis=-1) # stack into RGB color planes
  x2 = cv2.resize(x2,(orig_size*zoom,orig_size*zoom),interpolation=cv2.INTER_NEAREST)

  current = saccades[-1]
  current = (current+1.)/2.
  current = np.array(current)*(x2.shape[0])

  x3 = np.zeros((orig_size,orig_size))
  g = (255-64)//len(saccades)
  for i,s in enumerate(saccades):
    p = (s+1.)/2.
    p = np.array(p) * orig_size
    p = p.astype(np.uint8)

    x3[p[0],p[1]] = 255 - (g*(len(saccades)-i-1))
  x3 = cv2.resize(x3,(orig_size*zoom,orig_size*zoom),interpolation=cv2.INTER_NEAREST)

  xpad = 24*2
  ypad = 24*3
  xoffset = xpad
  yoffset = ypad
  viewwidth = orig_size*zoom
  viewheight = orig_size*zoom
  totalwidth = (xoffset + viewwidth)*3 + xoffset
  totalheight = yoffset + viewheight + yoffset+ glimpse_size*zoom
  b = PIL.Image.new('RGB',(totalwidth,totalheight),color=(255,255,255))
  xb = PIL.Image.fromarray(x2,)
  b.paste(xb,(xoffset,yoffset))
  xb = PIL.Image.fromarray(x3,)
  xoffset = xoffset+viewwidth+xpad
  b.paste(xb,(xoffset,yoffset))

  xoffset = xpad+4
  yoffset = ypad+viewheight+58

  # draw individual glimpses
  for s in range(scale):
    gm = glimpses[s]
    g = np.stack((gm,)*3,axis=-1)
    g = cv2.resize(g,(gm.shape[0]*zoom,gm.shape[0]*zoom),interpolation=cv2.INTER_NEAREST)
    xb = PIL.Image.fromarray(np.uint8(g*255.),'RGB')
    b.paste(xb,(xoffset,yoffset))
    xoffset += g.shape[0] + 10

  ctx = PIL.ImageDraw.Draw(b)

  ctx.text((xpad,6),title,font=font,fill=(0,0,0))
  ctx.text((xpad+viewwidth+xpad+viewwidth+xpad//2,ypad+viewheight//2),status,font=font,fill=status_color)
  ctx.text((xpad,ypad-24),'Visualization',font=fontsm,fill=(0,0,0))
  ctx.text((xpad+viewwidth+xpad,ypad-24),'Glimpse History',font=fontsm,fill=(0,0,0))
  ctx.text((xpad,ypad+viewheight+30),'Individual Glimpses',font=fontsm,fill=(0,0,0))

  xoffset = xpad+4
  yoffset = ypad+viewheight+58-5
  # draw colored boxes around individual glimpses
  for s in range(scale):
    color = colors[scale-np.clip(s,0,len(colors))-1]
    sz = glimpses[0].shape[0]*zoom
    ctx.rectangle(((xoffset-4,yoffset),(xoffset+sz+4,yoffset+sz+8)),width=3,outline=color)
    xoffset += g.shape[0] + 10

  xoffset = xpad
  yoffset = ypad
  posx = xoffset+current[1]
  posy = yoffset+current[0]
  # draw retina
  for s in range(scale):
    color = colors[np.clip(s,0,len(colors))]
    sc = 2**s
    hw = (glimpse_size*sc//2)*zoom
    ctx.rectangle(((posx-hw,posy-hw),(posx+hw,posy+hw)),width=1,outline=color)

  bgr = cv2.cvtColor(np.array(b),cv2.COLOR_RGB2BGR)
  return bgr
