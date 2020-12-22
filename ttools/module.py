import numbers
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Sequence
from typing import Tuple, List, Optional
from torch import Tensor


class GlobalAvgPool1d(nn.Module):
  def forward(self, inputs):
    return nn.functional.adaptive_avg_pool1d(inputs,1).view(inputs.size(0),-1)


class GlobalAvgPool2d(nn.Module):
  def forward(self, inputs):
    return nn.functional.adaptive_avg_pool2d(inputs,1).view(inputs.size(0),-1)


class GlobalMaxPool1d(nn.Module):
  def forward(self, inputs):
    return nn.functional.adaptive_max_pool1d(inputs,1).view(inputs.size(0),-1)


class GlobalMaxPool2d(nn.Module):
  def forward(self, inputs):
    return nn.functional.adaptive_max_pool2d(inputs,1).view(inputs.size(0),-1)


class Average(nn.ModuleList):
  def forward(self, inputs):
    output = []
    for module in self:
      output.append(module(inputs))
    output = torch.stack(output, -1)
    return output.mean(-1)


class Scan(nn.Module):
  def __init__(self, model, window=256, hop=128):
    super().__init__()
    self.model = model
    self.window = window
    self.hop = hop
    
  def forward(self, inputs):
    wdw = self.window
    hop = self.hop
    shp = inputs.shape[-1]
    
    output = []
    for i in  torch.arange(0, shp - wdw + hop + 1e-6, hop):
      x = inputs[...,int(i):int(i+wdw)]
      output.append(self.model(x))

    return torch.stack(output, -1)


class Residual(nn.Sequential):
  def __init__(self, transition, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.transition = transition
    
  def forward(self, x):
    if self.transition is not None:
      x = self.transition(x)
    return x + super().forward(x)


class ResConv2d(Residual):
  def __init__(self, dim):
    super().__init__(
      None, 
      nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), bias=False),
        
      nn.BatchNorm2d(dim), 
      nn.ReLU(),
       
      nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=False),
    )
    self.dim = dim

  def __repr__(self):
    return f'{self.__class__.__name__}({self.dim})'


class ChannelNorm(nn.Module):
  def forward(self, image):
    return channel_norm(image)

  def __repr__(self):
    return f'{self.__class__.__name__}()'


def channel_norm(image):
  img = image.flatten(2)
  avg = img.mean(-1)[:,:,None,None]
  var = img.var(-1)[:,:,None,None]
  return (image - avg) / torch.sqrt(var + 1e-6)


class ChannelScale(nn.Module):
  def forward(self, image):
    return channel_scale(image)
    
  def __repr__(self):
    return f'{self.__class__.__name__}()'


def channel_scale(image):
  img = image.flatten(2)
  vmin = img.min(-1)[0][:,:,None,None]
  vmax = img.max(-1)[0][:,:,None,None]
  return (image - vmin) / (vmax - vmin + 1e-6)


class Resize(torch.nn.Module):
  def __init__(self, size, mode='bilinear'):
    super().__init__()
    self.size = size
    self.mode = mode

  def forward(self, img):
    return F.interpolate(img, size=self.size, mode=self.mode, align_corners=False)

  def __repr__(self):
    return f'{self.__class__.__name__}(size={self.size}, interpolation={self.mode})'