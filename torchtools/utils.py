import numpy as np


def set_parameter_requires_grad(model, value):
  for param in model.parameters():
    param.requires_grad = value


class no_grad:
  def __init__(self, module):
    super().__init__()
    self.module = module
    self.prev = []

  def __enter__(self):
    self.prev = [p.requires_grad for p in self.module.parameters()]
    for p in self.module.parameters():
      p.requires_grad = False
    
  def __exit__(self, type, value, traceback):
    params = self.module.parameters()
    for p, v in zip(params, self.prev):
      p.requires_grad = v


def CosineWithStartup(lr_min, lr_max, ep_start, ep_last):
  def fn(ep):
    return cosine_with_startup(ep, lr_min, lr_max, ep_start, ep_last)
  return fn


def cosine_with_startup(ep, lr_min, lr_max, ep_start, ep_last):
  if ep <= ep_start:
    return lr_max / ep_start * ep
  elif ep > ep_last:
    return lr_min
  else:
    lr_delta = lr_max - lr_min
    ep_delta = ep_last - ep_start
    return lr_min + 1/2 * lr_delta * (1 + np.cos((ep - ep_start) * np.pi / ep_delta))