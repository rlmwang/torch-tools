import numpy as np


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