import torch
import numpy as np
from time import time
from tqdm import tqdm


def set_parameter_requires_grad(model, value):
  for param in model.parameters():
    param.requires_grad = value


def evaluate(metrics, output, labels):
  """
  Default model evaluation.
  """
  scores = {}
    
  with torch.no_grad():
    output = output.detach()
    
    for key in metrics:
      try:
        scores[key] = metrics[key](output, labels).item()
      except:
        scores[key] = metrics[key](output.cpu(), labels.cpu())

  return scores


class Checkpoint:
  def __init__(self, phase, metric):
    super().__init__()
    self.phase = phase
    self.metric = metric
    self.best = None
    self.state_dict = None

  def step(self, epoch, model):
    loss = epoch['score'][self.phase][self.metric][-1]

    if self.best is None or loss < self.best:
      self.state_dict = model.state_dict()
      self.best = loss

      print('Checkpoint.', flush=True)

  def load_state_dict(self, model):
    model.load_state_dict(self.state_dict)

  def reset(self):
    self.best = None
    self.state_dict = None

  def __repr__(self):
    return f'Best {self.phase} {self.metric}: {self.best:.4f}'


def epochs(start, stop):
  since = time()

  epoch = {
    'start': start,
    'stop': stop,
    'curr': None,
    'score': {},
  }

  for epoch['curr'] in range(start, stop):
    
    yield epoch

    for phase in epoch['score']:
      s = epoch['score'][phase]
      score_str = [f'{m}: {s[m][-1]:.4f}' for m in s]
      score_str = ', '.join(score_str).strip(', ')
    
      print(f'{phase} - {score_str}', flush=True)
  
  elapsed = time() - since
  m, s = elapsed // 60, elapsed % 60

  print(flush=True)
  print(f'Training complete in {m:.0f}m {s:.0f}s')


def steps(epoch, dataloader, with_tqdm=True):
  desc = f'Epoch {epoch["curr"] + 1}/{epoch["stop"]}'
  
  count_agg, score_agg, score = 0, {}, {}
  loader = tqdm(dataloader, desc=desc) if with_tqdm else dataloader
  
  for data in loader:
    try:
      count = data.shape[0]
    except:
      count = data[0].shape[0]
    count_agg += count
    
    yield score, data
    
    for p in score:
      if p not in score_agg:
        score_agg[p] = {}
      for m in score[p]:
        try:
          s = score[p][m].item()
        except:
          s = score[p][m]
        if m not in score_agg[p]:
          score_agg[p][m] = s
        else:
          score_agg[p][m] += s * count

  for p in score_agg:
    if p not in epoch['score']:
      epoch['score'][p] = {}
    for m in score_agg[p]:
      s = score_agg[p][m] / count_agg
      if m not in epoch['score'][p]:
        epoch['score'][p][m] = [s]
      else:
        epoch['score'][p][m].append(s)

    
def folds(splits):
  for k, split in enumerate(splits):
    print(12 * '-')
    print(f'  Fold {k:02d}:')
    print(12 * '-')

    yield k, split

