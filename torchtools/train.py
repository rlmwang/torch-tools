from copy import copy
from sys import stderr
from typing import Union
from collections import defaultdict
from time import time
from tqdm import tqdm
from torch import no_grad


def validate(metrics, scores, labels):
  """Validate model scores on a dictionary of metrics.
  """
  result = {}
  with no_grad():
    scores = scores.detach()
    for key in metrics:
      try:
        result[key] = metrics[key](scores, labels).item()
      except:
        result[key] = metrics[key](scores.cpu(), labels.cpu())
  return result


class Checkpoint:
  def __init__(self, phase, metric, file=stderr):
    super().__init__()
    self.phase = phase
    self.metric = metric
    self.file = file
    self.best = None
    self.state_dict = None

  def step(self, epoch, model):
    loss = epoch[self.phase][self.metric][-1]
    if self.best is None or loss < self.best:
      self.best = loss
      self.state_dict = model.state_dict()
      self.file.write('Checkpoint.\n')

  def load_state_dict(self, model):
    model.load_state_dict(self.state_dict)

  def reset(self):
    self.best = None
    self.state_dict = None

  def __repr__(self):
    return f'Best {self.phase} {self.metric}: {self.best:.4f}'


class Score(object):
  def __init__(self):
    self._values = []
    self._counts = []
  
  @property
  def values(self):
    return [v / c for v, c in zip(self._values, self._counts)]
    
  @property
  def counts(self):
    return self._counts

  def __len__(self):
    return len(self._values)

  def __getitem__(self, key):
    return self._values[key] / self._counts[key]

  def __setitem__(self, key, pair):
    value, count = pair
    self._values[key] = count * value
    self._counts[key] = count

  def __delitem__(self, key):
    del self._values[key]
    del self._counts[key]

  def __repr__(self):
    return repr(self.values)

  def append(self, value, count):
    self._values.append(count * value)
    self._counts.append(count)

  def reduce_(self):
    self._values = [sum(self._values)]
    self._counts = [sum(self._counts)]

  def argmin(self):
    s = self.values
    return s.index(min(s))

  def argmax(self):
    s = self.values
    return s.index(max(s))


class ScoreDict(defaultdict):
  def __init__(self):
    super().__init__(Score)


class epochs(object):
  def __init__(self, *args, file=stderr):

    assert len(args) <= 2
    if len(args) == 1:
      start = 0
      stop = args[0]
    elif len(args) == 2:
      start, stop = args

    self.start = start
    self.stop = stop
    self.curr = start - 1

    self.score = defaultdict(ScoreDict)

    self.since = time()
    self.file = file

  def __int__(self):
    return self.curr

  def __repr__(self):
    return f'Epoch {self.curr}/{self.stop-1}'
 
  def __iter__(self):
    return self

  def __next__(self):
    self.curr += 1

    self.file.write(f'Epoch {self.curr}/{self.stop-1}\n')
    
    if self.curr < self.stop:
      return self
    
    elapsed = time() - self.since
    m, s = elapsed // 60, elapsed % 60

    loss = self.score['train']['loss']
    index = loss.argmin()

    self.file.write('\n')
    self.file.write(f'Training complete in {m:.0f}m {s:.0f}s\n')
    self.file.write(f'Best train loss: {loss[index]:.4f} in epoch {index}')

    raise StopIteration

  def log(self, score, phase):
    for key in score:
      self.score[phase][key].append(score[key], 1)


class steps(object):
  def __init__(self, dataloader, epoch=None, phase=None, file=stderr, **tqdm_kwargs):
    lbar = '{desc}: {n_fmt}/{total_fmt} |'
    mbar = '{bar}'
    rbar = '| {elapsed_s:.0f}s,{rate_fmt}{postfix}'
    desc = phase.capitalize()

    self.epoch = epoch
    self.phase = phase
    self.score = ScoreDict()
    self.iterable = tqdm(dataloader, file=file, unit='step',
        bar_format=f'{lbar}{mbar}{rbar}', desc=desc, **tqdm_kwargs)
    
  def __len__(self):
    return len(self.iterable)

  def __iter__(self):
    self.iterator = iter(self.iterable)
    return self

  def __next__(self):
    for key in self.score:
      self.score[key].reduce_()

    score = {key: self.score[key][0] for key in self.score}
    self.iterable.set_postfix(score)

    try:
      return self, next(self.iterator)
    except StopIteration:
      if self.epoch is not None:
        self.epoch.log(score, self.phase)
      raise StopIteration

  def log(self, score, count):
    if not isinstance(score, dict):
      score = {None: score}
    for key in score:
      self.score[key].append(score[key], count)

