from sys import stderr


class Best:
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