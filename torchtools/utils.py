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