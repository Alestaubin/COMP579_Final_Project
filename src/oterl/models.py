import torch.nn as nn
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model


# Define models (categorical for discrete actions)
class Policy(CategoricalMixin, Model):
  def __init__(self, observation_space, action_space, device, clip_actions=False):
    Model.__init__(self, observation_space, action_space, device)
    CategoricalMixin.__init__(self, clip_actions)

    self.net = nn.Sequential(
      nn.Linear(self.num_observations, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, self.num_actions),
    )

  def compute(self, inputs, role):
    return self.net(inputs['states']), {}


class Value(DeterministicMixin, Model):
  def __init__(self, observation_space, action_space, device, clip_actions=False):
    Model.__init__(self, observation_space, action_space, device)
    DeterministicMixin.__init__(self, clip_actions)

    self.net = nn.Sequential(
      nn.Linear(self.num_observations, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
    )

  def compute(self, inputs, role):
    return self.net(inputs['states']), {}
