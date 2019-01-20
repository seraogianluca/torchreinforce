import torch
from .base import ReinforceDistribution

class Categorical(ReinforceDistribution, torch.distributions.Categorical):
    def __init__(self, probs, **kwargs):
        self.deterministic = kwargs.get("deterministic", False)
        self.probs = probs
        if "deterministic" in kwargs: del kwargs["deterministic"]
        torch.distributions.Categorical.__init__(self, probs, **kwargs)

    def sample(self):
        if self.deterministic:
            return self.probs.max(0)[1]
        else:
            return torch.distributions.Categorical.sample(self)
