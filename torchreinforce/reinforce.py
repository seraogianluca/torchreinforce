import torch
from functools import wraps

from .output import ReinforceOutput
from .distributions import *

class ReinforceModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ReinforceModule, self).__init__()
        self.gamma = kwargs["gamma"] if "gamma" in kwargs else 0.99
        self.distribution = kwargs["distribution"] if "distribution" in kwargs else Categorical
        
        if issubclass(self.distribution, torch.distributions.Distribution)\
        and not issubclass(self.distribution, ReinforceDistribution):
            self.distribution = getNonDeterministicWrapper(self.distribution)
        
        self.checkused = kwargs["checkused"] if "checkused" in kwargs else True
        self.history = []

    def forward(model_forward):
        @wraps(model_forward)
        def decorated(*args, **kwargs):
            self = args[0]
            if self.checkused: self._check_used()
            
            model_output = model_forward(*args, **kwargs)
            if type(model_output) != list: model_output = [model_output]
            dist = self.distribution(*model_output, deterministic=not self.training)
            output = ReinforceOutput(dist)

            if self.training: self.history.append(output)
            return output

        return decorated

    def _check_used(self):
        if len(self.history) != 0 and len(list(filter(lambda x: x.used, self.history))) != 0:
            raise Exception("One or more outputs seems to be already been used, you may want to reset() this module. If you know what you are doing you can ignore di this check by settig checkused to False")

    def loss(self, normalize=True):
        history = list(filter(lambda x: x.get_reward() is not None and x.action is not None, self.history))
        log_probs = torch.stack(list(map(lambda x: x._log_prob(), history)))
        rewards = list(map(lambda x: x.get_reward(), history))

        comulative = torch.tensor(0, dtype=torch.float32, device=log_probs.device)
        disconted_rewards = []
        for r in reversed(rewards):
            comulative = comulative*self.gamma + r
            disconted_rewards.append(comulative.unsqueeze(0))
        
        disconted_rewards = torch.tensor(disconted_rewards, device=log_probs.device)
        if normalize:
            disconted_rewards = (disconted_rewards - disconted_rewards.mean()) / disconted_rewards.std()
        
        loss = torch.mul(disconted_rewards, -1*log_probs)
        return loss.sum()


    def total_reward(self):
        history = list(filter(lambda x: x.get_reward() is not None and x.action is not None, self.history))
        rewards = list(map(lambda x: x.get_reward(), history))
        return sum(rewards)
    
    def reset(self):
        self.history = []
