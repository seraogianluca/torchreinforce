import torch
from functools import wraps



class ReinforceOutput:
    def __init__(self, distribution):
        self.distribution = distribution
        self.action = None
        self._reward = None
        self.used = False

    def get(self):
        self.action = self.distribution.sample()
        return self.action.item()
    
    def reward(self, reward):
        self._reward = reward
    
    def get_reward(self):
        return self._reward
    
    def _log_prob(self):
        self.used = True
        return -self.distribution.log_prob(self.action).unsqueeze(0)



class ReinforceModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ReinforceModule, self).__init__(**kwargs)
        self.gamma = kwargs["gamma"] if "gamma" in kwargs else 0.99
        self.distribution = kwargs["distribution"] if "distribution" in kwargs else torch.distributions.Categorical 
        self.checkused = kwargs["checkused"] if "checkused" in kwargs else True
        self.history = []

    def forward(model_forward):
        @wraps(model_forward)
        def decorated(*args, **kwargs):
            self = args[0]
            if self.checkused: self._check_used()
            
            model_output = model_forward(*args, **kwargs)
            c = self.distribution(model_output)
            output = ReinforceOutput(c)
            self.history.append(output)

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
        
        loss = torch.mul(disconted_rewards, log_probs)
        return loss.sum()


    def total_reward(self):
        history = list(filter(lambda x: x.get_reward() is not None and x.action is not None, self.history))
        rewards = list(map(lambda x: x.get_reward(), history))
        return sum(rewards)
    
    def reset(self):
        self.history = []



class Test(ReinforceModule):
    def __init__(self):
        super(Test, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Sigmoid()
        )
    
    @ReinforceModule.forward
    def forward(self, x):
        return self.net(x)

'''
a = Test().to(torch.device("cuda"))
b = a(torch.randn(2).to(torch.device("cuda")))
c = a(torch.randn(2).to(torch.device("cuda")))

print(b, b.get(), c.get())

b.reward(3)
c.reward(2)


loss = a.loss()

print(loss)
loss.backward()
'''