import torch

class ReinforceOutput:
    def __init__(self, distribution):
        self.distribution = distribution
        self.action = None
        self._reward = None
        self.used = False

    def get(self):
        self.action = self.distribution.sample()
        return self.action
    
    def reward(self, reward):
        self._reward = reward
    
    def get_reward(self):
        return self._reward
    
    def _log_prob(self):
        self.used = True
        return self.distribution.log_prob(self.action).unsqueeze(0)



class ReinforceModule(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ReinforceModule, self).__init__(**kwargs)
        self.gamma = kwargs["gamma"] if "gamma" in kwargs else 0.99
        self.distribution = kwargs["distribution"] if "distribution" in kwargs else torch.distributions.Categorical 
        self.checkused = kwargs["checkused"] if "checkused" in kwargs else True
        self.history = []

    def _check_used(self):
        if len(self.history) != 0 and len(list(filter(lambda x: x.used, self.history))) != 0:
            raise Exception("One or more outputs seems to be already been used, you may want to reset() this module. \
            If you know what you are doing you can ignore di this check by settig checkused to False")

    def get_action(self, x):
        if self.checkused: self._check_used()
        y = self(x)

        if self.training:
            c = self.distribution(y)
            output = ReinforceOutput(c)
            self.history.append(output)
            return output
        else:
            return y.max(0)[1]

    def loss(self):
        history = list(filter(lambda x: x.reward is not None and x.action is not None, self.history))
        log_probs = torch.stack(list(map(lambda x: x._log_prob(), history)))
        rewards = list(map(lambda x: x.get_reward(), history))

        comulative = torch.tensor(0, dtype=torch.float32, device=log_probs.device)
        disconted_rewards = []
        for r in rewards:
            comulative = comulative*self.gamma + r
            disconted_rewards.append(comulative.unsqueeze(0))
        
        disconted_rewards = torch.tensor(disconted_rewards, device=log_probs.device)
        if disconted_rewards.abs().sum() != 0:
            disconted_rewards = (disconted_rewards - disconted_rewards.mean()) / disconted_rewards.std()
        
        loss = torch.mul(disconted_rewards, log_probs)
        return loss.sum()

    
    def clear(self):
        self.rewards = []
        self.actions = []
        self.tot_reward = 0





class Test(ReinforceModule):
    def __init__(self):
        super(Test, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


a = Test().to(torch.device("cuda"))
b = a.get_action(torch.randn(2).to(torch.device("cuda")))
c = a.get_action(torch.randn(2).to(torch.device("cuda")))

print(b, b.get(), c.get())

b.reward(3)
c.reward(2)


loss = a.loss()

print(loss)
loss.backward()
