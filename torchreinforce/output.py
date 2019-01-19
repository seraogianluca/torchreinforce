class ReinforceOutput:
    def __init__(self, distribution, defaultreward=None):
        self.distribution = distribution
        self.action = None
        self._reward = defaultreward
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
        return self.distribution.log_prob(self.action).unsqueeze(0)
