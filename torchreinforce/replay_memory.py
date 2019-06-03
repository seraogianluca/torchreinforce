import random

from collections import namedtuple

Transition = namedtuple('Timestep', 'state action reward next_state')

class ReplayMemory(object):
    def __init__(self, size):
        self.memory_size = size
        self.memory = []
    
    def store(self, *args):
        """Save a transition into replay memory. When memory is full, 
        the oldest data will be overwritten."""
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append(Transition(*args))
    
    def sample(self):
        """Sample a random batch from the memory."""
        #return random.sample(self.memory, min(len(self.memory), batch_size))
        return random.sample(self.memory, 1)