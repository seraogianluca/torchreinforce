import random
import torch
import numpy as np
from collections import namedtuple

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayMemory(object):
    def __init__(self, action_size, memory_size, batch_size, seed):
        self.action_size = action_size
        self.memory = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.transition = namedtuple('Transition', 'state action reward next_state done')

    
    def store(self, state, action, reward, next_state, done):
        """Save a transition into replay memory. When memory is full, 
        the oldest data will be overwritten."""
        transition = self.transition(state, action, reward, next_state, done)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append(transition)
    
    def sample(self):
        """Sample a random batch from the memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        #states = torch.tensor([t.state for t in experiences if t is not None], dtype=torch.float).unsqueeze(1)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        #actions = torch.tensor([t.action for t in experiences if t is not None], dtype=torch.long).unsqueeze(1)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        #rewards = torch.tensor([t.reward for t in experiences if t is not None], dtype=torch.float).unsqueeze(1)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        #next_states = torch.tensor([t.next_state for t in experiences if t is not None], dtype=torch.float).unsqueeze(1)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        #dones = torch.tensor([t.done for t in experiences if t is not None], dtype=torch.float).unsqueeze(1)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of memory."""
        return len(self.memory)