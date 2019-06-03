import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import random
import math

from .replay_memory import *


class DeepReinforceModule(nn.Module):
    def __init__(self, **kwargs):
        super(DeepReinforceModule, self).__init__()
        #self.__dict__.update(kwargs)
        #Iperparametri
        self.gamma = kwargs.get("gamma", 0.99)
        self.memory_size = kwargs.get("memory_size", 1000)
        self.epsilon_init = kwargs.get("epsilon_init", 0.0001)
        self.epsilon_max = kwargs.get("epsilon_max", 0.1)
        self.epsilon_decay = kwargs.get("epsilon_decay", 200)
        self.learning_rate = kwargs.get("lr", 0.001)
        self.target_update_rate = kwargs.get("target_update", 10)
        self.batch_size = kwargs.get("batch_size", 32)
        self.counter = 0
        #Rete target, ottimizzatore e memoria
        self.memory = ReplayMemory(self.memory_size)
        self.target_net = kwargs.get("target_net", None)

        if self.target_net is not None:
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        

    def loss(self, **kwargs):
        '''Take a sampled batch from replay memory and compute Q(s, a) and V(s). Return a MSE loss between Q and V.'''
        loss = 0
        for count in range(self.batch_size):
            batch = self._get_data()
            state = torch.cat(batch.state) 
            action = torch.tensor(batch.action)
            reward = torch.tensor(batch.reward)
            Q = self(state).gather(0, action)

            if batch.next_state[0] is not None:
                next_state = torch.cat(batch.next_state)
                Q_exp = self.target_net(next_state)
                Q_exp, _ = torch.max(Q_exp, 0)
                Q_opt = reward + torch.mul(Q_exp.detach(), self.gamma)
            else:
                Q_opt = reward
            
            loss += functional.smooth_l1_loss(Q, Q_opt)

            if self.counter % self.target_update_rate == 0:
                self.target_net.load_state_dict(self.state_dict(), strict=False)
        return loss / self.batch_size


    def select_action(self):
        '''Perform an annealing epsilon-greedy policy'''
        sample = random.random()
        threshold = self.epsilon_max + (self.epsilon_init - self.epsilon_max) * math.exp(-1 * self.counter / self.epsilon_decay)
        self.counter += 1
        return sample < threshold

    
    def _get_data(self):
        '''Take a sample from the replay memory and transform it in a batch of arrays, one for every category of "Timestep".'''
        samples = self.memory.sample()
        batch = Transition(*zip(*samples)) 
        return batch


    def clipping_reward(self):
        '''Clipping positive and negative rewards to 1 and -1 respectively.'''
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)


    def update_target(self):
        '''Update params of the target net'''
        self.target_net.load_state_dict(self.state_dict(), strict=False)


    #def to(self, *args, **kwargs):
    #    self.target_net = self.target_net.to(*args, **kwargs)
    #    self.memory = self.memory.to(*args, **kwargs)