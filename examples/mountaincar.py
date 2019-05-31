import gym
import torch
import torch.nn as nn
from torchreinforce import DeepReinforceModule

env = gym.make('MountainCar-v0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPISODES = 1000
STATE_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n

class Policy(DeepReinforceModule):
    def __init__(self, **kwargs):
        super(Policy, self).__init__(**kwargs)
        self.net = torch.nn.Sequential(
            nn.Linear(STATE_SPACE, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SPACE),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


target_net = Policy()
policy_net = Policy(target_net=target_net)

#Weights initializer
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight)

policy_net.apply(init_normal)

for i_episode in range(EPISODES):
    state = torch.as_tensor(env.reset(), dtype=torch.float, device=device)

    done = False
    while not done:
        env.render()
        #Epsilon-greedy policy
        if policy_net.select_action():
            action = env.action_space.sample()
        else:
            action = policy_net(state).max(-1)[1]
            action = action.item()

        next_state, reward, done, _ = env.step(action)
        next_state = torch.as_tensor(next_state, dtype=torch.float, device=device) 

        if done:
            next_state = None

        policy_net.memory.store(state, action, reward, next_state)
        state = next_state 

        if i_episode % policy_net.counter == 0:
            policy_net.update_target()

    loss = policy_net.loss()
    policy_net.zero_grad()
    loss.backward()
    policy_net.optimizer.step()

    if i_episode % 50 == 0:
        print("Episode: %d  loss: %f  reward_threshold: %d" % (i_episode, loss.item(), env.spec.reward_threshold))

env.close()