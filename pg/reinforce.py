import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam

class ReinforceBuffer():
    def __init__(self):
        pass

    def store(self, o, a, next_o, r, d):
        self.buffer['obs'].append(o)
        self.buffer['act'].append(a)
        self.buffer['next_obs'].append(next_o)
        self.buffer['rew'].append(r)
        self.buffer['dones'].append(d)
    
    def clear(self):
        self.buffer = {'obs': [], 'act': [], 'next_obs': [], 'rew': [], 'dones': []}
    
    def get(self):
        return self.buffer['obs'],self.buffer['act'], self.buffer['rew']

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim = 128):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return F.softmax(x, dim=1)

class reinforce():
    def __init__(
        self, 
        env,
        episodes,
        seed = 0,
        lr = 1e-3, 
        gamma = 0.99):

        self.env = env()
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n 

        self.episodes = episodes
        self.seed = seed

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.policy_net = PolicyNet(obs_dim, act_dim).to(self.device)
        self.optimizer = Adam(self.policy_net.parameters(), lr)

        self.gamma = gamma
        
   
    def select_action(self, state):
        probs = self.policy_net(torch.FloatTensor([state]).to(self.device))
        selected_action = torch.distributions.Categorical(probs).sample()
        return selected_action.item()

    def update(self, buffer):
        obs, act, rew = buffer.get()

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(rew))):
            rew_i = rew[i]
            obs_i = torch.FloatTensor([obs[i]]).to(self.device)
            act_i = torch.LongTensor([act[i]]).view(-1,1).to(self.device)
            log_prob = torch.log(self.policy_net(obs_i)).gather(1, act_i)
            G = self.gamma * G + rew_i
            loss = -log_prob * G
            loss.backward()
        
        self.optimizer.step()

    def train(self):
        # Random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.env.seed(self.seed)

        ret_list = []
        buffer = ReinforceBuffer()
        for i in range(self.episodes):
            buffer.clear()
            o, ep_ret = self.env.reset(), 0
            d = False
            while not d:
                a = self.select_action(o)
                next_o, r, d, _ = self.env.step(a)
                buffer.store(o, a, next_o, r, d)
                o = next_o
                ep_ret += r
            ret_list.append(ep_ret)
            self.update(buffer)
            print('episode: ', i, ', ep_ret: ', ep_ret, '.')
        ep_list = list(range(len(ret_list)))
        plt.plot(ep_list, ret_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('Reinforce on {}'.format('CartPole-v0'))
        plt.show()
if __name__ =='__main__':
    env = 'CartPole-v0'
    agent = reinforce(lambda : gym.make(env), 500)
    agent.train()