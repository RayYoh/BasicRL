from dqn import DQN

from utils import *
from logx import *
import torch.nn.functional as F
import torch
import gym

class DoubleDQN(DQN):
    
    # Set up function for computing Q-loss
    def compute_loss_q(self, data):
        # tensor to cuda
        o, a, o2= data['obs'].to(self.device), data['act'].to(self.device), data['obs2'].to(self.device)
        r, d = data['rew'].to(self.device),  data['done'].to(self.device)
        
        q = self.dqn(o).gather(1, a)
        a2 = self.dqn(o2).argmax(dim=1, keepdim=True)
        next_q = self.dqn_targ(o2).gather(1, a2).detach()
        target = (r + self.gamma * (1 - d) * next_q).to(self.device)

        # calculate dqn loss
        loss_q = F.smooth_l1_loss(q, target)

        # Useful info for logging
        loss_info = dict(QVals=q.detach().cpu().numpy())

        return loss_q, loss_info

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--target_update', type=int, default=100)
    parser.add_argument('--epsilon_decay', type=float, default=1 / 2000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='CartPole-v0_double_dqn')
    args = parser.parse_args()

    from logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    agent = DoubleDQN(lambda : gym.make(args.env), target_update=args.target_update, 
         epsilon_decay=args.epsilon_decay, dqn=Network, seed=args.seed, logger_kwargs=logger_kwargs,
         epochs=args.epochs, gamma=args.gamma,steps_per_epoch=1000, max_ep_len=200)
    agent.train()