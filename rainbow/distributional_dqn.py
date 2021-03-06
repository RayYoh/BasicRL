import numpy as np
from dqn import DQN

from utils import *
from logx import *
import torch
from copy import deepcopy
import torch
from torch.optim import Adam
import gym
import time

class DistributionalDQN(DQN):
    def __init__(
        self, 
        env_fn, target_update,
        epsilon_decay, dqn = DistributionalNetwork, 
        seed=0, steps_per_epoch=4000, 
        epochs=100, replay_size=int(1e5), 
        gamma=0.99, q_lr=1e-3, 
        batch_size=32, start_steps=600, 
        update_after=600, num_test_episodes=5, 
        max_ep_len=300, logger_kwargs=dict(), 
        max_epsilon = 1.0, min_epsilon = 0.1,
        save_freq=10,
        # Distributional DQN parameters
        v_min: float = 0.0, v_max: float = 200.0,
        atom_size: int = 51,
        ):
        # Instantiate environment
        self.env, self.test_env = env_fn(), env_fn()

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n

        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update

        # Distributional DQN parameters
        self.v_min, self.v_max = v_min, v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # Create dqn module
        self.dqn = dqn(self.obs_dim, self.act_dim, atom_size, self.support).to(self.device)
        self.dqn_targ = deepcopy(self.dqn)
        # Freeze target networks with respect to optimizers
        for p in self.dqn_targ.parameters():
            p.requires_grad = False

        # Set up logger and save configuration
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.dqn])
        self.logger.log('\nNumber of parameters: \t pi: %d\n'%var_counts)

        self.replay_buffer = OffPolicyBuffer(self.obs_dim, 1, replay_size)

        # Set up optimizers for value function
        self.q_optimizer = Adam(self.dqn.parameters(), lr=q_lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.dqn)

    # Set up function for computing Q-loss
    def compute_loss_q(self, data):
        # tensor to cuda
        o, a, o2= data['obs'].to(self.device), data['act'].to(self.device), data['obs2'].to(self.device)
        r, d = data['rew'].to(self.device),  data['done'].to(self.device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            a2 = self.dqn_targ(o2).argmax(1)
            dist2 = self.dqn_targ.dist(o2)
            dist2 = dist2[range(self.batch_size), a2]

            t_z = r + (1 - d) * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(dist2.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (dist2 * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (dist2 * (b - l.float())).view(-1)
            )
        dist = self.dqn.dist(o)
        log_p = torch.log(dist[range(self.batch_size), a])
        loss_q = -(proj_dist * log_p).sum(1).mean()
    
        # loss_info = dict(QVals=q.detach().cpu().numpy())

        return loss_q
        
    def update(self, data):
        # run one gradient descent step for Q.
        loss_q = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item())
    def train(self):
        
        # Random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        update_cnt = 0

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= self.update_after:
                batch = self.replay_buffer.sample_batch(self.batch_size)
                self.update(data=batch)
                update_cnt += 1
                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self.target_hard_update()

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch
                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.logger.save_state({'env': self.env}, None)
                
                # Test the performance of the deterministic version of the agent.
                self.test_agent()
                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('TestEpLen', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--target_update', type=int, default=100)
    parser.add_argument('--epsilon_decay', type=float, default=1 / 2000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='CartPole-v0_distributional_dqn')
    args = parser.parse_args()

    from logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    agent = DistributionalDQN(lambda : gym.make(args.env), target_update=args.target_update, 
         epsilon_decay=args.epsilon_decay, dqn=DistributionalNetwork, seed=args.seed, logger_kwargs=logger_kwargs,
         epochs=args.epochs, gamma=args.gamma,steps_per_epoch=1000, max_ep_len=300)
    agent.train()