from utils import *
from logx import EpochLogger
from copy import deepcopy

import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time

class SAC():
    def __init__(
        self, env_fn, 
        actor_critic=MLPSACActorCritic, 
        ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, 
        epochs=100, replay_size=int(1e6), 
        gamma=0.99, polyak=0.995, 
        lr=1e-3, alpha=0.2, 
        batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, 
        num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1
        ):
        # Instantiate environment
        self.env, self.test_env = env_fn(), env_fn()

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create actor-critic module
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs).to(self.device)
        self.ac_targ = deepcopy(self.ac)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        
        
        print(self.device)

        # Set up logger and save configuration
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        
        self.replay_buffer = OffPolicyBuffer(self.obs_dim, self.act_dim, replay_size)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

    def compute_loss_q(self, data):
        o, a,= data['obs'].to(self.device), data['act'].to(self.device)
        r, o2, d = data['rew'].to(self.device), data['obs2'].to(self.device), data['done'].to(self.device)

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs'].to(self.device)
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(self.device), 
                      deterministic)

    def test_agent(self):
        for _ in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def train(self):
        # Random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

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
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)

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
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('Time', time.time()-start_time)
                self.logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    agent = SAC(lambda : gym.make(args.env), actor_critic=MLPSACActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
    agent.train()