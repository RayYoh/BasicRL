from dqn import DQN

from utils import *
from logx import *
import gym

class DuelingDQN(DQN):
    def __init__(
        self, 
        env_fn, 
        target_update,epsilon_decay,
        dqn = DuelingNetwork, seed=0, 
        steps_per_epoch=600, 
        epochs=100, replay_size=int(2000), 
        gamma=0.99, q_lr=1e-3, 
        batch_size=32, start_steps=200, 
        update_after=200, 
        num_test_episodes=5, 
        max_ep_len=200, logger_kwargs=dict(), 
        max_epsilon = 1.0, min_epsilon = 0.1,
        save_freq=10
        ):
        super(DuelingDQN, self).__init__(
            env_fn, 
            target_update,epsilon_decay,
            dqn, seed, 
            steps_per_epoch, 
            epochs, replay_size, 
            gamma, q_lr, 
            batch_size, start_steps, 
            update_after, 
            num_test_episodes, 
            max_ep_len, logger_kwargs, 
            max_epsilon, min_epsilon,
            save_freq
        )
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--target_update', type=int, default=100)
    parser.add_argument('--epsilon_decay', type=float, default=1 / 2000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='1008CartPole-v0_dqn_dueling')
    args = parser.parse_args()

    from logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    agent = DuelingDQN(lambda : gym.make(args.env), target_update=args.target_update, 
         epsilon_decay=args.epsilon_decay, dqn=DuelingNetwork, seed=args.seed, logger_kwargs=logger_kwargs,
         epochs=args.epochs, gamma=args.gamma,steps_per_epoch=600, max_ep_len=200)
    agent.train()