from double_dqn import DoubleDQN

from utils import *
from logx import *
import gym

class DuelingDoubleDQN(DoubleDQN):
    def __init__(
        self, 
        env_fn, target_update, 
        epsilon_decay, dqn=DuelingNetwork, 
        seed=0, steps_per_epoch=4000, 
        epochs=100, replay_size=int(1e5), 
        gamma=0.99, q_lr=0.001, 
        batch_size=32, start_steps=600, 
        update_after=600, num_test_episodes=5,
         max_ep_len=300, logger_kwargs=..., 
         max_epsilon=1, min_epsilon=0.1, save_freq=10):
        super().__init__(
            env_fn, target_update, 
            epsilon_decay, dqn=dqn, 
            seed=seed, steps_per_epoch=steps_per_epoch, 
            epochs=epochs, replay_size=replay_size, 
            gamma=gamma, q_lr=q_lr, batch_size=batch_size, 
            start_steps=start_steps, update_after=update_after, 
            num_test_episodes=num_test_episodes, 
            max_ep_len=max_ep_len, logger_kwargs=logger_kwargs, 
            max_epsilon=max_epsilon, min_epsilon=min_epsilon, 
            save_freq=save_freq)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--target_update', type=int, default=100)
    parser.add_argument('--epsilon_decay', type=float, default=1 / 2000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='CartPole-v0_dueling_double_dqn')
    args = parser.parse_args()

    from logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    agent = DuelingDoubleDQN(lambda : gym.make(args.env), target_update=args.target_update, 
         epsilon_decay=args.epsilon_decay, dqn=DuelingNetwork, seed=args.seed, logger_kwargs=logger_kwargs,
         epochs=args.epochs, gamma=args.gamma,steps_per_epoch=1000, max_ep_len=200)
    agent.train()