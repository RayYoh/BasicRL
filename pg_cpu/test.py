import gym
env = gym.make('Pendulum-v0')
act_limit = env.action_space.low[0]
print(act_limit)