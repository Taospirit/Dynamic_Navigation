import gym
env = gym.make('gym_gazebo:robot-v0')

obs = env.reset()
for _ in range(100):
    action = 2
    state_, reward, done, info = env.step(action)
    if done:
        obs = env.reset()