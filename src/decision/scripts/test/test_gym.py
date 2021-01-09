import gym
import time
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    gym_return = f'obs: {observation}, obs_type: {type(observation)}, \n\
        rew: {reward}, rew_type: {type(reward)}, \n\
        done: {done}, done_type: {type(done)}, \n\
        info: {info}, info_type: {type(info)}'

    print ('='*20+str(_)+'='*20)
    print(gym_return)
    
    if done:
        observation = env.reset()
env.close()