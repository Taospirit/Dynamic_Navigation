#!/usr/bin/env python
import rospy
import os
import time
# import sys
# sys.path.append('../')
from GazeboEnv import GazeboEnv
from DQN import DeepQNetwork
import matplotlib.pyplot as plt

num_episode = 10000
num_action = 80
gamma = 0.9
reward_list = []
folder = '/data_collection'
write_file_name = 'dqn_pass2'

agent_s, agent_g =[0, 0], [5, 0]
obs_s, obs_g = [[6, 0, 180]], [[-1, 0, 0]]
env_scene = [agent_s, agent_g, obs_s, obs_g]

def train():
    step = 0
    reward_list = []
    for episode in range(1, num_episode+1):
        if rospy.is_shutdown():
            rospy.loginfo('BREAK LEARNING!')
            env.step(-1, [0, 0])
            break
        # initial observation
        state = env.reset()
        # for n_action in range(num_action):
        action_n = 0
        discount_reward_ = []
        reward_ = []
        epsilon_ = []

        discount_reward, cumulate_reward = 0, 0
        while not rospy.is_shutdown():
            # action_n = 0
            # RL choose action by DQN law
            action = RL.choose_action(state)
            # RL take action and get next observation and reward
            state_, reward, done, _ = env.step(action)
            
            RL.store_transition(state, action, reward, state_)

            if (step > 5000) and (step % 5 == 0):
                RL.learn()
                
            # swap state
            state = state_
            # break while loop when end of this episode
            step += 1
            action_n += 1
            reward = round(reward, 3)
            discount_reward_.append(reward)
            reward_.append(reward)
            if done or action_n > num_action - 1:
                discount_reward = get_total_reward(discount_reward_, gamma)
                cumulate_reward = sum(reward_)
                # epsilon_.append()
                data = 'ep {}, a_n {}, eps {:.3f}, discount_r {:.2f}, cumulate_r {:.2f} step {}'.format(episode, \
                    action_n, RL.epsilon, discount_reward, cumulate_reward, step)
                print (data)
                write_data(folder+'/'+write_file_name, data)
                break
        reward_list.append(cumulate_reward)
        plot(episode, reward_list, 'Cumulate Reward')
        # plot(episode, discount_reward, 'Discount Reward')
        # plot(episode, cumulate_reward, 'Cumulate Reward')
    # end of game
    # print('game over')
    # env.destroy()
    rospy.loginfo('LEARNING OVER~')

def get_total_reward(r_l, g):
    if len(r_l) == 1:
        return r_l[0]
    else:
        return r_l.pop(0) + g * get_total_reward(r_l, g)

def write_data(file_name, data):
    dir_path = os.path.dirname(__file__)
    # file_path = file_name + '.txt'
    file_path = dir_path + file_name + '.txt'
    # file_path = os.path.join(dir_path, file_name)
    # if os.path.exists(file_path+'.txt'):
    #     if file_name[-1].isdigit():
    #         add = int(file_name[-1]) + 1
    #     else:
    #         add = 1
    #     file_name += str(add)
    # file_path = os.path.join(dir_path, file_name)
    # print (file_path)
    with open(file_path, 'a') as f:
        f.write(data+'\n')

def plot(ep, y, y_label):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel(y_label)
    ax.plot(y)
    # RunEpisode = len(ep)
    dir_path = os.path.dirname(__file__)

    path = dir_path + folder + '/' + write_file_name +'_'+y_label+'_RunStep' + str(ep) + '.jpg'
    if ep % 500 == 0 and ep != 0:
        plt.savefig(path)
    plt.pause(0.0000001)
        
if __name__ == "__main__":
    env = GazeboEnv()
    env.init_env(env_scene[0], env_scene[1], env_scene[2], env_scene[3])
    RL = DeepQNetwork(env.n_actions,
                      learning_rate=0.01,
                      reward_decay=gamma,
                      replace_target_iter=200,
                      memory_size=10000,
                      num_episode=num_episode,
                      # output_graph=True
                      )
    # RL.plot_cost()
    train()
    rospy.spin()
