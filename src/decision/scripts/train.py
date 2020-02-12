import rospy
import os
import time
# import sys
# sys.path.append('../')
from gazebo_env import gazebo_env
from DQN import DeepQNetwork
num_episode = 10000
num_action = 100
gamma = 0.9
reward_list = []
write_file_name = '/data_collection/dqn_no_obs_2'

def train():
    step = 0
    for episode in range(num_episode):
        if rospy.is_shutdown():
            rospy.loginfo('BREAK LEARNING!')
            env.step(-1, [0, 0])
            break
        # initial observation
        state = env.reset()
        # for n_action in range(num_action):
        action_n = 0
        reward_list = []
 
        while not rospy.is_shutdown():
            # action_n = 0
            # agent choose action by DQN law
            action = agent.choose_action(state)
            # agent take action and get next observation and reward
            state_, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, state_)

            if (step > 2000) and (step % 5 == 0):
                agent.learn()
                
            # swap state
            state = state_
            # break while loop when end of this episode
            step += 1
            action_n += 1
            reward_list.append(reward)
            if done or action_n > num_action - 1:
                total_reward = get_total_reward(reward_list, gamma)
                data = 'ep {}, a_n {}, epsilon {:.3f}, total_reward {:.2f}, step {}'.format(episode, action_n, agent.epsilon, total_reward, step)
                print (data)
                # write_data(write_file_name, data)
                break

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
    file_path = '.' + file_name + '.txt'
    # file_path = dir_path + file_name + '.txt'
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

        
if __name__ == "__main__":
    env = gazebo_env()
    agent = DeepQNetwork(env.n_actions,
                      learning_rate=0.01,
                      reward_decay=gamma,
                      replace_target_iter=200,
                      memory_size=5000,
                      num_episode=num_episode,
                      # output_graph=True
                      )
    # RL.plot_cost()
    train()
    rospy.spin()
