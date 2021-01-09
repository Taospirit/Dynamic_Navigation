import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time

env_list = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Acrobot-v1', 'MountainCar-v0']
env_name = env_list[0]
dir_path = os.path.abspath(os.path.dirname(__file__))
# dir_path +='/sac2.pkl'

def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
            smooth_data.append(d)
    else:
        smooth_data = data

    return smooth_data

def plot(file_name, sm_size, color_type):
    file_path = dir_path + '/' + file_name
    with open(file_path + '.pkl', 'rb') as f:
        data = pickle.load(f)
    data = smooth(data['means'], sm=sm_size)
    data = np.array(data)
    time = range(data.shape[-1])

    sns.set(style="darkgrid", font_scale=1)
    sns.tsplot(time=time, data=data, color=color_type, condition="SAC", linestyle='-')

    font_size = 13
    plt.ylabel("Average Reward", fontsize=font_size)
    plt.xlabel("Episodes Number", fontsize=font_size)
    plt.title(file_name, fontsize=font_size)

    save_path = file_path + '.jpg'
    # print (save_path)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # pkl_name = 'sac_Gazebo_pure_seed_'
    # file_name_dict = {'0', '10', '20', '30', '40'}
    # for i in file_name_dict:
    #     file_name = pkl_name + i
    #     plot(file_name, 1)
    #     time.sleep(1)



    # file_name = 'sac_Gazebo_pure'
    # # file_name = 'sac_Gazebo_PER_N-step_seed_40'
    # plot(file_name, 5, 'b')

    # name = 'actor_learn_freq_'
    # name = '_learn_freq_'
    # file = dir_path + '/sac_per_' + env_name + name + '1.pkl'


    file_name_dict = {'0', '10', '20', '30', '40'}
    file_name = '/sac_Gazebo_pure_seed_'
    data1 = {}
    # file_name_dict = {'20'}
    for seed in file_name_dict:
        file = dir_path + file_name + seed
        with open(file + '.pkl', 'rb') as f:
            data1[seed] = pickle.load(f)

    file_name = '/sac_Gazebo_PER_N-step_seed_'
    data2 = {}
    # file_name_dict = {'20'}
    for seed in file_name_dict:
        file = dir_path + file_name + seed
        with open(file + '.pkl', 'rb') as f:
            data2[seed] = pickle.load(f)

    size = -1
    data1 = np.array([item['mean'][:size] for item in data1.values()])
    data2 = np.array([item['mean'][:size] for item in data2.values()])
    # # print (np.array(x1).shape)
    time = range(data1.shape[-1])

    sm_size = 10
    data1 = smooth(data1, sm=sm_size)
    data2 = smooth(data2, sm=sm_size)

    # # x1 = smooth(data1["mean"], sm=1)
    # # x2 = smooth(data2['mean'], sm=5)
    # # x3 = smooth(data3['mean'], sm=5)
    # # x4 = smooth(data4['mean'], sm=5)
    # # x5 = smooth(data5['mean'], sm=5)
    plt.figure(figsize=(1280, 960), dpi=1)

    sns.set(style="darkgrid", font_scale=2)
    sns.tsplot(time=time, data=data1, color="b", condition="SAC", linestyle='-')
    sns.tsplot(time=time, data=data2, color="r", condition="MSAC", linestyle='-')

    # # sns.tsplot(time=time, data=x, color="b", condition="SAC", linestyle='-')

    # # sns.tsplot(time=time, data=x2, color="b", condition="actor_freq:3", linestyle='-')
    # # sns.tsplot(time=time, data=x3, color="r", condition="actor_freq:5", linestyle='-')
    # # sns.tsplot(time=time, data=x4, color="g", condition="actor_freq:7", linestyle='-')
    # # sns.tsplot(time=time, data=x5, color="pink", condition="actor_freq:9", linestyle='-') 
    
    # font_size = 25
    # plt.ylabel("Average Reward", fontsize=font_size)
    # plt.xlabel("Episodes Number", fontsize=font_size)
    # plt.title("SAC_Extensions_GazeboEnv", fontsize=font_size)

    plt.ylabel("Average Reward")
    plt.xlabel("Episodes Number")
    plt.title("SAC_Extensions_GazeboEnv")

    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)

    # save_name = file_name + 'compera.jpg'
    # save_path = dir_path + save_name
    # # print (save_path)
    # plt.savefig(save_path)
    plt.show()