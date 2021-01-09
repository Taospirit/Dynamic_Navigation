import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time

dir_path = os.path.abspath(os.path.dirname(__file__))

def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
            smooth_data.append(d)
    else:
        smooth_data = data

    return np.array(smooth_data)

def plot_data(data, sm_size, file_path, color_type, alg_name):
    file_name = file_path.split('/')[-1]
    data = smooth(data['mean'], sm=sm_size)
    time = range(data.shape[-1])

    sns.set(style="darkgrid", font_scale=1)
    sns.tsplot(time=time, data=data, color=color_type, condition=alg_name, linestyle='-')

    font_size = 13
    plt.ylabel("Average Reward", fontsize=font_size)
    plt.xlabel("Episodes Number", fontsize=font_size)
    if 'PER' in file_name:
        plt.title('MSAC_Gazebo', fontsize=font_size)
    else:
        plt.title('SAC_Gazebo', fontsize=font_size)
    save_path = file_path + '.jpg'
    plt.savefig(save_path)
    plt.cla()
    # plt.show()

if __name__ == '__main__':
    task_num = 0

    # alg_type = '_pure_seed_'n
    # alg_type = '_PER_N-step_seed_'n
    # dir_name = ''
    dir_path += '/task' + str(task_num)
    # pkl_path = 'sac_Gazebo_task' + str(task_num)
    pkl_path = 'sac_Gazebo'
# , '30', '40
    file_name_dict = {'0', '10', '20', '30', '40'}
    # file_name_dict = {'0', '10', '20', '30'}
    def get_data(pkl_name, size=-1, sm_size=10, plot=False, alg_name='SAC'):
        data_dict = {}
        for seed in file_name_dict:
            file_path = dir_path + '/' + pkl_name + seed
            with open(file_path + '.pkl', 'rb') as f:
                data_dict[seed] = pickle.load(f)
            if plot:
                if 'PER' in pkl_name:
                    plot_data(data_dict[seed], 1, file_path, 'b', alg_name)
                else:
                    plot_data(data_dict[seed], 1, file_path, 'g', alg_name)
        data = np.array([item['mean'][:size] for item in data_dict.values()])
        data = smooth(data, sm=sm_size)
        time = range(data.shape[-1])
        return data, time

    pkl_name = pkl_path + '_pure_seed_'
    data1, time1 = get_data(pkl_name, plot=0)
    pkl_name = pkl_path + '_PER_N-step_seed_'
    data2, time1 = get_data(pkl_name, plot=0, alg_name='MSAC')

    sns.set(style="darkgrid", font_scale=2)
    sns.tsplot(time=time1, data=data1, color="g", condition="SAC", linestyle='-')
    sns.tsplot(time=time1, data=data2, color="b", condition="MSAC", linestyle='-')

    plt.ylabel("Average Reward")
    plt.xlabel("Episodes Number")
    # plt.title("GazeboEnv_task"+str(task_num))
    plt.title("GazeboEnv")
    plt.show()