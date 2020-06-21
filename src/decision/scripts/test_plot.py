import os
import matplotlib.pyplot as plt

def read(store_list, filename):
    with open(filename, 'r') as f:
        while True:
            lines = f.readline()
            if not lines:
                break
            store_list.append(round(float(lines.split()[-1]), 2))
        print ('read over')

def plot(y, y_label, save_path):
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel(y_label)
    ax.plot(y)
    # plt.savefig(save_path)
    plt.show()

file_path = '/home/lintao/projects/Dynamic_Navigation/src/decision/scripts/data_collection/ts_dqn_3_crossing.txt'
save_path = '/'.join(file_path.split('/')[:-1]) + '/ts_dqn_3_crossing.jpg'

rew_list = []
read(rew_list, file_path)
plot(rew_list, 'Reward', save_path)