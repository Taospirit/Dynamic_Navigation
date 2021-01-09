import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import numpy as np

num = 7

file = os.path.abspath(os.path.dirname(__file__)) + '/' + str(num) + '.txt'
p_list = []
obs_goal_3 = [3.0, -2.0]
obs_goal_4 = [[-1, -2], [2, 2]]
obs_goal_5 = [[5, -2], [7, 2]]
obs_goal_6 = [[5, 2], [3, -2]]
obs_goal_7 = [[2, -3], [5, 5]]

def show_plot(p_list):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    def plot_robot(ax, p_list, color='r', marker='*', size=0.3, step=1):
        # p_list = [[x1, y1], [x2, y2], ... , [xn, yn]]
        count = 0
        for i, item in enumerate(p_list):
            ax.add_patch(mpathes.Circle(item, 0, color=color, fill=False))
            if i % step == 0 or i == len(p_list)-1:
                ax.add_patch(mpathes.Circle(item, size, color=color, fill=False))
                ax.text(item[0], item[1], str(round(i*0.1,2)), fontsize=14)
            count += 1
        x, y = [p[0] for p in p_list], [p[1] for p in p_list]
        ax.plot(x, y, marker=marker, color=color, markersize=2)

    ax.add_patch(mpathes.Circle([7.0, 0.0], 0.3, color='r', fill=True))
    ax.add_patch(mpathes.Circle(obs_goal_7[0], 0.3, color='g', fill=True))
    ax.add_patch(mpathes.Circle(obs_goal_7[-1], 0.3, color='g', fill=True))

    color = ['b', 'black', 'brown', 'red', 'sienna', 'darkorange', 'gold', 'y', '']
    marker = ['*', 'o', 'x', '1']

    plot_robot(ax, p_list[0], color='b', marker='*', size=0.3, step=5)
    for i in range(1, len(p_list)):
        plot_robot(ax, p_list[i], color=color[i], marker=marker[i], size=0.3, step=10)

    plt.xlim((-1, 9))
    plt.ylim((-5, 5))
    # 设置坐标轴刻度
    plt.xticks(np.arange(-1, 9, 1))
    plt.yticks(np.arange(-5, 5, 1))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.show()
    # plt.savefig(path)

if __name__ == "__main__":
    a_list = []
    o1_list = []
    o2_list = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
        # lines = f.readlines()
        # for i in range(len(lines)):
        #     if i % 5 == 0:
        #         line = lines[i]
        # lines = f.readlines()
            datas = line.split(';')
            data = datas[0].split(':')[-1]
            # print(datas)
            x, y, t, v, w = data.split(',')
            a_list.append([float(x), float(y)])

            data = datas[1].split(':')[-1]
            x, y, t, v, w = data.split(',')
            o1_list.append([float(x), float(y)])
            if num >= 4:
                data = datas[2].split(':')[-1]
                x, y, t, v, w = data.split(',')
                o2_list.append([float(x), float(y)])
        # a_list.append([7.0, 0.0])
        # print(a_list)
        # print(o_list)
        p_list.append(a_list)
        p_list.append(o1_list)
        p_list.append(o2_list)
        # print(p_list)
    print(a_list)
    print(np.array(p_list).shape)
    # print(o_list)
    show_plot(p_list)