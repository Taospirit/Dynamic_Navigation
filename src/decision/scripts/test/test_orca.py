#!/usr/bin/env python
from collections import namedtuple
import rvo2
"""
timeStep        The time step of the simulation.
                Must be positive.
neighborDist    The default maximum distance (center point
                to center point) to other agents a new agent
                takes into account in the navigation. The
                larger this number, the longer the running
                time of the simulation. If the number is too
                low, the simulation will not be safe. Must be
                non-negative.
maxNeighbors    The default maximum number of other agents a
                new agent takes into account in the
                navigation. The larger this number, the
                longer the running time of the simulation.
                If the number is too low, the simulation
                will not be safe.
timeHorizon     The default minimal amount of time for which
                a new agent's velocities that are computed
                by the simulation are safe with respect to
                other agents. The larger this number, the
                sooner an agent will respond to the presence
                of other agents, but the less freedom the
                agent has in choosing its velocities.
                Must be positive.
timeHorizonObst The default minimal amount of time for which
                a new agent's velocities that are computed
                by the simulation are safe with respect to
                obstacles. The larger this number, the
                sooner an agent will respond to the presence
                of obstacles, but the less freedom the agent
                has in choosing its velocities.
                Must be positive.
radius          The default radius of a new agent.
                Must be non-negative.
maxSpeed        The default maximum speed of a new agent.
                Must be non-negative.
velocity        The default initial two-dimensional linear
                velocity of a new agent (optional).

ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
Here set them to be large enough so that all agents will be considered as neighbors.
Time_horizon should be set that at least it's safe for one time step

In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

"""

# PyRVOSimulator(...)
# addAgent(...)
# loop:
#      setAgentPosition(index, position)
#      setAgentVelocity(index, velocity)
#      setAgentPrefVelocity(index, per_vel)
#      doStep()
#      action = getAgentVelocity(index)
params = 1.5, 5.0, 1.5, 2.0
sim = rvo2.PyRVOSimulator(1, 1.5, 5.0, 1.5, 2.0, 0.4, 2)
# PyRVOSimulator(time_step, neighbor_dist, max_neighbors, time_horizon, time_horizon_obst, radius, max_speed)

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
a0 = sim.addAgent((-4, -5)) # init position
a1 = sim.addAgent((4, -5))
a2 = sim.addAgent((0, 5))
# a2 = sim.addAgent((1, 1))
# a3 = sim.addAgent((0, 1), 1.5, 5, 1.5, 2, 0.4, 2, (0, 0))



# sim.addAgent(position, 
# neighbor_dist, max_neighbors, time_horizon, time_horizon_obst
# safe_dist, v_pref, velocity)

# Obstacles are also supported.
# o1 = sim.addObstacle([(1, 1), (-1, 1), (-1, -1)])
# sim.processObstacles()

# sim.setAgentVelocity(a0, (2, 2))
# sim.setAgentVelocity(a1, (1, 2))
# sim.setAgentVelocity(a2, (-1, 2))

sim.setAgentPrefVelocity(a0, (1, 1)) # pref_v = (vx, vy)
sim.setAgentPrefVelocity(a1, (-1, 1))
sim.setAgentPrefVelocity(a2, (1, -0.5))
# sim.setAgentPrefVelocity(a2, (-1, -1))
# sim.setAgentPrefVelocity(a3, (1, -1))

# print('Simulation has %i agents and %i obstacle vertices in it.' %
#       (sim.getNumAgents(), sim.getNumObstacleVertices()))

# print('Running simulation')

# for step in range(60):
#     sim.doStep()

#     positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(agent_no)
#                  for agent_no in (a0, a1, a2, a3)]
#     print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions)))

positions = [[] for _ in range(sim.getNumAgents())]

for _ in range(10):
      positions[0].append(sim.getAgentPosition(a0))
      positions[1].append(sim.getAgentPosition(a1))
      positions[2].append(sim.getAgentPosition(a2))
      sim.doStep()
      # for i in range():
      # positions[3].append(sim.getAgentPosition(a3))
      # print (info)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

p_list = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]

# fig = plt.figure()
# ax = fig.add_subplot(111)

# def plot_robot(ax, p_list, color='r', marker='*', size=0.3):
#       for i, item in enumerate(p_list):
#             ax.add_patch(mpathes.Circle(item, size, color='r', fill=False))
#             ax.text(item[0], item[1], str(i+1), fontsize=14)
#       x, y = [p[0] for p in p_list], [p[1] for p in p_list]
#       # ax.plot([1, 2, 3], [1, 3, 4], marker='o',color='lightgreen', markersize=10)
#       ax.plot(x, y, marker=marker, color='deeppink', markersize=5)

# for i in range(sim.getNumAgents()):
#       plot_robot(ax, positions[i])

# width = 10
# plt.xlim((-width, width))
# plt.ylim((-width, width))
# # 设置坐标轴刻度
# plt.xticks(np.arange(-width, width, 1))
# plt.yticks(np.arange(-width, width, 1))
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid()

# plt.show()



# PyRVOSimulator(...)
# addAgent(...)
# loop:
#      setAgentPosition(index, position)
#      setAgentVelocity(index, velocity)
#      setAgentPrefVelocity(index, per_vel)
#      doStep()
#      action = getAgentVelocity(index)
ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
state = namedtuple('state', ['px','py'])

s0 = [0, -5]
g0 = [0, 5]
s1 = [-2.5, 0]
g1 = [2.5, 0]
s2 = [-2, -2]
g2 = [2, 2]
s3 = [2, -2]
g3 = [-2, 2]

class orca_(object):
      def __init__(self):
            self.time_step = 1
            self.neighbor_dist = 10
            self.max_neighbors = 10
            # self.time_horizon = 1 # 向前看的时间步数
            self.time_horizon_obst = 5
            self.radius = 0.3
            self.max_speed = 1
            # other policy
            # def get_pref_vel(goal, start):
            #       velocity = np.array((goal[0] - start[0], goal[1] - start[1]))
            #       speed = np.linalg.norm(velocity)
            #       pref_vel = velocity / speed if speed > 1 else velocity
            #       return tuple(pref_vel)
            # self.a1_v = get_pref_vel(g1, s1)
            # self.a2_v = get_pref_vel(g2, s2)
            
      def set_(self, time_horizon):
            self.a1_p = [2, 2]
            self.a2_p = [0, 2.5]
            self.time_horizon = time_horizon
            params = self.neighbor_dist, self.max_neighbors, time_horizon, self.time_horizon_obst
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)

            self.a0 = self.sim.addAgent((s0[0], s0[1]))# init position
            self.a1 = self.sim.addAgent((s1[0], s1[1]))
            self.a2 = self.sim.addAgent((s2[0], s2[1]))
            self.a3 = self.sim.addAgent((s3[0], s3[1]))
            
            self.p_list = [[] for _ in range(self.sim.getNumAgents())]

      def do_action(self):
            pass

      def update(self): # get都是获取当前数据
            def get_pref_vel(goal, start):
                  velocity = np.array((goal[0] - start[0], goal[1] - start[1]))
                  speed = np.linalg.norm(velocity)
                  pref_vel = velocity / speed if speed > 1 else velocity
                  return tuple(pref_vel)

            self.p_list[0].append(self.sim.getAgentPosition(self.a0))
            # ORCA
            self.p_list[1].append(self.sim.getAgentPosition(self.a1))
            self.p_list[2].append(self.sim.getAgentPosition(self.a2))
            self.p_list[3].append(self.sim.getAgentPosition(self.a3))

            # other policy
            # self.p_list[1].append(tuple(self.a1_p))
            # self.p_list[2].append(tuple(self.a2_p))

            # 其他机器人是否遵守orca算法?
            # 是: 只用设置pref_vel即可, 其他部分不用刻意设置. 这里需要当前位置来计算pref_vel
            # 否: 手动设置它们的position和vel, 然后将pre_vel=(0, 0)
            p0 = self.sim.getAgentPosition(self.a0)
            # ORCA
            p1 = self.sim.getAgentPosition(self.a1)
            p2 = self.sim.getAgentPosition(self.a2)
            p3 = self.sim.getAgentPosition(self.a3)

            # other policy            
            # self.a1_p[0] = self.a1_p[0] + self.time_step * self.a1_v[0]
            # self.a1_p[1] = self.a1_p[1] + self.time_step * self.a1_v[1]
            # self.a2_p[0] = self.a2_p[0] +  self.time_step * self.a2_v[0]
            # self.a2_p[1] = self.a2_p[1] + self.time_step * self.a2_v[1]
            # self.a1_v = get_pref_vel(g1, self.a1_p)
            # self.a2_v = get_pref_vel(g2, self.a2_p)

            # self.sim.setAgentVelocity(self.a1, get_pref_vel(g1, p1))
            # self.sim.setAgentVelocity(self.a2, get_pref_vel(g2, p2))
            # self.sim.setAgentVelocity(self.a3, get_pref_vel(g3, p3))

            # self.sim.setAgentVelocity(self.a1, self.a1_v)
            # self.sim.setAgentVelocity(self.a2, self.a2_v)
            # self.sim.setAgentPosition(self.a1, tuple(self.a1_p))
            # self.sim.setAgentPosition(self.a2, tuple(self.a2_p))
            
            self.sim.setAgentPrefVelocity(self.a0, get_pref_vel(g0, p0))
            # ORCA
            self.sim.setAgentPrefVelocity(self.a1, get_pref_vel(g1, p1))
            self.sim.setAgentPrefVelocity(self.a2, get_pref_vel(g2, p2))
            self.sim.setAgentPrefVelocity(self.a3, get_pref_vel(g3, p3))

            # other policy
            # self.sim.setAgentPrefVelocity(self.a1, (0, 0))
            # self.sim.setAgentPrefVelocity(self.a2, (0, 0))
            # self.sim.setAgentPrefVelocity(self.a3, (0, 0))

            self.sim.doStep()

      def show_plot(self):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)

            def plot_robot(ax, p_list, color='r', marker='*', size=self.radius):
                  for i, item in enumerate(p_list):
                        ax.add_patch(mpathes.Circle(item, size, color=color, fill=False))
                        ax.text(item[0], item[1], str(i+1), fontsize=14)
                  x, y = [p[0] for p in p_list], [p[1] for p in p_list]
                  ax.plot(x, y, marker=marker, color=color, markersize=5)

            color = ['b', 'grey', 'brown', 'red', 'sienna', 'darkorange', 'gold', 'y', '']
            marker = ['*', 'o', 'x', '1']
            for i in range(len(self.p_list)):
                  # print (len(self.p_list))
                  # print (np.array(self.p_list).shape)
                  plot_robot(ax, self.p_list[i], color=color[i], marker=marker[i])

            width = 3
            plt.xlim((-width, width))
            plt.ylim((-width, width))
            # 设置坐标轴刻度
            plt.xticks(np.arange(-width, width, 1))
            plt.yticks(np.arange(-width, width, 1))
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid()
            # plt.show()
            path = '/home/lintao/medium_check/orca/'+str(self.sim.getNumAgents())+'-time_horizon:' + str(self.time_horizon) +'.png'
            plt.savefig(path)

      def is_done(self):
            a0_p = self.sim.getAgentPosition(self.a0)
            a1_p = self.sim.getAgentPosition(self.a1)
            a2_p = self.sim.getAgentPosition(self.a2)
            a3_p = self.sim.getAgentPosition(self.a3)
            # print ('a0_p {}, a1_p {}'.format(a0_p, a1_p))
            delta = []
            delta_0 = np.hypot(a0_p[0] - g0[0], a0_p[1] - g0[1])
            delta_1 = np.hypot(a1_p[0] - g1[0], a1_p[1] - g1[1])
            delta_2 = np.hypot(a2_p[0] - g2[0], a2_p[1] - g2[1])
            delta_3 = np.hypot(a3_p[0] - g3[0], a3_p[1] - g3[1])

            delta.append(delta_0)
            delta.append(delta_1)
            delta.append(delta_2)
            delta.append(delta_3)
            # print (delta_0, delta_1)
            if sum(delta)/len(delta) < 0.2:
                  # print (a0_p, a1_p)
                  # print (delta_0, delta_1)
                  self.p_list[0].append(self.sim.getAgentPosition(self.a0))
                  self.p_list[1].append(self.sim.getAgentPosition(self.a1))
                  self.p_list[2].append(self.sim.getAgentPosition(self.a2))
                  self.p_list[3].append(self.sim.getAgentPosition(self.a3))
                  return True
            return False
            

orca = orca_()
for i in range(1, 10):
      orca.set_(i)
      for _ in range(20):
            orca.update()
            if orca.is_done():
                  break
      orca.show_plot()