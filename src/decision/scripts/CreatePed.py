#!/usr/bin/env python
# coding=utf-8

import rospy
import rospkg
import os
import sys
pkg_path = rospkg.RosPack().get_path("actor_services")
class_path = os.path.join(pkg_path, 'src')
sys.path.append(class_path)
from PedSimWorld import PedSimWorld

if __name__ == "__main__":
    f = 'one_ped'
    s = [[1, 0]]
    g = [[7, 0]]
    sd = [1.2]

    f1 = 'crossing_wall_2'
    s1 = [(-2, 0), (-5, 0)]
    g1 = [(2, 0), (5, 0)]
    sd1 = [0.5, 1.2]

    f2 = 'crossing_wall_3'
    s2 = [(-2, 0), (0, 2), (0, 4)]
    g2 = [(2, 0), (0, -2), (0, -4)]
    sd2 = [0.8, 1.2, 0.7]

    f3 = 'crossing_wall_4'
    s3 = [(-2, 0), (-4, 0), (0, 2), (0, 4)]
    g3 = [(2, 0), (4, 0), (0, -2), (0, -4)]
    sd3 = [0.8, 1.2, 0.7, 1.2]

    f4 = 'overtaking_wall_2'
    s4 = [(-2, 0), (-5, 0)]
    g4 = [(2, 0), (5, 0)]
    sd4 = [0.5, 1.2]

    f5 = 'overtaking_wall_4'
    s5 = [(-2, 0), (-6, 0), (-3, 0), (-4, 0)]
    g5 = [(2, 0), (6, 0), (3, 0), (4, 0)]
    sd5 = [0.1, 1.2, 0.4, 0.7]

    f6 = 'passing_wall_2'
    s6 = [(-1, 0), (2.5, 0)]
    g6 = [(4, 0), (-2.5, 0)]
    sd6 = [1, 1]

    f7 = 'passing_wall_3'
    s7 = [(-3, -0.5), (3, -0.5), (2.5, 0.5)]
    g7 = [(2.5, -0.5), (-2.5, -0.5), (-3, 0.5)]
    sd7 = [1, 0.9, 1.1]

    f8 = 'passing_wall_4'
    s8 = [(-3, -0.5), (-2.5, 0.5), (3, 0.5), (2.5, 0.5)]
    g8 = [(2.5, -0.5), (3, 0.5), (-2.5, -0.5), (-3, 0.5)]
    sd8 = [1, 0.9, 1.1, 1.2]

    ped_world = PedSimWorld()
    ped_world.set_ped_info(s3, g3, sd3)
    ped_world.create_ped_world()