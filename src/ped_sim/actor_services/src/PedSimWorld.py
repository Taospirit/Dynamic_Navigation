#!/usr/bin/env python
# coding=utf-8

import os
import random
from lxml import etree
from lxml.etree import Element
import rospkg
import rospy

skin_list = ["moonwalk.dae",
        "run.dae",
        "sit_down.dae",
        "sitting.dae",
        "stand_up.dae",
        "stand.dae",
        "talk_a.dae",
        "talk_b.dae",
        "walk.dae"]

class PedSimWorld():
    def __init__(self):
        'Create SDF file: file_name.world'
        base_pkg_name = rospy.get_param("BASE_PKG")
        base_file_name = rospy.get_param("BASE_WORLD")
        save_pkg_name = rospy.get_param("SAVE_PKG")
        save_file_name = rospy.get_param("SAVE_WORLD")
        # self.rospack = rospkg.RosPack()
        base_pkg = rospkg.RosPack().get_path(base_pkg_name)
        self.save_pkg = rospkg.RosPack().get_path(save_pkg_name)
        self.base_file_path = base_pkg + "/worlds/" + base_file_name + ".world"
        self.save_file_path = self.save_pkg + "/worlds/" + save_file_name + ".world"
        self.plugin_path = rospkg.RosPack().get_path("actor_plugin") + "/lib/libactorplugin_ros.so"
        self.start_position = None
        self.goal_position = None
        self.ped_speed = None

    def set_ped_info(self, start_position, goal_position, ped_speed):
        self.set_start_position(start_position)
        self.set_goal_position(goal_position)
        self.set_ped_speed(ped_speed)

    def set_start_position(self, p_list):
        self.start_position = p_list

    def set_goal_position(self, g_list):
        self.goal_position = g_list

    def set_ped_speed(self, sd_list):
        self.ped_speed = sd_list

    def create_ped_world(self):
        tree = etree.parse(self.base_file_path)
        world = tree.getroot().getchildren()[0]
        assert not len(self.start_position) < len(self.goal_position)
        for i in range(len(self.start_position)):
            sx = str(self.start_position[i][0])
            sy = str(self.start_position[i][1])
            gx = str(self.goal_position[i][0])
            gy = str(self.goal_position[i][1])
            z = '1.02'
            s = str(self.ped_speed[i])
            ped = self.create_ped_sdf(i, sx, sy, gx, gy, z, s)
            world.append(ped)
    
        tree.write(self.save_file_path, xml_declaration=True, encoding="utf-8", pretty_print=True)

    def create_ped_sdf(self, index, sx, sy, gx, gy, z, s):
        ped = Element("actor", name="ped" + str(index))

        pose = Element("pose")
        pose.text = ' '.join([sx, sy, z, '0', '0', '0'])

        skin = Element("skin")
        skin_fn = Element("filename")
        skin_fn.text=random.choice(skin_list)
        skin_scale = Element("scale")
        skin_scale.text = "1"
        skin.append(skin_fn)
        skin.append(skin_scale)

        animation = Element("animation", name="walking")
        animate_fn = Element("filename")
        animate_fn.text = skin_list[-1]
        interpolate_x = Element("interpolate_x")
        interpolate_x.text = "true"
        animate_scale = Element("scale")
        animate_scale.text = "1"
        animation.append(animate_fn)
        animation.append(animate_scale)
        animation.append(interpolate_x)

        plugin = Element("plugin", name="None", filename=self.plugin_path)
        speed = Element("speed")
        speed.text = s
        target = Element("target")
        target.text = ' '.join([gx, gy, z])
        ignore_obstacle = Element("ignore_obstacles")
        model_cafe = Element("model")
        model_cafe.text = "caffe"
        model_ground_plane = Element("model")
        model_ground_plane.text = "ground_plane"
        ignore_obstacle.append(model_cafe)
        ignore_obstacle.append(model_ground_plane)
        plugin.append(speed)
        plugin.append(target)
        plugin.append(ignore_obstacle)

        ped.append(pose)
        ped.append(skin)
        ped.append(animation)
        ped.append(plugin)

        return ped

    def set_ped_world(self, open_file, save_file, s_list, g_list, sd_list):
        file_open = self.save_pkg + "/worlds/" + open_file + ".world"
        file_save = self.save_pkg + "/worlds/" + save_file + ".world"
        tree = etree.parse(file_open)
        world = tree.getroot().getchildren()[0]

        ped_num = 0
        for element in world.iter('actor'):
            ped_num += 1
            index = int(element.get('name')[-1])
            # print ("{}, {}".format(element.tag, index))
            for p in element.iter('pose'):
                # print ("{}, {}".format(p.tag, p.text))
                px = str(s_list[index][0])
                py = str(s_list[index][1])
                pose = [px, py, '1.02', '0', '0']
                p.text = ' '.join(pose)
            for t in element.iter('target'):
                # print ("{}, {}".format(t.tag, t.text))
                tx = str(g_list[index][0])
                ty = str(g_list[index][1])
                position = [tx, ty, '1.02']
                t.text = ' '.join(position)

        assert ped_num > len(s_list)

        tree.write(file_save, xml_declaration=True, encoding="utf-8", pretty_print=True)


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
    ped_world.set_ped_info(s, g, sd)
    ped_world.create_ped_world()