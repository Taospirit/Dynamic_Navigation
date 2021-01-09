#!/usr/bin/env python
import rospy
import math
import sys
import time
import numpy as np
import threading
# import pprint
from collections import namedtuple, deque
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image, PointCloud2
from sensor_msgs import point_cloud2
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates, ContactsState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

def euler_from_quaternion(pose):
    x = pose.orientation.x 
    y = pose.orientation.y 
    z = pose.orientation.z 
    w = pose.orientation.w 
    Roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2+y**2))
    Pitch = np.arcsin(2 * (w * y - z * x))
    Yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return [Roll, Pitch, Yaw] # [r p y]

import os
num = 7
file = os.path.abspath(os.path.dirname(__file__)) + '/task' + str(num) +'.txt'
# w(file_name, info)
info_type = '#agent_x, agent_y, agent_t, agent_v, agent_w, obs1_x, obs1_y, obs1_t, obs1_v, obs2_x, obs2_y, obs2_t, obs2_v'

with open(file, 'w') as f:
    f.write(info_type+'\n')

def gazebo_callback(data):
    agent_info = ''
    obs_info = ''
    info = ''
  
    for i in range(len(data.name)):
        x = data.pose[i].position.x
        y = data.pose[i].position.y
        vx = data.twist[i].linear.x
        vy = data.twist[i].linear.y
        v = np.hypot(vx, vy)
        w = data.twist[i].angular.z
        t  = np.degrees(euler_from_quaternion(data.pose[i])[2]) # normalized, 机器人朝向与x轴方向的夹角, 单位为度
        name = str(data.name[i])
        
        if abs(x) < 10 and abs(y) < 10:
            if name[:-1] == 'agent':
                agent_info = f'a:{x:.3f}, {y:.3f}, {t:.3f}, {v:.3f}, {w:.3f};'

            if name[:-1] == 'obs':
                obs_info += f'o:{x:.3f}, {y:.3f}, {t:.3f}, {v:.3f}, {w:.3f};'
           
    info = agent_info + obs_info
    print(info)

    with open(file, 'a') as f:
        f.write(info+'\n')
    time.sleep(0.1)
    # self.goal_dist = self.euclidean_distance(self.agent_pose, self.agent_goal)


# size {point_data.shape}, 
if __name__ == "__main__":
    # 'agent0/camera/depth/image_raw'
    topic1 = 'agent0/camera/rgb/image_raw'
    topic2 = 'agent0/camera/depth/points'
    topic3 = '/scan'
    gazebo_states_ = '/gazebo/model_states'
    rospy.init_node('test')

    while not rospy.is_shutdown():
        # bridge = CvBridge()

        rospy.Subscriber(gazebo_states_, ModelStates, gazebo_callback, queue_size=1)

        # rospy.Subscriber(topic1, Image, imagecallback)
        # rospy.Subscriber(topic3, LaserScan, lasercallback)
        rospy.spin()
    
