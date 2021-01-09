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

LEARNING_PATH = '/home/lintao/.conda/envs/learning'
CONDA_PATH = '/home/lintao/anaconda3'
ROS_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
VERSION = sys.version_info.major
if VERSION == 2:
    import cv2
elif ROS_PATH in sys.path:
    sys.path.remove(ROS_PATH)
    import cv2
    from cv_bridge import CvBridge, CvBridgeError
    sys.path.append(ROS_PATH)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(5, 5), stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2,2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2,2)),
        )
    def forward(self, x):
        x = x.transpose(2, 3)
        print(f'shape raw {x.shape}')
        x = self.conv1(x)
        print(f'shape conv1 {x.shape}')
        x = self.conv2(x)
        print(f'shape conv2 {x.shape}')
        x = self.conv3(x)
        print(f'shape conv3 {x.shape}')
        x = x.flatten(start_dim=1)
        print(f'shape flatten {x.shape}')
        return x

net = ConvNet()
img_ = deque(maxlen=4)
laser_ = deque(maxlen=4)
max_scan = 3
# assert 0

def imagecallback(data):
    print ('==========')
    # print(f'Image Height: {data.height}, Width: {data.width}, encoding: {data.encoding}')
    # img = np.array(data.data, dtype=np.uint8)
    img = bridge.imgmsg_to_cv2(data)
    # print(type(img))
    # print(f'img size {img.shape}')

    w, h, c = img.shape
    # print(w, h, c)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (w // 10, h // 10)) # 

    img_.append(img)
    if len(img_) < img_.maxlen:
        return
    # print(np.array(img_).shape)
    # assert 0
    img_tensor = torch.tensor(img_, dtype=torch.float32).unsqueeze(0)
    x = net(img_tensor)

def cloudcallback(data):
    print ('---------------')
    # 480, 640, 20480 = 640 * 32
    print(f'Cloud Height: {data.height}, Width: {data.width}, row_step {data.row_step}')
    point_data = point_cloud2.read_points(data)
    print(point_data)
    # point = np.array(data.data)
    print(f'point type {type(point_data)}')
    store = []
    for p in point_data:
        store.append(p)
    store = np.array(store)
    print(store.shape)

def lasercallback(data):
    laser = data.ranges
    laser = np.clip(laser, 0, max_scan) / max_scan
    laser_.append(laser)
    if len(laser_) < laser_.maxlen:
        return
    laser_np = np.array(laser_)
    print(laser_np.shape)
    laser_np = laser_np.flatten()
    print(laser_np.shape)

# size {point_data.shape}, 
if __name__ == "__main__":
    # 'agent0/camera/depth/image_raw'
    topic1 = 'agent0/camera/rgb/image_raw'
    topic2 = 'agent0/camera/depth/points'
    topic3 = '/scan'
    rospy.init_node('test')
    while not rospy.is_shutdown():
        bridge = CvBridge()

        rospy.Subscriber(topic1, Image, imagecallback)
        rospy.Subscriber(topic3, LaserScan, lasercallback)
        rospy.spin()
    
