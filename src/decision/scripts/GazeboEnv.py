#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import math
import sys
import time
import numpy as np
import threading
# import pprint
from collections import namedtuple
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates, ContactsState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from Planner import *

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


class GazeboEnv():
    def __init__(self):
        rospy.init_node('GazeboEnv')
        # self.point = namedtuple('point', ['name', 'x', 'y'])
        self.verbose = False
        self.agent_name = 'agent'
        self.obs_name = 'obs'
        # self.obs_name = 'ped'
        self.obs_goal_name = 'obs_goal'
        self.agent_goal_name = 'agent_goal'
        self.agent_goal = {'x':10, 'y':10}
        self.agent_pose = {'x':0, 'y':0, 'v':0, 'w':0, 't':0}
        self.gazebo_obs_states = [{'x':0, 'y':0, 'v':0, 'w':0, 't':0} for _ in range(10)]
        
        self.start_pose = None
        self.obs_start_list = None
        self.obs_goal_list = None
        self.goal_pose = None
        # self.agent_point_set = None
        state_type = ['raw_sensor', 'raw_env']
        self.state_type = state_type[1]

        self.start_default_list = [[-16, y, 0] for y in range(-10, 11, 2) if y != 0]
        self.goal_default_list = [[16, y, 0] for y in range(-10, 11, 2) if y != 0]
        self.agent_start_default = [0, 0]
        self.agent_goal_default = [5, 0]

        self.bridge = CvBridge()
        self.odom, self.rgb_image_raw, self.laser_scan_raw = None, None, None
        self.image_data_set, self.laser_data_set = [], []
        self.info = 0
        self.done = False
        self.collapsed = False
        speed_x, speed_z = 1.5, 1.5
        self.observation_space = np.array([])
        self.action_space = np.array([[speed_x, speed_z], [speed_x, speed_z/2], [speed_x, 0.0],
                        [speed_x, -speed_z/2], [speed_x, -speed_z], [speed_x/2, speed_z],
                        [speed_x/2, 0.0], [speed_x/2, -speed_z], [0.0, -speed_z],
                        [0.0, 0.0], [0.0, speed_z/2]])
        # self.n_actions = len(self.action_space)
        self.goal_dist_last, self.goal_dist = 0, 0
        self.step_count = 0
        self.step_count_limit = 50 # max_step_count

        self.euclidean_distance = lambda p1, p2: np.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
        
        self.laser_sacn_clip = rospy.get_param("/dist/laser_sacn_clip")
        self.dist_near_goal = rospy.get_param("/dist/near_goal")
        self.dist_near_obs = rospy.get_param("/dist/near_obs")
        self.dist_scan_set = rospy.get_param("/dist/min_scan")

        self.laser_size = rospy.get_param("/params/laser_size")
        self.img_size = rospy.get_param("/params/img_size")
        self.num_sikp_frame = rospy.get_param("/params/num_sikp_frame")
        self.num_stack_frame = rospy.get_param("/params/num_stack_frame")
        self.reward_near_goal = rospy.get_param("/params/reward_near_goal")
        self.reward_near_obs = rospy.get_param("/params/reward_near_obs")

        # get topic name from param server
        odom_ = rospy.get_param('/topics/odom')
        laser_ = rospy.get_param('/topics/laser_scan')
        agent_cmd_ = rospy.get_param('/topics/agent_cmd')
        gazebo_states_ = rospy.get_param('/topics/gazebo_states')
        rgb_image_ = rospy.get_param('/topics/rgb_image')
        gazebo_set_ = rospy.get_param('/topics/gazebo_set')
        dumper_ = '/agent0/base_bumper'
        self._check_all_sensors_ready(odom_, laser_, rgb_image_, gazebo_states_)
        rospy.Subscriber(odom_, Odometry, self._odom_callback)
        rospy.Subscriber(laser_, LaserScan, self._laser_callback)
        rospy.Subscriber(rgb_image_, Image, self._image_callback)
        rospy.Subscriber(gazebo_states_, ModelStates, self._gazebo_states_callback, queue_size=1)
        rospy.Subscriber(dumper_, ContactsState, self._dumper_callback, queue_size=1)
        self.pub_agent = rospy.Publisher(agent_cmd_, Twist, queue_size=1)
        self.pub_gazebo = rospy.Publisher(gazebo_set_, ModelState, queue_size=1)

        self.cmd_vel = Twist()
        self.cmd_vel_last = {'v':0, 'w':0}
        self.agent_state_ = ModelState()
        self.obs_state_ = ModelState()
        self.planner = APFM()
        # parameter to be fixed
        self.planner.set_katt(2.0)
        self.planner.set_krep(5)
        self.planner.set_linear_max(0.8)
        self.planner.set_linear_min(0.2)
        self.planner.set_angluar_max(2.0)
        self.planner.set_angluar_min(0.2)
        self.planner.set_safe_dist(2.0)
        self.planner.set_arrived_dist(self.dist_near_goal)
        self.planner_agent_influence = False # not consider agent when obs planning 

    #region check_topic
    def _check_all_sensors_ready(self, odom, scan, image, gazebo):
        rospy.logdebug('START ALL TOPIC READY')
        self._check_odom_ready(odom, 5)
        self._check_laser_ready(scan, 5)
        self._check_rgb_image_raw(image, 5)
        self._check_gazebo_state_info(gazebo, 5)
        rospy.logdebug('ALL TOPIC READY!')

    def _check_odom_ready(self, topic, time_out):
        self.odom = None
        rospy.logdebug("Waiting for {} to be READY...".format(topic))
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message(topic, Odometry, timeout=time_out)
                rospy.logdebug("Current {} READY=>".format(topic))
            except:
                rospy.logerr("Current {} not ready yet, retrying...".format(topic))
        return self.odom

    def _check_laser_ready(self, topic, time_out):
        self.laser_scan = None
        rospy.logdebug("Waiting for {} to be READY...".format(topic))
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message(topic, LaserScan, timeout=time_out)
                rospy.logdebug("Current {} READY".format(topic))
            except:
                rospy.logerr("Current {} not ready yet, retrying...".format(topic))
        return self.laser_scan

    def _check_rgb_image_raw(self, topic, time_out):
        self.rgb_image_raw = None
        rospy.logdebug("Waiting for {} to be READY...".format(topic))
        while self.rgb_image_raw is None and not rospy.is_shutdown():
            try:
                self.rgb_image_raw = rospy.wait_for_message(topic, Image, timeout=time_out)
                rospy.logdebug("Current {} READY".format(topic))
            except:
                rospy.logerr("Current {} not ready yet, retrying...".format(topic))
        return self.rgb_image_raw

    def _check_gazebo_state_info(self, topic, time_out):
        self.gazebo_state_info = None
        rospy.logdebug("Waiting for {} to be READY...".format(topic))
        while self.gazebo_state_info is None and not rospy.is_shutdown():
            try:
                self.gazebo_state_info = rospy.wait_for_message(topic, ModelStates, timeout=time_out)
                rospy.logdebug("Current {} READY".format(topic))
            except:
                rospy.logerr("Current {} not ready yet, retrying...".format(topic))
        return self.gazebo_state_info
    #endregion

    #region topic_callback & publisher
    def _odom_callback(self, data):
        self.odom = data

    def _laser_callback(self, data):
        self.laser_scan_raw = data.ranges
        laser_clip = np.clip(self.laser_scan_raw, 0, self.laser_sacn_clip) / self.laser_sacn_clip # normalization laser data
        laser_data = [(laser_clip[i] + laser_clip[i+1]) / 2 for i in range(0, len(laser_clip), 2)]    
        self.laser_data_set.append(laser_data)

        if len(self.laser_data_set) > self.num_sikp_frame * self.num_stack_frame: del self.laser_data_set[0] 

    def _image_callback(self, data):
        self.rgb_image_raw = self.bridge.imgmsg_to_cv2(data)
        img_data = cv2.resize(self.rgb_image_raw, (self.img_size, self.img_size)) # 80x80x3
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        img_data = np.reshape(img_data, (self.img_size, self.img_size))
        self.image_data_set.append(img_data)

        if len(self.image_data_set) > self.num_sikp_frame * self.num_stack_frame: del self.image_data_set[0]

    def _gazebo_states_callback(self, data):
        self.gazebo_state_info = data
        self.gazebo_obs_states = [{'x':0, 'y':0, 'v':0, 'w':0, 't':0} for name in data.name if name[:-1] == self.obs_name]
        for i in range(len(data.name)):
            x = data.pose[i].position.x
            y = data.pose[i].position.y
            vx = data.twist[i].linear.x
            vy = data.twist[i].linear.y
            v = np.hypot(vx, vy)
            w = data.twist[i].angular.z
            t = np.degrees(self.euler_from_quaternion(data.pose[i])[2]) # normalized, 机器人朝向与x轴方向的夹角, 单位为度
            name = str(data.name[i])

            #======test obs info correct,
            if name == 'obs0':
                self.x, self.y = x, y
                self.vx, self.vy, self.w = vx, vy, w
                self.robot_v, self.robot_theta = v, t
            #======test obs info correct,
                
            if name[:-1] == self.obs_name:
                index = int(data.name[i][-1])
                self.gazebo_obs_states[index]['x'] = x
                self.gazebo_obs_states[index]['y'] = y
                self.gazebo_obs_states[index]['v'] = v
                self.gazebo_obs_states[index]['w'] = w
                self.gazebo_obs_states[index]['t'] = t
                # if abs(x) < 10:
                #     print ('name is {}, position.x is {:.2f}, linear.x is {:.2f}, linear.y is {:.2f}, angular.z is {:.2f}'.format(name, x, v, data.twist[i].linear.y, w))
            elif name[:-1] == self.agent_name:
                self.agent_pose['x'] = x
                self.agent_pose['y'] = y
                self.agent_pose['v'] = v
                self.agent_pose['w'] = w
                self.agent_pose['t'] = t
            elif name[:-1] == self.agent_goal_name:
                self.agent_goal['x'] = x
                self.agent_goal['y'] = y
            
        self.goal_dist = self.euclidean_distance(self.agent_pose, self.agent_goal)

        # print ('state_x is {}'.format(self.gazebo_obs_states[-1]['x']))

        # info test
        # print ('========')
        # for item in self.gazebo_obs_states:
        #     print ('index {}, x={:.2f}, y={:.2f}'.format(self.gazebo_obs_states.index(item), item['x'], item['y']))
        # print(self.agent_goal)
        # print(self.agent_pose)

    def _dumper_callback(self, data):
        # self_collision_name = 'agent::base_footprint::base_footprint_fixed_joint_lump__base_link_collision_1'
        contacts = data.states
        self.collapsed = False
        for info in contacts:
            if 'agent' in info.collision1_name or 'agent' in info.collision2_name:
                self.collapsed = True
                break

    def gazebo_pub(self, state): # pub gazebo ped state for env pde state reset
        self.pub_gazebo.publish(state)

    def agent_pub(self, cmd_vel): # pub cmd_vel for agent robot in env step
        self.pub_agent.publish(cmd_vel)
    #endregion topic_callback & publisher

    #region math function
    def euler_from_quaternion(self, pose):
        x = pose.orientation.x 
        y = pose.orientation.y 
        z = pose.orientation.z 
        w = pose.orientation.w 
        Roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2+y**2))
        Pitch = np.arcsin(2 * (w * y - z * x))
        Yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return [Roll, Pitch, Yaw] # [r p y]

    def quaternion_from_euler(self, y, p=0, r=0):
        q3 = np.cos(r / 2) * np.cos(p / 2) * np.cos(y / 2) + \
            np.sin(r / 2) * np.sin(p / 2) * np.sin(y / 2)
        q0 = np.sin(r / 2) * np.cos(p / 2) * np.cos(y / 2) - \
            np.cos(r / 2) * np.sin(p / 2) * np.sin(y / 2)
        q1 = np.cos(r / 2) * np.sin(p / 2) * np.cos(y / 2) + \
            np.sin(r / 2) * np.cos(p / 2) * np.sin(y / 2)
        q2 = np.cos(r / 2) * np.cos(p / 2) * np.sin(y / 2) - \
            np.sin(r / 2) * np.sin(p / 2) * np.cos(y / 2)
        return [q0, q1, q2, q3] # [x y z w]

    def normalize_theta(self, theta):
        return math.degrees(math.atan2(math.sin(math.radians(theta)), math.cos(math.radians(theta))))

    #endregion math function

    #region: env setting and function related
    def init_default_env(self):
        'init obs to default start_list & goal_list for given attrib'
        self.init_env(self.agent_start_default, self.agent_goal_default, self.start_default_list, self.goal_default_list)

    def init_env(self, start_point, goal_point, obs_start_list, obs_goal_list):
        '''init new env: init obs start poses & goal poses for given list
        @start_point: start point (x, y) for agent
        @goal_point: goal point (x, y) for agent
        @obs_start_list: given list for obs start
        @obs_goal_list: given list for obs goal
        '''
        assert len(obs_start_list) == len(obs_goal_list)
        assert not len(obs_start_list) > len(self.gazebo_obs_states)
        self.set_agent_start_pose(start_point)
        self.set_obs_init_pose(obs_start_list)
        self.set_obs_goal_pose(obs_goal_list)
        self.set_agent_goal_pose(goal_point)

    def init_env_random(self, num=10):
        'init new env: init random obs for given number, default value is 10'
        # print (len(self.gazebo_obs_states))
        assert not num > len(self.gazebo_obs_states)
        self.set_agent_start_pose()
        self.set_obs_init_pose(num=num)
        self.set_obs_goal_pose(num=num)
        self.set_agent_goal_pose()

    def reset_env(self):
        'reset env to begin state for start_list & goal_list taken'
        assert self.obs_start_list is not None and self.obs_goal_list is not None
        # self.init_env(self.start_pose[:-1], self.goal_pose[:-1], self.obs_start_list, self.obs_goal_list)
        self.set_agent_start_pose(self.start_pose)
        self.set_obs_init_pose(self.obs_start_list)
        self.set_obs_goal_pose(self.obs_goal_list)

    def set_agent_start_pose(self, start_point=None):
        if start_point is None:
            start_point = np.random.uniform(-8, 8, 2)
        self.start_pose = list(np.append(start_point, 0)) # set theta to np.pi

        if self.verbose:
            print ("=====set agent start pose!=====")
        self._pub_gazebo_states(self.agent_name, [self.start_pose])

    def set_obs_init_pose(self, obs_start_list=None, num=10):
        # get init position
        if obs_start_list is None:
            obs_start_list = self._get_random_pose(num)

        obs_start_list.extend(self.start_default_list[len(obs_start_list):])
        # set init position
        # rospy.logdebug("reset obs init position!")
        self.obs_start_list = obs_start_list
        if self.verbose:
            print ("=====set obs init pose!=====")
        self._pub_gazebo_states(self.obs_name, self.obs_start_list)

    def set_obs_goal_pose(self, obs_goal_list=None, num=10):
        assert self.obs_start_list is not None
        if obs_goal_list is None:
            while True:  # keep the goal point is not too close to start point for obs
                done = True
                obs_goal_list = self._get_random_pose(num)
                for i in range(num):
                    start_x = self.obs_start_list[i][0]
                    start_y = self.obs_start_list[i][1]
                    goal_x = obs_goal_list[i][0]
                    goal_y = obs_goal_list[i][1]
                    dist_start2goal = np.hypot(start_x - goal_x, start_y - goal_y)
                    if dist_start2goal < 2:
                        done = False
                        break
                if done:
                    break

        obs_goal_list.extend(self.goal_default_list[len(obs_goal_list):])
        # rospy.logdebug("reset obs goal position!")
        self.obs_goal_list = obs_goal_list
        if self.verbose:
            print ("=====set obs goal pose!=====")
        self._pub_gazebo_states(self.obs_goal_name, self.obs_goal_list)

    def set_agent_goal_pose(self, goal_point=None):
        if goal_point is None:
            while True:
                done = True
                goal_point = np.random.uniform(-10, 10, 2)
                pose_list = np.vstack((self.obs_start_list, self.obs_goal_list))

                dist_agent = np.hypot(goal_point[0] - self.agent_pose['x'], goal_point[0] - self.agent_pose['y'])
                if dist_agent < 7:
                    done = False
                    continue
                for pose in pose_list:
                    dist_obs = np.hypot(goal_point[0] - pose[0], goal_point[1] - pose[1])
                    if dist_obs < 0.5:
                        done = False
                        break
                if done:
                    break
        self.goal_pose = list(np.append(goal_point, 0)) # set theta to np.pi
        # rospy.logdebug("reset agent goal position!")
        if self.verbose:
            print ("=====set agent goal pose!=====")
        self._pub_gazebo_states(self.agent_goal_name, [self.goal_pose])

    def _get_random_pose(self, num, width=10, safe_dist=2):
        while True:
            x_list = np.random.uniform(-width, width, num)
            y_list = np.random.uniform(-width, width, num)
            done = True
            for i in range(num):
                for j in range(i+1, num):
                    dist_obs = np.hypot(x_list[i]-x_list[j], y_list[i]-y_list[j])
                    dist_agent = np.hypot(x_list[j]-self.agent_pose['x'], y_list[j] - self.agent_pose['y'])
                    if dist_obs < safe_dist or dist_agent < safe_dist:
                        done = False
                        break
                if not done:
                    break
            if done:
                theta_list = np.random.uniform(-180, 180, num)
                return list(np.dstack((x_list, y_list, theta_list))[0])

    def _pub_gazebo_states(self, gazebo_state_name, pose_list, index_list=None):
        # pose_list = [[p1_x, p2_y, yaw1], [p2_x, p2_y, yaw2], [p3_x, p3_y, yaw3], ...]
        # print (pose_list)
        theta_list = [pose[2] for pose in pose_list]
        quat_list = [self.quaternion_from_euler(np.radians(theta)) for theta in theta_list]
        model_state = ModelState()

        pub_index_list = list(range(len(pose_list)))
        if index_list is not None:
            assert len(set(index_list)) == len(pose_list)
            for index in index_list:
                assert index < len(self.gazebo_obs_states)
            pub_index_list = index_list

        for i in pub_index_list:
            model_state.model_name = gazebo_state_name + str(i)
            model_state.pose.position.x = pose_list[i][0]
            model_state.pose.position.y = pose_list[i][1]
            model_state.pose.orientation.x = quat_list[i][0]
            model_state.pose.orientation.y = quat_list[i][1]
            model_state.pose.orientation.z = quat_list[i][2]
            model_state.pose.orientation.w = quat_list[i][3]
            if self.verbose:
                print ('=====model name is {}====='.format(gazebo_state_name + str(i)))
                print ('x is {:.2f}, y is {:.2f}'.format(pose_list[i][0], pose_list[i][1]))
                print ('theta is {:.2f}'.format(theta_list[i]))
            model_state.twist.linear.x = 0
            model_state.twist.linear.y = 0       
            model_state.twist.angular.z = 0
            self.gazebo_pub(model_state)

            time.sleep(0.02)


    #region get_env_info
    def _get_state(self):# sensor data collection
        # state_stack = []
        self.observation_space = np.array([])

        assert self.state_type, \
            'You should set state_type for raw_sensor or other.'
        if self.state_type == 'raw_sensor':
            assert self.image_data_set and self.laser_data_set
            image_stack = np.zeros((self.img_size, self.img_size, self.num_stack_frame)) # (80, 80, 4)
            laser_stack = np.zeros((self.num_stack_frame, self.laser_size)) # (4, 360)

            for i in range(self.num_stack_frame):
                index = -1 - i * self.num_sikp_frame
                image_stack[:, :, -1 - i] = self.image_data_set[index if abs(index) < len(self.image_data_set) else 0]
                laser_stack[-1 - i, :] = self.laser_data_set[index if abs(index) < len(self.laser_data_set) else 0]
            self.observation_space = np.array([image_stack, laser_stack])

        if self.state_type == 'raw_env':
            r = 0.5
            obs_num = len([obs for obs in self.obs_start_list if -10 < obs[0] < 10])

            point2goal_theta = math.degrees(math.atan2(self.agent_goal['y'] - self.agent_pose['y'],\
                 self.agent_goal['x'] - self.agent_pose['x']))
            agent2goal_theta = self.normalize_theta(self.agent_pose['t'] - point2goal_theta) # 左正右负
            agent_vx = self.agent_pose['v'] * math.cos(math.radians(self.agent_pose['t']))
            agent_vy = self.agent_pose['v'] * math.sin(math.radians(self.agent_pose['t']))

            self_state_stack = [self.goal_dist, agent2goal_theta, agent_vx, agent_vy, r]
            # s = [d_g, theta, v_ax, v_ay, r_a]

            obs_state_stack = []
            for i in range(obs_num):
                delta_x = self.gazebo_obs_states[i]['x'] - self.agent_pose['x']
                delta_y = self.gazebo_obs_states[i]['y'] - self.agent_pose['y']
                agent2obs_dist = math.hypot(delta_x, delta_y)
                agent2obs_theta = math.degrees(math.atan2(delta_y, delta_x))
                agent2obs_theta = self.normalize_theta(self.agent_pose['t'] - agent2obs_theta) # 左正右负
                obs_vx = self.gazebo_obs_states[i]['v'] * math.cos(math.radians(self.gazebo_obs_states[i]['t']))
                obs_vy = self.gazebo_obs_states[i]['v'] * math.sin(math.radians(self.gazebo_obs_states[i]['t']))
                obs_state = [agent2obs_dist, agent2obs_theta, obs_vx, obs_vy, r]
                # s_obs = [d_a, a_theta, v_x, v_y, r]
                obs_state_stack += obs_state
                # s_prime = [p_x, p_y, v_x, v_y, r_prime, d_a, r_peimr + r_a]
                        
            self.observation_space = np.array(self_state_stack + obs_state_stack) # shape=10
        
        return self.observation_space

    def _get_reward(self):
        reward = 0
        reward += self.reward_cmd_vel_change()
        reward += self.reward_close_to_goal()

        if self.info == 1: # coll
            reward += self.reward_near_obs
        if self.info == 2: # arrive at goal
            reward += self.reward_near_goal
        return reward

    def _get_done(self):# arrived or collsiped or time_out
        return self.info != 0

    def _get_info(self):
        self._set_info(0)

        if self.check_agent_collision():
            # print ('-----!!!robot collapsed!!!-----')
            self._set_info(1)
        if self.check_agent_at_goal(self.dist_near_goal):
            # print ('-----!!!agent get goal at {:.2f}!!!-----'.format(self.goal_dist))
            self._set_info(2)
        # self.check_agent_obs(self.dist_near_obs, self.dist_scan_set, 100)
        if self.check_obs_at_goal(self.dist_near_obs):
            # print ('-----!!!obs at goal!!!-----')
            self._set_info(3)

        if self.step_count > self.step_count_limit:
            # print (f'-----!!!step_count {self.step_count} over {self.step_count_limit}!!!-----')
            self.step_count = 0
            # print ('------!!!step_count over num!!!------')
            self._set_info(4)

        # print ('----info is {}'.format(self.info))
        return { }

    def reward_cmd_vel_change(self, rate=-0.1):
        # print (type(self.cmd_vel_last['v']), type(self.cmd_vel.linear.x))
        cmd_vel_reward = abs(self.cmd_vel_last['v'] - self.cmd_vel.linear.x) + abs(self.cmd_vel_last['w'] - self.cmd_vel.angular.z)
        # print ('cmd_vel_change: {:.3f} / '.format(cmd_vel_reward))
        return rate * cmd_vel_reward

    def reward_close_to_goal(self, rate=5):
        delta_dist = self.goal_dist_last - self.goal_dist if self.goal_dist_last != 0 else 0
        # print ('goal_dist: {:.3f}, goal_dist_last: {:.3f}'.format(self.goal_dist, self.goal_dist_last))
        # print ('delta_dist: {:.3f}'.format(delta_dist))
        return rate * delta_dist

    def check_agent_at_goal(self, dist_min):
        return self.goal_dist < dist_min

    def check_agent_collision(self):
        return self.collapsed

    def check_obs_at_goal(self, arrived_dist):
        obs_num = len([obs for obs in self.obs_start_list if -10 < obs[0] < 10])
        if obs_num == 0: # for no obs test only
            return 0
        done = True
        for i in range(obs_num):
            p_x = self.gazebo_obs_states[i]['x']
            p_y = self.gazebo_obs_states[i]['y']
            g_x = self.obs_goal_list[i][0]
            g_y = self.obs_goal_list[i][1]
            if np.hypot(p_x - g_x, p_y - g_y) > arrived_dist:
                done = False
                break
        return done

    def _set_info(self, num):
        self.info = num

    # def wait_until_twist_achieved(self, cmd_vel_value, epsilon=0.15):
    #     linear_speed = cmd_vel_value.linear.x
    #     angular_speed = cmd_vel_value.angular.z

    #     linear_speed_plus = linear_speed + epsilon
    #     linear_speed_minus = linear_speed - epsilon
    #     angular_speed_plus = angular_speed + epsilon
    #     angular_speed_minus = angular_speed - epsilon

    #     while not rospy.is_shutdown():

    #         # current_odometry = self._check_odom_ready()
    #         current_odometry = self.get_odom()
    #         odom_linear_vel = current_odometry.twist.twist.linear.x
    #         odom_angular_vel = current_odometry.twist.twist.angular.z
    #         # rospy.loginfo()
    #         print ('Current is {:.2f}/{:.2f}, goal is {:.2f}/{:.2f}'.format(odom_linear_vel, odom_angular_vel, linear_speed, angular_speed))
    #         linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
    #         angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (odom_angular_vel > angular_speed_minus)

    #         if linear_vel_are_close and angular_vel_are_close:
    #             print ('Achieved speed {:.2f}/{:.2f}, goal is {:.2f}/{:.2f}'.format(odom_linear_vel, odom_angular_vel, linear_speed, angular_speed))
    #             # rospy.loginfo('Have achieve twist speed')
    #             break

    def get_odom(self):
        return self.odom

    def set_verbose(self, vel:bool) -> bool:
        self.verbose = vel
    #endregion

    def reset(self): # init state and env
        if self.verbose:
            print ('=============env_reset==============')
        start_point = np.append(self.agent_start_default, 0)
        self._pub_gazebo_states(self.agent_name, [start_point])
        self.reset_env()
        # init state
        return self._get_state()
        # TODO-dynamic obs

    def step(self, action_index, tele_input=None):
        # rate = rospy.Rate(1)
        self.cmd_vel.linear.x = tele_input[0] if tele_input else self.action_space[action_index][0]
        self.cmd_vel.angular.z = tele_input[-1] if tele_input else self.action_space[action_index][1]
        if self.verbose:
            print ('set linear vel {}, angular vel {}, index {}'.format(self.cmd_vel.linear.x, self.cmd_vel.angular.z, action_index))
        self.agent_pub(self.cmd_vel)
        # self.wait_until_twist_achieved(self.cmd_vel)
        self.obs_planner_pub()

        # rate.sleep()
        start = time.time()
        end = time.time()
        
        step_interval = 0.5 # openai_ros 0.2
        # time.sleep(during)
        
        begin = time.time()
        while time.time() - begin < step_interval: # trade_off 
            info = self._get_info() # set diff info
            done = self._get_done() # judge done or not
            if done:
                break

        reward = self._get_reward()
        state_ = self._get_state()

        self.cmd_vel_last['v'] = self.cmd_vel.linear.x
        self.cmd_vel_last['w'] = self.cmd_vel.angular.z
        self.goal_dist_last = self.goal_dist
        self.step_count += 1

        return state_, reward, done, info
    
    def obs_planner_pub(self):
        obs_num = len([obs for obs in self.obs_start_list if -10 < obs[0] < 10])
        for i in range(obs_num):
            cmd_vel = Twist()
            self_pose = self.gazebo_obs_states[i]
            goal_pose = {'x':self.obs_goal_list[i][0], 'y':self.obs_goal_list[i][1]}
            if self.planner_agent_influence:
                obs_pose_list = [self.agent_pose if p is self_pose else p for p in self.gazebo_obs_states] # consider the agent influence actively
            else:
                obs_pose_list = [p for p in self.gazebo_obs_states if p is not self_pose] # not consider the agent influence
            self.planner.init_env(self_pose, goal_pose, obs_pose_list)
        #==== robot tf
            cmd_vel.linear.x, cmd_vel.angular.z = self.planner.get_cmd()
            # print (f'cmd_vel_x  {round(cmd_vel.linear.x, 2)}, cmd_vel_z {round(cmd_vel.angular.z, 2)}')
            topic = '/' + self.obs_name + str(i) + '/cmd_vel'
            pub = rospy.Publisher(topic, Twist, queue_size=1)
            pub.publish(cmd_vel)


def get_key():
    key = input()
    if key == 'p':
        return -1
    if key == 'w':
        print ('step forward')
    elif key == 's':
        print ('step backward')
    elif key == 'a':
        print ('turn left')
    elif key == 'd':
        print ('turn right')
    # print ('get key:{} from tele, type is {}'.format(key, type(key)))
    return key

def get_totoal_reward(r_l, gamma):
    if len(r_l) == 1:
        return r_l[0]
    else:
        return r_l.pop(0) + gamma * get_totoal_reward(r_l, gamma)

if __name__ == "__main__":
    env = GazeboEnv()
    env.set_verbose(False)
    print ('---before while---')
    # env.init_default_env()
    print ('=====befor reset=====')
    # env.reset()
    agent_start = [0, 0]
    agent_goal = [8, 0]

    # s1 = [[4, 4, 0]]
    s1 = [[4, 4, -90]]
    g1 = [[4, -4, 0]]

    s1_1 = [[2, 4, 0]]
    g1_1 = [[2, -4, 0]]

    s2 = [[3, 6, 0]]
    g2 = [[-6, -5, 0]]

    s3 = [[5, 3, 0], [-1, 3, 0]] # 双机平行
    g3 = [[6, -3, 0], [0, -3, 0]]

    s4 = [[2, 1.5, -45], [3, -2, 45]] # 双机交叉
    g4 = [[6, -3, 0], [8, 4, 0]]

    s5 = [[-2, 6, 0]] # 斜边，过静态
    g5 = [[2, -6, 0]]

    s6 = [[-1, -5, 0], [4, -5, 0]] # 平行， 过静态 对照
    g6 = [[-1, 5, 0], [4, 5, 0]]

    s_ = [[1, 0.4, 0]]
    g_ = [[15, 0.4, 0]]

    s7 = [[1, 1, -45], [3, -3, 45]]
    g7 = [[3, -2, 0], [5, 3, 0]]

    s_cross = [[8, 0, 180]]
    g_cross = [[0, 0, 0]]



    #=======test APFM planner=====
    # env.init_env(agent_start, agent_goal, s_cross, g_cross)
    # print ('tttttttttttttttttttt')
    # state_ = env.reset()
    # while not rospy.is_shutdown():
    #     action = [0, 0]
    #     state_, reward, done, info = env.step(0, action)
    #     if done:
    #         state_ = env.reset()

    #=========test obs info get correct or not, velocity is not correct!
    # env.init_env_random(1)
    # cmd_vel = Twist()
    # cmd_vel.linear.x = 1.0
    # count = 0
    # while not rospy.is_shutdown():
    #     obs_0_x = env.x
    #     # print (f'obs_0_x {obs_0_x}')

    #     if obs_0_x > 3:
    #         cmd_vel.linear.x = -1.0
    #     elif obs_0_x < -3:
    #         cmd_vel.linear.x = 1.0
    #     # if abs(obs_0_x) > 3:
    #     #     cmd_vel.linear.x *= -1


    #     pub = rospy.Publisher('/obs0/cmd_vel', Twist, queue_size=1)
    #     pub.publish(cmd_vel)
    #     # print (count)
    #     # time_list = [0, time.time()]
    #     # px_ = [0, 0]
    #     # py_ = [0, 0]
    #     if count % 30000 == 0:
    #         print ('--------------info-----------------')
    #         print (f'obs_0_x {obs_0_x}')
    #         print (f'pub x {cmd_vel.linear.x}')
    #         print (f'x: {round(env.x, 2)}, y: {round(env.y, 2)}')
    #         print (f'gz vx: {round(env.vx, 2)}, gz vy: {round(env.vy, 2)}, gz w: {round(env.w, 2)}')
    #         print (f'robot v: {round(env.robot_v, 2)}, robot theta: {round(env.robot_theta, 2)}')

    #         # theta = math.atan2(env.vy, env.vx)
    #         # t = math.degrees(math.atan2(math.sin(theta), math.cos(theta)))
    #         # print (f'x: {env.x}, y: {env.y}')
    #         # print (f'gz vx: {env.vx}, gz vy: {env.vy}, gz w: {env.w}, clc t: {t}')
    #         # print (f'gz v: {env.robot_v}, gz theta: {env.robot_theta}')
    #         # if count > 3:
    #         #     during = time_list[1]-time_list[0]
    #         #     real_x_speed = (px_[1] - px_[0])/during
    #         #     real_y_speed = (py_[1] - py_[0])/during
    #         #     real_v_speed = np.hypot(real_x_speed, real_y_speed)
    #         #     print (f'during {during}')
    #         #     print (f'real_x_speed {real_x_speed}, real_y_speed {real_y_speed}, real_v_speed {real_v_speed}')
    #         print (f'speed rate: really / simu {1/env.robot_v}')
    #         # print (env._get_state())
        
    #     count += 1


    #=======test twist speed======

    # agent_s, agent_g =[0, 0], [5, 0]
    # obs_s, obs_g = [[6, 0, 180]], [[-1, 0, 0]]
    # env_scene = [agent_s, agent_g, obs_s, obs_g]
    # env.init_env(env_scene[0], env_scene[1], env_scene[2], env_scene[3])

    # # start = time.time()
    # # sd_list = []
    # # count = 0
    # while not rospy.is_shutdown():
    #     state_, reward, done, info = env.step(0, [1, 0])
    #     if done:
    #         # end = time.time()
    #         # during = end - start
    #         # speed = (env.agent_goal['x'] - 0.5) / during
    #         # print ('during {:.2f}s, speed is {:.2f}'.format(during, speed))

    #         # sd_list.append(speed)
    #         # count += 1
    #         # start = end

    #         # if count % 10 == 0:
    #         #     print ('mean speed is {:.2f}'.format(speed))
    #         env.reset()

    #======test basic_logic======
    #     if env.step_count > 1000:
    #         env.reset()
    #     choose_action = np.random.randint(1, 10)
    #     print ('choose_action {}, count {}'.format(choose_action, env.step_count))
    #     s_, r, d, i = env.step(choose_action)

    #======test reward=======#
        # key = get_key()
        # speed = 0.2
        # move = {'w': [speed, 0],
        #         'a': [0, speed],
        #         's': [-speed, 0],
        #         'd': [0, -speed]}
        # r_list = []
        # action_n = 0
        # print ('===test reward, wait for tele_input===')
        # if key in move.keys():
        #     state_, reward, done, info = env.step(0, move[key])
        #     action_n += 1
        #     r_list.append(reward)
        #     # print ('reward {:.3f}, info {}'.format(reward, info))

        #     if done:
        #         r = get_totoal_reward(r_list, 0.9)
        #         print ('----action_n: {}, total_reward: {:.3f}'.format(action_n, r))
        #         r_list = []
        #         action_n = 0
        # else:
        #     print ('!!!stop move!!!')
        #     env.step(0, [0, 0])
    
    #=======test env_update======#
        # key = get_key()
        # if key == 'r': # random set position for num, num default value is 10
        #     print ('random init all obs, 10!')
        #     env.init_env_random()
        # if key == 't':
        #     print ('random init 4 obs!')
        #     env.init_env_random(4)
        # if key == 'y':
        #     print ('reset env~')
        #     env.reset_env()
        # if key == 'u':
        #     print ('set given points')
        #     obs_start = [[5, 5, 0], [7, 5, 0]]
        #     obs_goal = [[5, -5, 0], [7, -5, 0]]
        #     goal = [8, 0]
        #     env.init_env(obs_start, obs_goal, goal)
        # if key == 'm':
        #     env.reset()
           
    #====== test odom =======#
    #     env._check_odom_ready()
    #     odom = env.get_odom()
    #     linear_vel = odom.twist.twist.linear.x
    #     angular_vel = odom.twist.twist.angular.z
    #     print ('linear_vel is {:.2f}, angular_vel is {:.2f}'.format(linear_vel, angular_vel))

    #====== test safe_dist======#   
    #     env.check_agent_obs(0, 0, 1000)
    #     time.sleep(0.5)

    #====== check gazebo state info=======#
        # v_x = env.gazebo_obs_states[-1]['vx']
        # v_y = env.gazebo_obs_states[-1]['vy']
        # env.pub_gazebo()
        # rospy.loginfo('linear.x is {:.2f}, linear.y is {:.2f}'.format(v_x, v_y))
        # time.sleep(0.5)
    print ('stop~')
    rospy.spin()