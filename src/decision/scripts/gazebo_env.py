#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import math
import sys
import time
import numpy as np
import threading
import numpy as np
from collections import namedtuple
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from Planner import *

ROS_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
LEARNING_PATH = '/home/lintao/.conda/envs/learning'
CONDA_PATH = '/home/lintao/anaconda3'

VERSION = sys.version_info.major
if VERSION == 2:
    import cv2
elif ROS_PATH in sys.path:
    sys.path.remove(ROS_PATH)
    import cv2
    from cv_bridge import CvBridge, CvBridgeError
    sys.path.append(ROS_PATH)


class gazebo_env():
    def __init__(self):
        rospy.init_node('gazebo_env')
        # self.point = namedtuple('point', ['name', 'x', 'y'])

        self.agent_name = 'agent'
        self.obs_name = 'obs'
        self.obs_goal_name = 'obs_goal'
        self.agent_goal_name = "agent_goal"
        self.agent_goal = {'x':10, 'y':10}
        self.agent_position = {'x':0, 'y':0}
        self.gazebo_obs_states = [{'x':0, 'y':0, 'vx':0, 'vy':0, 'yaw':0}]
        self.obs_start_list = None
        self.obs_goal_list = None
        self.agent_point_set = None

        self.bridge = CvBridge()
        self.odom, self.rgb_image_raw, self.laser_scan_raw = None, None, None
        self.image_data_set, self.laser_data_set = [], []
        self.info = 0
        self.done = False

        self.actions = [[0.5, 0.5], [0.5, 0.2], [0.5, 0.0],
                        [0.5, -0.2], [0.5, -0.5], [0.2, 0.5],
                        [0.2, 0.0], [0.2, -0.5], [0.0, -0.5],
                        [0.0, 0.0], [0.0, 0.5]]
        self.n_actions = len(self.actions)
        self.goal_dist_last, self.goal_dist = 0, 0
        self.euclidean_distance = lambda p1, p2: math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
        
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
        # self.store_data_size = self.num_sikp_frame * self.num_stack_frame

        odom_ = rospy.get_param('/topics/odom')
        laser_ = rospy.get_param('/topics/laser_scan')
        agent_cmd_ = rospy.get_param('/topics/agent_cmd')
        gazebo_states_ = rospy.get_param('/topics/gazebo_states')
        rgb_image_ = rospy.get_param('/topics/rgb_image')
        gazebo_set_ = rospy.get_param('/topics/gazebo_set')

        self._check_all_sensors_ready(odom_, laser_, rgb_image_, gazebo_states_)
        rospy.Subscriber(odom_, Odometry, self._odom_callback)
        rospy.Subscriber(laser_, LaserScan, self._laser_callback)
        rospy.Subscriber(rgb_image_, Image, self._image_callback)
        rospy.Subscriber(gazebo_states_, ModelStates, self._gazebo_states_callback, queue_size=1)
        self.pub_agent = rospy.Publisher(agent_cmd_, Twist, queue_size=1)
        self.pub_gazebo = rospy.Publisher(gazebo_set_, ModelState, queue_size=1)

        self.cmd_vel = Twist()
        # self.cmd_vel.linear.x = 0
        # self.cmd_vel.angular.z = 0
        self.cmd_vel_last = {'v':0, 'w':0}
        self.agent_state_ = ModelState()
        self.obs_state_ = ModelState()

        self.action = [0, 0] # init action
        self.action_done = False
        self.action_count = 0

        # add_thread = threading.Thread(target = self.thread_job)
        # add_thread.start()
        # rospy.spin()

    def thread_job(self):
        print('start spin!')
        while not rospy.is_shutdown():
            rospy.spin()

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


    #region topic_callback
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
        # TODO: 逻辑欠妥
        self.gazebo_obs_states = [{'x':0, 'y':0, 'vx':0, 'vy':0, 'yaw':0} for name in data.name if self.obs_name in name and not self.obs_goal_name in name]

        for i in range(len(data.name)):
            p_x = data.pose[i].position.x
            p_y = data.pose[i].position.y
            name = str(data.name[i])
            if self.obs_name in name and not self.obs_goal_name in name:
                index = int(data.name[i][-1])
                v_x = data.twist[i].linear.x
                v_y = data.twist[i].linear.y
                self.gazebo_obs_states[index]['x'] = p_x
                self.gazebo_obs_states[index]['y'] = p_y
                self.gazebo_obs_states[index]['vx'] = v_x
                self.gazebo_obs_states[index]['vy'] = v_y
                self.gazebo_obs_states[index]['yaw'] = self.euler_from_quaternion(data.pose[i])[2]

            elif name == self.agent_goal_name:
                self.agent_goal['x'] = p_x
                self.agent_goal['y'] = p_y
            elif name == self.agent_name:
                self.agent_position['x'] = p_x
                self.agent_position['y'] = p_y

        self.goal_dist = self.euclidean_distance(self.agent_position, self.agent_goal)
        # print ('state_x is {}'.format(self.gazebo_obs_states[-1]['x']))
        # self.pub_gazebo_states()

        # info test
        # print ('========')
        # for item in self.gazebo_obs_states:
        #     print ('index {}, x={:.2f}, y={:.2f}'.format(self.gazebo_obs_states.index(item), item['x'], item['y']))
        # print(self.agent_goal)
        # print(self.agent_position)
    #endregion

    def gazebo_pub(self, state):
        self.pub_gazebo.publish(state)

    def agent_pub(self, cmd_vel):
        self.pub_agent.publish(cmd_vel)

    def pub_gazebo_states(self):
        self.obs_state_.model_name = 'obs0'
        self.obs_state_.reference_frame = 'world'
        # self.obs_state_.twist.linear.x = 5
        # self.obs_state_.twist.linear.y = 5

        self.obs_state_.pose.position.x += 1
        self.obs_state_.pose.position.y = 0
        self.obs_state_.pose.position.z = 0
        if self.gazebo_obs_states[0]['x'] > 8:
            self.obs_state_.pose.position.x = 2

        # self.agent_state_.model_name = self.agent_name
        # self.agent_state_.pose.position.x = 0
        # self.agent_state_.pose.position.y = 0
        # self.agent_state_.twist.linear.x = 0
        # self.agent_state_.twist.angular.z = 0
        # self.pub_state.publish(self.obs_state_)

        print ('pub state position ')
        time.sleep(0.2)

    def euler_from_quaternion(self, pose):
        x = pose.orientation.x 
        y = pose.orientation.y 
        z = pose.orientation.z 
        w = pose.orientation.w 
        Roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2+y**2))
        Pitch = math.asin(2 * (w * y - z * x))
        Yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return [Roll, Pitch, Yaw] # [r p y]

    def quaternion_from_euler(self, y, p=0, r=0):
        q3 = math.cos(r / 2) * math.cos(p / 2) * math.cos(y / 2) + \
            math.sin(r / 2) * math.sin(p / 2) * math.sin(y / 2)
        q0 = math.sin(r / 2) * math.cos(p / 2) * math.cos(y / 2) - \
            math.cos(r / 2) * math.sin(p / 2) * math.sin(y / 2)
        q1 = math.cos(r / 2) * math.sin(p / 2) * math.cos(y / 2) + \
            math.sin(r / 2) * math.cos(p / 2) * math.sin(y / 2)
        q2 = math.cos(r / 2) * math.cos(p / 2) * math.sin(y / 2) - \
            math.sin(r / 2) * math.sin(p / 2) * math.cos(y / 2)
        return [q0, q1, q2, q3] # [x y z w]

    # reinit start list & goal list for random or given list
    def reinit_env(self, obs_start_lis=None, obs_goal_list=None): 
        self.obs_start_list = obs_start_lis
        self.obs_goal_list = obs_goal_list
        self.set_obs_init_position()
        self.set_obs_goal_position()
        self.set_agent_goal_point()

    # reset env to begin state for start list & goal list taken
    def reset_env(self):
        assert self.obs_start_list.any() and self.obs_goal_list.any()
        self.set_obs_init_position()
        # self.set_obs_goal_position()

    def set_obs_init_position(self):
        obs_num = len(self.gazebo_obs_states)
        # get init position
        if self.obs_start_list is None:
            self.obs_start_list = self._get_random_position(obs_num)
        # set init position
        # rospy.logdebug("reset obs init position!")
        print ("reset obs init position!")
        self._pub_gazebo_states(self.obs_name, self.obs_start_list, obs_num)

    def set_obs_goal_position(self):
        obs_num = len(self.gazebo_obs_states)
        assert self.obs_start_list.any() # assert start_list is not None
        if self.obs_goal_list is None:
            while True:  # keep the goal point is not too close to start point for obs
                done = True
                self.obs_goal_list = self._get_random_position(obs_num)
                for i in range(obs_num):
                    start_x = self.obs_start_list[i][0]
                    start_y = self.obs_start_list[i][1]
                    goal_x = self.obs_goal_list[i][0]
                    goal_y = self.obs_goal_list[i][1]
                    dist_star_to_goal = math.hypot(start_x - goal_x, start_y - goal_y)
                    if dist_star_to_goal < 2:
                        done = False
                        break
                if done:
                    break
        # rospy.logdebug("reset obs goal position!")
        print ("reset obs goal position!")
        self._pub_gazebo_states(self.obs_goal_name, self.obs_goal_list, obs_num)

    def set_agent_goal_point(self, goal_point=None):
        obs_num = len(self.gazebo_obs_states)
        if not goal_point:
            while True:
                done = True
                point = np.random.uniform(-10, 10, 2)
                goal_point = np.append(point, math.pi) # set theta = pi
                pose_list = np.vstack((self.obs_start_list, self.obs_goal_list))
                for i in range(obs_num * 2):
                    dist_obs = math.hypot(goal_point[0] - pose_list[i][0], goal_point[1] - pose_list[i][1])
                    dist_agent = math.hypot(goal_point[0]-self.agent_position['x'], goal_point[0] - self.agent_position['y'])
                    if dist_obs < 1 or dist_agent < 7:
                        done = False
                        break
                if done:
                    break
        # rospy.logdebug("reset agent goal position!")
        print ("reset agent goal position!")
        self._pub_gazebo_states(self.agent_goal_name, [goal_point], 1)

    def _get_random_position(self, num, width=10, safe_dist=2):
        while True:
            x_list = np.random.uniform(-width, width, num)
            y_list = np.random.uniform(-width, width, num)
            done = True
            for i in range(num):
                for j in range(i+1, num):
                    dist_obs = math.hypot(x_list[i]-x_list[j], y_list[i]-y_list[j])
                    dist_agent = math.hypot(x_list[j]-self.agent_position['x'], y_list[j] - self.agent_position['y'])
                    if dist_obs < safe_dist or dist_agent < safe_dist:
                        done = False
                        break
                if not done:
                    break
            if done:
                theta_list = np.random.uniform(-math.pi, math.pi, num)
                return np.dstack((x_list, y_list, theta_list))[0]

    def _pub_gazebo_states(self, gazebo_state_name, pose_list, num):
        theta_list = [item[2] for item in pose_list]
        quat_list = [self.quaternion_from_euler(y) for y in theta_list]
        for i in range(num):
            self.obs_state_.model_name = gazebo_state_name + str(i)
            # print ('====model name is {}===='.format(gazebo_state_name + str(i)))
            self.obs_state_.pose.position.x = pose_list[i][0]
            self.obs_state_.pose.position.y = pose_list[i][1]
            # print ('x is {:.2f}, y is {:.2f}'.format(pose_list[i][0], pose_list[i][1]))
            self.obs_state_.pose.orientation.x = quat_list[i][0]
            self.obs_state_.pose.orientation.y = quat_list[i][1]
            self.obs_state_.pose.orientation.z = quat_list[i][2]
            self.obs_state_.pose.orientation.w = quat_list[i][3]
            self.gazebo_pub(self.obs_state_)
            # print ('reset obs{} / '.format(i))
            time.sleep(0.01)

    #region get_env_info
    def _get_state(self):# sensor data collection
        state_stack = []
        
        if self.image_data_set and self.laser_data_set:
            image_stack = np.zeros((self.img_size, self.img_size, self.num_stack_frame)) # (80, 80, 4)
            laser_stack = np.zeros((self.num_stack_frame, self.laser_size)) # (4, 360)

            for i in range(self.num_stack_frame):
                index = -1 - i * self.num_sikp_frame
                # if abs(index) > len(self.image_data_set):
                #     index = 0
                # image_stack[:, :, -1 - i] = self.image_data_set[index]
                
                image_stack[:, :, -1 - i] = self.image_data_set[index if abs(index) < len(self.image_data_set) else 0]

                # index = -1 - i * self.num_sikp_frame
                # if abs(index) > len(self.laser_data_set):
                #     index = 0
                # laser_stack[-1 - i, :] = self.laser_data_set[index]
                laser_stack[-1 - i, :] = self.laser_data_set[index if abs(index) < len(self.laser_data_set) else 0]
            state_stack = [image_stack, laser_stack]
        
        return state_stack

    def _get_reward(self):
        reward = 0
        reward += self.reward_cmd_vel_change()
        reward += self.reward_close_to_goal()

        if self.info == 1: # coll
            reward += self.reward_near_obs
        if self.info == 2: # arrive at goal
            reward += self.reward_near_goal
        return reward

    def reward_cmd_vel_change(self, rate=-0.1):
        # print (type(self.cmd_vel_last['v']), type(self.cmd_vel.linear.x))
        cmd_vel_reward = abs(self.cmd_vel_last['v'] - self.cmd_vel.linear.x) + abs(self.cmd_vel_last['w'] - self.cmd_vel.angular.z)
        # print ('cmd_vel_change: {:.3f} / '.format(cmd_vel_reward))
        return rate * cmd_vel_reward
        
    def reward_close_to_goal(self, rate=2):
        delta_dist = self.goal_dist_last - self.goal_dist if self.goal_dist_last != 0 else 0
        # print ('goal_dist: {:.3f}, goal_dist_last: {:.3f}'.format(self.goal_dist, self.goal_dist_last))
        # print ('delta_dist: {:.3f}'.format(delta_dist))
        return rate * delta_dist

    def _get_done(self):# arrived or collsiped or time_out
        self.done = False
        if self.info == 1 or self.info == 2: 
            self.done = True
        # print ('----done is {}'.format(self.done))
        return self.done

    def _get_info(self):
        self._set_info(0)
        self.check_near_goal(self.dist_near_goal)
        self.check_near_obs(self.dist_near_obs, self.dist_scan_set, 100)
        # print ('----info is {}'.format(self.info))
        return self.info

    def check_near_goal(self, dist_min):
        if self.goal_dist < dist_min:
            print ('=====!!!agent get goal at {:.2f}!!!====='.format(self.goal_dist))
            return self._set_info(2)
       
    def check_near_obs(self, dist_min, laser_min_set, scan_num):
        laser_min_count, laser_min, obs_dist_min = 0, 1000, 1000
        for obs in self.gazebo_obs_states:
            obs_dist = self.euclidean_distance(self.agent_position, obs)
            if obs_dist < obs_dist_min:
                obs_dist_min = obs_dist
        if obs_dist_min < dist_min:
            print ('----!!!robot too close to the obs for {:.2f}!!!-----'.format(obs_dist_min))
            return self._set_info(1)

        for r in self.laser_scan_raw:
            if r < laser_min:
                laser_min = r
            if r < laser_min_set:
                laser_min_count += 1
        # print ('------laser_min is {}'.format(laser_min))
        if laser_min_count > scan_num:
            print ('----!!!laser too close to the obs for {:.2f}, count {}!!!-----'.format(laser_min, laser_min_count))
            return self._set_info(1)

    def _set_info(self, num):
        self.info = num
    #endregion

    def reset(self): # init state and env
        print ('=============reset_env==============')
        self.agent_state_.model_name = self.agent_name
        self.agent_state_.pose.position.x = 0
        self.agent_state_.pose.position.y = 0
        self.agent_state_.twist.linear.x = 0
        self.agent_state_.twist.angular.z = 0
        self.gazebo_pub(self.agent_state_)
        # self.pub_gazebo.publish(self.agent_state_)
        self.action_count = 0
        # init state
        return self._get_state()
        # TODO-dynamic obs

    def step(self, action_index, tele_input=None):
        # rate = rospy.Rate(1)
        self.cmd_vel.linear.x = tele_input[0] if tele_input else self.actions[action_index][0]
        self.cmd_vel.angular.z = tele_input[-1] if tele_input else self.actions[action_index][1]
        print ('set linear vel {}, angular vel {}, index {}'.format(self.cmd_vel.linear.x, self.cmd_vel.angular.z, action_index))
        self.agent_pub(self.cmd_vel)
        # self.wait_until_twist_achieved(self.cmd_vel)
        self.action_count += 1
        
        start = time.time()
        # rate.sleep()
        end = time.time()
        # print ('during {}'.format(end - start))
        #XXX: to be tested
        during = 0.2
        # print ('wait for {} seconds'.format(during))
        time.sleep(during)

        info = self._get_info()
        done = self._get_done()
        if done: self.reset()
        reward = self._get_reward()
        state_ = self._get_state()

        self.cmd_vel_last['v'] = self.cmd_vel.linear.x
        self.cmd_vel_last['w'] = self.cmd_vel.angular.z
        self.goal_dist_last = self.goal_dist

        return state_, reward, done, info

    def wait_until_twist_achieved(self, cmd_vel_value, epsilon=0.15):
        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z

        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        angular_speed_plus = angular_speed + epsilon
        angular_speed_minus = angular_speed - epsilon

        while not rospy.is_shutdown():

            # current_odometry = self._check_odom_ready()
            current_odometry = self.get_odom()
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = current_odometry.twist.twist.angular.z
            # rospy.loginfo()
            print ('Current is {:.2f}/{:.2f}, goal is {:.2f}/{:.2f}'.format(odom_linear_vel, odom_angular_vel, linear_speed, angular_speed))
            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (odom_angular_vel > angular_speed_minus)

            if linear_vel_are_close and angular_vel_are_close:
                print ('Achieved speed {:.2f}/{:.2f}, goal is {:.2f}/{:.2f}'.format(odom_linear_vel, odom_angular_vel, linear_speed, angular_speed))
                # rospy.loginfo('Have achieve twist speed')
                break

    def get_odom(self):
        return self.odom

    def env_destory(self):
        self.cmd_vel.linear.x = 0
        self.cmd_vel.angular.z = 0
        self.agent_pub(self.cmd_vel)


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
    # gazebo_env = gazebo_env()
    # env = env()
    env = gazebo_env()
    print ('---before while---')
    env.reset()

    while not rospy.is_shutdown():

    #======test basic_logic======
    #     if env.action_count > 1000:
    #         env.reset()
    #     choose_action = np.random.randint(1, 10)
    #     print ('choose_action {}, count {}'.format(choose_action, env.action_count))
    #     s_, r, d, i = env.step(choose_action)

    #======test reward=======#
        speed = 0.2
        move = {'w': [speed, 0],
                'a': [0, speed],
                's': [-speed, 0],
                'd': [0, -speed]}
        r_list = []
        action_n = 0
        print ('===test reward, wait for tele_input===')
        key = get_key()
        if key in move.keys():
            state_, reward, done, info = env.step(0, move[key])
            action_n += 1
            r_list.append(reward)
            # print ('reward {:.3f}, info {}'.format(reward, info))

            if done:
                r = get_totoal_reward(r_list, 0.9)
                print ('----action_n: {}, total_reward: {:.3f}'.format(action_n, r))
                r_list = []
                action_n = 0

        else:
            print ('!!!stop move!!!')
            env.step(0, [0, 0])

        if key == 'r':
            # env.set_obs_init_position()
            env.reinit_env()
        if key == 't':
            env.reset_env()
            # p = [[2, 3], [4, -6], [8, -5]]
            # env.set_obs_init_position(p)
    #====== test odom =======#
    #     env._check_odom_ready()
    #     odom = env.get_odom()
    #     linear_vel = odom.twist.twist.linear.x
    #     angular_vel = odom.twist.twist.angular.z
    #     print ('linear_vel is {:.2f}, angular_vel is {:.2f}'.format(linear_vel, angular_vel))

    #====== test safe_dist======#   
    #     env.check_near_obs(0, 0, 1000)
    #     time.sleep(0.5)

    #====== check gazebo state info=======#
        # v_x = env.gazebo_obs_states[-1]['vx']
        # v_y = env.gazebo_obs_states[-1]['vy']
        # env.pub_gazebo()
        # rospy.loginfo('linear.x is {:.2f}, linear.y is {:.2f}'.format(v_x, v_y))
        # time.sleep(0.5)

    rospy.spin()