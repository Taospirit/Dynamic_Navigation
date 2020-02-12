#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import math
import sys
import time
import numpy as np
import threading
from collections import namedtuple
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
learning_path = '/home/lintao/.conda/envs/learning'
conda_path = '/home/lintao/anaconda3'

VERSION = sys.version_info.major
if VERSION == 2:
    import cv2
elif ros_path in sys.path:
    sys.path.remove(ros_path)
    import cv2
    from cv_bridge import CvBridge, CvBridgeError
    sys.path.append(ros_path)


class gazebo_env():
    def __init__(self):
        rospy.init_node('gazebo_env')
        # self.point = namedtuple('point', ['name', 'x', 'y'])

        self.agent_name = 'agent'
        self.agent_goal = {'x':10, 'y':10}
        self.agent_position = {'x':0, 'y':0}
        self.gazebo_obs_states = [{'x':0, 'y':0}]

        self.bridge = CvBridge()
        self.odom, self.rgb_image_raw, self.laser_scan_raw = None, None, None
        self.image_data_set, self.laser_data_set = [], []
        
        self.laser_sacn_clip = rospy.get_param("/dist/laser_sacn_clip")
        self.dist_near_goal = rospy.get_param("/dist/near_goal")
        self.dist_near_obs = rospy.get_param("/dist/near_obs")
        self.dist_min_scan = rospy.get_param("/dist/min_scan")

        self.laser_size = rospy.get_param("/params/laser_size")
        self.img_size = rospy.get_param("/params/img_size")

        self.num_sikp_frame = rospy.get_param("/params/num_sikp_frame")
        self.num_stack_frame = rospy.get_param("/params/num_stack_frame")
        self.reward_near_goal = rospy.get_param("/params/reward_near_goal")
        self.reward_near_obs = rospy.get_param("/params/reward_near_obs")

        # self.laser_sacn_clip = 5.0
        # self.dist_near_goal = 1.0
        # self.dist_near_obs = 0.7
        # self.dist_min_scan = 0.3
        
        # self.laser_size = 360
        # self.img_size = 80
        
        # self.reward_near_goal = 1000
        # self.reward_near_obs = -10
        # self.num_sikp_frame = 2
        # self.num_stack_frame = 4
        
        self.info = 0
        self.done = False

        self.actions = [[0.5, 0.5], [0.5, 0.2], [0.5, 0.0],
                        [0.5, -0.2], [0.5, -0.5], [0.2, 0.5],
                        [0.2, 0.0], [0.2, -0.5], [0.0, -0.5],
                        [0.0, 0.0], [0.0, 0.5]]
        self.n_actions = len(self.actions)
        self.store_data_size = self.num_sikp_frame * self.num_stack_frame

        self.euclidean_distance = lambda p1, p2: math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
        self.goal_dist_last, self.goal_dist = 0, 0

        # image_topic = '/' + self.agent_name + '/front/left/image_raw'
        

        # laser_ = '/' + self.agent_name + '/front/scan'
        odom_ = rospy.get_param('/topics/odom')
        laser_ = rospy.get_param('/topics/laser_scan')
        agent_cmd_ = rospy.get_param('/topics/agent_cmd')
        gazebo_states_ = rospy.get_param('/topics/gazebo_states')
        rgb_image_ = rospy.get_param('/topics/rgb_image')
        gazebo_set_ = rospy.get_param('/topics/gazebo_set')
        # agent_cmd_ = '/agent/jackal_velocity_controller/cmd_vel'

        # gazebo_states_ = '/gazebo/model_states'
        # gazebo_set_topic = '/gazebo/set_model_state'

        self._check_all_sensors_ready(odom_, laser_, rgb_image_, gazebo_states_)
        rospy.Subscriber(odom_, Odometry, self._odom_callback)
        rospy.Subscriber(laser_, LaserScan, self._laser_callback)
        rospy.Subscriber(rgb_image_, Image, self._image_callback)
        rospy.Subscriber(gazebo_states_, ModelStates, self._gazebo_states_callback, queue_size=1)
        self.pub_agent = rospy.Publisher(agent_cmd_, Twist, queue_size=1)
        self.pub_state = rospy.Publisher(gazebo_set_, ModelState, queue_size=1)

        self.cmd_vel = Twist()
        # self.cmd_vel.linear.x = 0
        # self.cmd_vel.angular.z = 0
        self.cmd_vel_last = {'v':0, 'w':0}
        self.pose_msg = ModelState()

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

        if len(self.laser_data_set) > self.store_data_size: 
            del self.laser_data_set[0]

    def _image_callback(self, data):
        self.rgb_image_raw = self.bridge.imgmsg_to_cv2(data)
        img_data = cv2.resize(self.rgb_image_raw, (self.img_size, self.img_size)) # 80x80x3
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        img_data = np.reshape(img_data, (self.img_size, self.img_size))
        self.image_data_set.append(img_data)

        if len(self.image_data_set) > self.store_data_size: 
            del self.image_data_set[0]

    def _gazebo_states_callback(self, data):
        self.gazebo_state_info = data
        self.gazebo_obs_states = [{'x':0, 'y':0} for name in data.name if 'obs' in name]

        for i in range(len(data.name)):
            p_x = data.pose[i].position.x
            p_y = data.pose[i].position.y
            name = str(data.name[i])
            if 'obs' in name:
                self.gazebo_obs_states[int(data.name[i][-1])] = {'x':p_x, 'y':p_y}
            elif name == 'agent_point_goal':
                self.agent_goal['x'] = p_x
                self.agent_goal['y'] = p_y
            elif name == 'agent':
                self.agent_position['x'] = p_x
                self.agent_position['y'] = p_y

        self.goal_dist = self.euclidean_distance(self.agent_position, self.agent_goal)
        # info test
        # print ('========')
        # for item in self.gazebo_obs_states:
        #     print ('index {}, x={:.2f}, y={:.2f}'.format(self.gazebo_obs_states.index(item), item['x'], item['y']))
        # print(self.agent_goal)
        # print(self.agent_position)
    #endregion


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
        print ('----done is {}'.format(self.done))
        return self.done

    def _get_info(self):
        self._set_info(0)
        self.check_near_goal(self.dist_near_goal)
        self.check_near_obs(self.dist_near_obs, self.dist_min_scan, 100)
        print ('----info is {}'.format(self.info))
        return self.info

    def check_near_goal(self, dist_min):
        if self.goal_dist < dist_min:
            print ('=====!!!agent get goal at {:.2f}!!!====='.format(self.goal_dist))
            return self._set_info(2)
       
    def check_near_obs(self, dist_min, laser_dist_min, scan_num):
        laser_min_count, laser_min, obs_dist_min = 0, 1000, 1000
        for obs in self.gazebo_obs_states:
            obs_dist = self.euclidean_distance(self.agent_position, obs)
            if obs_dist < obs_dist_min:
                obs_dist_min = obs_dist
        # print ('------obs_dist_min is {}'.format(obs_dist_min))
        if obs_dist_min < dist_min:
            print ('----!!!agent near the obs at {:.2f}!!!----'.format(obs_dist))
            return self._set_info(1)
                
        for r in self.laser_scan_raw:
            if r < laser_dist_min:
                laser_min_count += 1
            if r < laser_min:
                laser_min = r
        # print ('------laser_min is {}'.format(laser_min))
        if laser_min_count > scan_num:
            print ('----!!!laser too close to the obs for {:.2f}, count {}!!!-----'.format(laser_min, laser_min_count))
            return self._set_info(1)

    def _set_info(self, num):
        self.info = num
    #endregion

    def reset(self): # init state and env
        print ('=============reset_env==============')
        self.pose_msg.model_name = self.agent_name
        self.pose_msg.pose.position.x = 0
        self.pose_msg.pose.position.y = 0
        self.pub_state.publish(self.pose_msg)
        self.action_count = 0
        # init state
        return self._get_state()
        # TODO-dynamic obs

    def step(self, action_index, tele_input=None):
        # rate = rospy.Rate(1)
        self.cmd_vel.linear.x = tele_input[0] if tele_input else self.actions[action_index][0]
        self.cmd_vel.angular.z = tele_input[-1] if tele_input else self.actions[action_index][1]

        self.pub_agent.publish(self.cmd_vel)
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

    #======test basic_logic======
    # while not rospy.is_shutdown():
    #     if env.action_count > 1000:
    #         env.reset()
    #     choose_action = np.random.randint(1, 10)
    #     print ('choose_action {}, count {}'.format(choose_action, env.action_count))
    #     s_, r, d, i = env.step(choose_action)
    # rospy.spin()

    #======test reward=======#
    speed = 0.1
    move = {'w': [speed, 0, 0, 0],
            'a': [0, 0, 0, speed],
            's': [-speed, 0, 0, 0],
            'd': [0, 0, 0, -speed]}
    r_list = []
    action_n = 0
    while not rospy.is_shutdown():
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

    #====== test odom =======#
    # while not rospy.is_shutdown():
    #     env._check_odom_ready()
    #     odom = env.get_odom()
    #     linear_vel = odom.twist.twist.linear.x
    #     angular_vel = odom.twist.twist.angular.z
    #     print ('linear_vel is {:.2f}, angular_vel is {:.2f}'.format(linear_vel, angular_vel))

    #====== test safe_dist======#
    # while not rospy.is_shutdown():
   
    #     env.check_near_obs(0, 0, 1000)
    #     time.sleep(0.5)

    rospy.spin()