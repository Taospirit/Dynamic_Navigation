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
        self.agent_goal = {'x':0, 'y':0}
        self.agent_position = {'x':0, 'y':0}
        self.gazebo_obs_states = [{'x':0, 'y':0}]

        self.bridge = CvBridge()
        self.image_raw, self.laser_raw = [], []
        self.image_data_set, self.laser_data_set = [], []
        
        self.laser_clip_dist = 5.0
        self.dist_goal_arrive = 1.0
        self.dist_obs_safe = 1.2
        self.laser_size = 360
        # self.laser_clip = 5
        self.img_size = 80

        self.info = 0
        self.done = False

        self.actions = [[0.5, 0.5], [0.5, 0.2], [0.5, 0.0],
                        [0.5, -0.2], [0.5, -0.5], [0.2, 0.5],
                        [0.2, 0.0], [0.2, -0.5], [0.0, -0.5],
                        [0.0, 0.0], [0.0, 0.5]]
        self.n_actions = len(self.actions)
        # paraments
        self.reward_near_goal = 1000
        self.reward_near_obs = -10
        self.num_sikp_frame = 2
        self.num_stack_frame = 4
        self.store_data_size = self.num_sikp_frame * self.num_stack_frame


        self.euclidean_distance = lambda p1, p2: math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
        self.goal_dist_last, self.goal_dist = 0, 0

        # image_topic = '/' + self.agent_name + '/front/left/image_raw'
        image_topic = '/camera/rgb/image_raw'

        # laser_topic = '/' + self.agent_name + '/front/scan'
        laser_topic = '/scan'

        # agent_pub_topic = '/agent/jackal_velocity_controller/cmd_vel'
        agent_pub_topic = '/cmd_vel'

        gazebo_topic = '/gazebo/model_states'
        gazebo_set_topic = '/gazebo/set_model_state'
        odom_topic = '/odom'

        rospy.Subscriber(image_topic, Image, self._image_callback)
        rospy.Subscriber(laser_topic, LaserScan, self._laser_callback)
        rospy.Subscriber(gazebo_topic, ModelStates, self._gazebo_states_callback, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self._odom_callback)
        self.pub_agent = rospy.Publisher(agent_pub_topic, Twist, queue_size=1)
        self.pub_state = rospy.Publisher(gazebo_set_topic, ModelState, queue_size=1)

        self.cmd_vel = Twist()
        # self.cmd_vel.linear.x = 0
        # self.cmd_vel.angular.z = 0
        self.cmd_vel_last = {'v':0, 'w':0}
        self.pose_msg = ModelState()
        # self.pose_msg.model_name = self.agent_name

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

    #region ros_callback
    def _image_callback(self, data):
        try:
            self.image_raw = self.bridge.imgmsg_to_cv2(data)
            img_data = cv2.resize(self.image_raw, (self.img_size, self.img_size)) # 80x80x3
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
            img_data = np.reshape(img_data, (self.img_size, self.img_size))
            self.image_data_set.append(img_data)

            if len(self.image_data_set) > self.store_data_size: del self.image_data_set[0]

        except CvBridgeError as e:
            print (e)

    def _laser_callback(self, data):
        # sample_size:=720 update_rate:=50
        # min_angle:=-2.35619 max_angle:=2.35619 
        # min_range:=0.1 max_range:=30.0
        # range_min = 0.1 range_max = 10.0 meters
        self.laser_raw = data.ranges
        laser_clip = np.clip(self.laser_raw, 0, self.laser_clip_dist) / self.laser_clip_dist # normalization laser data
        laser_data = [(laser_clip[i] + laser_clip[i+1]) / 2 for i in range(0, len(laser_clip), 2)]    
        self.laser_data_set.append(laser_data)

        if len(self.laser_data_set) > self.store_data_size: del self.laser_data_set[0]

    def _gazebo_states_callback(self, data):
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

    def _odom_callback(self, data):
        self.odom = data
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

    def _get_reward(self, dist_rate=2, cmd_rate=0.1):
        reward = 0
        # print (type(self.cmd_vel_last['v']), type(self.cmd_vel.linear.x))
        cmd_vel_reward = abs(self.cmd_vel_last['v'] - self.cmd_vel.linear.x) + abs(self.cmd_vel_last['w'] - self.cmd_vel.angular.z)
        reward += -cmd_rate * cmd_vel_reward
        # delta_dist = 0 if self.goal_dist_last == 0 else self.goal_dist_last - self.goal_dist
        delta_dist = self.goal_dist_last - self.goal_dist if self.goal_dist_last != 0 else 0
        reward += dist_rate * delta_dist
        # print ('goal_dist: {:.3f}, goal_dist_last: {:.3f}'.format(self.goal_dist, self.goal_dist_last))
        # print ('delta_dist: {:.3f}'.format(delta_dist))
        # print ('cmd_vel_change: {:.3f} / '.format(cmd_vel_reward))

        if self.info == 1: # coll
            reward += self.reward_near_obs
        if self.info == 2: # arrive at goal
            reward += self.reward_near_goal
        # print ('cal_reward: {}'.format(delta_dist*dist_rate - cmd_vel_reward*cmd_rate))
        return reward

    def _get_info(self):
        self.info = 0
        # print ('goal_dist from _get_info(): {:.3f}'.format(self.goal_dist))
        if self.goal_dist < self.dist_goal_arrive:
            print ('=====!!!agent get goal at {:.2f}!!!====='.format(self.goal_dist))
            self.info = 2
            return self.info
        # min_dist = 1000
        for obs in self.gazebo_obs_states:
            obs_dist = self.euclidean_distance(self.agent_position, obs) 
            # if obs_dist < min_dist:
            #     min_dist = obs_dist
            if obs_dist < self.dist_obs_safe:
                print ('----!!!agent collision with the obs at {:.2f}!!!----'.format(obs_dist))
                # print ('obs is ', obs)
                # print ('index is ', self.gazebo_obs_states.index(obs))
                self.info = 1
        # print ('obs_min_dist {:.3f}'.format(min_dist))
        # print ('goal_dist {:.3f}'.format(self.goal_dist))
        return self.info

    def _get_done(self):# arrived or collsiped or time_out
        self.done = False
        if self.info == 1 or self.info == 2: self.done = True
        return self.done
    #endregion

    def reset(self): # init state and env
        print ('----reset_env-----')
        self.pose_msg.model_name = 'agent'
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
        self.wait_until_twist_achieved(self.cmd_vel)
        self.action_count += 1
        
        start = time.time()
        # rate.sleep()
        end = time.time()
        # print ('during {}'.format(end - start))
        #XXX: to be tested
        during = 0.2
        # print ('wait for {} seconds'.format(during))
        # time.sleep(during)

        info = self._get_info()
        done = self._get_done()
        if done: self.reset()
        reward = self._get_reward()
        state_ = self._get_state()

        self.cmd_vel_last['v'] = self.cmd_vel.linear.x
        self.cmd_vel_last['w'] = self.cmd_vel.angular.z
        self.goal_dist_last = self.goal_dist

        return state_, reward, done, info

    def wait_until_twist_achieved(self, cmd_vel_value, epsilon=0.2):
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
            print ('Current is {:.3f}\{:.3f}, goal is {:.3f}\{:.3f}'.format(odom_linear_vel, odom_angular_vel, linear_speed, angular_speed))
            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (odom_angular_vel > angular_speed_minus)

            if linear_vel_are_close and angular_vel_are_close:
                rospy.loginfo('Have achieve twist speed')
                break

    def _check_odom_ready(self):
        self.odom = None
        # rospy.logdebug("Waiting for /odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
                rospy.logdebug("Current /odom READY=>")

            except:
                rospy.logerr("Current /odom not ready yet, retrying for getting odom")

        return self.odom

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
    move = {'w': [1, 0, 0, 0],
            'a': [0, 0, 0, 1],
            's': [-1, 0, 0, 0],
            'd': [0, 0, 0, -1]}
    r_list = []
    action_n = 0
    while not rospy.is_shutdown():
        print ('===test reward, wait for tele_input===')
        key = get_key()

        if key in move.keys():
            state_, reward, done, info = env.step(0, move[key])
            action_n += 1
            r_list.append(reward)
            print ('reward {:.3f}, info {}'.format(reward, info))

            if done:
                r = get_totoal_reward(r_list, 0.9)
                print ('----action_n: {}, total_reward: {:.3f}'.format(action_n, r))
                r_list = []
                action_n = 0

        else:
            env.step(0, [0, 0])

    rospy.spin()