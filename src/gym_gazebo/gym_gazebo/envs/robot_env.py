import gym
from gym import error, spaces, utils
from gym.utils import seeding

import gazebo_env
import time
import numpy as np
import planner

class RobotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.gazebo_env = gazebo_env.GazeboEnv()
        self.obs_name = self.gazebo_env.obs_name
        self.obs_goal_name = self.gazebo_env.obs_goal_name
        self.agent_name = self.gazebo_env.agent_name
        self.agent_pose = self.gazebo_env.agent_pose
        self.agent_goal_name = self.gazebo_env.agent_goal_name

        self.verbose = False
        self.info = 0
        self.done = False

        self.Planner = planner.APFM()
        self.actions = [[1.6, 1.6], [1.6, 0.8], [1.6, 0.0],
                        [1.6, -0.8], [1.6, -1.6], [0.8, 1.6],
                        [0.8, 0.0], [0.8, -1.6], [0.0, -1.6],
                        [0.0, 0.0], [0.0, 0.8]]
        self.n_actions = len(self.actions)
        self.action_count = 0

        self.start_default_list = [[-16, y, 0] for y in range(-10, 11, 2) if y != 0]
        self.goal_default_list = [[16, y, 0] for y in range(-10, 11, 2) if y != 0]
        self.agent_start_default = [0, 0]
        self.agent_goal_default = [5, 0]

        s0 = [0, 0]
        g0 = [5, 0]
        s1 = [[4, 4, -90]]
        g1 = [[4, -4, 0]]
        self.init_env(s0, g0, s1, g1)

    def step(self, action_index):
        self.gazebo_env.agent_step(self.actions[action_index])
        if self.verbose:
            print('set linear/angular vel {}, index {}'.format(self.actions[action_index], action_index))
        self.gazebo_env.obs_step(self.Planner, self.obs_start_list, self.obs_goal_list)

        time.sleep(0.15)

        self.action_count += 1
        info = self.gazebo_env._get_info(self.start_default_list, self.goal_default_list)
        done = self.gazebo_env._get_done()
        # if done:
        #     self.reset()
        reward = self.gazebo_env._get_reward()
        state_ = self.gazebo_env._get_state()

        self.gazebo_env.update_data()

        return state_, reward, done, info

    def reset(self):
        print('=============env_reset==============')
        self.reset_env()
        self.action_count = 0

        return self.gazebo_env._get_state()

    def render(self, mode='human'):
        print('this is render')

    def close(self):
        print('this is close')

    # region: env setting and function related
    def init_default_env(self):
        self.init_env(self.start_default_list, self.goal_default_list,
                      self.agent_start_default, self.agent_goal_default)

    def init_env(self, start_point, goal_point, obs_start_list, obs_goal_list):
        assert len(obs_start_list) == len(obs_goal_list)
        assert not len(obs_start_list) > len(self.gazebo_env.gazebo_obs_states)
        self.get_agent_start_pose(start_point)
        self.get_obs_start_pose(obs_start_list)
        self.get_obs_goal_pose(obs_goal_list)
        self.get_agent_goal_pose(goal_point)

        self.reset_env()

    def init_env_random(self, num=10):
        assert not num > len(self.gazebo_env.gazebo_obs_states)
        self.get_agent_start_pose(self.agent_start_default)
        self.get_agent_goal_pose(self.agent_goal_default)
        self.get_obs_start_pose(num=num)
        self.get_obs_goal_pose(num=num)

    def reset_env(self):
        self.gazebo_env.gazebo_pub_states(self.agent_name, [self.agent_start_pose])
        self.gazebo_env.gazebo_pub_states(self.agent_goal_name, [self.agent_goal_pose])
        self.gazebo_env.gazebo_pub_states(self.obs_name, self.obs_start_list)
        self.gazebo_env.gazebo_pub_states(self.obs_goal_name, self.obs_goal_list)

        # start_point = np.append(self.agent_start_default, 0)
        # self.gazebo_env._pub_gazebo_states(self.agent_name, [start_point])
        # assert self.obs_start_list is not None and self.obs_goal_list is not None
        # self.get_obs_start_pose(self.obs_start_list)

    # region: get pose properly
    def get_agent_start_pose(self, start_point=None):
        if start_point is None:
            start_point = np.random.uniform(-8, 8, 2)
        self.agent_start_pose = list(np.append(start_point, np.pi))  # set theta to np.pi
        if self.verbose:
            print("=====get agent start pose!=====")

    def get_obs_start_pose(self, obs_start_list=None, num=10):
        if obs_start_list is None:
            obs_start_list = self._get_random_pose(num)

        obs_start_list.extend(self.start_default_list[len(obs_start_list):])
        self.obs_start_list = obs_start_list
        if self.verbose:
            print("=====get obs init pose!=====")

    def get_obs_goal_pose(self, obs_goal_list=None, num=10):
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
        self.obs_goal_list = obs_goal_list
        if self.verbose:
            print("=====get obs goal pose!=====")

    def get_agent_goal_pose(self, goal_point=None):
        if goal_point is None:
            while True:
                done = True
                goal_point = np.random.uniform(-10, 10, 2)
                pose_list = np.vstack((self.obs_start_list, self.obs_goal_list))
                for pose in pose_list:
                    dist_obs = np.hypot(goal_point[0] - pose[0], goal_point[1] - pose[1])
                    dist_agent = np.hypot(goal_point[0] - self.agent_pose['x'], goal_point[0] - self.agent_pose['y'])
                    if dist_obs < 1 or dist_agent < 7:
                        done = False
                        break
                if done:
                    break
        self.agent_goal_pose = list(np.append(goal_point, np.pi))  # set theta to np.pi
        # rospy.logdebug("reset agent goal position!")
        if self.verbose:
            print("=====get agent goal pose!=====")

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
    # endregion get pose properly

    # def set_states(self):
        # self.gazebo_env._pub_gazebo_states(self.agent_name, [self.agent_start_pose])
        # self.gazebo_env._pub_gazebo_states(self.agent_goal_name, [self.agent_goal_pose])
        # self.gazebo_env._pub_gazebo_states(self.obs_name, self.obs_start_list)
        # self.gazebo_env._pub_gazebo_states(self.obs_goal_name, self.obs_goal_list)

    # endregion env setting and function related
