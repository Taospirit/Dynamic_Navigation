from collections import namedtuple
import numpy as np
import rvo2

class BasicInfo():
    def __init__(self):
        self.state = namedtuple('state', ['x', 'y', 'v', 'w', 't'])
        # self._self_state = ['x': 0, 'y': 0]
        self._self_state = None
        self._goal_state = None
        self._env_states = None

    def _set_robot_state(self, x, y, v, w, t):
        return self.state(x=x, y=y, v=v, w=w, t=t)

    def set_self_state(self, pose):
        x, y = pose['x'], pose['y']
        v, w = pose['v'], pose['w']
        t = pose['t']
        self._self_state = self._set_robot_state(x, y, v, w, t)

    def set_goal_state(self, pose):
        x, y = pose['x'], pose['y']
        self._goal_state = self._set_robot_state(x, y, 0, 0, 0)

    def set_env_state(self, pose_list):
        self._env_states = [None for _ in range(len(pose_list))]
        for i in range(len(pose_list)):
            x, y = pose_list[i]['x'], pose_list[i]['y']
            v, w = pose_list[i]['v'], pose_list[i]['w']
            t = pose_list[i]['t']
            self._env_states[i] = self._set_robot_state(x, y, v, w, t)

    def get_self_state(self):
        return self._self_state

    def get_goal_state(self):
        return self._goal_state

    def get_env_states(self):
        return self._env_states

    def get_obs_num(self):
        return len(self._env_states)

    def get_dist(self, p1, p2):
        return np.hypot((p1.x - p2.x), (p1.y - p2.y))

    def get_cmd(self):
        pass

class APFM(BasicInfo):
    def __init__(self):
        super().__init__()
        self.Katt = 1.0 # 引力增益
        self.Krep = 1.0 # 斥力增益
        self.linear_max = 0.5
        self.angluar_max = 0.5
        self.angluar_min = 0.1
        self.safe_dist = 4.0
        self.arrived_dist = 0.1

    def init_env(self, self_pose, goal_pose, obs_poses):
        self.set_self_state(self_pose)
        self.set_goal_state(goal_pose)
        self.set_env_state(obs_poses)
        self.self_state = self.get_self_state()
        self.obs_states = self.get_env_states()
        self.goal_state = self.get_goal_state()

    def set_katt(self, num):
        self.Katt = num

    def set_krep(self, num):
        self.Krep = num

    def set_linear_max(self, num):
        self.linear_max = num

    def set_angluar_max(self, num):
        self.angluar_max = num

    def set_angluar_min(self, num):
        self.angluar_min = num

    def set_safe_dist(self, num):
        self.safe_dist = num

    def set_arrived_dist(self, num):
        self.arrived_dist = num

    def limvel(self, vel, lim):
        sign = 1 if vel > 0 else -1
        return vel if abs(vel) < abs(lim) else sign * abs(lim)

    def _get_force(self):
        '''
        U_att = 0.5 * k_att * (d^2)  d = ||q - q_g||
        F_att = - k_att * (q - q_goal)
        U_rep = 0.5 * k_rep * (1/d - 1/d_)^2
        F_rep = k_rep * (1/d_ - 1/d) / d^2 * (q - q_g) / d
        '''
        F_att_x, F_att_y = self.calc_attractive_potential()
        F_rep_x, F_rep_y = self.calc_repulsive_potential()

        F_x = F_att_x + F_rep_x
        F_y = F_att_y + F_rep_y
        return F_x, F_y

    def calc_attractive_potential(self):
        delta_x = self.self_state.x - self.goal_state.x
        delta_y = self.self_state.y - self.goal_state.y
        if np.hypot(delta_x, delta_y) < self.arrived_dist:
            print ('arrived at goal, stop at {:.2f}, {:.2f}!!'.format(delta_x, delta_y))
            return 0, 0

        return -self.Katt * delta_x, -self.Katt * delta_y

    def calc_repulsive_potential(self):
        F_rep_x = 0
        F_rep_y = 0
        for i in range(self.get_obs_num()):
            f_rep_x = 0
            f_rep_y = 0
            obs_dist = self.get_dist(self.self_state, self.obs_states[i])
            if obs_dist < self.safe_dist:
                delta_x = self.self_state.x - self.obs_states[i].x
                delta_y = self.self_state.y - self.obs_states[i].y
                f_rep_x = self.calc_rep(delta_x, obs_dist)
                f_rep_y = self.calc_rep(delta_y, obs_dist)
            F_rep_x += f_rep_x
            F_rep_y += f_rep_y

        return F_rep_x, F_rep_y

    def calc_rep(self, delta, dist):
        return self.Krep * (1/dist - 1/self.safe_dist) * delta / dist ** 3

    def get_cmd(self):
        F_x, F_y = self._get_force()
        print ('F_x is {:.2f}, F_y is {:.2f}'.format(F_x, F_y))
        linear_norm = np.hypot(F_x, F_y)
        cmd_linear = self.limvel(linear_norm, self.linear_max)
        ###==== robot tf
        # yaw_goal = np.arctan2(F_y, F_x) # radicus
        # print ('yaw_goal degrees is {:.2f}'.format(np.degrees(yaw_goal)))
        # yaw_current = np.radians(self.self_state.t) # radicus
        # print ('yaw_current degrees is {:.2f}'.format(np.degrees(yaw_current)))
        # delta = np.degrees(yaw_goal) - np.degrees(yaw_current)
        delta = np.arctan2(F_y, F_x) - np.radians(self.self_state.t)
        theta_norm = np.degrees(np.arctan2(np.sin(delta), np.cos(delta)))
        sign = 1 if theta_norm > 0 else -1
        print ('theta_norm is {:.2f}'.format(theta_norm))
        t1 = 10
        t2 = 90
        cmd_max = self.angluar_max
        cmd_min = self.angluar_min

        if abs(theta_norm) > t2:
            cmd_angular = cmd_max
        elif abs(theta_norm) > t1:
            k = (cmd_max - cmd_min) / (t2 - t1)
            cmd_angular = k * (abs(theta_norm) - t1) + cmd_min
        else:
            cmd_angular = 0
        cmd_angular *= sign
        # print ('linear.x is {:.2f}, angular.z is {:.2f}'.format(cmd_linear, cmd_angular))

        if self.get_dist(self.self_state, self.goal_state) < 1.0 and cmd_angular != 0:
            cmd_linear = 0

        return cmd_linear, cmd_angular


class ORCA(BasicInfo):
    def __init__(self, self_pose, goal_pose, obs_poses):
        super().__init__(self_pose, goal_pose, obs_poses)

    def _get_cmd(self):
        pass


if __name__ == "__main__":
    obs0 = {'x':111, 'y':222, 'v':333, 'w':444, 't':0}
    agent = {'x':1, 'y':2, 'v':3, 'w':4, 't':0}
    agent_goal = {'x':10, 'y':10}
    obs_list = []
    obs_list.append(obs0)

    planner = APFM()
   
