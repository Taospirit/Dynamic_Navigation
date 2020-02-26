from collections import namedtuple
import math

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
        return math.hypot((p1.x - p2.x), (p1.y - p2.y))

    def get_cmd(self):
        pass

class APFM(BasicInfo):
    def __init__(self):
        super().__init__()
        self.Katt = 1.0 # 引力增益
        self.Krep = 1.0 # 斥力增益
        self.D_obs = 4.0
        self.lim_linear = 0.5
        self.lim_angluar = 0.5
        self.safe_dist = 0.1

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

    def set_lim_linear(self, num):
        self.lim_linear = num

    def set_lim_angluar(self, num):
        self.lim_angluar = num

    def limvel(self, vel, lim):
        sign = 1 if vel > 0 else -1
        return vel if abs(vel) < abs(lim) else sign * abs(lim)

    def _get_force(self):
        delta_x = self.self_state.x - self.goal_state.x
        delta_y = self.self_state.y - self.goal_state.y

        if delta_x < self.safe_dist and delta_y < self.safe_dist:
            print ('arrived at goal, stop!!')
            return 0, 0

        F_att_x = -self.Katt * delta_x
        F_att_y = -self.Katt * delta_y
        # F_att_x = self.limvel(F_att_x, self.lim_vel)
        # F_att_y = self.limvel(F_att_y, self.lim_vel)
        F_rep_x = 0
        F_rep_y = 0
        for i in range(self.get_obs_num()):
            f_rep_x = 0
            f_rep_y = 0
            obs_dist = self.get_dist(self.self_state, self.obs_states[i])
            if obs_dist < self.D_obs:
                # f_rep_x = self.Krep * (1/obs_dist - 1/self.D_obs)*(1/obs_dist**2)*(self_x - self.obs_states[i].x)/obs_dist
                delta_x = self.self_state.x - self.obs_states[i].x
                delta_y = self.self_state.y - self.obs_states[i].y
                # f_rep_x = self.Krep * (self.D_obs - obs_dist) * delta_x / obs_dist**4 / self.D_obs
                # f_rep_y = self.Krep * (self.D_obs - obs_dist) * delta_y / obs_dist**4 / self.D_obs
                f_rep_x = self.calc_rep(delta_x, obs_dist)
                f_rep_y = self.calc_rep(delta_y, obs_dist)
            
            F_rep_x += f_rep_x
            F_rep_y += f_rep_y
            
        F_x = F_att_x + F_rep_x
        F_y = F_att_y + F_rep_y
        return F_x, F_y

    def calc_rep(self, delta, dist):
        return self.Krep * (self.D_obs - dist) * delta / dist**4 / self.D_obs

    def get_cmd(self):
        f_x, f_y = self._get_force()
        # print ('f_x is {:.2f}, f_y is {:.2f}'.format(f_x, f_y))
        linear_norm = math.hypot(f_x, f_y)
        cmd_linear = self.limvel(linear_norm, self.lim_linear)

        yaw_goal = math.atan2(f_y, f_x) # radicus
        # print ('yaw_goal degrees is {:.2f}'.format(math.degrees(yaw_goal)))
        yaw_current = math.radians(self.self_state.t) # radicus
        # print ('yaw_current degrees is {:.2f}'.format(math.degrees(yaw_current)))
        # delta = math.degrees(yaw_goal) - math.degrees(yaw_current)
        delta = yaw_goal - yaw_current
        theta_norm = math.degrees(math.atan2(math.sin(delta), math.cos(delta)))
        sign = 1 if theta_norm > 0 else -1
        print ('theta_norm is {:.2f}'.format(theta_norm))
        t1 = 10
        t2 = 90
        cmd_max = self.lim_angluar
        cmd_min = 0.2
        cmd_angular = 0

        delta_theta = abs(theta_norm)
        if delta_theta > t2:
            cmd_angular = self.lim_angluar
        elif delta_theta > t1:
            k = (cmd_max - cmd_min) / (t2 - t1)
            cmd_angular = k * (delta_theta - t1) + cmd_min
        cmd_angular *= sign
        print ('linear.x is {:.2f}, angular.z is {:.2f}'.format(cmd_linear, cmd_angular))
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
   
