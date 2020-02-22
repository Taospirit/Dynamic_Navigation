from collections import namedtuple
import math

class BasicInfo():
    def __init__(self):
        self.state = namedtuple('state', ['x', 'y', 'v', 'w', 'yaw'])
        # self._self_state = ['x': 0, 'y': 0]
        self._self_state = None
        self._obs_states = None

    def _set_robot_state(self, x, y, v, w, yaw):
        return self.state(x=x, y=y, v=v, w=w, yaw=yaw)

    def set_self_state(self, x, y, v, w, yaw=0):
        self._self_state = self._set_robot_state(x, y, v, w, yaw)

    def set_goal_state(self, x, y, v=0, w=0, yaw=0):
        self._goal_state = self._set_robot_state(x, y, v, w, yaw)

    def set_obs_state(self, states_list):
        self._obs_states = [None for _ in range(len(states_list))]
        for i in range(len(states_list)):
            x, y = states_list[i]['x'], states_list[i]['y']
            v, w = states_list[i]['v'], states_list[i]['w']
            yaw = states_list[i]['yaw']
            self._obs_states[i] = self._set_robot_state(x, y, v, w, yaw)

    def get_self_state(self):
        return self._self_state

    def get_goal_state(self):
        return self._goal_state

    def get_obs_states(self):
        return self._obs_states

    def get_obs_num(self):
        return len(self._obs_states)

    def get_dist(self, p1, p2):
        return math.hypot((p1.x - p2.x), (p1.y - p2.y))

    def get_cmd(self):
        pass

class APFM(BasicInfo):
    def __init__(self):
        super().__init__()
        self.Katt = 1.0 # 引力增益
        self.Krep = 1.0 # 斥力增益
        self.D_obs = 3.0
        self.lim_linear = 0.5
        self.lim_angluar = 0.5
        self.self_state = self.get_self_state()
        self.obs_states = self.get_obs_states()
        self.goal_state = self.get_goal_state()

    def set_katt(self, num):
        self.Katt = num

    def set_krep(self, num):
        self.Krep = num

    def limvel(self, vel, lim):
        sign = 1 if vel > 0 else -1
        return vel if math.fabs(vel) < math.fabs(lim) else sign * math.fabs(lim)

    def _get_force(self):
        F_att_x = -self.Katt * (self.self_state.x - self.goal_state.x)
        F_att_y = -self.Katt * (self.self_state.y - self.goal_state.y)
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

    def _get_cmd(self):
        f_x, f_y = self._get_force()
        linear_norm = math.hypot(f_x, f_y)
        cmd_linear = self.limvel(linear_norm, self.lim_linear)

        yaw_goal = math.atan2(f_y, f_x) # radicus
        yaw_current = self.self_state.yaw # radicus
        # delta = math.degrees(yaw_goal) - math.degrees(yaw_current)
        delta = yaw_goal - yaw_current
        theta_norm = math.degrees(math.atan2(math.sin(delta), math.cos(delta)))
        
        t1 = 10
        t2 = 90
        cmd_max = self.lim_angluar
        cmd_min = 0.2
        cmd_angular = 0
        if theta_norm > t2:
            cmd_angular = self.lim_angluar
        elif theta_norm > t1:
            k = (cmd_max - cmd_min) / (t2 - t1)
            cmd_angular = k * (theta_norm - t1) + cmd_min

        return cmd_linear, cmd_angular


class ORCA(BasicInfo):
    def __init__(self):
        super().__init__()

    def _get_cmd(self):
        pass


if __name__ == "__main__":
    obs0 = {'x':111, 'y':222, 'v':333, 'w':444}
    obs_list = []
    obs_list.append(obs0)

    planner = APFM()
    planner.set_self_state(1, 2, 3, 4)
    planner.set_goal_state(10, 10)
    planner.set_obs_state(obs_list)
