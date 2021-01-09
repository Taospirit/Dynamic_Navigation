#!/usr/bin/env python

# from test_tool import policy_test
#region
import gym
import os
from os.path import abspath, dirname
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
import numpy as np

from drl.algorithm import MSAC as SAC
from drl.algorithm import TD3
from drl.utils import ZFilter
from utils.plot import plot
from utils.config import config

from GazeboEnv import GazeboEnv

#config
config = config['msac']

# env_name = config['env_name']
task_num = 5
env_name = 'Gazebo_task'+str(task_num)
# only for eval
env_name = 'task'+str(task_num)+'_save'
#region
buffer_size = int(config['buffer_size'])
actor_learn_freq = config['actor_learn_freq']
update_iteration = config['update_iteration']
target_update_freq = config['target_update_freq']
batch_size = config['batch_size']
hidden_dim = config['hidden_dim']
episodes = config['episodes'] + 10
max_step = config['max_step']
a_lr = config['actor_lr']
c_lr = config['critic_lr']
n_step = 5
# rand_theta = True

LOG_DIR = config['LOG_DIR']

POLT_NAME = config['POLT_NAME']
SAVE_DIR = config['SAVE_DIR']
PKL_DIR = config['PKL_DIR']

POLT_NAME += env_name
SAVE_DIR += env_name
PKL_DIR += env_name

file_path = abspath(dirname(__file__))
pkl_dir = file_path + PKL_DIR
model_save_dir = file_path + SAVE_DIR
save_file = model_save_dir.split('/')[-1]
writer_path = model_save_dir + LOG_DIR


TRAIN = False
PLOT = True
WRITER = False
#endregion
# 4, 5, 6
#region
task_s, task_g = {}, {}
task_dict = {}
# scence 1:
# passing , 单机相对而行
task_s[1] = [[5, 0, -180]]
task_g[1] = [[1, 0, -180]]

# scence 2:
# overtaking, 单机同向二行
task_s[2] = [[1, 0, 0]]
task_g[2] = [[5, 0, 0]]

# scence 3:
# crossing close: 单机左侧到右侧, 初始距离近
task_s[3] = [[3, 2, -90]]
task_g[3] = [[3, -2, -90]]

# scence 4:
# passing 2, 双机相向而行,发生碰撞
task_s[4] = [[2, 1, -135], [6, -2, 135]]
task_g[4] = [[-1, -2, -135], [2, 2, 135]]

# scence 5:
# overtaking 2, 双机交叉同向而行
task_s[5] = [[2, 1, -45], [3, -2, 45]]
task_g[5] = [[5, -2, -45], [7, 2, 45]]

# scence 6:
# crossing 2, 双机,一个斜着从左到右同向, 一个斜着从右到左相向
task_s[6] = [[2, -1, 45], [7, 2, -135]]
task_g[6] = [[5, 2, 45], [3, -2, -135]]

# scence 7:
# crossing 3, 双机,一个垂直从左到右, 一个垂直从右到左
task_s[7] = [[2, 3, -45], [5, -5, 0]]
task_g[7] = [[2, -3, -45], [5, 5, 0]]

for i in range(len(task_s)):
    task_dict[i+1] = (task_s[i+1], task_g[i+1]) 

env = GazeboEnv()
# env.init_default_env()
env.init_env([0, 0], [7, 0], task_dict[task_num][0], task_dict[task_num][1])

# Parameters
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
print (f'state space {state_space}, action space {action_space}')
# action_max = env.action_space.high[0]
action_scale = env.action_scale
action_bias = env.action_bias
print (f'env scale {action_scale}, bias {action_bias}')
# print (f'state_space {state_space}, action_space {action_space}')
# assert 0
#endregion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#region network
def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

class ConvNet(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class ActorTD3(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        layer_norm(self.fc1, std=1.0)
        layer_norm(self.fc2, std=1.0)
        layer_norm(self.out, std=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = torch.tanh(self.out(x))
        return out

    def action(self, state, noise_std=0, noise_clip=0.5):
        action = self.forward(state)
        if noise_std:
            noise_norm = torch.ones_like(action).data.normal_(0, noise_std).to(device)
            action += noise_norm.clamp(-noise_clip, noise_clip)
        action = action.clamp(-1, 1)
        return action.data.cpu().numpy()

    
class ActorSAC(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)
        layer_norm(self.fc1, std=1.0)
        layer_norm(self.fc2, std=1.0)
        layer_norm(self.mean, std=1.0)
        layer_norm(self.log_std, std=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def action(self, state, test=False):
        mean, log_std = self.forward(state)
        if test:
            return torch.tanh(mean).detach().cpu().numpy()
        std = log_std.exp()
        normal = Normal(mean, std)
        
        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()
        return action

    # Use re-parameterization tick
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0,1)
        
        z = noise.sample()
        action = torch.tanh(mean + std*z.to(device))
        log_prob = normal.log_prob(mean + std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        
        return action, log_prob

class CriticModel(nn.Module):
    def __init__(self, obs_dim, mid_dim, act_dim, use_dist=False, v_min=-20, v_max=0, num_atoms=51):
        super().__init__()
        # print(f'obs_dim {obs_dim}, mid_dim {mid_dim}, act_dim {act_dim}')
        # assert 0
        self.use_dist = use_dist
        if use_dist:
            self.v_min = v_min
            self.v_max = v_max
            self.num_atoms = num_atoms
            self.net1 = self.build_network(obs_dim, mid_dim, act_dim, num_atoms)
            self.net2 = self.build_network(obs_dim, mid_dim, act_dim, num_atoms)
        else:
            self.net1 = self.build_network(obs_dim, mid_dim, act_dim, act_dim)
            self.net2 = self.build_network(obs_dim, mid_dim, act_dim, act_dim)
    
    def build_network(self, obs_dim, mid_dim, act_dim, out_dim):
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim , mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, out_dim), )
        return self.net

    def twinQ(self, obs, act):
        x = torch.cat((obs, act), dim=1)
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2

    def forward(self, obs, act):
        x = torch.cat((obs, act), dim=1)
        q1 = self.net1(x)
        return q1

    def get_probs(self, obs, act, log=False):
        z1, z2 = self.forward(obs, act)
        if log:
            z1 = torch.log_softmax(z1, dim=1)
            z2 = torch.log_softmax(z2, dim=1)
        else:
            z1 = torch.softmax(z1, dim=1)
            z2 = torch.softmax(z2, dim=1)
        return z1, z2
#endregion

#region training
def map_action(action):
    if isinstance(action, torch.Tensor):
        action = action.data.cpu().numpy()
    return action * action_scale + action_bias

def sample(env, policy, max_step, eval=False, rand_theta=False):
    rewards = []
    state = env.reset(rand_theta)
    for step in range(max_step):
        #==============choose_action==============
        action = policy.choose_action(state, test=eval)
        next_state, reward, done, info = env.step(map_action(action))
        if not eval:
            mask = 0 if done else 1
            policy.process(s=state, a=action, r=reward, m=mask, s_=next_state)
        rewards.append(reward)
        if done:
            break
        state = next_state
    return rewards

kwargs = {
    'action_dim': 2;
    'buffer_size': int(5e4);
    'batch_size': 1024;
    'actor_learn_freq': 2;
    'actor_lr': 3e-4;
    'critic_lr': 3e-4;
    'use_priority': True;
    'use_m': True;
    'n_step': 5
}

def build_policy():
    model = namedtuple('model', ['policy_net', 'value_net'])
    actor = ActorSAC(state_space, hidden_dim, action_space)
    critic = CriticModel(state_space, hidden_dim, action_space, use_dist=False)
    rl_agent = model(actor, critic)
    policy = SAC(rl_agent, **kwargs)
    return policy

def train():
    policy = build_policy()
    writer = SummaryWriter(writer_path)
    mean, std = [], []
    for i_eps in range(episodes):
        rewards = sample(env, policy, max_step, rand_theta=True)

        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        mean.append(reward_mean)
        std.append(reward_std)

        #==============learn==============
        pg_loss, q_loss, a_loss = policy.learn()
        if PLOT:
            plot(mean, POLT_NAME, model_save_dir, 50)
        if WRITER:
            writer.add_scalar('reward', reward_mean, global_step=i_eps)
            writer.add_scalar('loss/pg_loss', pg_loss, global_step=i_eps)
            writer.add_scalar('loss/q_loss', q_loss, global_step=i_eps)
            writer.add_scalar('loss/alpha_loss', a_loss, global_step=i_eps)
        if i_eps % 1 == 0:
            print (f'EPS:{i_eps}, reward:{reward_mean:.3f}, pg_loss:{pg_loss:.3f}, q_loss:{q_loss:.3f}, alpha_loss:{a_loss:.3f}', end='\r')
        if i_eps % 50 == 0 and i_eps > 0:
            policy.save_model(model_save_dir, save_file, save_actor=True, save_critic=True)
            print(f'save model at {i_eps}')
    writer.close()
    return mean, std

def eval():
    policy = build_policy()
    policy.load_model(model_save_dir, save_file, load_actor=True)
   
    for i_eps in range(episodes):
        rewards = sample(env, policy, max_step, eval=True, rand_theta=False)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        if i_eps % 1 == 0:
            # env.random_goal()
            pass
        print (f'EPS:{i_eps + 1}, reward:{reward_mean:.3f}')

def train_and_save(pkl_name):
    import pickle
    means, stds = [], []
    for seed in range(5):
        torch.manual_seed(seed * 10)
        mean, std = train()
        save = {'mean': mean, 'std': std}
        with open(pkl_dir + pkl_name + f'_seed_{seed * 10}.pkl', 'wb') as f:
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        means.append(mean)
        stds.append(std)
        print (f'-----finish traing in seed {seed * 10} at priority : {use_priority}')

    d = {'means': means, 'stds': stds}
    with open(pkl_dir + pkl_name + '.pkl', 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    print ('finish all learning!')
#endregion 

if __name__ == '__main__':
    off_on = [False, True]
    TRAIN = off_on[0]
    for i in range(2):
        # i = 1
        use_munchausen = off_on[i]
        use_priority = off_on[i]

        if TRAIN:
            try:
                os.makedirs(model_save_dir)
            except FileExistsError:
                import shutil
                shutil.rmtree(model_save_dir)
                os.makedirs(model_save_dir)

            pkl_name = ''
            if use_priority:
                p = '_PER_N-step'
                pkl_name += p
                # global SAVE_DIR, POLT_NAME, PKL_DIR
                SAVE_DIR += p
                POLT_NAME += p
                PKL_DIR += p
            else:
                pkl_name += '_pure'
            train_and_save(pkl_name)
        else:
            eval()
    # torch.manual_seed(1)
    # train()
    
    # train_and_save()