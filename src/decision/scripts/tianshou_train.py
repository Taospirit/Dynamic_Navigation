#!/usr/bin/env python
import gym, os
import tianshou as ts
import torch, numpy as np
from torch import nn
from GazeboEnv import GazeboEnv

agent_s, agent_g =[0, 0], [6, 0]
obs_s, obs_g = [[6, 0, 180]], [[-1, 0, 0]]

s3_crossing_1 = [[2, 3, -45], [3, -3, 45], [2, 1, -90]] # 双机交叉
g3_crossing_1 = [[5, -3, 0], [6, 3, 0], [2, -1, 0]]

s1, g1 = [[3, 3, -90]], [[3, -3, 0]]
s3_crossing, g3_crossing = [[4, 3, -90], [3, -2, 90], [2, 1, -90]], [[4, -3, 0], [3, 2, 0], [2, -1, 0]]

env_scene = [agent_s, agent_g, s3_crossing_1, g3_crossing_1]
save_name = 'ts_dqn_3_crossing_1.pth'
env = GazeboEnv()
env.init_env(env_scene[0], env_scene[1], env_scene[2], env_scene[3])
# env.init_default_env()

print ('after init env')

train_envs = env
test_envs = env

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Linear(np.prod(state_shape), 150), nn.ReLU(inplace=True),
            nn.Linear(150, 100), nn.ReLU(inplace=True),
            nn.Linear(100, 100), nn.ReLU(inplace=True),
            nn.Linear(100, np.prod(action_shape))
        ])
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

state_shape = env.observation_space.shape or env.observation_space.n # 10
action_shape = env.action_space.shape or env.action_space.n # 11
print (f'state_shape{state_shape}, action_shape{action_shape}')
# action_shape = env.n_actions
ped_num = len(s3_crossing)
net = Net(20, 10)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(net, optim, 
    discount_factor=0.9, estimation_step=3,
    use_target_network=True, target_update_freq=320)

save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'saved_network/')
save_path = save_dir + save_name

train_type = ['train', 'eval']
state = train_type[0]

if state == 'train':
    goal_change_count = 0

    train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=20000))
    test_collector = ts.data.Collector(policy, test_envs)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('log/dqn2')
    
    while goal_change_count < 20:
        result = ts.trainer.offpolicy_trainer(
            policy, train_collector, test_collector,
            max_epoch=10, step_per_epoch=1000, collect_per_step=10,
            episode_per_test=20, batch_size=100,
            train_fn=lambda e: policy.set_eps(0.1),
            test_fn=lambda e: policy.set_eps(0.05),
            stop_fn=lambda x: x >= 70,
            writer=writer
        )
        print (f'Finished traing! Use {result["duration"]}')
        goal_change_count += 1
        # env.set_agent_goal_pose(np.random.uniform(-5, 5, 2))
        # print ('===============================================')
    torch.save(policy.state_dict(), save_path)

    state = 'eval'

def _make_batch(data):
    if isinstance(data, np.ndarray):
        return data[None]
    else:
        return np.array([data])

if state == 'eval':
    obs = env.reset()
    state = act = rew = done = info = None
    # reload and show
    policy.load_state_dict(torch.load(save_path))
    show_episode = 0
    while show_episode < 100:
        batch_data = ts.data.Batch(
                    obs=_make_batch(obs),
                    act=_make_batch(act),
                    rew=_make_batch(rew),
                    done=_make_batch(done),
                    obs_next=None,
                    info=_make_batch(info))

        result = policy(batch_data, state)

        if isinstance(result.act, torch.Tensor):
                act = result.act.detach().cpu().numpy()
        elif not isinstance(act, np.ndarray):
            act = np.array(result.act)
        else:
            act = result.act
        # print(act)
        obs_next, rew, done, info = env.step(act[-1])

        if done: # end of one trajectory
            # cur_episode += 1
            # reward_sum += self.reward
            # length_sum += self.length
            # self.reward, self.length = 0, 0
            # self.state = None
            obs_next = env.reset()
            show_episode += 1
        obs = obs_next
        print (f'show model eval epsiode: {show_episode}')
    # collect = ts.data.Collector(policy, env)
    # collect.collect(n_episode=1)
    # collect.close()