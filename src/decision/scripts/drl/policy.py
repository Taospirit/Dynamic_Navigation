import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from abc import ABC, abstractmethod
from drl.utils import ReplayBuffer
# import untils

class BasePolicy(ABC):
    def __init__(self, **kwargs):
        super().__init__()
    
    @abstractmethod
    def choose_action(self, state, **kwargs):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def process(self, **kwargs):
        pass
    
    def save_model(self, save_dir, save_file_name):
        assert isinstance(save_dir, str) and isinstance(save_file_name, str)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, save_file_name)
        actor_save = save_path + '_actor.pth'
        critic_save = save_path + '_critic.pth'

        torch.save(self.actor_eval.state_dict(), actor_save)
        torch.save(self.critic_eval.state_dict(), critic_save)
        print (f'Save actor-critic model in {save_path}!')

    def load_model(self, save_dir, save_file_name, test=False):
        save_path = os.path.join(save_dir, save_file_name)
        actor_save = save_path + '_actor.pth'
        assert os.path.exists(actor_save), f'No {actor_save} file to load'

        self.actor_eval.load_state_dict(torch.load(actor_save))
        print (f'Loading actor model success in {actor_save}!')
        
        if test: return
        critic_save = save_path + '_critic.pth'
        assert os.path.exists(critic_save), f'No {critic_save} file to load'
        self.critic_eval.load_state_dict(torch.load(critic_save))
        print (f'Loading critic model success in {critic_save}!')

    @staticmethod
    def soft_sync_weight(target, source, tau=0.01):
        with torch.no_grad():
            for target_param, eval_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def GAE(rewards, v_evals, next_v_eval=0, masks=None, gamma=0.99, lam=1): # [r1, r2, ..., rT], [V1, V2, ... ,VT, VT+1]
        r'''
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

        :param gamma: (float) Discount factor
        :param lam: (float) GAE factor
        '''
        # GAE(gamma, lam=0) = td_error
        # GAE(gamma, lam=1) = MC
        assert isinstance(rewards, np.ndarray), 'rewards must be np.ndarray'
        assert isinstance(v_evals, np.ndarray), 'v_evals must be np.ndarray'
        assert len(rewards) == len(v_evals), 'V_pred length must equal rewards length'

        rew_len = len(rewards)
        masks = np.ones(rew_len) if masks is None else masks # nonterminal
        v_evals = np.append(v_evals, next_v_eval)
        adv_gae = np.empty(rew_len, 'float32')
        last_gae = 0
        for i in reversed(range(rew_len)):
            delta = rewards[i] + gamma * v_evals[i+1] * masks[i] - v_evals[i]
            adv_gae[i] = last_gae = delta + gamma * lam * masks[i] * last_gae

        return adv_gae


class A2CPolicy(BasePolicy): #option: double
    def __init__(
        self, 
        actor_net, 
        critic_net,
        buffer_size=1000,
        actor_learn_freq=1,
        target_update_freq=0,
        target_update_tau=5e-3,
        learning_rate=0.01,
        discount_factor=0.99,
        gae_lamda=1,
        verbose = False
        ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau
        self.gae_lamda = gae_lamda

        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._sync_cnt = 0
        # self._learn_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = verbose
        self.buffer = ReplayBuffer(buffer_size, replay=False)
        # assert not self.buffer.allow_replay, 'PPO buffer cannot be replay buffer'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_eval = actor_net.to(self.device)
        self.critic_eval = critic_net.to(self.device)
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        
        self.actor_eval.train()
        self.critic_eval.train()

        if self._target:
            self.actor_target = deepcopy(self.actor_eval)
            self.critic_target = deepcopy(self.critic_eval)
            self.actor_target.load_state_dict(self.actor_eval.state_dict())
            self.critic_target.load_state_dict(self.critic_eval.state_dict())

            self.actor_target.eval()
            self.critic_target.eval()

        self.criterion = nn.SmoothL1Loss()

    def choose_action(self, state, test=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if test:
            self.actor_eval.eval()
            return Categorical(self.actor_eval(state)).sample().item(), 0
        
        dist = self.actor_eval(state)
        m = Categorical(dist)
        action = m.sample()
        log_prob = m.log_prob(action)
        state_value = self.critic_eval(state)

        return action.item(), log_prob

    def learn(self):
        memory_split = self.buffer.split(self.buffer.all_memory()) # s, r, l, m
        S = torch.tensor(memory_split['s'], dtype=torch.float32, device=self.device)
        R = torch.tensor(memory_split['r'], dtype=torch.float32).view(-1, 1)
        M = torch.tensor(memory_split['m'], dtype=torch.float32).view(-1, 1)
        Log = torch.stack(memory_split['l']).view(-1, 1)
    
        v_eval = self.critic_eval(S)

        v_evals = v_eval.detach().cpu().numpy()
        rewards = R.numpy()
        masks = M.numpy()
        adv_gae_mc = self.GAE(rewards, v_evals, next_v_eval=0, masks=masks, gamma=self._gamma, lam=self.gae_lamda) # MC adv
        advantage = torch.from_numpy(adv_gae_mc).to(self.device).reshape(-1, 1)

        v_target = advantage + v_eval.detach()
        # critic_core
        critic_loss = self.criterion(v_eval, v_target)
        self.critic_eval_optim.zero_grad()
        critic_loss.backward()
        self.critic_eval_optim.step()
        self._learn_critic_cnt += 1

        if self._learn_critic_cnt % self.actor_learn_freq == 0:
            # actor_core
            actor_loss = (-Log * advantage).sum()
            self.actor_eval.train()
            self.actor_eval_optim.zero_grad()
            actor_loss.backward()
            self.actor_eval_optim.step()
            self._learn_actor_cnt += 1

        if self._target:
            if self._learn_critic_cnt % self.target_update_freq == 0:
                if self._verbose: print (f'=======Soft_sync_weight of AC=======')
                self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)
                self._sync_cnt += 1
        
        self.buffer.clear()
        assert self.buffer.is_empty()

    def process(self, **kwargs):
        self.buffer.append(**kwargs)


class DDPGPolicy(BasePolicy):
    def __init__(
        self, 
        actor_net, 
        critic_net, 
        buffer_size=1000,
        actor_learn_freq=1,
        target_update_freq=0,
        target_update_tau=5e-3,
        learning_rate=0.01,
        discount_factor=0.99,
        batch_size=100,
        verbose = False
        ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau

        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._update_iteration = 10
        self._sync_cnt = 0
        # self._learn_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = verbose
        self._batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        # assert buffer.allow_replay, 'DDPG buffer must be replay buffer'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_eval = actor_net.to(self.device) # pi(s)
        self.critic_eval = critic_net.to(self.device) # Q(s, a)
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        
        self.actor_eval.train()
        self.critic_eval.train()

        if self._target:
            self.actor_target = deepcopy(self.actor_eval)
            self.critic_target = deepcopy(self.critic_eval)
            self.actor_target.load_state_dict(self.actor_eval.state_dict())
            self.critic_target.load_state_dict(self.critic_eval.state_dict())

            self.actor_target.eval()
            self.critic_target.eval()
            
        self.criterion = nn.MSELoss() # why mse?

    def choose_action(self, state, test=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if test:
            self.actor_eval.eval()
        action = self.actor_eval(state) # out = tanh(x)
        action = action.clamp(-1, 1)

        return action.item()

    def learn(self):

        loss_actor_avg = 0
        loss_critic_avg = 0

        for _ in range(self._update_iteration):
            memory_batch = self.replay_buffer.random_sample(self._batch_size)
            batch_split = self.replay_buffer.split(memory_batch)
            
            S = torch.tensor(batch_split['s'], dtype=torch.float32, device=self.device) # [batch_size, S.feature_size]
            A = torch.tensor(batch_split['a'], dtype=torch.float32, device=self.device).unsqueeze(-1) # [batch_size, 1]
            S_ = torch.tensor(batch_split['s_'], dtype=torch.float32, device=self.device)
            R = torch.tensor(batch_split['r'], dtype=torch.float32, device=self.device).unsqueeze(-1)

            with torch.no_grad():
                q_target = self.critic_eval(S_, self.actor_eval(S_))
                if self._target:
                    q_target = self.critic_target(S_, self.actor_target(S_))
                q_target = R + self._gamma * q_target
            print (f'SIZE S {S.size()}, A {A.size()}, S_ {S_.size()}, R {R.size()}')
            q_eval = self.critic_eval(S, A) # [batch_size, q_value_size]
            critic_loss = self.criterion(q_eval, q_target)
            loss_critic_avg += critic_loss.item()

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1

            if self._learn_critic_cnt % self.actor_learn_freq == 0:
                actor_loss = -self.critic_eval(S, self.actor_eval(S)).mean()
                loss_actor_avg += actor_loss.item()

                self.actor_eval_optim.zero_grad()
                actor_loss.backward()
                self.actor_eval_optim.step()
                self._learn_actor_cnt += 1
                if self._verbose: print (f'=======Learn_Actort_Net=======')

            if self._target:
                if self._learn_critic_cnt % self.target_update_freq == 0:
                    if self._verbose: print (f'=======Soft_sync_weight of DDPG=======')
                    self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                    self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)
        
        loss_actor_avg /= (self._update_iteration/self.actor_learn_freq)
        loss_critic_avg /= self._update_iteration

        return loss_actor_avg, loss_critic_avg

    def process(self, **kwargs):
        self.replay_buffer.append(**kwargs)


class PPOPolicy(BasePolicy): #option: double
    def __init__(
        self, 
        actor_net, 
        critic_net, 
        buffer_size=1000,
        actor_learn_freq=1,
        target_update_freq=0,
        target_update_tau=5e-3,
        learning_rate=0.0001,
        discount_factor=0.99,
        batch_size=100,
        verbose = False
        ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau
        self.ratio_clip = 0.2
        self.lam_entropy = 0.01
        self.adv_norm = True
        self.rew_norm = False
        self.schedule_clip = False
        self.schedule_adam = False

        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._update_iteration = 10
        self._sync_cnt = 0
        # self._learn_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0

        self._verbose = verbose
        self._batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, replay=False)
        # assert not self.buffer.allow_replay, 'PPO buffer cannot be replay buffer'
        self._normalized=lambda x, e: (x - x.mean()) / (x.std() + e)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_eval = actor_net.to(self.device)
        self.critic_eval = critic_net.to(self.device)
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        
        self.actor_eval.train()
        self.critic_eval.train()

        if self._target:
            self.actor_target = deepcopy(self.actor_eval)
            self.critic_target = deepcopy(self.critic_eval)
            self.actor_target.load_state_dict(self.actor_eval.state_dict())
            self.critic_target.load_state_dict(self.critic_eval.state_dict())

            self.actor_target.eval()
            self.critic_target.eval()

        self.criterion = nn.SmoothL1Loss()

    def choose_action(self, state, test=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if test:
            self.actor_eval.eval()
        with torch.no_grad():
            mu, sigma = self.actor_eval(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        # print (f'mu:{mu}, sigma:{sigma}, dist: {dist}, action sample before clamp: {action}')
        action = action.clamp(-2, 2)
        # print (f'action after clamp {action}')
        log_prob = dist.log_prob(action)
        assert  abs(action.item())<=2, f'ERROR: action out of {action}'

        return action.item(), log_prob.item()

    def get_batchs_indices(self, buffer_size, batch_size, replace=True, batch_num=None):
        indices = [i for i in range(buffer_size)]
        if replace: # 有放回的采样
            if not batch_num:
                batch_num = round(buffer_size / batch_size + 0.5) * 2
            return [np.random.choice(indices, batch_size, replace=False) for _ in range(batch_num)]
        else:# 无放回的采样
            np.random.shuffle(indices)
            return [indices[i: i + batch_size] for i in range(0, buffer_size, batch_size)]

    def learn(self, i_episode=0, num_episode=100):
        if not self.buffer.is_full(): 
            print (f'Waiting for a full buffer: {len(self.buffer)}\{self.buffer.capacity()} ', end='\r')
            return 0, 0

        loss_actor_avg = 0
        loss_critic_avg = 0

        memory_split = self.buffer.split(self.buffer.all_memory())
        S = torch.tensor(memory_split['s'], dtype=torch.float32, device=self.device)
        A = torch.tensor(memory_split['a'], dtype=torch.float32, device=self.device).view(-1, 1)
        S_ = torch.tensor(memory_split['s_'], dtype=torch.float32, device=self.device)
        R = torch.tensor(memory_split['r'], dtype=torch.float32).view(-1, 1)
        Log = torch.tensor(memory_split['l'], dtype=torch.float32, device=self.device).view(-1, 1)
        
        # print (f'Size S {S.size()}, A {A.size()}, S_ {S_.size()}, R {R.size()}, Log {Log.size()}')
        # print (f'S {S}, A {A}, S_ {S_}, R {R}, Log {Log}')
        with torch.no_grad():
            v_evals = self.critic_eval(S).cpu().numpy()
            end_v_eval = self.critic_eval(S_[-1]).cpu().numpy()

        rewards = self._normalized(R, self.eps).numpy() if self.rew_norm else R.numpy()
        # rewards = rewards.cpu().numpy()
        adv_gae_td = self.GAE(rewards, v_evals, next_v_eval=end_v_eval, gamma=self._gamma, lam=0) # td_error adv
        advantage = torch.from_numpy(adv_gae_td).to(self.device).unsqueeze(-1)
        advantage = self._normalized(advantage, 1e-10) if self.adv_norm else advantage

        # indices = [i for i in range(len(self.buffer))]
        for _ in range(self._update_iteration):
 
            v_eval = self.critic_eval(S)
            v_target = advantage + v_eval.detach()

            critic_loss = self.criterion(v_eval, v_target)
            loss_critic_avg += critic_loss.item()

            self.critic_eval_optim.zero_grad()
            critic_loss.backward()
            self.critic_eval_optim.step()
            self._learn_critic_cnt += 1
            
            if self._learn_critic_cnt % self.actor_learn_freq == 0:
                # actor_core
                mu, sigma = self.actor_eval(S)
                dist = Normal(mu, sigma)
                new_log_prob = dist.log_prob(A)
                
                pg_ratio = torch.exp(new_log_prob - Log) # size = [batch_size, 1]
                clipped_pg_ratio = torch.clamp(pg_ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)

                surrogate_loss = -torch.min(pg_ratio * advantage, clipped_pg_ratio * advantage).mean()
               
                # policy entropy
                loss_entropy = -torch.mean(torch.exp(new_log_prob) * new_log_prob)
                
                actor_loss = surrogate_loss - self.lam_entropy * loss_entropy

                loss_actor_avg += actor_loss.item()

                self.actor_eval_optim.zero_grad()
                actor_loss.backward()
                self.actor_eval_optim.step()
                self._learn_actor_cnt += 1
                if self._verbose: print (f'=======Learn_Actort_Net=======')

            if self._target:
                if self._learn_critic_cnt % self.target_update_freq == 0:
                    if self._verbose: print (f'=======Soft_sync_weight of DDPG=======')
                    self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                    self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)
    
        self.buffer.clear()
        assert self.buffer.is_empty()

        # update param
        ep_ratio = 1 - (i_episode / num_episode)
        if self.schedule_clip:
            self.ratio_clip = 0.2 * ep_ratio

        if self.schedule_adam:
            new_lr = self.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in self.actor_eval_optim.param_groups:
                g['lr'] = new_lr
            for g in self.critic_eval_optim.param_groups:
                g['lr'] = new_lr

        print (f'critic_cnt {self._learn_critic_cnt}, actor_cnt {self._learn_actor_cnt}')
        loss_actor_avg /= (self._update_iteration/self.actor_learn_freq)
        loss_critic_avg /= self._update_iteration

        return loss_actor_avg, loss_critic_avg

    def process(self, **kwargs):
        self.buffer.append(**kwargs)