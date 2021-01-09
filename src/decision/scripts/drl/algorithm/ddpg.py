import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from drl.algorithm import BasePolicy
from drl.utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG(BasePolicy):
    def __init__(
        self,
        model,
        buffer_size=1000,
        actor_learn_freq=1,
        target_update_freq=0,
        target_update_tau=1,
        learning_rate=1e-3,
        discount_factor=0.99,
        batch_size=100,
        update_iteration=10,
        verbose=False,
    ):
        super().__init__()
        self.lr = learning_rate
        self.eps = np.finfo(np.float32).eps.item()
        self.tau = target_update_tau

        self.actor_learn_freq = actor_learn_freq
        self.target_update_freq = target_update_freq
        self._gamma = discount_factor
        self._target = target_update_freq > 0
        self._update_iteration = update_iteration
        self._sync_cnt = 0
        self._learn_critic_cnt = 0
        self._learn_actor_cnt = 0
        self._verbose = verbose
        self._batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)

        self.actor_eval = model.policy_net.to(device).train()  # pi(s)
        self.critic_eval = model.value_net.to(device).train()  # Q(s, a)
        self.actor_eval_optim = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_eval_optim = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        if self._target:
            self.actor_target = self.copy_net(self.actor_eval)
            self.critic_target = self.copy_net(self.critic_eval)

        self.criterion = nn.MSELoss()  # why mse?

    def learn(self):
        loss_actor_avg, loss_critic_avg = 0, 0

        for _ in range(self._update_iteration):
            batch_split = self.buffer.split_batch(self._batch_size)
            S = torch.tensor(batch_split['s'], dtype=torch.float32, device=device)  # [batch_size, S.feature_size]
            A = torch.tensor(batch_split['a'], dtype=torch.float32, device=device).view(-1, 1)  # [batch_size, 1]
            M = torch.tensor(batch_split['m'], dtype=torch.float32).view(-1, 1)
            R = torch.tensor(batch_split['r'], dtype=torch.float32).view(-1, 1)
            S_ = torch.tensor(batch_split['s_'], dtype=torch.float32, device=device)

            with torch.no_grad():
                q_next = self.critic_eval(S_, self.actor_eval(S_))
                if self._target:
                    q_next = self.critic_target(S_, self.actor_target(S_))
                q_target = R + M * self._gamma * q_next.cpu()
                q_target = q_target.to(device)
            # print (f'SIZE S {S.size()}, A {A.size()}')
            q_eval = self.critic_eval(S, A)  # [batch_size, q_value_size]
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
                if self._verbose:
                    print(f'=======Learn_Actort_Net=======')

            if self._target:
                if self._learn_critic_cnt % self.target_update_freq == 0:
                    if self._verbose:
                        print(f'=======Soft_sync_weight of DDPG=======')
                    self.soft_sync_weight(self.critic_target, self.critic_eval, self.tau)
                    self.soft_sync_weight(self.actor_target, self.actor_eval, self.tau)

        loss_actor_avg /= (self._update_iteration/self.actor_learn_freq)
        loss_critic_avg /= self._update_iteration
        return loss_actor_avg, loss_critic_avg
