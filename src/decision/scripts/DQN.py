"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features=4,
            learning_rate=0.01,
            reward_decay=0.9,
            # e_greedy=0.9,
            e_greedy_max=0.9,
            e_greedy_min=0.1,
            replace_target_iter=300,
            memory_size=5000,
            batch_size=500,
            num_episode=50000,
            num_rate=0.5,
            # e_greedy_increment=None,
            output_graph=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        # self.epsilon_max = e_greedy
        self.e_greedy_max = e_greedy_max
        self.e_greedy_min = e_greedy_min
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.num_episode = num_episode
        self.num_rate = num_rate
        # self.epsilon_increment = e_greedy_increment
        # self.epsilon = 1.0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon = e_greedy_max
        # total learning step
        self.learn_step_counter = 0
        self.double_q = True
        # initialize zero memory [s, a, r, s_]
        # self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        # self.memory = np.array([])
        self.memory = []

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=config)

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.saver = tf.train.Saver()
        self.model_path = './'
        self.model_name = 'saved_network_DQN_tf'

    def _build_net(self):
        # self.state_img = tf.placeholder(tf.float32, [None, 80, 80], name='s')  # input
        # self.state_laser = tf.placeholder(tf.float32, [None, 4, 360])
        # self.s = [self.state_image, self.state_laser]

        # self.s = (tf.placeholder(tf.float32, [None, 80, 80, 4], name='image_s'),
        #           tf.placeholder(tf.float32, [None, 4, 360], name='laser_s'))
        # self.s_ = (tf.placeholder(tf.float32, [None, 80, 80, 4], name='image_s_'),
        #            tf.placeholder(tf.float32, [None, 4, 360], name='laser_s_'))

        self.image_s = tf.placeholder(tf.float32, [None, 80, 80, 4], name='image_s')
        self.laser_s = tf.placeholder(tf.float32, [None, 4, 360], name='laser_s')
        self.image_s_ = tf.placeholder(tf.float32, [None, 80, 80, 4], name='image_s_')
        self.laser_s_ = tf.placeholder(tf.float32, [None, 4, 360], name='laser_s_')

        # ------------------ build evaluate_net ------------------# Behavior Network
        self.q_eval = self._net_model([self.image_s, self.laser_s], 'eval_net', 'eval_net_params', 'conv', 'fc')
        # ------------------ build target_net ------------------# Target Network
        self.q_next = self._net_model([self.image_s_, self.laser_s_], 'target_net', 'target_net_params', 'conv_', 'fc_')

        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def _net_model(self, net_input, net_name, col_name, conv_name, fc_name):
        with tf.variable_scope(net_name):
            c_names = [col_name, tf.GraphKeys.GLOBAL_VARIABLES]
            # net_input = [self.state_image, self.state_laser]
            image_data = (net_input[0] - (255.0 / 2)) / (255.0)  # normalization [80, 80, 4]
            laser_data = tf.unstack(net_input[1], axis=1)  # [360, 4]

            with tf.variable_scope(conv_name):
                w1 = self.get_variable(conv_name+'w1', [8, 8, 4, 32], c_names)
                b1 = self.get_variable(conv_name+'b1', [32], c_names)
                h1 = tf.nn.relu(self.conv2d(image_data, w1, 4) + b1)

                w2 = self.get_variable(conv_name+'w2', [4, 4, 32, 64], c_names)
                b2 = self.get_variable(conv_name+'b2', [64], c_names)
                h2 = tf.nn.relu(self.conv2d(h1, w2, 2) + b2)

                w3 = self.get_variable(conv_name+'w3', [3, 3, 64, 64], c_names)
                b3 = self.get_variable(conv_name+'b3', [64], c_names)
                h3 = tf.nn.relu(self.conv2d(h2, w3, 1) + b3)  # [1, 10, 10, 64]

            cell = tf.nn.rnn_cell.LSTMCell(num_units=512)
            laser_out, _ = tf.nn.static_rnn(inputs=laser_data, cell=cell, dtype=tf.float32)  # [360, 4]

            img_out = tf.reshape(h3, [-1, 10 * 10 * 64])  # [1, 6400]
            h_concat = tf.concat([img_out, laser_out[-1]], axis=1)

            with tf.variable_scope(fc_name):
                w_fc1 = self.get_variable(fc_name+'w1', [6400 + 512, 512], c_names)
                b_fc1 = self.get_variable(fc_name+'b1', [512], c_names)
                h_fc1 = tf.nn.relu(tf.matmul(h_concat, w_fc1) + b_fc1)

                w_fc2 = self.get_variable(fc_name+'w2', [512, self.n_actions], c_names)
                b_fc2 = self.get_variable(fc_name+'b2', [self.n_actions], c_names)
                return tf.matmul(h_fc1, w_fc2) + b_fc2

    def get_variable(self, name, shape, col):
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), collections=col)

    def conv2d(self, img, img_filter, stride):
        return tf.nn.conv2d(img, img_filter, strides=[1, stride, stride, 1], padding='SAME')

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # transition = np.hstack((s, [a, r], s_)) # transition = [s, a, r, s_]
        transition = [s, a, r, s_]
        if self.memory_counter < self.memory_size:
            # self.memory = np.append(self.memory, transition)
            self.memory.append(transition)
        else:
            index = self.memory_counter % self.memory_size
            self.memory[index] = transition
        # replace the old memory with new memory
        # index = self.memory_counter % self.memory_size
        # self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, state):
        if np.random.uniform() > self.epsilon:  # choose Q_max action
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, 
                feed_dict={self.image_s: [state[0]], self.laser_s: [state[1]]})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # print ('======learning======')
        # time.sleep(1)
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:  # 300次一更新
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # batch_memory = self.memory[sample_index, :]

        # batch_memory = []
        # for index in sample_index:
        #     batch_memory.append(self.memory[index])

        # print ('---get batch_memory---')
        
        # print ('self.memory shape is {}'.format(np.array(self.memory).shape))

        # batch_memory = np.array([np.array(self.memory[index]) for index in sample_index])

        batch_memory = [self.memory[index] for index in sample_index]
        # print ('batch_memory shape is {}'.format(np.array(batch_memory).shape))
        '''
            >>> a = np.random.choice(3, 5)
            a = [0 2 2 1 2]
            so we get a uniform random from (0, self.memory_size) for size of self.bathc_size
            batch_memory is a list of self.memory by index of item in sample_index

            batch_memory = [
                            [s1, a1, r1, s_1],
                            [s3, a3, r3, s_3],
                            ...
                            [s7, a7, r7, s_7],
                            ...
                            ]

            which s1 and s_1 are a features list for self.n_features length w.r.t s1 = [2, 3, 1, 0] contain 4 features
        '''
        # batch_s = []
        # for item in batch_memory:
        #     batch_s.append(item[0])

        # batch_s = np.array([np.array(item[0]) for item in batch_memory])
        # batch_s_ = np.array([np.array(item[-1]) for item in batch_memory])

        # batch_s = np.array([item[0] for item in batch_memory])
        # batch_s_ = np.array([item[-1] for item in batch_memory])
        # print ('batch_s shape is {}'.format(batch_s.shape))

        batch_s = [item[0] for item in batch_memory]
        batch_s_ = [item[-1] for item in batch_memory]
        # print (np.array(batch_s).shape)

        # batch_image = np.array([item[0] for item in batch_s])
        # batch_laser = np.array([item[1] for item in batch_s])

        batch_image = np.array([item[0] for item in batch_s])
        batch_laser = np.array([item[1] for item in batch_s])
        batch_image_ = np.array([item[0] for item in batch_s_])
        batch_laser_ = np.array([item[1] for item in batch_s_])

        # print ('batch_image shape:{}, batch_laser shape:{}'.format(batch_image.shape, batch_laser.shape))
        # print ('---sess.run q_next, q_eval for batch---')
        # q_next, q_eval = self.sess.run(
        #     [self.q_next, self.q_eval],
        #     feed_dict={
        #         self.s_: batch_s_,  # fixed params
        #         self.s: batch_s,  # newest params
        #     })
        # print (type(batch_s))
        # print (batch_s)

        # print (batch_s_[:, 0])
        # print (batch_s_[:, 0][0].shape)

        # print (batch_s_[:, 0].shape)

        # print (batch_s_[:, 1][0])
        # print (batch_s_[:, 1][0].shape)
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.image_s_: batch_image_, self.laser_s_: batch_laser_,
                self.image_s: batch_image, self.laser_s: batch_laser  # newest params
            })
        # q_eval = self.sess.run(self.q_eval, feed_dict={self.image_s: batch_image, self.laser_s: batch_laser})
        # print ('----after sess.run ----, sleep for ')
        # time.sleep(3)
        '''
        batch_memory = [
                        [s1, a1, r1, s_1],
                        [s3, a3, r3, s_3],
                        [s7, a7, r7, s_7],
                        ......
                        ]
        >>> a[:, -n_features:]
        [s_1, s_3, s_7, ...]

        >>> a[:, :n_features]
        [s1, s3, s7, ...]
        '''
        # change q_target w.r.t q_behavior's action
        q_target = q_eval.copy()
        # print ('---copy q_eval to q_target---, sleep for ')
        # time.sleep(3)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # eval_act_index = batch_memory[:, self.n_features].astype(int)
        eval_act_index = np.array([item[1] for item in batch_memory]).astype(int)
        # reward = batch_memory[:, self.n_features + 1]
        reward = np.array([item[2] for item in batch_memory])
        '''
        bathch_index = [0, 1, 2, ...]
        eval_act_index = [a1, a3, a7, ...]
        reward = [r1, r3, r7, ...]
        '''
        if self.double_q:
            max_act = np.argmax(q_eval, axis=1)
            selected_q = q_next[batch_index, max_act]
        else:
            selected_q = np.max(q_next, axis=1)
        # print ('---update q_target---')
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q

        """
        For example in this batch I have 2 samples and 3 actions:
        q_behavior =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_behavior =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_behavior's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_behavior) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.image_s: batch_image, self.laser_s: batch_laser,
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        pot = np.log(self.e_greedy_max / self.e_greedy_min) / self.num_rate / self.num_episode

        # if self.epsilon < 0.1:
        #     self.epsilon = 0.1
        # else:
        #     self.epsilon = np.exp(-pot * self.learn_step_counter)
        if self.epsilon < self.e_greedy_min:
            self.epsilon = self.e_greedy_min
        else:
            self.epsilon = self.e_greedy_max * np.exp(-pot*self.learn_step_counter/5)
        
        # self.epsilon = self.e_greedy_min if self.epsilon < self.e_greedy_min else (self.e_greedy_max * np.exp(-pot * self.num_episode))

        self.learn_step_counter += 1
        if self.learn_step_counter % 10000 == 0:
            save_path = self.model_path + self.model_name
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.saver.save(self.sess, save_path, global_step=self.learn_step_counter)

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


# DoubleDQN
class DoubleDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=3000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            double_q=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.double_q = double_q    # decide to use double q or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2
            return out
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


# DQN_PR:
class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized    # decide to use double q or not

        self.learn_step_counter = 0

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer, trainable):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names,  trainable=trainable)
                out = tf.matmul(l1, w2) + b2
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer, True)

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer, False)

    def store_transition(self, s, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: batch_memory[:, -self.n_features:],
                           self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
