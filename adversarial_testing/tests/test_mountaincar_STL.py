#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:37:24 2022

@author: josh
"""

'''
Here we consider a controller trained  for the mountain-car environment in
OpenAI Gym. The controller was taken from the website.
Originally this controller is trained to be non-deterministic sampled from a
gaussian distribution, but we make it deterministic by considering the most
likely control, the mean.
'''
import os
import time
import math
from datetime import datetime
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
import gym
#import torch

import logging
import numpy as np


import tensorflow as tf
import numpy as np
import os
import gym
import time
import itertools
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

from baselines.common import set_global_seeds, tf_util as U
import gym, logging
from baselines import logger
import numpy as np
import tensorflow as tf
from baselines.ppo1 import mlp_policy, pposgd_simple
from baselines.ppo1.pposgd_simple import *

"""
# Training phase
def exec_time(func):
    def new_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Cost {} seconds.".format(end - start))
        return result

    return new_func

seed = 8902077161928034768
env = gym.envs.make("MountainCarContinuous-v0")
seed = env.seed()
video_dir = os.path.abspath("./videos")
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
#env = gym.wrappers.Monitor(env, video_dir, force=True)

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))


def process_state(state):
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]


class PolicyEstimator:
    def __init__(self, env, lamb=1e-5, learning_rate=0.01, scope="policy_estimator"):
        self.env = env
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.train =True

        with tf.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [400], name="state")

        self.mu = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.mu = tf.squeeze(self.mu)

        self.sigma = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.sigma = tf.squeeze(self.sigma)
        self.sigma = tf.nn.softplus(self.sigma) + 1e-5

        self.norm_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = self.norm_dist.sample(1)
        self.action = tf.clip_by_value(self.action, self.env.action_space.low[0],
                                       self.env.action_space.high[0])

    def _build_train_op(self):
        self.action_train = tf.placeholder(tf.float32, name="action_train")
        self.advantage_train = tf.placeholder(tf.float32, name="advantage_train")

        self.loss = -tf.log(
            self.norm_dist.prob(self.action_train) + 1e-5) * self.advantage_train \
                    - self.lamb * self.norm_dist.entropy()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        feed_dict = {self.state: process_state(state)}
        return sess.run(self.action, feed_dict=feed_dict)

    def run_deterministic(self, state, sess):
        feed_dict= {self.state:process_state(state)}
        return sess.run(self.mu, feed_dict=feed_dict)

    def update(self, state, action, advantage, sess):
        feed_dict = {
            self.state: process_state(state),
            self.action_train: action,
            self.advantage_train: advantage
        }
        sess.run([self.train_op], feed_dict=feed_dict)


class ValueEstimator:
    def __init__(self, env, learning_rate=0.01, scope="value_estimator"):
        self.env = env
        self.learning_rate = learning_rate

        with tf.variable_scope(scope):
            self._build_model()
            self._build_train_op()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [400], name="state")

        self.value = tf.contrib.layers.fully_connected(
            inputs=tf.expand_dims(self.state, 0),
            num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer()
        )
        self.value = tf.squeeze(self.value)

    def _build_train_op(self):
        self.target = tf.placeholder(tf.float32, name="target")
        self.loss = tf.reduce_mean(tf.squared_difference(self.value, self.target))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.value, feed_dict={self.state: process_state(state)})

    def update(self, state, target, sess):
        feed_dict = {
            self.state: process_state(state),
            self.target: target
        }
        sess.run([self.train_op], feed_dict=feed_dict)


@exec_time
def actor_critic(episodes=100, gamma=0.95, display=False, lamb=1e-5,
                 policy_lr=0.001, value_lr=0.1):
    tf.reset_default_graph()
    policy_estimator = PolicyEstimator(env, lamb=lamb, learning_rate=policy_lr)
    value_estimator = ValueEstimator(env, learning_rate=value_lr)
    config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    stats = []
    for i_episode in range(episodes):
        state = env.reset()
        reward_total = 0
        for t in itertools.count():
            action = policy_estimator.predict(state, sess)
            next_state, reward, done, _ = env.step(action)
            reward_total += reward

            if display:
                env.render()

            target = reward + gamma * value_estimator.predict(next_state, sess)
            td_error = target - value_estimator.predict(state, sess)

            policy_estimator.update(state, action, advantage=td_error, sess=sess)
            value_estimator.update(state, target, sess=sess)

            if done:
                break
            state = next_state
        stats.append(reward_total)
        if np.mean(stats[-100:]) > 90 and len(stats) >= 101:
            print(np.mean(stats[-100:]))
            print("Solved.")
        print("Episode: {}, reward: {}.".format(i_episode, reward_total))
    return np.mean(stats[-100:]), policy_estimator, sess


def controller_training(episodes=200):
    policy_lr, value_lr, lamb, gamma = [0.0001, 0.00046415888336127773,
                                        2.782559402207126e-05, 0.98999999999999999]
    loss, policy_estimator, sess = actor_critic(episodes=episodes, gamma=gamma,
                                                display=False, lamb=lamb,
                                                policy_lr=policy_lr, value_lr=value_lr)
    print(-loss)
    env.close()
    return policy_estimator, sess


pe, sess= controller_training(200)
"""
#PPO Controller
def learn_return(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vfloss1 = tf.square(pi.vpred - ret)
    vpredclipped = oldpi.vpred + tf.clip_by_value(pi.vpred - oldpi.vpred, -clip_param, clip_param)
    vfloss2 = tf.square(vpredclipped - ret)
    vf_loss = .5 * U.mean(tf.maximum(vfloss1, vfloss2)) # we do the same clipping-based trust region for the value function
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        print(sum(seg['rew']),seg['rew'], len(seg['rew']))
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))
        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far+=1

    return pi

U.make_session(num_cpu=1).__enter__()
seed = 8902077161928034768
env = gym.make("MountainCarContinuous-v0")
env.seed(seed)
num_timesteps=5e6

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=2)

env.seed(seed)
gym.logger.setLevel(logging.WARN)
pi = learn_return(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_batch=2048,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
        gamma=0.99, lam=0.95,
    )


from gym import spaces
def compute_traj(**kwargs):
    env.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        env.env.state = ob
    if 'goal_pos' in kwargs:
        gp = kwargs['goal_pos']
        env.env.goal_position = gp
    if 'max_speed' in kwargs:
        ms = kwargs['max_speed']
        env.env.max_speed = ms
        env.env.low_state = \
            np.array([env.env.min_position, - env.env.max_speed])
        env.env.high_state = \
            np.array([env.env.max_position, env.env.max_speed])
        env.env.observation_space = \
            spaces.Box(env.env.low_state, env.env.high_state)
    if 'power' in kwargs:
        pow = kwargs['power']
        env.env.power = pow
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf

    iter_time = 0
    reward = 0
    done=False
    traj = [ob]
    while done==False:
        iter_time += 1
        action = pe.run_deterministic(ob, sess)
        ob, r, done, _ = env.step(np.array([action]))
        traj.append(ob)
        reward += r
        done = done or (iter_time >= max_steps)
        if done:
            break
    return traj, {'reward':reward, 'iter_time': iter_time}
"""
def sut(x0, **kwargs):
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf
    return compute_traj(max_steps=max_steps, init_state=x0[0:2],goal_pos=x0[2],
                        max_speed=x0[3], power=x0[4])

def sut_nv(x0, **kwargs):
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf
    return compute_traj(max_steps=max_steps, init_state=np.array([x0[0], 0.]),goal_pos=x0[1],
                        max_speed=x0[2], power=x0[3])
"""
def sut(x0, **kwargs):
    if 'max_steps' in kwargs:
        max_steps = kwargs['max_steps']
    else:
        max_steps = np.inf
    return compute_traj(max_steps=max_steps, init_state=x0[0:2],
                        max_speed=x0[2], power=x0[3])

from scipy.stats import chi2, norm

def cost_func(X):
    goal_rv = chi2(5, loc=0.3999999, scale=0.05/3.)
    speed_rv = chi2(5, scale=0.005/3.)
    power_rv = norm(0.0015, 0.00075)
    goal_pdf = goal_rv.pdf(X.T[2]) / goal_rv.pdf(0.45)
    speed_pdf = speed_rv.pdf(0.075- X.T[3]) / speed_rv.pdf(0.005)
    power_pdf = power_rv.pdf(X.T[4]) / power_rv.pdf(0.0015)
    goal_pdf.resize(len(goal_pdf), 1)
    speed_pdf.resize(len(speed_pdf), 1)
    power_pdf.resize(len(power_pdf), 1)
    return goal_pdf*speed_pdf*power_pdf

#------------------------------------------------------------------------------------------------------------------
from adversarial_testing import pred_node, max_node, min_node, test_module
from adversarial_testing.utils import sample_from
rand_nums = [
 3188388221,
 1954593344,
 2154016205,
 3894811078,
 3493033583,
 3248332584,
 1304673443,
 3857496775,
 2668478815,
 278535713
 ]
rand_nums2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rand_nums3=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
rand_nums4=[101, 113, 134, 156, 194, 202, 213, 111, 129, 200]
rand_nums5=[5085, 8991, 1635, 7805, 7187, 8645, 8888, 5520, 6446, 1714,]
rand_nums6=[1461, 8194, 6927, 5075, 4903, 3799, 6268, 8155, 5502, 1187]
rand_nums7=[64846, 28856, 43210, 70661, 14700, 21044, 58191, 17243, 24958, 80194]
rand_nums8=[54239, 69118, 51184, 57468, 57945, 78075, 34142, 78062, 33150,
            64048]
rand_nums9=[63951, 36835, 59249, 17176, 32123, 54118, 79720, 64639, 81307, 16913]
rand_nums10=[347957, 510020, 545416, 613511, 673274, 619204, 630790, 627544,
       127016, 390172]
rand_nums11=[61,18,2,33,31,49,81,17,11,131]
rand_nums12=[65,13,19,38,32,99,84,22,41,143]
rand_nums_test=[172857]

# Requirement 1: Find the initial configuration that minimizes the reward
# We need only one node for the reward. The reward is a smooth function
# given that the closed loop system is deterministic
bounds = [(-0.6, -0.4)] # Bounds on the position
bounds.append((-0.025, 0.025)) # Bounds on the velocity
#bounds.append((0.4, 0.6)) # Bounds on the goal position
bounds.append((0.055, 0.075)) # Bounds on the max speed
bounds.append((0.0005, 0.0025)) # Bounds on the power magnitude


smooth_details=[]
ns_details= []
random_details= []

smooth_results=[]
ns_results=[]
random_results=[]

smooth_Failure_count=[]
rand_Failure_count=[]
ns_Failure_count=[]

def pred1(traj):
    traj1 = traj[0]
    Robustness=[]
    for i in range (len(traj1)):
        #x_pos=np.array(traj1[0]).T[i]
        #angle=np.array(traj[0]).T[i]
        x_pos=traj1[i][0]
        velocity=traj1[i][1]
        if x_pos <= -1.1 or x_pos>=0.5:
            Robustness.append(0.03-abs(velocity))
        if x_pos>-1.1 and x_pos<0.5:
            Robustness.append(1/abs(x_pos))
    return min(Robustness)

def pred2(traj):
    traj1 = traj[0]
    Robustness=[]
    Until_Con=0
    for i in range (len(traj1)):
        #x_pos=np.array(traj1[0]).T[i]
        #angle=np.array(traj[0]).T[i]
        x_pos=traj1[i][0]
        velocity=traj1[i][1]
        if x_pos>0.15:
            Until_Con+=1
        if x_pos<=0.15:
            Until_Con+=0
        if Until_Con<1:
            Robustness.append(0.04-abs(velocity))
        if Until_Con<1:
            Robustness.append(1)
    return min(Robustness)

for r in rand_nums:
    """
    np.random.seed(r)
    node0=pred_node(f=pred1)
    node1=pred_node(f=pred2)
    node2=min_node(children=[node0,node1])

    TM_smooth = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=1000),
                     f_tree=node2, init_samples=20, with_smooth=True,
                     with_random=False, with_ns=False,
                     optimize_restarts=1, exp_weight=2)
    TM_smooth.initialize()
    
    TM_smooth.run_BO(100)
    
    smooth_Failure_count.append(TM_smooth.smooth_count)
    
    smooth_vals = np.array(TM_smooth.f_acqu.find_GP_func())
    smooth_details.append([TM_smooth.smooth_count,
                              TM_smooth.smooth_min_x,
                              TM_smooth.smooth_min_val, TM_smooth.smooth_min_loc])
    
    #smooth_results.append(TM_smooth.smooth_count, )
    #del TM_smooth
    #print(r, smooth_details_r1[-1])
    """
#for r in rand_nums:
    
    np.random.seed(r)
    node0_rand=pred_node(f=pred1)
    node1_rand=pred_node(f=pred2)
    node2_rand=min_node(children=[node0_rand,node1_rand])

    TM_rand = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=1000),
                     f_tree=node2_rand, init_samples=20, with_smooth=False,
                     with_random=True, with_ns=False,
                     optimize_restarts=1, exp_weight=2)
    TM_rand.initialize()
    
    TM_rand.run_BO(100)
    
    rand_Failure_count.append(TM_rand.rand_count)
    
    rand_vals = np.array(TM_rand.random_Y)
    random_details.append([TM_rand.rand_count,
                          TM_rand.rand_min_x,
                          TM_rand.rand_min_val, TM_rand.rand_min_loc])
    #print(r, random_details_r3[-1])
    #del TM_rand
#for r in rand_nums:
    
    np.random.seed(r)
    node0_ns=pred_node(f=pred1)
    node1_ns=pred_node(f=pred2)
    node2_ns=min_node(children=[node0_ns,node1_ns])

    TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(x0,max_steps=1000),
                     f_tree=node2_ns, init_samples=20, with_smooth=False,
                     with_random=False, with_ns=True,
                     optimize_restarts=1, exp_weight=2)
    TM_ns.initialize()
    
    TM_ns.run_BO(100)
    
    ns_Failure_count.append(TM_ns.ns_count)
    
    ns_smooth_vals = np.array(TM_ns.ns_GP.Y)
    ns_details.append([TM_ns.ns_count,
                        TM_ns.ns_min_x,
                        TM_ns.ns_min_val, TM_ns.ns_min_loc])

Random_mean=np.mean(rand_Failure_count)

#Smooth_mean=np.mean(smooth_Failure_count)

NS_mean=np.mean(ns_Failure_count)

#Smooth_std=np.std(smooth_Failure_count)

NS_std=np.std(ns_Failure_count)

Random_std=np.std(rand_Failure_count)

Method=['Random Sampling',  'BO']
x_pos=np.arange(len(Method))

Means_Failure_Modes=[Random_mean,NS_mean]

Error=[Random_std,NS_std]

plt.bar(x_pos, Means_Failure_Modes, yerr=Error, align='center', alpha=0.5, ecolor='black', capsize=10),\
    plt.ylabel('Failure Modes Found'),plt.xticks(x_pos,Method),\
    plt.title('Failure Modes Found: BO v. Random Sampling'),\
    plt.grid(True,axis='y'), plt.show()

   
#How to Plot failure modes per test
"""
Test_num=['1','2','3','4','5','6','7','8','9','10']
X_axis=np.arange(len(Test_num))

plt.bar(X_axis-0.2,smooth_Failure_count,0.2,label='Smooth BO')
plt.bar(X_axis+0,rand_Failure_count,0.2,label='Random Sampling BO')
plt.bar(X_axis+0.2,ns_Failure_count,0.2,label='Non-Smooth BO')
plt.xticks(X_axis,Test_num)
plt.xlabel("Tests")
plt.ylabel("Failure Modes Found")
plt.title("Failure Modes Per Test")
plt.legend()
plt.show()

#To plot robustness over BO iteration
time_range_iteration=np.array(list(range(1,len(TM_smooth.f_acqu.Y)+1)))
fig4=plt.figure()
ex=fig4.add_subplot()
ex.set_xlim(left=1,right=len(TM_smooth.f_acqu.Y)+1)
ex.scatter(time_range_iteration,TM_smooth.f_acqu.Y)

ex.set_title('Smooth BO Robustness v. Iteration')
ex.legend(['Smooth_BO_Robustness'])
ex.set_xlabel('BO Iteration')
ex.set_ylabel('Robustness')

time_range_iteration2=np.array(list(range(1,len(TM_ns.ns_GP.Y)+1)))
fig5=plt.figure()
fx=fig5.add_subplot()
fx.set_xlim(left=1,right=len(TM_ns.ns_GP.Y)+1)
fx.scatter(time_range_iteration2,TM_ns.ns_GP.Y)

fx.set_title('NS BO Robustness v. Iteration')
fx.legend(['NS_BO_Robustness'])
fx.set_xlabel('BO Iteration')
fx.set_ylabel('Robustness')

time_range_iteration3=np.array(list(range(1,len(TM_rand.random_Y)+1)))
fig6=plt.figure()
gx=fig6.add_subplot()
gx.set_xlim(left=1,right=len(TM_rand.random_Y)+1)
gx.scatter(time_range_iteration3,TM_rand.random_Y)

gx.set_title('Random Sampling Robustness v. Iteration')
gx.legend(['Random_Sampling_Robustness'])
gx.set_xlabel('BO Iteration')
gx.set_ylabel('Robustness')

#To create safest and most dangerous trajectories of Smooth-GP BO
smooth_safest_params=TM_smooth.smooth_X[smooth_vals.argmax()]
traj_smooth_safe=[]
cart_pos_safe=[]
cart_vel_safe=[]

traj_smooth_safe.append(TM_smooth.system_under_test(smooth_safest_params))
for i in range(len(np.array(traj_smooth_safe[0][0]))):
    cart_pos_safe.append(np.array(traj_smooth_safe[0][0]).T[0][i])
    cart_vel_safe.append(np.array(traj_smooth_safe[0][0]).T[1][i])

    
import operator
enumerate_obj=enumerate(smooth_vals)
sorted_smooth_vals=sorted(enumerate_obj,key=operator.itemgetter(1))
sorted_indices_smooth_vals=[index for index, element in sorted_smooth_vals]
print(sorted_indices_smooth_vals)
Four_DangerousEnv_Param=TM_smooth.smooth_X[sorted_indices_smooth_vals[0:4]]

traj_smooth_traj1=[]
cart_pos_traj1=[]
cart_vel_traj1=[]

smooth_traj1_params=Four_DangerousEnv_Param[0]
traj_smooth_traj1.append(TM_smooth.system_under_test(smooth_traj1_params))
for i in range(len(np.array(traj_smooth_traj1[0][0]))):
    cart_pos_traj1.append(np.array(traj_smooth_traj1[0][0]).T[0][i])
    cart_vel_traj1.append(np.array(traj_smooth_traj1[0][0]).T[1][i])

    
#To Save data, change 1B and date at end
DF=pd.DataFrame(rand_Failure_count)
DF.to_csv("Experiment_7_Time_Random_Failure_Count_07-27.csv")

DF=pd.DataFrame(ns_Failure_count)
DF.to_csv("Experiment_7_Time_NS_Failure_Count_07-27.csv")

DF=pd.DataFrame(smooth_Failure_count)
DF.to_csv("Experiment_7_Time_Smooth_Failure_Count_07-27.csv")

#To Save Params and Trajs for best smooth
DF=pd.DataFrame(smooth_safest_params)
DF.to_csv("Experiment_7_STL_Smooth_Safe_Param_07-27.csv")
DF=pd.DataFrame(cart_pos_safe)
DF.to_csv("Experiment_7_STL_Smooth_Safe_Cart_Pos_07-27.csv")
DF=pd.DataFrame(cart_vel_safe)
DF.to_csv("Experiment_7_STL_Smooth_Safe_Cart_Vel_07-27.csv")


#To Save Params and Trajs for Worst smooth
DF=pd.DataFrame(smooth_traj1_params)
DF.to_csv("Experiment_7_STL_Smooth_traj1_Param_07-27.csv")
DF=pd.DataFrame(cart_pos_traj1)
DF.to_csv("Experiment_7_STL_Smooth_traj1_Cart_Pos_07-27.csv")
DF=pd.DataFrame(cart_vel_traj1)
DF.to_csv("Experiment_7_STL_Smooth_traj1_Cart_Vel_07-27.csv")


#To load files into lists
BO_Failure_count=pd.read_csv("/home/josh/Cart_Pole_Results/July_19_2022/Test 6: Both/Experiment_6_Time_Smooth_Failure_Count_07-18.csv")
BO=BO_Failure_count.to_numpy()
BO_count=[]
for i in range(10):
    rand_count.append(np.array(rand).T[1][i])
    
    
#Best for Random Sampling and BO
Random_mean=np.mean(rand_count)
BO_mean=np.mean(BO_count)
BO_std=np.std(BO_count)
Random_std=np.std(rand_count)
Method=['Random Sampling', 'Bayeisan Optimization']
x_pos=np.arange(len(Method))
Means_Failure_Modes=[Random_mean,BO_mean]
Error=[Random_std,BO_std]
plt.bar(x_pos, Means_Failure_Modes, yerr=Error, align='center', alpha=0.5, ecolor='black', capsize=10),\
    plt.ylabel('Failure Modes Found'),plt.xticks(x_pos,Method),\
    plt.title('Failure Modes Found for Cart Pole Using Random Sampling and BO'),\
    plt.grid(True,axis='y'), plt.show()

Test_num=['1','2','3','4','5','6','7','8','9','10']
X_axis=np.arange(len(Test_num))
plt.bar(X_axis+0,rand_count,0.2,label='Random Sampling')
plt.bar(X_axis-0.2,BO_count,0.2,label='Bayesian Optimization')
plt.xticks(X_axis,Test_num)
plt.xlabel("Tests")
plt.ylabel("Failure Modes Found")
plt.title("Failure Modes Per Test: Cart Pole")
plt.legend()
plt.show()
"""
