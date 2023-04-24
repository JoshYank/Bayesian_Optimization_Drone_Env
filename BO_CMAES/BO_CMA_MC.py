#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 14:47:21 2023

@author: josh
"""


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

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

from BO_CMAES_Coding_1_0 import BO_CMAES
from BO_CMAES_Coding_1_0 import set_bounds

seed = 8902077161928034768
env = gym.make('MountainCarContinuous-v0')
env.seed(seed)
model = DDPG.load("ddpg_mountain")

from gym import spaces
def compute_traj(x):
    env.reset()
    ob = x[0:2]
    env.env.state = ob
    #gp = kwargs['goal_pos']
    #env.env.goal_position = gp
    ms = x[2]
    env.env.max_speed = ms
    env.env.low_state = \
        np.array([env.env.min_position, - env.env.max_speed])
    env.env.high_state = \
        np.array([env.env.max_position, env.env.max_speed])
    env.env.observation_space = \
        spaces.Box(env.env.low_state, env.env.high_state)
    pow = x[3]
    env.env.power = pow
    max_steps = 200
    #max_steps = np.inf

    iter_time = 0
    reward = 0
    done=False
    traj = [ob]
    while done==False:
        iter_time += 1
        action, _states = model.predict(ob)
        ob, rewards, dones, info = env.step(action)
        
        traj.append(ob)
        #reward += r
        done = done or (iter_time >= max_steps)
        if done:
            break
    return traj

def pred1(x):
    Y=[]
    for _ in x:
        traj=compute_traj(_)
        Robustness=[]
        for i in range (len(traj)):
            #x_pos=np.array(traj1[0]).T[i]
            #angle=np.array(traj[0]).T[i]
            x_pos=traj[i][0]
            velocity=traj[i][1]
            if x_pos <= -1.1 or x_pos>=0.5:
                Robustness.append(0.0735-abs(velocity))
            if x_pos>-1.1 and x_pos<0.5:
                Robustness.append(1/abs(x_pos))
        Y.append(min(Robustness))
    return np.array(Y)

def pred2(x):
    Y=[]
    for _ in x:
        traj=compute_traj(_)
        Robustness=[]
        Until_Con=0
        for i in range (len(traj)):
            #x_pos=np.array(traj1[0]).T[i]
            #angle=np.array(traj[0]).T[i]
            x_pos=traj[i][0]
            velocity=traj[i][1]
            if x_pos>0.1:
                Until_Con+=1
            if x_pos<=0.1:
                Until_Con+=0
            if Until_Con<1:
                Robustness.append(0.055-abs(velocity))
            if Until_Con>=1:
                Robustness.append(1)
        Y.append(min(Robustness))
    return np.array(Y)

bounds=np.array([[-0.6,-0.4],[-0.025, 0.025],[0.040, 0.075],[0.0005, 0.0025]])

Test=BO_CMAES(spec=pred1,boundary=bounds,budget=120,population_size=10)
Test.initialize()
Test.run_BO_CMA()

