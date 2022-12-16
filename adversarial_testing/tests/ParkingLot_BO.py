#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:17:13 2022

@author: josh

Parking Lot Env
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


import logging
import numpy as np

import gym
import highway_env
from stable_baselines3 import HerReplayBuffer, SAC
from sb3_contrib import TQC

import gym
from matplotlib import pyplot as plt

from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from gym.wrappers import RecordVideo
from pathlib import Path
import base64
from tqdm.notebook import trange

env = gym.make("parking-v0")
seed = 8902077161928034768
env.seed(seed)
model=TQC.load("./her_parking",env=env)

def compute_traj(max_steps,**kwargs):
    env.reset()
    if 'car_position' in kwargs:
        c_pos = kwargs['car_position']
        env.env.controlled_vehicles[0].position=c_pos
    if 'car_heading' in kwargs:
        c_head=kwargs['car_heading']
        env.env.controlled_vehicles[0].heading=c_head
    if 'car_speed' in kwargs:
        c_speed = kwargs['car_speed']
        env.env.controlled_vehicles[0].speed=c_speed
    if 'goal_position' in kwargs:
        g_pos = kwargs['goal_position']
        env.env.goal.position=g_pos
    obs=env.env.observation_type_parking.observe()
    traj = [obs]

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        #env.render()
        traj.append(obs)
    return traj

def sut(max_steps,x0):
    return compute_traj(max_steps,car_position=x0[0:2],car_heading=x0[2],
                        car_speed=x0[3],goal_position=x0[4:6])
        #max_steps,init_state=x0[0:4],ead=ead)#, masspole=x0[4],
                        #length=x0[5], force_mag=x0[6], ead=ead)
#------------------------------------------------------------------------------
#BO
from adversarial_testing import pred_node, max_node, min_node, test_module
from adversarial_testing.utils import sample_from
from adversarial_testing import optimizers

rand_nums = [2440271967,
 3816968049,
 3160626546,
 636413671,
 3105544786,
 646012482,
 3406852803,
 1769141240,
 109713304,
 3433822084]

rand_nums2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rand_nums3=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
rand_nums4=[101, 113, 134, 156, 194, 202, 213, 111, 129, 200]
rand_nums5=[5085, 8991, 1635, 7805, 7187, 8645, 8888, 5520, 6446, 1714]
rand_nums6=[1461, 8194, 6927, 5075, 4903, 3799, 6268, 8155, 5502, 1187]
rand_nums7=[64846, 28856, 43210, 70661, 14700, 21044, 58191, 17243, 24958, 80194]
rand_nums8=[54239, 69118, 51184, 57468, 57945, 78075, 34142, 78062, 33150,
            64048]
rand_nums9=[63951, 36835, 59249, 17176, 32123, 54118, 79720, 64639, 81307, 16913]
rand_nums10=[347957, 510020, 545416, 613511, 673274, 619204, 630790, 627544,
       127016, 390172]
rand_nums11=[61,18,2,33,31,49,81,17,11,131]
rand_nums12=[65,13,19,38,32,99,84,22,41,143]
rand_nums_test=[1732]

#bounds of uncertainty
#bounds=[(-50,50)] #x-pos car
bounds=[(-50,50)] #x-pos car
bounds.append((-30,30)) #y-pos car
bounds.append((-1.57,1.57)) #car heading
bounds.append((-5,5)) #car speed (m/s)
#bounds.append((-50,50)) #x-pos goal
bounds.append((-50,50)) #x-pos goal
bounds.append((-30,30)) #y-pos goal

#predicates
def pred1(traj):
    traj1=traj
    Robustness=[]
    X_Car=[]
    Y_Car=[]
    X_Goal=traj1[0]['desired_goal'][0]
    Y_Goal=traj1[0]['desired_goal'][1]
    for i in range (len(traj1)):
        x_pos=traj1[i]['observation'][0]
        y_pos=traj1[i]['observation'][1]
        X_Car.append(x_pos)
        Y_Car.append(y_pos)
        rob_x=.02-abs(X_Goal-x_pos)
        rob_y=.02-abs(Y_Goal-y_pos)
        Robustness.append(max(rob_x,rob_y))
    return max(Robustness)

"""
Alternative for pred 1
def pred1(traj):
    traj1=traj
    Robustness=[]
    X_Car=[]
    Y_Car=[]
    X_Goal=traj1[0]['desired_goal'][0]
    Y_Goal=traj1[0]['desired_goal'][1]
    for i in range (len(traj1)):
        x_pos=traj1[i]['observation'][0]
        y_pos=traj1[i]['observation'][1]
        X_Car.append(x_pos)
        Y_Car.append(y_pos)
        rob_x=.02-abs(X_Goal-x_pos)
        rob_y=.02-abs(Y_Goal-y_pos)
        Robustness.append(np.sqrt(8)-np.sqrt((X_Goal-x_pos)**2+(Y_Goal-y_pos)**2))
    return max(Robustness)
"""
def pred2(traj):
    traj1=traj
    Robustness=[]
    Tot_Ang=[]
    old_theta=np.arcsin(traj1[0]['observation'][5])
    tot_theta=0
    #Max_Ang=4.712
    #Max_Ang=12.6
    Max_Ang=9.60
    for i in range (len(traj1)):
        theta=np.arcsin(traj1[i]['observation'][5])
        d_theta=abs(old_theta-theta)
        tot_theta=tot_theta+d_theta
        old_theta=theta
        Tot_Ang.append(tot_theta)
        Robustness.append(Max_Ang-tot_theta)
    return min(Robustness)

#BO set up
#C=[rand_nums,rand_nums2,rand_nums3,rand_nums4,rand_nums5,rand_nums6,rand_nums7,rand_nums8,rand_nums9,rand_nums10]
C=[rand_nums,rand_nums2,rand_nums3,rand_nums4,rand_nums5]
#C=[rand_nums_test]
NS_Details=[]
NS_Param=[]
NS_Robust=[]
NS_worst_sim=[]
Rand_Details=[]
Rand_Param=[]
Rand_Rob=[]
Rand_worst_sim=[]

Start_time=[]
End_time=[]

for a in range(len(C)):
    
    start_t=time.time()
    
    smooth_details_r1=[]
    ns_details = []
    random_details = []
    
    smooth_results=[]
    ns_results=[]
    random_results=[]
    
    smooth_Failure_count=[]
    rand_Failure_count=[]
    ns_Failure_count=[]
    
    ns_param=[]
    ns_worst_sim=[]
    ns_robust=[]
    random_param=[]
    random_robust=[]
    rand_worst_sim=[]
    
    for r in C[a]:
        #for r in rand_nums:
        
        np.random.seed(r)
        node0_rand=pred_node(f=pred1)
        node1_rand=pred_node(f=pred2)
        node2_rand=min_node(children=[node0_rand,node1_rand])
    
        TM_rand = test_module(bounds=bounds, sut=lambda x0: sut(30,x0),
                         f_tree=node2_rand, init_samples=20, with_smooth=False,
                         with_random=True, with_ns=False,
                         optimize_restarts=1, exp_weight=2)
        TM_rand.initialize()
        
        TM_rand.run_BO(500)
        
        rand_Failure_count.append(TM_rand.rand_count)
        rand_worst_sim.append(TM_rand.rand_min_loc)
        
        rand_vals = np.array(TM_rand.random_Y)
        random_details.append([TM_rand.rand_count,
                              TM_rand.rand_min_x,
                              TM_rand.rand_min_val, TM_rand.rand_min_loc])
        random_param.append(TM_rand.rand_min_x)
        random_robust.append(TM_rand.rand_min_val)
        
        #print(r, random_details_r3[-1])
        #del TM_rand
    #for r in rand_nums:
        
        np.random.seed(r)
        node0_ns=pred_node(f=pred1)
        node1_ns=pred_node(f=pred2)
        node2_ns=min_node(children=[node0_ns,node1_ns])
    
        TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(30,x0),
                         f_tree=node2_ns, init_samples=20, with_smooth=False,
                         with_random=False, with_ns=True, #optimizer=optimizers.direct_opt(bounds),
                         optimize_restarts=1, exp_weight=2)
        TM_ns.initialize()
        
        TM_ns.run_BO(500)
        
        ns_Failure_count.append(TM_ns.ns_count)
        ns_worst_sim.append(TM_ns.ns_min_loc)
        
        ns_smooth_vals = np.array(TM_ns.ns_GP.Y)
        ns_details.append([TM_ns.ns_count,
                            TM_ns.ns_min_x,
                            TM_ns.ns_min_val, TM_ns.ns_min_loc])
        ns_param.append(TM_ns.ns_min_x)
        ns_robust.append(TM_ns.ns_min_val)
    end_t=time.time()
    Start_time.append(start_t)
    End_time.append(end_t)
    NS_Details.append(ns_Failure_count)
    NS_Param.append(ns_param)
    NS_Robust.append(ns_robust)
    NS_worst_sim.append(ns_worst_sim)
    Rand_Details.append(rand_Failure_count)
    Rand_Param.append(random_param)
    Rand_Rob.append(random_robust)
    Rand_worst_sim.append(rand_worst_sim)
    print('###################################################################')
    
#---------------------------------------------------------------------------------------
#plot results
Random_mean=np.mean(rand_Failure_count)

#Smooth_mean=np.mean(smooth_Failure_count)

NS_mean=np.mean(ns_Failure_count)

#Smooth_std=np.std(smooth_Failure_count)

NS_std=np.std(ns_Failure_count)

Random_std=np.std(rand_Failure_count)

Method=['Random Sampling', 'BO']
x_pos=np.arange(len(Method))

Means_Failure_Modes=[Random_mean,NS_mean]

Error=[Random_std,NS_std]

plt.bar(x_pos, Means_Failure_Modes, yerr=Error, align='center', alpha=0.5, ecolor='black', capsize=10),\
    plt.ylabel('Failure Modes Found'),plt.xticks(x_pos,Method),\
    plt.title('Failure Modes Found: BO v. Random Sampling'),\
    plt.grid(True,axis='y'), plt.show()
"""
example code to run sim
traj=[]
obs=env.reset()
traj.append(obs)
for episode in trange(1, desc="Test episodes"):
    obs, done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        traj.append(obs)

#to check observations
traj[i]['observation']
traj1=[]
for i in range(len(traj)):
    traj1.append(traj[i]['observation'])
"""    
    
