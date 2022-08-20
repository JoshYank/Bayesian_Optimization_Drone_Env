#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:41:38 2022

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
#import torch

import logging
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

env = gym.make('CartPole-v1')
model = PPO1.load("ppo1_cartpole")


def compute_traj(max_steps,ead=False, **kwargs):
    env.reset()
    if 'init_state' in kwargs:
        ob = kwargs['init_state']
        env.env.state = ob
    if 'masspole' in kwargs:
        env.env.masspole = kwargs['masspole']
        env.env.total_mass = env.env.masspole + env.env.masscart
        env.env.polemass_length = env.env.masspole * env.env.length
    if 'length' in kwargs:
        env.env.length = kwargs['length']
        env.env.polemass_length = env.env.masspole * env.env.length
    if 'force_mag' in kwargs:
        env.env.force_mag = kwargs['force_mag']
    traj = [ob]
    reward = 0
    iters= 0
    for _ in range(max_steps):
        iters+=1
        action, _states = model.predict(ob)
        ob, r, done, _ = env.step(action)
        reward += r
        traj.append(ob)
        if ead and done:
            break
    additional_data = {'reward':reward, 'mass':env.env.total_mass, 'iters':iters}
    return traj, additional_data

def sut(max_steps,x0, ead=False):
    return compute_traj(max_steps,init_state=x0[0:4],ead=ead)#, masspole=x0[4],
                        #length=x0[5], force_mag=x0[6], ead=ead)

from scipy.stats import norm
def cost_func(X):
    mass_rv = norm(0.1, 0.05)
    length_rv = norm(0.5, 0.05)
    force_rv = norm(10,2)
    mass_pdf = mass_rv.pdf(X.T[4])/mass_rv.pdf(0.1)
    length_pdf = length_rv.pdf(X.T[5])/length_rv.pdf(0.5)
    force_pdf = force_rv.pdf(X.T[6])/force_rv.pdf(10)
    mass_pdf.resize(len(mass_pdf), 1)
    length_pdf.resize(len(length_pdf), 1)
    force_pdf.resize(len(force_pdf), 1)
    return mass_pdf*length_pdf*force_pdf

# ------------------------------------------------------------------------------
from adversarial_testing import pred_node, max_node, min_node, test_module
from adversarial_testing.utils import sample_from
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
#bounds = [(-0.175, 0.175)] * 4 # Bounds on the state
bounds=[(-0.3,0.3)] #x pos
bounds.append((-0.8,0.8)) #velocity
bounds.append((-0.26,0.26)) #angle
bounds.append((-0.05,0.05)) #ang vel
#bounds.append((0.05, 0.15)) # Bounds on the mass of the pole
#bounds.append((0.4, 0.6)) # Bounds on the length of the pole
#bounds.append((8.00, 12.00)) # Bounds on the force magnitude

smooth_details=[]
ns_details= []
random_details= []

smooth_results=[]
ns_results=[]
random_results=[]

smooth_Failure_count=[]
rand_Failure_count=[]
ns_Failure_count=[]

# maneuver is deemed unsafe if until some time the pole angle is larger than 20
#while the cart’s horizontal position is more than 0.3.
def pred1(traj):
    traj1 = traj[0]
    Robustness=[]
    for i in range (len(traj1)):
        #x_pos=np.array(traj1[0]).T[i]
        #angle=np.array(traj[0]).T[i]
        x_pos=traj1[i][0]
        angle=traj1[i][2]
        if x_pos <= -0.3 or x_pos>=0.3:
            Robustness.append(0.349066-abs(angle))
        if x_pos>-0.3 and x_pos<0.3:
            Robustness.append(1/abs(angle))
    return min(Robustness)

def pred2(traj):
    traj1 = traj[0]
    Robustness=[]
    for i in range (len(traj1)):
        #x_pos=np.array(traj1[0]).T[i]
        #angle=np.array(traj[0]).T[i]
        x_vel=traj1[i][1]
        Robustness.append(2-abs(x_vel))
    return min(Robustness)

for r in rand_nums10:
    """
    np.random.seed(r)
    node0=pred_node(f=pred1)
    node1=pred_node(f=pred2)
    node2=min_node(children=[node0,node1])

    TM_smooth = test_module(bounds=bounds, sut=lambda x0: sut(200,x0),
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

    TM_rand = test_module(bounds=bounds, sut=lambda x0: sut(200,x0),
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

    TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(200,x0),
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

Method=['Random Sampling', 'BO']
x_pos=np.arange(len(Method))

Means_Failure_Modes=[Random_mean,NS_mean]

Error=[Random_std,NS_std]

plt.bar(x_pos, Means_Failure_Modes, yerr=Error, align='center', alpha=0.5, ecolor='black', capsize=10),\
    plt.ylabel('Failure Modes Found'),plt.xticks(x_pos,Method),\
    plt.title('Failure Modes Found: BO v. Random Sampling'),\
    plt.grid(True,axis='y'), plt.show()

    
"""   
Test_num=['1','2','3','4','5','6','7','8','9','10']
X_axis=np.arange(len(Test_num))

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
pole_ang_safe=[]
pole_vel_safe=[]
traj_smooth_safe.append(TM_smooth.system_under_test(smooth_safest_params))
for i in range(len(np.array(traj_smooth_safe[0][0]))):
    cart_pos_safe.append(np.array(traj_smooth_safe[0][0]).T[0][i])
    cart_vel_safe.append(np.array(traj_smooth_safe[0][0]).T[1][i])
    pole_ang_safe.append(np.array(traj_smooth_safe[0][0]).T[2][i])
    pole_vel_safe.append(np.array(traj_smooth_safe[0][0]).T[3][i])
    
import operator
enumerate_obj=enumerate(smooth_vals)
sorted_smooth_vals=sorted(enumerate_obj,key=operator.itemgetter(1))
sorted_indices_smooth_vals=[index for index, element in sorted_smooth_vals]
print(sorted_indices_smooth_vals)
Four_DangerousEnv_Param=TM_smooth.smooth_X[sorted_indices_smooth_vals[0:4]]

traj_smooth_traj1=[]
cart_pos_traj1=[]
cart_vel_traj1=[]
pole_ang_traj1=[]
pole_vel_traj1=[]
smooth_traj1_params=Four_DangerousEnv_Param[0]
traj_smooth_traj1.append(TM_smooth.system_under_test(smooth_traj1_params))
for i in range(len(np.array(traj_smooth_traj1[0][0]))):
    cart_pos_traj1.append(np.array(traj_smooth_traj1[0][0]).T[0][i])
    cart_vel_traj1.append(np.array(traj_smooth_traj1[0][0]).T[1][i])
    pole_ang_traj1.append(np.array(traj_smooth_traj1[0][0]).T[2][i])
    pole_vel_traj1.append(np.array(traj_smooth_traj1[0][0]).T[3][i])
    
#To Save data, change 1B and date at end
DF=pd.DataFrame(rand_Failure_count)
DF.to_csv("CP_Experiment_2_Random_Failure_Count_07-19.csv")

DF=pd.DataFrame(ns_Failure_count)
DF.to_csv("CP_Experiment_2_Time_NS_Failure_Count_07-19.csv")

DF=pd.DataFrame(smooth_Failure_count)
DF.to_csv("Experiment_6_Time_Smooth_Failure_Count_07-18.csv")

#To Save Params and Trajs for best smooth
DF=pd.DataFrame(smooth_safest_params)
DF.to_csv("Experiment_6_STL_Smooth_Safe_Param_07-19.csv")
DF=pd.DataFrame(cart_pos_safe)
DF.to_csv("Experiment_6_STL_Smooth_Safe_Cart_Pos_07-19.csv")
DF=pd.DataFrame(cart_vel_safe)
DF.to_csv("Experiment_6_STL_Smooth_Safe_Cart_Vel_07-19.csv")
DF=pd.DataFrame(pole_ang_safe)
DF.to_csv("Experiment_6_STL_Smooth_Safe_Pole_Ang_07-19.csv")
DF=pd.DataFrame(pole_vel_safe)
DF.to_csv("Experiment_6_STL_Smooth_Safe_Pole_Vel_07-19.csv")

#To Save Params and Trajs for Worst smooth
DF=pd.DataFrame(smooth_traj1_params)
DF.to_csv("Experiment_6_STL_Smooth_traj1_Param_07-19.csv")
DF=pd.DataFrame(cart_pos_traj1)
DF.to_csv("Experiment_6_STL_Smooth_traj1_Cart_Pos_07-19.csv")
DF=pd.DataFrame(cart_vel_traj1)
DF.to_csv("Experiment_6_STL_Smooth_traj1_Cart_Vel_07-19.csv")
DF=pd.DataFrame(pole_ang_traj1)
DF.to_csv("Experiment_6_STL_Smooth_traj1_Pole_Ang_07-19.csv")
DF=pd.DataFrame(pole_vel_traj1)
DF.to_csv("Experiment_6_STL_Smooth_traj1_Pole_Vel_07-19.csv")
"""
