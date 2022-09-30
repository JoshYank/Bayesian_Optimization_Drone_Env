#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 08:18:48 2022

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


env = gym.make('MountainCarContinuous-v0')
model = DDPG.load("ddpg_mountain")

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
        action, _states = model.predict(ob)
        ob, rewards, dones, info = env.step(action)
        
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
rand_nums13=[375686, 31957, 26589, 180738, 281078, 88509, 499107, 466051, 478311, 69059]
rand_nums14=[55831, 297066, 108997, 401197, 377974, 111910, 185704, 116438, 414932, 310477]
rand_nums15=[320,919,8132,30911,127,471,459,5228,99,12222]
rand_nums16=[1304673443,3857496775,36,234,1819,194,101,200,6268,6927]
rand_nums_test=[172857]

# Requirement 1: Find the initial configuration that minimizes the reward
# We need only one node for the reward. The reward is a smooth function
# given that the closed loop system is deterministic
bounds = [(-0.6, -0.4)] # Bounds on the position
bounds.append((-0.025, 0.025)) # Bounds on the velocity
#bounds.append((0.4, 0.6)) # Bounds on the goal position
bounds.append((0.040, 0.075)) # Bounds on the max speed
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
            Robustness.append(0.0735-abs(velocity))
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
        if x_pos>0.1:
            Until_Con+=1
        if x_pos<=0.1:
            Until_Con+=0
        if Until_Con<1:
            Robustness.append(0.055-abs(velocity))
        if Until_Con>=1:
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
"""
Method=['Random Sampling',  'BO']
x_pos=np.arange(len(Method))

Means_Failure_Modes=[Random_mean,NS_mean]

Error=[Random_std,NS_std]

plt.bar(x_pos, Means_Failure_Modes, yerr=Error, align='center', alpha=0.5, ecolor='black', capsize=10),\
    plt.ylabel('Failure Modes Found'),plt.xticks(x_pos,Method),\
    plt.title('Failure Modes Found: BO v. Random Sampling'),\
    plt.grid(True,axis='y'), plt.show()

"""
Method=['BO']
x_pos=np.arange(len(Method))
Means_Failure_Modes=[Random_mean]
Error=[Random_std]

plt.bar(x_pos, Means_Failure_Modes, yerr=Error, align='center', alpha=0.5, ecolor='black', capsize=10),\
    plt.ylabel('Failure Modes Found'),plt.xticks(x_pos,Method),\
    plt.title('Failure Modes Found: BO'),\
       plt.grid(True,axis='y'), plt.show()
#How to Plot failure modes per test

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
smooth_safest_params=TM_ns.ns_X[smooth_vals.argmax()]
traj_smooth_safe=[]
cart_pos_safe=[]
cart_vel_safe=[]

traj_smooth_safe.append(TM_ns.system_under_test(smooth_safest_params))
for i in range(len(np.array(traj_smooth_safe[0][0]))):
    cart_pos_safe.append(np.array(traj_smooth_safe[0][0]).T[0][i])
    cart_vel_safe.append(np.array(traj_smooth_safe[0][0]).T[1][i])

    
import operator
enumerate_obj=enumerate(smooth_vals)
sorted_smooth_vals=sorted(enumerate_obj,key=operator.itemgetter(1))
sorted_indices_smooth_vals=[index for index, element in sorted_smooth_vals]
print(sorted_indices_smooth_vals)
Four_DangerousEnv_Param=TM_ns.smooth_X[sorted_indices_smooth_vals[0:4]]

traj_smooth_traj1=[]
cart_pos_traj1=[]
cart_vel_traj1=[]

smooth_traj1_params=Four_DangerousEnv_Param[0]
traj_smooth_traj1.append(TM_ns.system_under_test(smooth_traj1_params))
for i in range(len(np.array(traj_smooth_traj1[0][0]))):
    cart_pos_traj1.append(np.array(traj_smooth_traj1[0][0]).T[0][i])
    cart_vel_traj1.append(np.array(traj_smooth_traj1[0][0]).T[1][i])

    
#To Save data, change 1B and date at end
DF=pd.DataFrame(ns_Failure_count)
DF.to_csv("MC_Experiment_1_BO_Failure_Count_08-20.csv")

DF=pd.DataFrame(rand_Failure_count)
DF.to_csv("MC_Experiment_1_Random_Failure_Count_08-20.csv")

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