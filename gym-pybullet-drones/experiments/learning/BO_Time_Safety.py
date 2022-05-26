#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:06:21 2022

@author: josh
"""

"""
Parts of Code come from:
    MIT License

Copyright (c) 2017 Shromona Ghosh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

and

MIT License

Copyright (c) 2020 Jacopo Panerati

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This scripts runs the best model found by one of the executions of `singleagent.py`

Example
-------
To run the script, type in a terminal:

    $ python BO_Singleagent_February3.py --exp ./results/save-<env>-<algo>-<obs>-<act>-<time_date>


Boundary conditions and Safety functions are for Hover Aviary only
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
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
from gym_pybullet_drones.envs.single_agent_rl.TuneAviary import TuneAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

import shared_constants

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--exp',                           type=str,            help='The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>', metavar='')
    ARGS = parser.parse_args()

    #### Load the model from file ##############################
    algo = ARGS.exp.split("-")[2]

    if os.path.isfile(ARGS.exp+'/success_model.zip'):
        path = ARGS.exp+'/success_model.zip'
    elif os.path.isfile(ARGS.exp+'/best_model.zip'):
        path = ARGS.exp+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", ARGS.exp)
    if algo == 'a2c':
        model = A2C.load(path)
    if algo == 'ppo':
        model = PPO.load(path)
    if algo == 'sac':
        model = SAC.load(path)
    if algo == 'td3':
        model = TD3.load(path)
    if algo == 'ddpg':
        model = DDPG.load(path)

    #### Parameters to recreate the environment ################
    env_name = ARGS.exp.split("-")[1]+"-aviary-v0"
    OBS = ObservationType.KIN if ARGS.exp.split("-")[3] == 'kin' else ObservationType.RGB
    if ARGS.exp.split("-")[4] == 'rpm':
        ACT = ActionType.RPM
    elif ARGS.exp.split("-")[4] == 'dyn':
        ACT = ActionType.DYN
    elif ARGS.exp.split("-")[4] == 'pid':
        ACT = ActionType.PID
    elif ARGS.exp.split("-")[4] == 'vel':
        ACT = ActionType.VEL
    elif ARGS.exp.split("-")[4] == 'tun':
        ACT = ActionType.TUN
    elif ARGS.exp.split("-")[4] == 'one_d_rpm':
        ACT = ActionType.ONE_D_RPM
    elif ARGS.exp.split("-")[4] == 'one_d_dyn':
        ACT = ActionType.ONE_D_DYN
    elif ARGS.exp.split("-")[4] == 'one_d_pid':
        ACT = ActionType.ONE_D_PID

      #### Show, record a video, and log the model's performance #
    test_env = gym.make(env_name,
                        #initial_xyzs=(1,[0,0,.5]),
                        #initial_rpys=(0,[math.pi/4, math.pi/2, -math.pi/4]),
                        gui=False,
                        record=False,
                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=OBS,
                        act=ACT
                        )
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                    num_drones=1
                    )


from gym import spaces

"def compute_traj(**kwargs):"      
def compute_traj(initial_xyz,initial_rpy):   
        #This sets initial xyz position

    test_env.INIT_XYZS=initial_xyz
    test_env.INIT_RPYS=initial_rpy
    
    test_env.reset()
    iter_time=0
    traj_iter=[iter_time]
    r=0
    done=False
    obs=test_env._computeObs()
    traj=[obs[[0,1,2,6,7,8]]]
    for i in range(6*int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)): # Up to 6''
        action, _states = model.predict(obs,
                                        deterministic=True # OPTIONAL 'deterministic=False'
                                        )
        obs, reward, done, additional_data = test_env.step(action)
        r += reward
        traj.append(5*obs[[0,1,2,6,7,8]])
        iter_time+=1
        traj_iter.append(iter_time)
        additional_data['reward']=r
        additional_data['iters'] = iter_time
        test_env.render()
    return traj, {'reward':r, 'iter_time': iter_time}
    
    

def sut(x0, **kwargs):
    return compute_traj([[0,0,x0[0]]],[[x0[1],x0[2],x0[3]]])
 
#--------------------------------------------------------------
#Bayesian Optimization
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
 278535713,
 1762150547,
 788841329,
 2525132954,
 677754898,
 754758634,
 ]

rand_nums2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
rand_nums3=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]
rand_nums4=[101, 113, 134, 156, 194, 202, 213, 111, 129, 200, 91, 81, 82, 71, 78]
rand_nums5=[5085, 8991, 1635, 7805, 7187, 8645, 8888, 5520, 6446, 1714, 7053,
       4131, 7929, 7799, 5766]
rand_nums6=[1461, 8194, 6927, 5075, 4903, 3799, 6268, 8155, 5502, 1187, 7833,
       3916, 7906, 3815, 3587]
rand_nums7=[64846, 28856, 43210, 70661, 14700, 21044, 58191, 17243, 24958, 80194,
       65943, 58561, 24073, 68194, 69265]
rand_nums8=[54239, 69118, 51184, 57468, 57945, 78075, 34142, 78062, 33150,
            64048, 65056, 48293, 35515, 50506, 20161]
rand_nums9=[63951, 36835, 59249, 17176, 32123, 54118, 79720, 64639, 81307, 16913, 
       66005, 22091, 78671, 29591, 74848]
rand_nums10=[347957, 510020, 545416, 613511, 673274, 619204, 630790, 627544,
       127016, 390172, 231790, 414417, 561875, 376595, 632379]
rand_nums_test=[172857]


#Bounds on Environmental Parameters
bounds=[(0.2,1.0)] #bounds on initial z-position
bounds.append((-math.pi/3,math.pi/3)) #bounds on roll angle in radians
bounds.append((-math.pi/3,math.pi/3)) #bounds on pitch angle in radians
bounds.append((-math.pi/3,math.pi/3)) #bounds on yaw angle in radians

#Requirements: Find initial configuration to minimize the requirement:
#1.The drone does not change altitude by more than a specified amount over a time period
#       Does not change altitude by more than 0.2m over 10 itertions
#2.The drone doesn't change velocity by more than 0.3m/s within 10 iterations
#3. Over time ranges of 96 iterations, we have a max height for each time segment
#       We test if the trajectory in that time range breaks its safety spec
smooth_details_r1=[]
ns_details_r3 = []
random_details_r3 = []

smooth_results=[]
ns_results=[]
random_results=[]

smooth_Failure_count=[]
rand_Failure_count=[]
ns_Failure_count=[]

def pred1(traj):
     time_z_safety=[]
     traj=traj[0]
     iter_time1=0
     for i in range(len(traj)-9):
         iter_time1+=1
         time_z=np.array(traj).T[2][iter_time1-1:iter_time1+9]
#        time_z_safety.append(0.2-(np.abs(time_z.max()-time_z.min())))
         time_z_max=time_z.max()
         time_z_min=time_z.min()
         time_z_funct=0.2-(np.abs(time_z.max()-time_z.min()))
         time_z_safety.append(time_z_funct)
     return min(time_z_safety)
 
def pred2(traj):
     time_vel_x_safety=[]
     time_vel_y_safety=[]
     time_vel_z_safety=[]
     
     traj=traj[0]
     iter_time2=0
     for i in range(len(traj)-9):
         iter_time2+=1
         time_velocity_x=np.array(traj).T[3][iter_time2-1:iter_time2+9]
         time_velocity_y=np.array(traj).T[4][iter_time2-1:iter_time2+9]
         time_velocity_z=np.array(traj).T[5][iter_time2-1:iter_time2+9]
#        time_z_safety.append(0.2-(np.abs(time_z.max()-time_z.min())))
         """
         time_vel_x_max=time_velocity_x.max()
         time_vel_x_min=time_velocity_x.min()
         time_vel_y_max=time_velocity_y.max()
         time_vel_y_min=time_velocity_y.min()
         time_vel_z_max=time_velocity_z.max()
         time_vel_z_min=time_velocity_z.min()
         """
         time_vel_x_funct=0.3-(np.abs(time_velocity_x.max()-time_velocity_x.min()))
         time_vel_y_funct=0.3-(np.abs(time_velocity_y.max()-time_velocity_y.min()))
         time_vel_z_funct=0.3-(np.abs(time_velocity_z.max()-time_velocity_z.min()))
         time_vel_x_safety.append(time_vel_x_funct)
         time_vel_y_safety.append(time_vel_y_funct)
         time_vel_z_safety.append(time_vel_z_funct)
     vel_x_safe_min=min(time_vel_x_safety)
     vel_y_safe_min=min(time_vel_y_safety)
     vel_z_safe_min=min(time_vel_z_safety)
         
     return min(vel_x_safe_min,vel_y_safe_min,vel_z_safe_min)
 
 #pred1 & pred2 do not work as intended. Since the policy acts in a similar manner each time,
     #either all trajectories are going to fail or succeed.
 #pred3 to pred6 are an updated approach to time safety spec, we look over a time
     #period (iteration) and see if the max height is broken

def pred3(traj):
    traj=traj[0]
    time_z1_safe=[]
    iter_time1=0
    for i in range(3):
        iter_time1+=1
        time_z1=np.array(traj).T[2][96*iter_time1-96:96*iter_time1]
        if iter_time1==1:    
            time_z1_f=1.08-time_z1.max()
        if iter_time1==2:
            time_z1_f=1.44-time_z1.max()
        if iter_time1>=3:
            time_z1_f=1.65-time_z1.max()
        time_z1_safe.append(time_z1_f)
    return min(time_z1_safe)

#Or this predicate can be broken down into 3 different predicates
def pred4(traj):
    traj=traj[0]
    time_z1_safe=[]
    iter_time1=0
    for i in range(96):
        iter_time1+=1
        time_z1=np.array(traj).T[2][iter_time1]
        time_z1_f=1.08-time_z1
        time_z1_safe.append(time_z1_f)
    return min(time_z1_safe)

def pred5(traj):
    traj=traj[0]
    time_z2_safe=[]
    iter_time2=0
    for i in range(96):
        iter_time2+=1
        time_z2=np.array(traj).T[2][iter_time2]
        time_z2_f=1.08-time_z2
        time_z2_safe.append(time_z2_f)
    return min(time_z2_safe)

def pred6(traj):
    traj=traj[0]
    time_z3_safe=[]
    iter_time3=0
    for i in range(96):
        iter_time3+=1
        time_z3=np.array(traj).T[2][iter_time3]
        time_z3_f=1.08-time_z3
        time_z3_safe.append(time_z3_f)
    return min(time_z3_safe)

#This predicate defines the function: we want to eventually be in the area of 1.2>=z>=1.4
    #for at least 50 iterations (50/240=00.208s)
def pred7(traj):
    traj=traj[0]
    goal_area_high=1.4
    goal_area_low=1.2
    goal_time=60
    time_in_goal_area=[]
    for i in range(len(traj)):
        if np.array(traj).T[2][i] >=goal_area_low and np.array(traj).T[2][i]<=goal_area_high:
            time_in_goal_area.append(1)
        else:
            time_in_goal_area.append(0)
    return (-goal_time+np.array(time_in_goal_area).sum())/100

#We always want to follow the speed limits: velocity<0.25m/s when z<=0.5m and
    #vel<0.4m/s when z>0.5
def pred8(traj):
    traj=traj[0]
    velocity=[]
    for i in range(len(traj)):
        if np.array(traj).T[2][i]<=0.5:
            vel_limit=0.25
            max_vel=max(np.array(traj).T[3][i],np.array(traj).T[4][i],np.array(traj).T[5][i])
            velocity.append(vel_limit-max_vel)
        else:
            vel_limit=0.45
            max_vel=max(np.array(traj).T[3][i],np.array(traj).T[4][i],np.array(traj).T[5][i])
            velocity.append(vel_limit-max_vel)
    return np.array(velocity).min()

#node0=pred_node(f=pred1)
#node1=pred_node(f=pred2)
#node2=pred_node(f=pred3)
#node3=min_node(children=[node1,node2])
#node4=min_node(children=[node3,node0])
    
    

for r in rand_nums:
    
    np.random.seed(r)
    node0=pred_node(f=pred1)
    node1=pred_node(f=pred2)
    node2=min_node(children=[node0,node1])
    
    node3=pred_node(f=pred3)
    
    node4=pred_node(f=pred4)
    node5=pred_node(f=pred5)
    node6=pred_node(f=pred6)
    node7=min_node(children=[node4,node5])
    node8=min_node(children=[node6,node7])
    
    node9=pred_node(f=pred7)
    node10=pred_node(f=pred8)
    node11=min_node(children=[node9,node10])
    
    TM_smooth = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree=node9, init_samples=20, with_smooth=True,
                     with_random=False, with_ns=False,
                     optimize_restarts=1, exp_weight=2)
    TM_smooth.initialize()
    
    TM_smooth.run_BO(150)
    
    smooth_Failure_count.append(TM_smooth.smooth_count)
    
    smooth_vals = np.array(TM_smooth.f_acqu.find_GP_func())
    smooth_details_r1.append([TM_smooth.smooth_count,
                              TM_smooth.smooth_min_x,
                              TM_smooth.smooth_min_val, TM_smooth.smooth_min_loc])
    
    #smooth_results.append(TM_smooth.smooth_count, )
    #del TM_smooth
    #print(r, smooth_details_r1[-1])
    
#for r in rand_nums:
    
    np.random.seed(r)
    node0_rand=pred_node(f=pred1)
    node1_rand=pred_node(f=pred2)
    node2_rand=min_node(children=[node0_rand,node1_rand])
    
    node3_rand=pred_node(f=pred3)
    
    node4_rand=pred_node(f=pred4)
    node5_rand=pred_node(f=pred5)
    node6_rand=pred_node(f=pred6)
    node7_rand=min_node(children=[node4_rand,node5_rand])
    node8_rand=min_node(children=[node6_rand,node7_rand])
    
    node9_rand=pred_node(f=pred7)
    node10_rand=pred_node(f=pred8)
    node11_rand=min_node(children=[node9_rand,node10_rand])
    
    TM_rand = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree=node9_rand, init_samples=20, with_smooth=False,
                     with_random=True, with_ns=False,
                     optimize_restarts=1, exp_weight=2)
    TM_rand.initialize()
    
    TM_rand.run_BO(150)
    
    rand_Failure_count.append(TM_rand.rand_count)
    
    rand_vals = np.array(TM_rand.random_Y)
    random_details_r3.append([TM_rand.rand_count,
                          TM_rand.rand_min_x,
                          TM_rand.rand_min_val, TM_rand.rand_min_loc])
    #print(r, random_details_r3[-1])
    #del TM_rand
#for r in rand_nums:
    
    np.random.seed(r)
    node0_ns=pred_node(f=pred1)
    node1_ns=pred_node(f=pred2)
    node3_ns=min_node(children=[node0_ns,node1_ns])
    
    node3_ns=pred_node(f=pred3)
    
    node4_ns=pred_node(f=pred4)
    node5_ns=pred_node(f=pred5)
    node6_ns=pred_node(f=pred6)
    node7_ns=min_node(children=[node4_ns,node5_ns])
    node8_ns=min_node(children=[node6_ns,node7_ns])
    
    node9_ns=pred_node(f=pred7)
    node10_ns=pred_node(f=pred8)
    node11_ns=min_node(children=[node9_ns,node10_ns])
    
    TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree=node9_ns, init_samples=20, with_smooth=False,
                     with_random=False, with_ns=True,
                     optimize_restarts=1, exp_weight=2)
    TM_ns.initialize()
    
    TM_ns.run_BO(150)
    
    ns_Failure_count.append(TM_ns.ns_count)
    
    ns_smooth_vals = np.array(TM_ns.ns_GP.Y)
    ns_details_r3.append([TM_ns.ns_count,
                        TM_ns.ns_min_x,
                        TM_ns.ns_min_val, TM_ns.ns_min_loc])
    
    #print(r,ns_details_r3[-1])
    #del TM_ns
#print(smooth_Failure_count,rand_Failure_count,ns_Failure_count)
#Making bar plots with error bars
Random_mean=np.mean(rand_Failure_count)

Smooth_mean=np.mean(smooth_Failure_count)

NS_mean=np.mean(ns_Failure_count)

Smooth_std=np.std(smooth_Failure_count)

NS_std=np.std(ns_Failure_count)

Random_std=np.std(rand_Failure_count)

Method=['Random Sampling', 'Smooth GP', 'Non-Smooth GP']
x_pos=np.arange(len(Method))

Means_Failure_Modes=[Random_mean,Smooth_mean,NS_mean]

Error=[Random_std,Smooth_std,NS_std]

plt.bar(x_pos, Means_Failure_Modes, yerr=Error, align='center', alpha=0.5, ecolor='black', capsize=10),\
    plt.ylabel('Failure Modes Found'),plt.xticks(x_pos,Method),\
    plt.title('Failure Modes Found of Three Methods Based Off 15 Runs'),\
    plt.grid(True,axis='y'), plt.show()

##These are arrays for each specific position value through trajectory
smooth_safest_params=TM_smooth.smooth_X[smooth_vals.argmax()]
traj_smooth_safe=[]
traj_smooth_safe.append(TM_smooth.system_under_test(smooth_safest_params))
x_vals_smooth_safe=np.array(traj_smooth_safe[0][0]).T[0]
y_vals_smooth_safe=np.array(traj_smooth_safe[0][0]).T[1]
z_vals_smooth_safe=np.array(traj_smooth_safe[0][0]).T[2]
xvel_vals_smooth_safe=np.array(traj_smooth_safe[0][0]).T[3]
yvel_vals_smooth_safe=np.array(traj_smooth_safe[0][0]).T[4]
zvel_vals_smooth_safe=np.array(traj_smooth_safe[0][0]).T[5]
velocity_smooth_safe=[]
for i in range(len(xvel_vals_smooth_safe)):
    velocity_smooth_safe.append(max(np.abs(xvel_vals_smooth_safe[i]),np.abs(yvel_vals_smooth_safe[i]),np.abs(zvel_vals_smooth_safe[i])))

import operator
enumerate_obj=enumerate(smooth_vals)
sorted_smooth_vals=sorted(enumerate_obj,key=operator.itemgetter(1))
sorted_indices_smooth_vals=[index for index, element in sorted_smooth_vals]
print(sorted_indices_smooth_vals)
Four_DangerousEnv_Param=TM_smooth.smooth_X[sorted_indices_smooth_vals[0:4]]

traj_smooth_traj1=[]
smooth_traj1_params=Four_DangerousEnv_Param[0]
traj_smooth_traj1.append(TM_smooth.system_under_test(smooth_traj1_params))
x_vals_smooth_traj1=np.array(traj_smooth_traj1[0][0]).T[0]
y_vals_smooth_traj1=np.array(traj_smooth_traj1[0][0]).T[1]
z_vals_smooth_traj1=np.array(traj_smooth_traj1[0][0]).T[2]
xvel_vals_smooth_traj1=np.array(traj_smooth_traj1[0][0]).T[3]
yvel_vals_smooth_traj1=np.array(traj_smooth_traj1[0][0]).T[4]
zvel_vals_smooth_traj1=np.array(traj_smooth_traj1[0][0]).T[5]
velocity_smooth_traj1=[]
for i in range(len(xvel_vals_smooth_traj1)):
    velocity_smooth_traj1.append(max(np.abs(xvel_vals_smooth_traj1[i]),np.abs(yvel_vals_smooth_traj1[i]),np.abs(zvel_vals_smooth_traj1[i])))

"""
traj_smooth_traj2=[]
smooth_traj2_params=Four_DangerousEnv_Param[1]
traj_smooth_traj2.append(TM_smooth.system_under_test(smooth_traj2_params))
x_vals_smooth_traj2=np.array(traj_smooth_traj2[0][0]).T[0]
y_vals_smooth_traj2=np.array(traj_smooth_traj2[0][0]).T[1]
z_vals_smooth_traj2=np.array(traj_smooth_traj2[0][0]).T[2]

traj_smooth_traj3=[]
smooth_traj3_params=Four_DangerousEnv_Param[2]
traj_smooth_traj3.append(TM_smooth.system_under_test(smooth_traj3_params))
x_vals_smooth_traj3=np.array(traj_smooth_traj3[0][0]).T[0]
y_vals_smooth_traj3=np.array(traj_smooth_traj3[0][0]).T[1]
z_vals_smooth_traj3=np.array(traj_smooth_traj3[0][0]).T[2]

traj_smooth_traj4=[]
smooth_traj4_params=Four_DangerousEnv_Param[3]
traj_smooth_traj4.append(TM_smooth.system_under_test(smooth_traj4_params))
x_vals_smooth_traj4=np.array(traj_smooth_traj4[0][0]).T[0]
y_vals_smooth_traj4=np.array(traj_smooth_traj4[0][0]).T[1]
z_vals_smooth_traj4=np.array(traj_smooth_traj4[0][0]).T[2]
"""
#This code is to graph the trajcetory in respect to safety spec pred7
time_range=np.array(list(range(0,289)))
fig=plt.figure()
ax=fig.add_subplot()
ax.set_xlim(left=0,right=289)
ax.scatter(time_range,z_vals_smooth_safe,color='green')
ax.scatter(time_range,z_vals_smooth_traj1,color='Black')


#code to plot safety specs
ax.axhline(y=1.4,color='r')
ax.axhline(y=1.2,color='r')



ax.legend(['Safe Traj','Most Dangerous'])
ax.set_title('Safest and Most Dangerous Trajectories: Goal Area')
ax.set_xlabel('Time [iteration]')
ax.set_ylabel('Z-Position')

#pred8
fig1=plt.figure()
bx=fig1.add_subplot()
bx.scatter(z_vals_smooth_safe,velocity_smooth_safe,color='green')
bx.scatter(z_vals_smooth_traj1,velocity_smooth_traj1,color='Black')
bx.axvline(x=0.5,color='blue')
bx.axhline(y=0.25,xmin=0,xmax=0.23,color='r')
bx.axhline(y=0.45,xmin=0.23,color='r')
bx.legend(['Safe Traj','Most Dangerous'])
bx.set_title('Safest and Most Dangerous Trajectories: Velocity')
bx.set_xlabel('Z Position')
bx.set_ylabel('Velocity')

