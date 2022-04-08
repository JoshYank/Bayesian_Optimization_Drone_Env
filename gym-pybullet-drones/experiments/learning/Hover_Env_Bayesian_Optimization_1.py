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
    r=0
    done=False
    obs=test_env._computeObs()
    traj=[obs[0:3]]
    for i in range(6*int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)): # Up to 6''
        action, _states = model.predict(obs,
                                        deterministic=True # OPTIONAL 'deterministic=False'
                                        )
        obs, reward, done, additional_data = test_env.step(action)
        r += reward
        traj.append(5*obs[0:3])
        iter_time+=1
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

#Bounds on Environmental Parameters
bounds=[(0.2,1.0)] #bounds on initial z-position
bounds.append((-math.pi/3,math.pi/3)) #bounds on roll angle in radians
bounds.append((-math.pi/3,math.pi/3)) #bounds on pitch angle in radians
bounds.append((-math.pi/3,math.pi/3)) #bounds on yaw angle in radians

#Requirements: Find initial configuration to minimize the requirement:
#1.that the drone doesn't fly over a certain altitude
#2. the drone stays close to original xy position
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
    #traj_array = np.array(traj)#Here we want the z position
    #z_s=traj_array[:,2]
    traj=traj[0]
    z_s = np.array(traj).T[2]
    return min(1.8-z_s)



def pred2(traj):
    #traj_array = np.array(traj) #Here we want the x position
    #x_s = traj_array[:,0]
    traj=traj[0]
    x_s = np.array(traj).T[0]
    return min(2-np.abs(x_s))

def pred3(traj):
    #traj_array = np.array(traj) #Here we want the y position
    #y_s = traj_array[:,1]
    traj=traj[0]
    y_s = np.array(traj).T[1]
    return min(2-np.abs(y_s))

#node0=pred_node(f=pred1)
#node1=pred_node(f=pred2)
#node2=pred_node(f=pred3)
#node3=min_node(children=[node1,node2])
#node4=min_node(children=[node3,node0])
    
    

for r in rand_nums:
    
    np.random.seed(r)
    node0=pred_node(f=pred1)
    node1=pred_node(f=pred2)
    node2=pred_node(f=pred3)
    node3=min_node(children=[node1,node2])
    node4=min_node(children=[node3,node0])
    TM_smooth = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree=node4, init_samples=20, with_smooth=True,
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
    node2_rand=pred_node(f=pred3)
    node3_rand=min_node(children=[node1_rand,node2_rand])
    node4_rand=min_node(children=[node3_rand,node0_rand])
    TM_rand = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree=node4_rand, init_samples=20, with_smooth=False,
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
    node2_ns=pred_node(f=pred3)
    node3_ns=min_node(children=[node1_ns,node2_ns])
    node4_ns=min_node(children=[node3_ns,node0_ns])
    TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree=node4_ns, init_samples=20, with_smooth=False,
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
    