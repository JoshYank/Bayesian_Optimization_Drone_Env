#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 16:55:13 2022

@author: josh
"""
import math

import os
import time
from datetime import datetime
import argparse
import re
import numpy as np
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

#loading the saved policy
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
test_env = gym.make(env_name,
                        gui=True,
                        record=True,
                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=OBS,
                        act=ACT
                        )
logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                        num_drones=1
                        )
from gym import spaces

def compute_traj(**kwargs):      
    

    test_env.reset()
    #This sets initial xyz position
    if 'initial_xyzs' in kwargs:
        test_env.INIT_XYZS=kwargs['initial_xyzs']
        #This sets initial roll, pitch, yaw angle
    if 'initial_rpys' in kwargs:
        test_env.INIT_RPYS=kwargs['initial_rpys']
    
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
    return traj, additional_data
    
    

def sut(x0, **kwargs):
    return compute_traj(intital_xyzs=[0,0,x0[0]], intital_rpys=[x0[1],x0[2],x0[3]])
 

    

#--------------------------------------------------------------
#Bayesian Optimization
from adversarial_testing import pred_node, max_node, min_node, test_module
from adversarial_testing.utils import sample_from
rand_nums = [1 
 #3188388221,
 #1954593344,
 #2154016205,
 #3894811078,
 #3493033583,
 #3248332584,
 #1304673443,
 #3857496775,
 #2668478815,
 #278535713,
 #1762150547,
 #788841329,
 #2525132954,
 #677754898,
 #754758634,
 ]

#Bounds on Environmental Parameters
bounds=[(0.0,1.0)] #bounds on initial z-position
bounds.append((-math.pi,math.pi)) #bounds on roll angle in radians
bounds.append((-math.pi,math.pi)) #bounds on pitch angle in radians
bounds.append((-math.pi,math.pi)) #bounds on yaw angle in radians

#Requirement 1: Find initial configuration to minimize the requirement:
#1.that the drone doesn't fly over a certain altitude
smooth_details_r1=[]
"""
def pred1(traj):
    traj_array = np.array(traj)#Here we want the z position
    z_s=traj_array[:,2]
    return min(1.1-z_s)



def pred2(traj):
    traj_array = np.array(traj) #Here we want the x position
    x_s = traj_array[:,0]
    return min(2-np.abs(x_s))

def pred3(traj):
    traj_array = np.array(traj) #Here we want the y position
    y_s = traj_array[:,1]
    return min(2-np.abs(y_s))

node0=pred_node(f=pred1)
node1=pred_node(f=pred2)
node2=pred_node(f=pred3)
    
    

for r in rand_nums:
    np.random.seed(r)
    TM = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                     f_tree=node0, init_samples=60,
                     optimize_restarts=5, exp_weight=2)
    TM.initialize()
    
    TM.run_BO(150)
    
    smooth_vals = np.array(TM.f_acqu.find_GP_func())
    smooth_details_r1.append([TM.smooth_count,
                              TM.smooth_min_x,
                              TM.smooth_min_val, TM.smooth_min_loc])
    print(r, smooth_details_r1[-1])
"""