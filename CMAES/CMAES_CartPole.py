#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:53:47 2022

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
from cmaes import CMA

import logging
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

env = gym.make('CartPole-v1')
seed = 8902077161928034768
env.seed(seed)
model = PPO1.load("ppo1_cartpole")

def compute_traj(max_steps,ead=True, **kwargs):
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
    env.env.theta_threshold_radians=np.pi/2
    env.env.x_threshold=10
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
    return traj

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

def pred1(traj):
    traj1 = traj
    Robustness=[]
    XP=[]
    ANG=[]
    for i in range (len(traj1)):
        #x_pos=np.array(traj1[0]).T[i]
        #angle=np.array(traj[0]).T[i]
        x_pos=traj1[i][0]
        angle=traj1[i][2]
        XP.append(x_pos)
        ANG.append(angle)
        
        if x_pos <= -0.3 or x_pos>=0.3:
            Robustness.append(0.349066-abs(angle))
        if x_pos>-0.3 and x_pos<0.3:
            Robustness.append(1/abs(angle))
    #XP_All.append(XP)
    #ANG_All.append(ANG)
    return min(Robustness)

def pred2(traj):
    traj1 = traj
    Robustness=[]
    VEL=[]
    for i in range (len(traj1)):
        #x_pos=np.array(traj1[0]).T[i]
        #angle=np.array(traj[0]).T[i]
        x_vel=traj1[i][1]
        VEL.append(x_vel)
        Robustness.append(2-abs(x_vel))
    #VEL_ALL.append(VEL)
    return min(Robustness)


bounds=np.array([[-0.3,0.3],[-0.05,0.05],[-0.26,0.26],[-0.05,0.05]])

#optimizer=CMA(mean=np.zeros(4), sigma=1.3,bounds=bounds,population_size=(5))
print(" g    f(x1,x2)     x1      x2  ")
print("===  ==========  ======  ======")

C=[rand_nums,rand_nums2,rand_nums3,rand_nums4,rand_nums5,rand_nums6,
   rand_nums7,rand_nums8,rand_nums9,rand_nums10]
#=[rand_nums_test]
Exp_First_Failure_Gen=[] #Gen first Failure is found in test
Exp_Min_Rob=[]
Exp_Min_Gen=[]
Exp_Min_Params=[] #params for lowest robustness in test
Exp_Failures_Found=[] #number of generations where failures were found
Exp_Failures_Count=[] 
Exp_Gens_till_Stop=[]

for a in range(len(C)):
    Test_min_gen=[] #Gen where lowest robustness was found in test
    Test_min_rob=[] #lowest Robustness Score in test
    Test_Params=[] #list of params for worst one in each gen
    First_Failure_Gen=[] #Gen first Failure is found in test
    Test_Robust_Gens=[] #lowest robustness in each gen
    Test_Min_Params=[] #params for lowest robustness in test
    Test_Failure_Count=[] #number of counter examples found
    Test_Failures_Found=[] #number of generations with counter examples
    Gens_till_Stop=[]
    I=0
    
    for r in C[a]:
        np.random.seed(r)
        optimizer=CMA(mean=np.zeros(4), sigma=1.3,bounds=bounds,population_size=(5),seed=r)
        Generation=[]
        while True:
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                traj = compute_traj(max_steps=200,init_state=x)
                value1=pred1(traj)
                value2=pred2(traj)
                value=min(value1,value2)
                solutions.append((x, value))
                print(
                    f"{optimizer.generation:3d}  {value} (x1={x[0]}, x2 = {x[1]},x3 = {x[2]},x4 = {x[3]})"
                )
            optimizer.tell(solutions)
            Generation.append(solutions)
    
            if optimizer.should_stop():
                break
        Robust=[]
        Min_Param=[]
        Details=[]
        for a in range(len(Generation)):
            Robust.append(np.array(Generation[a]).T[1].min())
            min_arg=np.array(Generation[a]).T[1].argmin()
            Min_Param.append(np.array(Generation[a]).T[0][min_arg])
            for s in range(optimizer.population_size):
                Details.append(np.array(Generation[a]).T[1][s])
        Failure_Count=np.sum(np.array(Details)<0)
        Test_Failure_Count.append(Failure_Count)
        Failures_Found=np.sum(np.array(Robust)<0)
        Test_Failures_Found.append(Failures_Found)
        Test_Robust_Gens.append(Robust)
        Test_Params.append(Min_Param)
        Test_min_rob.append(np.array(Robust).min())
        Test_min_gen.append(np.array(Robust).argmin())
        Test_Min_Params.append(Test_Params[I][Test_min_gen[I]])
        if np.sum(np.array(Robust)<0)==0:
            First_Failure_Gen.append(0)
        else:
            First_Failure_Gen.append(np.where(np.array(Robust)<0)[0][0])
        Gens_till_Stop.append(len(Generation))
        I+=1
    Exp_Failures_Found.append(Test_Failures_Found)
    Exp_Failures_Count.append(Test_Failure_Count)
    Exp_First_Failure_Gen.append(First_Failure_Gen)
    Exp_Gens_till_Stop.append(Gens_till_Stop)
    
    Exp_Min_Params.append(Test_Min_Params)
    Exp_Min_Gen.append(Test_min_gen)
    Exp_Min_Rob.append(Test_min_rob)
    
    
    #Running with same number of simulations or designated # of generations
    """
    Test_min_gen=[]
    Test_min_rob=[]
    First_Failure_Gen=[]
    Test_Params=[]
    Test_Robust_Gens=[]
    Test_Min_Params=[]
    """
    """
    for r in C[a]:
        #np.random.seed(r)
        optimizer=CMA(mean=np.zeros(4), sigma=1.3,bounds=bounds,population_size=(5),seed=r)
        Generation=[]
        for generation in range(30):
            solutions=[]
            for i in range(optimizer.population_size):
                x = optimizer.ask()
                traj = compute_traj(max_steps=200,init_state=x)
                value1=pred1(traj)
                value2=pred2(traj)
                value=min(value1,value2)
                solutions.append((x, value))
                print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]},x3 = {x[2]},x4 = {x[3]})")
            optimizer.tell(solutions)
            Generation.append(solutions)
    #Min robustness and parameters per generation
        Robust=[]
        Min_Param=[]
        Details=[]
        for a in range(len(Generation)):
            Robust.append(np.array(Generation[a]).T[1].min())
            min_arg=np.array(Generation[a]).T[1].argmin()
            Min_Param.append(np.array(Generation[a]).T[0][min_arg])
            for s in range(optimizer.population_size):
                Details.append(np.array(Generation[a]).T[1][s])
        Test_Failure_Count.append(np.sum(np.array(Details)<0))
        Test_Failures_Found.append(np.sum(np.array(Robust)<0))
        Test_Robust_Gens.append(Robust)
        Test_Params.append(Min_Param)
        Test_min_rob.append(np.array(Robust).min())
        Test_min_gen.append(np.array(Robust).argmin())
        Test_Min_Params.append(Test_Params[I][Test_min_gen[I]])
        if np.sum(np.array(Robust)<0)==0:
            First_Failure_Gen.append(0)
        else:
            First_Failure_Gen.append(np.where(np.array(Robust)<0)[0][0])
        I+=1
    Exp_Failures_Found.append(Test_Failures_Found)
    Exp_Failures_Count.append(Test_Failure_Count)
    Exp_First_Failure_Gen.append(First_Failure_Gen)
    Exp_Gens_till_Stop.append(Gens_till_Stop)
    
    Exp_Min_Params.append(Test_Min_Params)
    Exp_Min_Gen.append(Test_min_gen)
    Exp_Min_Rob.append(Test_min_rob)
    """