#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:55:19 2023

@author: josh
"""

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
import matlab.engine

import os
import time
import math
from datetime import datetime
import argparse
import re
import operator
from cmaes import CMA
import logging

eng = matlab.engine.start_matlab() #connect matlab
def compute_traj(**kwargs):
    if 'param' in kwargs:
        inp = kwargs['param']
    traj=[]
    param_convert=np.divide(inp,100) #divide by 100 for correct scale
    param_convert2=param_convert.tolist()
    Input=matlab.double(param_convert2) #convert to matlab format
    Out=eng.sim_AT(Input,nargout=4) #run sim
    time=np.array(Out[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    velocity=np.array(Out[1])
    rpm_out=np.array(Out[2])
    traj.append(time)
    traj.append(velocity)
    traj.append(rpm_out)
    return traj
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
rand_nums_test=[172857]

#predicates
#output velocity should always be below 90mph
def pred1(traj):
    traj1=traj
    time=np.asarray(traj1[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    velocity=np.asarray(traj1[1])
    rpm_out=np.asarray(traj1[2])
    Max_Speed=90
    Robustness=[]
    for i in range(len(time)):
        Robustness.append(Max_Speed-velocity[i])
    return min(Robustness)
    
#output engine speed should always be below 1400rpm
def pred2(traj):
    traj1=traj
    time=np.asarray(traj1[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    velocity=np.asarray(traj1[1])
    rpm_out=np.asarray(traj1[2])
    Max_RPM=1400
    Robustness=[]
    for i in range(len(time)):
        Robustness.append(Max_RPM-rpm_out[i])
    return min(Robustness)

#The car's velocity should remain below 65mph until 20s have passed
def pred3(traj):
    traj1=traj
    time=np.asarray(traj1[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    velocity=np.asarray(traj1[1])
    rpm_out=np.asarray(traj1[2])
    Max_Speed=65
    Robustness=[]
    for i in range(944):#20 s
        Robustness.append(Max_Speed-velocity[i])
    return min(Robustness)

#The car's velocity should eventually reach 60mph in 20s
def pred4(traj):
    traj1=traj
    time=np.asarray(traj1[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    velocity=np.asarray(traj1[1])
    rpm_out=np.asarray(traj1[2])
    Max_Speed=65
    Robustness=[]
    for i in range(944):#20 s
        Robustness.append(velocity[i]-Max_Speed)
    return max(Robustness)

#bounds=np.array([[0.,100.],[0.,100.],[0,100],[0.,325.],[0.,325.],[0.,325.]])
bounds=np.array([[0.,100.],[0.,100.],[0.,325.],[0.,325.]])

print(" g    f(x1,x2)     x1      x2  ")
print("===  ==========  ======  ======")

#C=[rand_nums,rand_nums2,rand_nums3,rand_nums4,rand_nums5,rand_nums6,
   #rand_nums7,rand_nums8,rand_nums9,rand_nums10]
C=[rand_nums_test]
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
        optimizer=CMA(mean=np.array([50,50,160,160]), sigma=1.3,bounds=bounds,population_size=(20),seed=r)
        Generation=[]
        while True:
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                traj=compute_traj(max_steps=30,car_position=x[0:2],car_heading=x[2],
                                    car_speed=x[3],goal_position=x[4:6])
                value1=pred1(traj)
                value2=pred2(traj)
                value=min(value1,value2)
                solutions.append((x, value))
                #solutions.append((x, value1))
                print(
                    f"{optimizer.generation:3d}  {value1} (x1={x[0]}, x2 = {x[1]},x3 = {x[2]},x4 = {x[3]},x5 = {x[4]},x6 = {x[5]})"
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
        print('****************************************************************')
        print(I)
    Exp_Failures_Found.append(Test_Failures_Found)
    Exp_Failures_Count.append(Test_Failure_Count)
    Exp_First_Failure_Gen.append(First_Failure_Gen)
    Exp_Gens_till_Stop.append(Gens_till_Stop)
    
    Exp_Min_Params.append(Test_Min_Params)
    Exp_Min_Gen.append(Test_min_gen)
    Exp_Min_Rob.append(Test_min_rob)
    print('####################################################################')