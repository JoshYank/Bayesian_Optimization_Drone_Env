#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:34:47 2023

@author: josh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:44:09 2023

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

from GPyOpt.methods import BayesianOptimization

from BO_CMAES_Coding_1_2 import BO_CMAES
from BO_CMAES_Coding_1_2 import set_bounds


eng = matlab.engine.start_matlab() #connect matlab
#matlab.engine.find_matlab()
#eng=matlab.engine.connect_matlab()

def compute_traj(**kwargs):
    if 'param' in kwargs:
        inp = kwargs['param']
    traj=[]
    param_convert=inp.tolist() #divide by 100 for correct scale
    Input=matlab.double(param_convert) #convert to matlab format
    Out=eng.NN_Sim(Input,nargout=3) #run sim
    time=np.array(Out[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    Position=np.array(Out[1])
    Ref=np.array(Out[2])
    traj.append(time)
    traj.append(Position)
    traj.append(Ref)
    return traj

def sut(x0):
    return compute_traj(param=x0[0:8])

rand_nums = [
 #3188388221,
 #1954593344,
 #2154016205,
 #3894811078,
 #3493033583,
 #3248332584,
 #1304673443,
 3857496775,
 2668478815,
 278535713
 ]
rand_nums2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#rand_nums3=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
rand_nums3=[34, 36, 38]
#rand_nums4=[1221, 113, 134, 156, 19344, 22102, 23413, 1511, 12239, 29800]
rand_nums4=[156, 19344, 22102, 23413, 1511, 12239, 29800]
#rand_nums5=[5085, 8991, 1635, 7805, 7187, 8645, 8888, 5520, 6446, 171452]
rand_nums5=[6446, 171452]
rand_nums6=[1461, 8194, 6927, 5075, 4903, 3799, 6268, 8155, 5502, 1187]
rand_nums7=[64846, 28856, 43210, 70661, 14700, 21044, 58191, 17243, 24958, 80194]
rand_nums8=[54239, 69118, 51184, 57468, 57945, 78075, 34142, 78062, 33150,6148]

rand_nums9=[63951, 36835, 59249, 17176, 32123, 54118, 79720, 64639, 81307, 16913]
rand_nums10=[347957, 510020, 545416, 613511, 673274, 619204, 630790, 627544,
       127016, 390172]
rand_nums11=[61,18,2,33,31,49,81,17,11,131]
rand_nums12=[65,13,19,38,32,99,84,22,41,143]
rand_nums13=[375686, 31957, 26589, 180738, 281078, 88509, 499107, 466051, 478311, 69059]
rand_nums14=[55831, 297066, 108997, 401197, 377974, 111910, 185704, 116438, 414932, 310477]
rand_nums15=[320,919,8132,30911,127,471,459,5228,99,12222]
rand_nums16=[1304673443,3857496775,36,234,1819,194,101,200,6268,6927]
rand_nums_test=[13857,3726]

#output height should always remain below 3.9
def pred1(x):
    Y=[]
    for _ in x:
        traj=compute_traj(param=_)
        time=np.asarray(traj[0]) #starting at time[172]=1 sec, every 25 iters=1sec
        Position=np.asarray(traj[1])
        Ref=np.asarray(traj[2])
        Max_Height=3.9
        Robustness=[]
        for i in range(len(time)):
            Robustness.append(Max_Height-Position[i])
        Y.append(min(Robustness))
    return np.array(Y)
"""
def pred1(traj):
    traj1=traj
    time=np.asarray(traj1[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    Position=np.asarray(traj1[1])
    Ref=np.asarray(traj1[2])
    Max_Height=3.9
    Robustness=[]
    for i in range(len(time)):
        Robustness.append(Max_Height-Position[i])
    return min(Robustness)
"""

def pred2(x):
    Y=[]
    for _ in x:
        traj=compute_traj(param=_)
        in_time=[5,10,15,20,25,30,35]
        time=np.asarray(traj[0]) #starting at time[172]=1 sec, every 25 iters=1sec
        Position=np.asarray(traj[1])
        Ref=np.asarray(traj[2])
        Settle_Time=100
        Robustness=[]
        for i in range(len(in_time)):
            segment_start=time==in_time[i]
            segment_end=time==in_time[i]+2
            Iter_start=np.argmax(segment_start)
            Iter_end=np.argmax(segment_end)
            settle=0
            for a in range(Iter_start,Iter_end):
                if abs(Position[a]-Ref[a])<=(0.005+0.04*abs(Ref[a])):
                    settle+=1
                if abs(Position[a]-Ref[a])>(0.005+0.04*abs(Ref[a])):
                    settle+=0
            Robustness.append(settle-Settle_Time)
        Robustness=np.divide(Robustness,100)
        Y.append(min(Robustness))
    return np.array(Y)
        
"""  
def pred2(traj):
    traj1=traj
    in_time=[5,10,15,20,25,30,35]
    time=np.asarray(traj1[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    Position=np.asarray(traj1[1])
    Ref=np.asarray(traj1[2])
    Settle_Time=100
    Robustness=[]
    for i in range(len(in_time)):
        segment_start=time==in_time[i]
        segment_end=time==in_time[i]+2
        Iter_start=np.argmax(segment_start)
        Iter_end=np.argmax(segment_end)
        settle=0
        for a in range(Iter_start,Iter_end):
            if abs(Position[a]-Ref[a])<=(0.005+0.04*abs(Ref[a])):
                settle+=1
            if abs(Position[a]-Ref[a])>(0.005+0.04*abs(Ref[a])):
                settle+=0
        Robustness.append(settle-Settle_Time)
    Robustness=np.divide(Robustness,100)
    return min(Robustness)
"""
bounds=np.array([[1.,3.],[1.,3.],[1.,3.],[1.,3.],[1.,3.],[1.,3.],[1.,3.],[1.,3.]])
B=set_bounds(bounds)

rand_num=[16245,18762,199,921,2817];
BO_Robust=[]
BO_Counter=[]
BCMA_Robb=[]
BCMA_Counter=[]
BCMA_Jump=[]
for r in rand_num:
    np.random.seed(r)
    
    Test=BO_CMAES(spec=pred1,boundary=bounds,budget=240,population_size=20)
    Test.initialize()
    Test.run_BO_CMA()
    Test.get_violation_count()
    BCMA_Counter.append(Test.Violation_Count)
    BCMA_Robb.append(Test.Global_Min_Robust)
    BCMA_Jump.append(Test.Record_jump)
    
    BO_Test=BayesianOptimization(f=pred1,domain=B, acquisition_type='LCB',initial_design_numdata=20)
    BO_Test.run_optimization(max_iter=120)
    BO_Counter.append(np.sum(BO_Test.Y<0))
    BO_Robust.append(BO_Test.Y.min())
    
"""
#---------------------------------------------------------------------------------
#BO-CMAES
Test=BO_CMAES(spec=pred2,boundary=bounds,budget=440,population_size=40)
Test.initialize()
Test.run_BO_CMA()

#-------------------------------------------------------------------------------
#BO

B=set_bounds(bounds)
BO_Test=BayesianOptimization(f=pred2,domain=B, acquisition_type='LCB',initial_design_numdata=20)
BO_Test.run_optimization(max_iter=200)
"""
