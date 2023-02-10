#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:19:39 2023

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
#------------------------------------------------------------------------------   
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
rand_nums4=[1221, 113, 134, 156, 19344, 22102, 23413, 1511, 12239, 29800]
rand_nums5=[5085, 8991, 1635, 7805, 7187, 8645, 8888, 5520, 6446, 171452]
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

#Uncertainty bounds input
bounds = [(1.00, 3.00)] # Bounds on the reference height at t=0
bounds.append((1.00, 3.00)) # Bounds on reference height at t=5
bounds.append((1.00, 3.00)) # Bounds on reference height at t=10
bounds.append((1.00, 3.00)) # Bounds on reference height at t=15
bounds.append((1.00, 3.00)) # Bounds on reference height at t=20
bounds.append((1.00, 3.00)) # Bounds on reference height at t=25
bounds.append((1.00, 3.00)) #Bounds on reference height at t=30
bounds.append((1.00, 3.00)) #Bounds on reference height at t=35


#predicates
#The system should settle within 2 seconds to a new input
def pred1(traj):
    traj1=traj
    time=np.asarray(traj1[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    Position=np.asarray(traj1[1])
    Ref=np.asarray(traj1[2])
    Robustness=[]
    for i in range(len(time)):
        Robustness.append((0.2+0.05*abs(Ref[i]))-(abs(Position[i]-Ref[i])))
    return min(Robustness)
    
#output height should always remain below 3.7
def pred2(traj):
    traj1=traj
    time=np.asarray(traj1[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    Position=np.asarray(traj1[1])
    Ref=np.asarray(traj1[2])
    Max_Height=3.9
    Robustness=[]
    for i in range(len(time)):
        Robustness.append(Max_Height-Position[i])
    return min(Robustness)
#The system should settle within 2 seconds to a new input
def pred3(traj):
    traj1=traj
    time=np.asarray(traj1[0]) #starting at time[172]=1 sec, every 25 iters=1sec
    Position=np.asarray(traj1[1])
    Ref=np.asarray(traj1[2])
    time_5=time==5.00
    time_5=time==5.00
    time_5=time==5.00
    time_5=time==5.00
    time_5=time==5.00
    time_5=time==5.00
    time_5=time==5.00
    time_5=time==5.00
    Iter_5=np.argmax(time_5)
    robustness=0
    Settle_Time=100 #1 second
    for i in range(Iter_5,Iter_2):
        if abs(Position[i]-Ref[i])<=(0.005+0.04*abs(Ref[i])):
            robustness+=1
        if abs(Position[i]-Ref[i])>(0.005+0.04*abs(Ref[i])):
            robustness+=0
    robustness/100
    
def pred4(traj):
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
        

#BO set up
#C=[rand_nums,rand_nums2,rand_nums3,rand_nums4,rand_nums5,rand_nums6,rand_nums7,rand_nums8,rand_nums9,rand_nums10]
C=[rand_nums,rand_nums2]
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
    AA=0
    
    for r in C[a]:
        #for r in rand_nums:
        
        np.random.seed(r)
        node0_rand=pred_node(f=pred2)
        node1_rand=pred_node(f=pred4)
        node2_rand=min_node(children=[node0_rand,node1_rand])
    
        TM_rand = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                         f_tree=node2_rand, init_samples=20, with_smooth=False,
                         with_random=True, with_ns=False,
                         optimize_restarts=1, exp_weight=2)
        TM_rand.initialize()
        
        TM_rand.run_BO(200)
        
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
        node0_ns=pred_node(f=pred2)
        node1_ns=pred_node(f=pred4)
        node2_ns=min_node(children=[node0_ns,node1_ns])
    
        TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                         f_tree=node2_ns, init_samples=20, with_smooth=False,
                         with_random=False, with_ns=True, #optimizer=optimizers.direct_opt(bounds),
                         optimize_restarts=1, exp_weight=2)
        TM_ns.initialize()
        
        TM_ns.run_BO(200)
        
        ns_Failure_count.append(TM_ns.ns_count)
        ns_worst_sim.append(TM_ns.ns_min_loc)
        
        ns_smooth_vals = np.array(TM_ns.ns_GP.Y)
        ns_details.append([TM_ns.ns_count,
                            TM_ns.ns_min_x,
                            TM_ns.ns_min_val, TM_ns.ns_min_loc])
        ns_param.append(TM_ns.ns_min_x)
        ns_robust.append(TM_ns.ns_min_val)
        print('***************************************************************')
        AA+=1
        print(AA)
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