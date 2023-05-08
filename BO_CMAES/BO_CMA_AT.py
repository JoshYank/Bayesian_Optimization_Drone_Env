#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:06:48 2023

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

from BO_CMAES_Coding_1_0 import BO_CMAES
from BO_CMAES_Coding_1_0 import set_bounds

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

def sut(x0):
    return compute_traj(param=x0[0:4])

def pred1(x):
    Y=[]
    for _ in x:
        traj=compute_traj(param=_)
        time=np.asarray(traj[0]) #starting at time[172]=1 sec, every 25 iters=1sec
        velocity=np.asarray(traj[1])
        rpm_out=np.asarray(traj[2])
        Max_Speed=80
        Robustness=[]
        for i in range(len(time)):
            Robustness.append(Max_Speed-velocity[i])
        Y.append(min(Robustness))
    return np.array(Y)
    
#output engine speed should always be below 1400rpm
def pred2(x):
    Y=[]
    for _ in x:
        traj=compute_traj(param=_)
        time=np.asarray(traj[0]) #starting at time[172]=1 sec, every 25 iters=1sec
        velocity=np.asarray(traj[1])
        rpm_out=np.asarray(traj[2])
        Max_RPM=1400
        Robustness=[]
        for i in range(len(time)):
            Robustness.append(Max_RPM-rpm_out[i])
        Y.append(min(Robustness))
    return np.array(Y)

bounds=np.array([[0.,100.],[0.,100.],[0.,100.],[0.,100.]])
#---------------------------------------------------------------------------------
#BO-CMAES
Test=BO_CMAES(spec=pred1,boundary=bounds,budget=220,population_size=20)
Test.initialize()
Test.run_BO_CMA()

#-------------------------------------------------------------------------------
#BO
B=set_bounds(bounds)
BO_Test=BayesianOptimization(f=pred1,domain=B, acquisition_type='LCB',initial_design_numdata=20)
BO_Test.run_optimization(max_iter=200)