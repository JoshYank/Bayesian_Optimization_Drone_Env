#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:17:33 2023

@author: josh
"""

import numpy as np
from GPyOpt.methods import BayesianOptimization
from cmaes import CMA
import matplotlib.pyplot as plt

#Code for BO_CMAES
class BO_CMAES:
    def __init__(
            self,
            spec,
            boundary,
            budget,
            population_size):
        self.spec=spec              #Robustness function with built in simulation
        self.global_bounds=boundary     #global boundaries of variables [(),(),..]
        self.population_size=population_size #number of iterations per localized model
        
        
    def initialize(self):
        format_glob_bound=set_bounds(self.global_bounds)
        self.global_opt=BayesianOptimization(f=self.spec,domain=format_glob_bound,
                                        acquisition_type='LCB',
                                        initial_design_numdata=self.population_size)
        best_ind_loc=[]
        for i in range(5):
            best_ind_loc.append(self.global_opt.Y.T.argsort()[0][i])
        

#function to define boundaries (both global and local)
def set_bounds(bounds):
    Bounds=[]
    for i in range(len(bounds)):
        Bounds.append({'name':'x'+str(i+1),'type':'continuous','domain':bounds[i]})
    return Bounds
