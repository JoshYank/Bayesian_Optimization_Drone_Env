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
            boundary,       #bounds=np.array([[-10,10],[-10,10]])
            budget,
            population_size):
        self.spec=spec              #Robustness function with built in simulation
        self.global_bounds=boundary     #global boundaries of variables
        self.population_size=population_size #number of iterations per localized model
        self.budget=budget
        
        
    def initialize(self):
        #initiate CMAES to calculate variance
        Sim_count=0
        self.CMA_optimizer=CMA(mean=np.zeros(len(self.global_bounds)),sigma=1.3,
                          bounds=self.global_bounds,
                          population_size=(self.population_size))
        self.CMA_optimizer._mu=self.population_size//4       #set number of best ind for eval
        #initialize fist sampling
        format_glob_bound=set_bounds(self.global_bounds)
        self.global_opt=BayesianOptimization(f=self.spec,domain=format_glob_bound,
                                        acquisition_type='LCB',
                                        initial_design_numdata=self.population_size)
        Sim_count+=len(self.global_opt.X)
        solutions=[]
        for _ in range(self.population_size):
            solutions.append((self.global_opt.X[_],self.global_opt.Y[_]))
        self.CMA_optimizer.tell(solutions)
        self.Global_Min_Robust=self.global_opt.Y.min()
        
        sigma=self.CMA_optimizer._sigma
        mean=self.CMA_optimizer._mean
        variance=[]
        for i in range(len(self.CMA_optimizer._C)):
            variance.append(self.CMA_optimizer._C[i][i])
        dev=np.sqrt(variance)
        self.local_bounds=[]
        for i in range(len(self.global_bounds)):
            lower_bounds=mean[i]-sigma*dev[i]
            upper_bounds=mean[i]+sigma*dev[i]
            self.local_bounds.append([lower_bounds,upper_bounds])
    """
        while Sim_count<self.budget:
            sigma=self.CMA_optimizer._sigma
            mean=self.CMA_optimizer._mean
            variance=[]
            for i in range(len(self.CMA_optimizer._C)):
                variance.append(self.CMA_optimizer._C[i][i])
            dev=np.sqrt(variance)
            self.local_bounds=[]
            for i in range(len(self.global_bounds)):
                lower_bounds=mean[i]-sigma*dev[i]
                upper_bounds=mean[i]+sigma*dev[i]
                self.local_bounds.append([lower_bounds,upper_bounds])
        """     
        

#function to define boundaries (both global and local)
def set_bounds(bounds):
    Bounds=[]
    for i in range(len(bounds)):
        Bounds.append({'name':'x'+str(i+1),'type':'continuous','domain':bounds[i]})
    return Bounds
