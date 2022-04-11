#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:41:43 2022

@author: josh
"""

run Hover_Env_Bayesian_Optimization_1.py --exp ./results/save-hover-ppo-kin-vel-02.22.2022_14.53.33

"Notes on getting the trajectory and making graphs"
"This is plugging in evironmental parameter to get a trajectory"
trajs=[]
x0=[1,2,3,4]
trajs.append(self.system_under_test(x0))

smooth_safest_params=TM_smooth.smooth_X[smooth_vals.argmax()]

smooth_dangerous_params=TM_smooth.smooth_X[smooth_vals.argmin()]

#These are arrays for each specific position value through trajectory
x_vals_min=np.array(trajs[0][0]).T[0]

y_vals_min=np.array(trajs[0][0]).T[1]

z_vals_min=np.array(trajs[0][0]).T[2]

#THis is to plot xy position with boundaries
plt.scatter(x_vals_min,y_vals_min), plt.title('Scatter Plot of positions in XY Plane'),\
    plt.xlabel('X Position'), plt.ylabel('Y Position'), plt.xlim([-2.5,2.5]),plt.ylim([-2.5,2.5]),\
    plt.axhline(y=-2,xmin=-2,xmax=2, color='r'),plt.axhline(y=2,xmin=-2,xmax=2, color='r'),\
    plt.axvline(x=-2,ymin=-2,ymax=2,color='r'), plt.axvline(x=2,ymin=-2,ymax=2,color='r')
    
plt.scatter(np.zeros_like(z_vals_min),z_vals_min),plt.title('Scatter Plot of Height Position'),\ 
    plt.ylabel('Z Position'),plt.axhline(y=1.4,color='r')
    
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
    
#To save data
DF=pd.DataFrame(rand_Failure_count)
DF.to_csv("Experiment_1_Random_Failure_Count_04-11.csv")

DF=pd.DataFrame(ns_Failure_count)
DF.to_csv("Experiment_1_NS_Failure_Count_04-11.csv")

DF=pd.DataFrame(smooth_Failure_count)
DF.to_csv("Experiment_1_Smooth_Failure_Count_04-11.csv")