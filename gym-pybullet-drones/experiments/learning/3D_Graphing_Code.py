#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:28:04 2022

@author: josh

To load and plot saved data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#This is to load files storing x,y,z position data through a trajectory
x_vals=pd.read_csv('/home/josh/BO_Results_Hover_Velocity_PPO/Experiment Results April 10 -16/Experiment 1/Safe Traj/Experiment_1_Smooth_Safe_X_Position_04-11.csv')
x_vals=x_vals.to_numpy()
x_vals=x_vals.T[1]

y_vals=pd.read_csv('/home/josh/BO_Results_Hover_Velocity_PPO/Experiment Results April 10 -16/Experiment 1/Safe Traj/Experiment_1_Smooth_Safe_X_Position_04-11.csv')
y_vals=y_vals.to_numpy()
y_vals=y_vals.T[1]

z_vals=pd.read_csv('/home/josh/BO_Results_Hover_Velocity_PPO/Experiment Results April 10 -16/Experiment 1/Safe Traj/Experiment_1_Smooth_Safe_X_Position_04-11.csv')
z_vals=z_vals.to_numpy()
z_vals=z_vals.T[1]

#To make 3D plot
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.scatter3D(x_vals,y_vals,z_vals)

#Safe Traj and other dangerous trajectory
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.scatter3D(x_vals_smooth_safe,y_vals_smooth_safe,z_vals_smooth_safe,color='green')
ax.scatter3D(x_vals_smooth_traj1,y_vals_smooth_traj1,z_vals_smooth_traj1,color='Black')
ax.scatter3D(x_vals_smooth_traj2,y_vals_smooth_traj2,z_vals_smooth_traj2,color='Purple')
ax.scatter3D(x_vals_smooth_traj3,y_vals_smooth_traj3,z_vals_smooth_traj3,color='Blue')
ax.scatter3D(x_vals_smooth_traj4,y_vals_smooth_traj4,z_vals_smooth_traj4,color='yellow')

ax.legend(['Safe Traj','Most Dangerous','Trajectory 2','Trajectory 3','Trajectory 4'])
ax.set_title('Safest and 4 Most Dangerous Trajectories')
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis')

#boundary plane for height safety Spec f(z)<1.8
xs=np.linspace(-.25,.25,100)
ys=np.linspace(-.25,.25,100)
zs=np.linspace(0,3,100)

xx,yy=np.meshgrid(xs,ys)
Z_Bounds_Plane=1.8+xx-xx
ax.plot_surface(xx,yy,Z_Bounds_Plane)

#x safety bounds
yyy,zzz=np.meshgrid(ys,zs)
X_Bounds_Plane1=0.16+yyy-yyy
X_Bounds_Plane2=-0.16+yyy-yyy
ax.plot_surface(X_Bounds_Plane1,yyy,zzz)
ax.plot_surface(X_Bounds_Plane2,yyy,zzz)

#y safety bounds
xxxx,zzzz=np.meshgrid(xs,zs)
Y_Bounds_Plane1=0.16+xxxx-xxxx
Y_Bounds_Plane2=-0.16+xxxx-xxxx
ax.plot_surface(xxxx,Y_Bounds_Plane1,zzzz)
ax.plot_surface(xxxx,Y_Bounds_Plane2,zzzz)