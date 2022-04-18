#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:53:08 2022

@author: josh
"""

#To Save data, change 1B and date at end
DF=pd.DataFrame(rand_Failure_count)
DF.to_csv("Experiment_1B_Random_Failure_Count_04-18.csv")

DF=pd.DataFrame(ns_Failure_count)
DF.to_csv("Experiment_1B_NS_Failure_Count_04-18.csv")

DF=pd.DataFrame(smooth_Failure_count)
DF.to_csv("Experiment_1B_Smooth_Failure_Count_04-18.csv")

#To Save Params and Trajs for best smooth
DF=pd.DataFrame(smooth_safest_params)
DF.to_csv("Experiment_1B_Smooth_Safe_Param_04-18.csv")
DF=pd.DataFrame(x_vals_smooth_safe)
DF.to_csv("Experiment_1B_Smooth_Safe_X_Pos_04-18.csv")
DF=pd.DataFrame(y_vals_smooth_safe)
DF.to_csv("Experiment_1B_Smooth_Safe_Y_Pos_04-18.csv")
DF=pd.DataFrame(z_vals_smooth_safe)
DF.to_csv("Experiment_1B_Smooth_Safe_Z_Pos_04-18.csv")

#To Save Params and Trajs for worst Trajs of Smooth
DF=pd.DataFrame(smooth_traj1_params)
DF.to_csv("Experiment_1B_Smooth_traj1_Param_04-18.csv")
DF=pd.DataFrame(x_vals_smooth_traj1)
DF.to_csv("Experiment_1B_Smooth_Traj1_X_Pos_04-18.csv")
DF=pd.DataFrame(y_vals_smooth_traj1)
DF.to_csv("Experiment_1B_Smooth_Traj1_Y_Pos_04-18.csv")
DF=pd.DataFrame(z_vals_smooth_traj1)
DF.to_csv("Experiment_1B_Smooth_Traj1_Z_Pos_04-18.csv")

DF=pd.DataFrame(smooth_traj2_params)
DF.to_csv("Experiment_1B_Smooth_traj2_Param_04-18.csv")
DF=pd.DataFrame(x_vals_smooth_traj2)
DF.to_csv("Experiment_1B_Smooth_Traj2_X_Pos_04-18.csv")
DF=pd.DataFrame(y_vals_smooth_traj2)
DF.to_csv("Experiment_1B_Smooth_Traj2_Y_Pos_04-18.csv")
DF=pd.DataFrame(z_vals_smooth_traj2)
DF.to_csv("Experiment_1B_Smooth_Traj2_Z_Pos_04-18.csv")

DF=pd.DataFrame(smooth_traj3_params)
DF.to_csv("Experiment_1B_Smooth_traj3_Param_04-18.csv")
DF=pd.DataFrame(x_vals_smooth_traj3)
DF.to_csv("Experiment_1B_Smooth_Traj3_X_Pos_04-18.csv")
DF=pd.DataFrame(y_vals_smooth_traj3)
DF.to_csv("Experiment_1B_Smooth_Traj3_Y_Pos_04-18.csv")
DF=pd.DataFrame(z_vals_smooth_traj3)
DF.to_csv("Experiment_1B_Smooth_Traj3_Z_Pos_04-18.csv")

DF=pd.DataFrame(smooth_traj4_params)
DF.to_csv("Experiment_1B_Smooth_traj4_Param_04-18.csv")
DF=pd.DataFrame(x_vals_smooth_traj4)
DF.to_csv("Experiment_1B_Smooth_Traj4_X_Pos_04-18.csv")
DF=pd.DataFrame(y_vals_smooth_traj4)
DF.to_csv("Experiment_1B_Smooth_Traj4_Y_Pos_04-18.csv")
DF=pd.DataFrame(z_vals_smooth_traj4)
DF.to_csv("Experiment_1B_Smooth_Traj4_Z_Pos_04-18.csv")

