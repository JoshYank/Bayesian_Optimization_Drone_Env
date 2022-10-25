#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:04:13 2022

@author: josh
"""

"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import pandas as pd

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=12,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[0*R*np.cos((i/6)*2*np.pi+np.pi/2), 0*R*np.sin((i/6)*2*np.pi+np.pi/2)-R, 0*H+i*H_STEP] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  0*i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(ARGS.num_drones)])

    #### Debug trajectory ######################################
    #### Uncomment alt. target_pos in .computeControlFromState()
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(ARGS.num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    # NUM_WP = ARGS.control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    # wp_counters = np.array([0 for i in range(ARGS.num_drones)])

    #### Create the environment with or without video capture ##
    if ARGS.vision: 
        env = VisionAviary(drone_model=ARGS.drone,
                           num_drones=ARGS.num_drones,
                           initial_xyzs=INIT_XYZS,
                           initial_rpys=INIT_RPYS,
                           physics=ARGS.physics,
                           neighbourhood_radius=10,
                           freq=ARGS.simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           #gui=ARGS.gui,
                           gui=False,
                           record=ARGS.record_video,
                           obstacles=ARGS.obstacles
                           )
    else: 
        env = CtrlAviary(drone_model=ARGS.drone,
                         num_drones=ARGS.num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=ARGS.physics,
                         neighbourhood_radius=10,
                         freq=ARGS.simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         #gui=ARGS.gui,
                         gui=False,
                         record=ARGS.record_video,
                         obstacles=ARGS.obstacles,
                         user_debug_gui=ARGS.user_debug_gui
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    #### Initialize the controllers ############################
    if ARGS.drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
    elif ARGS.drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]

    """
    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
    START = time.time()
    for i in range(0, 5*int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        
        obs_traj.append([obs[str(0)]["state"][0],obs[str(0)]["state"][1],obs[str(0)]["state"][2],
                         obs[str(0)]["state"][7],obs[str(0)]["state"][8]])

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(ARGS.num_drones):
                action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(j)]["state"],
                                                                       target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                                                                       # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                       target_rpy=INIT_RPYS[j, :]
                                                                       )

            #### Go to the next way point and loop #####################
            for j in range(ARGS.num_drones): 
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0


        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
   # env.close()
   """
from gym import spaces
def compute_traj(initial_xyz,initial_rpy):   
        #This sets initial xyz position

    env.INIT_XYZS=initial_xyz
    env.INIT_RPYS=initial_rpy
    
    env.reset()
    iter_time=0
    traj_iter=[iter_time]
    r=0
    done=False
    obs=[[env.pos[0][0],env.pos[0][1],env.pos[0][2],env.rpy[0][0],env.rpy[0][1],env.rpy[0][2]]]
    traj=[obs[0]]
    
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
    START = time.time()
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, done, additional_data = env.step(action)
        
        traj.append([obs[str(0)]["state"][0],obs[str(0)]["state"][1],obs[str(0)]["state"][2],
                         obs[str(0)]["state"][7],obs[str(0)]["state"][8]])
        
        iter_time+=1
        traj_iter.append(iter_time)

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(ARGS.num_drones):
                action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(j)]["state"],
                                                                       target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                                                                       # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                       target_rpy=INIT_RPYS[j, :]
                                                                       )

            #### Go to the next way point and loop #####################
            for j in range(ARGS.num_drones): 
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0


        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

        iter_time+=1
        traj_iter.append(iter_time)
        additional_data['reward']=r
        additional_data['iters'] = iter_time
        env.render()
    return traj, {'reward':r, 'iter_time': iter_time}
    
    
def sut(x0, **kwargs):
    return compute_traj(np.array([[x0[0],x0[1],x0[2]] for i in range(ARGS.num_drones)]), np.array([[x0[3], x0[4], 0] for i in range(ARGS.num_drones)]))

#--------------------------------------------------------------
#Bayesian Optimization
from adversarial_testing import pred_node, max_node, min_node, test_module
from adversarial_testing.utils import sample_from
from adversarial_testing import optimizers
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
 278535713,
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
rand_nums11=[61,18,2,33,31,49,81,17,11,131]
rand_nums12=[65,13,19,38,32,99,84,22,41,143]
rand_nums13=[375686, 31957, 26589, 180738, 281078, 88509, 499107, 466051, 478311, 69059]
rand_nums9_A=[64639, 81307, 16913]
rand_nums_test=[172857]


#Bounds on Environmental Parameters
bounds=[(-0.5,0.5)] #init x pos
bounds.append((-0.5,0.5)) #init y pos
bounds.append((0.2,1.0)) #bounds on z
bounds.append((-math.pi/1.75,math.pi/1.75)) #bounds on roll angle in radians pi/4 before
bounds.append((-math.pi/1.75,math.pi/1.75)) #bounds on pitch angle in radians
#bounds.append((-math.pi/3,math.pi/3)) #bounds on yaw angle in radians

#Requirements: Find initial configuration to minimize the requirement:
    #1. Until 1 second of time has passed, the drone must remain close to its hovering state (initial position)
    #2. For the whole trajectory, the drone must remain upright
        #The drone does not go beyond the angle of pi/3 radians or 60 degrees
    
smooth_details_r1=[]
ns_details_r3 = []
random_details_r3 = []

smooth_results=[]
ns_results=[]
random_results=[]

smooth_Failure_count=[]
rand_Failure_count=[]
ns_Failure_count=[]

def pred1(traj):
    hover_displacement=[]
    traj1=traj[0]
    iter_time1=0
    #checking over time range [0,1] seconds
    for i in range(len(traj1)):
        iter_time1+=1
        x_pos=np.array(traj1).T[i][0]
        y_pos=np.array(traj1).T[i][1]
        z_pos=np.array(traj1).T[i][2]
        displacement=np.linalg.norm(np.array([0, 0, 0.5])-[x_pos,y_pos,z_pos])
        hover_displacement.append(1.5-displacement)
    return min(hover_displacement)

def pred2(traj):
    roll_robust=[]
    pitch_robust=[]
    traj1=traj[0]
    iter_time2=0
    #check over full flight [0,TIME] seconds
    for i in range(len(traj)):
        iter_time2+=1
        roll_ang=np.array(traj1).T[i][3]
        pitch_ang=np.array(traj1).T[i][4]
        roll_robust.append((np.pi/1.5)-np.abs(roll_ang))
        pitch_robust.append((np.pi/1.5)-np.abs(pitch_ang))
    min_roll=min(roll_robust)
    min_pitch=min(pitch_robust)
    orientation=[min(roll_robust),min(pitch_robust)]
    return min(min_roll,min_pitch)


C=[rand_nums,rand_nums2,rand_nums3,rand_nums4,rand_nums5,rand_nums6,
   rand_nums7,rand_nums8,rand_nums9,rand_nums10]
NS_Details=[]
NS_Param=[]
NS_Robust=[]
Rand_Details=[]
Rand_Param=[]
Rand_Rob=[]
for a in range(len(C)):

    
    smooth_details_r1=[]
    ns_details_r3 = []
    random_details_r3 = []
    
    smooth_results=[]
    ns_results=[]
    random_results=[]
    
    smooth_Failure_count=[]
    rand_Failure_count=[]
    ns_Failure_count=[]
    
    ns_param=[]
    ns_robust=[]
    random_param=[]
    random_robust=[]
    
    for r in C[a]:
        """
        np.random.seed(r)
        node0=pred_node(f=pred1)
        node1=pred_node(f=pred2)
        node2=min_node(children=[node0,node1])
        
        #node3=pred_node(f=pred3)
        #node4=min_node(children=[node0,node3])
    
        TM_smooth = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                         f_tree=node2, init_samples=20, with_smooth=True,
                         with_random=False, with_ns=False,
                         optimize_restarts=1, exp_weight=2)
        TM_smooth.initialize()
        
        TM_smooth.run_BO(150)
        
        smooth_Failure_count.append(TM_smooth.smooth_count)
        
        smooth_vals = np.array(TM_smooth.f_acqu.find_GP_func())
        smooth_details_r1.append([TM_smooth.smooth_count,
                                  TM_smooth.smooth_min_x,
                                  TM_smooth.smooth_min_val, TM_smooth.smooth_min_loc])
        
        #smooth_results.append(TM_smooth.smooth_count, )
        #del TM_smooth
        #print(r, smooth_details_r1[-1])
        """
    #for r in rand_nums:
        
        np.random.seed(r)
        node0_rand=pred_node(f=pred1)
        
        node1_rand=pred_node(f=pred2)
        node2_rand=min_node(children=[node0_rand,node1_rand])
        
       # node3_rand=pred_node(f=pred3)
        #node4_rand=min_node(children=[node0_rand,node3_rand])
        """
        """
        """
        TM_rand = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                         f_tree=node2_rand, init_samples=20, with_smooth=False,
                         with_random=True, with_ns=False, optimizer=optimizers.direct_opt(bounds),
                         optimize_restarts=1, exp_weight=2)
        TM_rand.initialize()
        
        TM_rand.run_BO(100)
        
        rand_Failure_count.append(TM_rand.rand_count)
        
        rand_vals = np.array(TM_rand.random_Y)
        random_details_r3.append([TM_rand.rand_count,
                              TM_rand.rand_min_x,
                              TM_rand.rand_min_val, TM_rand.rand_min_loc])
        random_param.append(TM_rand.rand_min_x)
        random_robust.append(TM_rand.rand_min_val)
        """
        #print(r, random_details_r3[-1])
        #del TM_rand
    #for r in rand_nums:
        
        np.random.seed(r)
        node0_ns=pred_node(f=pred1)
        
        node1_ns=pred_node(f=pred2)
        node2_ns=min_node(children=[node0_ns,node1_ns])
        
        """
        node3_ns=pred_node(f=pred3)
        node4_ns=min_node(children=[node0_ns,node3_ns])
        """
        
        
        TM_ns = test_module(bounds=bounds, sut=lambda x0: sut(x0),
                         f_tree=node2_ns, init_samples=20, with_smooth=False,
                         with_random=False, with_ns=True,  optimizer=optimizers.direct_opt(bounds),
                         optimize_restarts=1, exp_weight=2,)
        TM_ns.initialize()
        
        TM_ns.run_BO(100)
        
        ns_Failure_count.append(TM_ns.ns_count)
        
        ns_smooth_vals = np.array(TM_ns.ns_GP.Y)
        ns_details_r3.append([TM_ns.ns_count,
                            TM_ns.ns_min_x,
                            TM_ns.ns_min_val, TM_ns.ns_min_loc])
        ns_param.append(TM_ns.ns_min_x)
        ns_robust.append(TM_ns.ns_min_val)
        
    NS_Details.append(ns_Failure_count)
    NS_Param.append(ns_param)
    NS_Robust.append(ns_robust)
    Rand_Details.append(rand_Failure_count)
    Rand_Param.append(random_param)
    Rand_Rob.append(random_robust)
    print("##################################################################")
        
    
#print(smooth_Failure_count,rand_Failure_count,ns_Failure_count)
#Making bar plots with error bars
#Random_mean=np.mean(rand_Failure_count)

#Smooth_mean=np.mean(smooth_Failure_count)

NS_mean=np.mean(ns_Failure_count)

#Smooth_std=np.std(smooth_Failure_count)

NS_std=np.std(ns_Failure_count)

#Random_std=np.std(rand_Failure_count)

#Method=['Random Sampling','Bayesian Optimization']
Method=['Bayesian Optimization']
x_pos=np.arange(len(Method))

#Means_Failure_Modes=[Random_mean,NS_mean]
#Error=[Random_std,NS_std]
Means_Failure_Modes=[NS_mean]
Error=[NS_std]

plt.bar(x_pos, Means_Failure_Modes, yerr=Error, align='center', alpha=0.5, ecolor='black', capsize=10),\
    plt.ylabel('Failure Modes Found'),plt.xticks(x_pos,Method),\
    plt.title('Failure Modes Found:  BO v. Random Sampling'),\
    plt.grid(True,axis='y'), plt.show()
"""
Method=['Random Sampling']
Means_Failure_Modes=[Random_mean]
x_pos=np.arange(len(Method))
Error=[Random_std]
plt.bar(x_pos, Means_Failure_Modes, yerr=Error, align='center', alpha=0.5, ecolor='black', capsize=10),\
    plt.ylabel('Failure Modes Found'),plt.xticks(x_pos,Method),\
    plt.title('Failure Modes Found: Random Sampling'),\
    plt.grid(True,axis='y'), plt.show()
"""
"""
Test_num=['1','2','3','4','5','6','7','8','9','10']
X_axis=np.arange(len(Test_num))

#plt.bar(X_axis-0.2,smooth_Failure_count,0.2,label='Smooth BO')
plt.bar(X_axis+0.2,rand_Failure_count,0.2,label='Random Sampling')
plt.bar(X_axis+0,ns_Failure_count,0.2,label='Bayesian Optimization')
plt.xticks(X_axis,Test_num)
plt.xlabel("Tests")
plt.ylabel("Failure Modes Found")
plt.title("Failure Modes Per Test")
plt.legend()
plt.show()
"""

#To plot robustness over BO iteration
"""
time_range_iteration=np.array(list(range(1,len(TM_smooth.f_acqu.Y)+1)))
fig4=plt.figure()
ex=fig4.add_subplot()
ex.set_xlim(left=1,right=len(TM_smooth.f_acqu.Y)+1)
ex.scatter(time_range_iteration,TM_smooth.f_acqu.Y)

ex.set_title('Smooth BO Robustness v. Iteration')
ex.legend(['Smooth_BO_Robustness'])
ex.set_xlabel('BO Iteration')
ex.set_ylabel('Robustness')

time_range_iteration2=np.array(list(range(1,len(TM_ns.ns_GP.Y)+1)))
fig5=plt.figure()
fx=fig5.add_subplot()
fx.set_xlim(left=1,right=len(TM_ns.ns_GP.Y)+1)
fx.scatter(time_range_iteration2,TM_ns.ns_GP.Y)

fx.set_title('NS BO Robustness v. Iteration')
fx.legend(['NS_BO_Robustness'])
fx.set_xlabel('BO Iteration')
fx.set_ylabel('Robustness')

time_range_iteration3=np.array(list(range(1,len(TM_rand.random_Y)+1)))
fig6=plt.figure()
gx=fig6.add_subplot()
gx.set_xlim(left=1,right=len(TM_rand.random_Y)+1)
gx.scatter(time_range_iteration3,TM_rand.random_Y)

gx.set_title('Random Sampling Robustness v. Iteration')
gx.legend(['Random_Sampling_Robustness'])
gx.set_xlabel('BO Iteration')
gx.set_ylabel('Robustness')
"""
#To Save data, change 1B and date at end
DF=pd.DataFrame(rand_Failure_count)
DF.to_csv("Drone_Experiment_4_Safety_Random_Failure_Count_08-14.csv")

DF=pd.DataFrame(ns_Failure_count)
DF.to_csv("Drone_Experiment_4_Safety_BO_Failure_Count_08-14.csv")

DF=pd.DataFrame(smooth_Failure_count)
DF.to_csv("Drone_Experiment_1_Safety_Smooth_Failure_Count_08-10.csv")

"""
# safest traj
smooth_safest_params=TM_smooth.smooth_X[smooth_vals.argmax()]
traj_smooth_safe=[]
x_vals_smooth_safe=[]
y_vals_smooth_safe=[]
z_vals_smooth_safe=[]
roll_vals_smooth_safe=[]
pitch_vals_smooth_safe=[]
yaw_vals_smooth_safe=[]
traj_smooth_safe.append(TM_smooth.system_under_test(smooth_safest_params))
for i in range(len(np.array(traj_smooth_safe[0][0]))):
    x_vals_smooth_safe.append(np.array(traj_smooth_safe[0][0][i]).T[0])
    y_vals_smooth_safe.append(np.array(traj_smooth_safe[0][0][i]).T[1])
    z_vals_smooth_safe.append(np.array(traj_smooth_safe[0][0][i]).T[2])
    roll_vals_smooth_safe.append(np.array(traj_smooth_safe[0][0][i]).T[3])
    pitch_vals_smooth_safe.append(np.array(traj_smooth_safe[0][0][i]).T[4])
    #yaw_vals_smooth_safe.append(np.array(traj_smooth_safe[0][0][i]).T[5])


"""
