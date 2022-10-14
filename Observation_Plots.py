#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 12:23:32 2022

@author: josh
"""

import os
import time
import math
from datetime import datetime
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
import gym

#Cart Pole Results
BO1=[19,92,30,100,63,52,68,34,76,45]
RS1=[21,32,19,28,17,22,31,22,22,22]
BO2=[45,31,30,25,12,19,21,96,25,31]
RS2=[32,23,28,31,35,26,23,21,24,18]
BO3=[62,84,27,28,24,62,41,23,99,44]
RS3=[31,22,25,24,21,29,22,30,23,21]
BO4=[32,59,70,26,27,26,29,23,25,63]
RS4=[29,25,27,23,24,26,24,19,22,23]
BO5=[88,26,98,40,25,43,102,51,36,19]
RS5=[23,23,20,27,19,19,23,26,29,28]
BO6=[16,30,100,31,22,25,26,32,46,30]
RS6=[19,20,20,24,29,23,21,26,23,23]
BO7=[31,38,57,98,31,68,20,31,34,71]
RS7=[25,25,24,28,29,30,19,28,26,30]
BO8=[32,76,29,27,90,39,51,34,35,26]
RS8=[24,22,29,27,18,26,29,30,26,28]
BO9=[49,29,101,24,25,19,34,27,27,45]
RS9=[21,29,25,23,36,26,19,28,21,31]
BO10=[24,27,29,30,20,22,89,24,28,95]
RS10=[23,22,19,16,21,28,20,25,15,30]

BO1A=[10, 44, 6, 19, 9, 7, 8, 9, 13, 9]
RS1A=[4, 7, 8, 3, 10, 8, 9, 7, 9, 4]
BO2A=[47, 24, 11, 9, 13, 12, 12, 10, 13, 6]
RS2A=[3, 6, 8, 12, 4, 7, 10, 6, 7, 7]
BO3A=[12, 8, 7, 7, 12, 15, 9, 49, 25, 13]
RS3A=[4, 8, 5, 12, 6, 5, 7, 7, 5, 7]
BO4A=[7, 50, 33, 7, 16, 5, 22, 6, 9, 15]
RS4A=[7, 5, 5, 1, 10, 13, 7, 10, 3, 5]
BO5A=[8, 27, 7, 9, 9, 6, 10, 3, 14, 4]
RS5A= [4, 4, 2, 4, 8, 8, 11, 5, 4, 10]
BO6A=[6,3,6,14,7,9,11,20,5,6]
RS6A=[11,7,6,8,7,3,6,3,5,3]
BO7A=[13,12,9,16,14,11,10,10,8,9]
RS7A=[5,7,13,7,6,6,2,6,4,4]
BO8A=[19,8,13,12,19,6,35,17,7,7]
RS8A=[4,3,7,5,9,5,8,4,5,9]
BO9A=[11,14,42,8,3,20,6,9,51,14]
RS9A=[8,4,5,5,8,5,5,4,7,4]
BO10A=[5,31,9,16,8,14,9,4,4,8]
RS10A=[8,7,7,5,3,3,4,5,4,7]

BO1B= [68, 35, 51, 49, 22, 53, 37, 38, 40, 23]
RS1B=[28, 22, 20, 18, 18, 25, 25, 30, 26, 27]
BO2B=[42, 22, 30, 67, 45, 21, 75, 67, 86, 23]
RS2B=[17, 25, 23, 29, 32, 28, 27, 26, 25, 28]
BO3B=[99, 46, 33, 82, 95, 59, 65, 55, 37, 54]
RS3B=[26, 27, 25, 24, 30, 26, 24, 27, 30, 29]
BO4B=[64, 39, 30, 61, 28, 30, 23, 31, 23, 87]
RS4B=[22, 30, 24, 23, 23, 26, 21, 25, 21, 32]
BO5B=[57, 46, 41, 43, 23, 79, 47, 40, 39, 25]
RS5B=[23, 23, 23, 30, 27, 28, 20, 24, 22, 24]
BO6B=[19,89,77,71,35,64,50,21,69,33]
RS6B=[17,22,30,22,25,22,18,30,31,19]
BO7B=[41,93,29,103,22,28,75,27,23,102]
RS7B=[25,26,30,35,28,30,17,27,27,21]
BO8B=[25,42,98,26,67,53,42,28,29,30]
RS8B=[24,24,32,24,21,24,26,23,28,27]
BO9B=[37,89,26,24,29,29,35,34,70,40]
RS9B=[23,23,28,20,20,28,28,38,25,23]
BO10B=[72,37,69,28,35,31,32,38,26,101]
RS10B=[20,20,24,20,24,23,20,20,24,32]

"""
Mountain Car Results
BO1=[89,34,45,77,23,84,79,71,97,46]
RS1=[24, 32, 23, 24, 43, 22, 21, 29, 29, 32]
BO2=[98, 48, 101, 96, 98, 34, 75, 94, 30, 58]
RS2=[26, 24, 25, 31, 23, 23, 29, 25, 28, 22]
BO3=[21, 99, 74, 90, 101, 84, 43, 100, 99, 40]
RS3=[33, 21, 28, 28, 23, 24, 28, 21, 23, 23]
BO4=[96, 77, 101, 101, 93, 19, 44, 64, 56, 24]
RS4=[28, 26, 28, 23, 23, 27, 21, 30, 28, 23]
BO5=[101, 39, 24, 101, 59, 52, 86, 99, 91, 100]
RS5=[25, 19, 25, 22, 29, 26, 29, 29, 24, 19]
BO6=[42, 28, 99, 85, 46, 92, 99, 103, 101, 88]
RS6=[26, 27, 32, 26, 25, 27, 23, 31, 30, 25]
BO7=[63, 31, 84, 79, 102, 102, 43, 25, 58, 46]
RS7=[26, 28, 23, 26, 23, 27, 26, 40, 37, 22]
BO8=[43, 82, 26, 81, 24, 36, 54, 31, 16, 48]
RS8=[21, 26, 24, 32, 21, 28, 27, 29, 27, 24]
BO9=[40, 97, 104, 98, 46, 46, 37, 31, 48, 52]
RS9=[24, 19, 28, 23, 17, 25, 26, 28, 23, 31]
BO10=[73, 98, 95, 77, 34, 74, 55, 45, 29, 101]
RS10=[25, 24, 22, 27, 22, 27, 27, 25, 20, 25]

BO1A=[100,101,100,100,100,100,77,52,101,101]
RS1A=[15,11,9,11,15,7,11,13,9,14]
BO2A=[93,100,101,102,100,100,100,100,101,100]
RS2A=[11,9,8,15,9,14,8,13,10,6]
BO3A=[100,96,102,101,100,101,100,100,74,101]
RS3A=[13,8,16,13,14,17,13,12,13,9]
BO4A=[86,100,101,100,37,102,100,101,100,101]
RS4A=[10,8,11,7,8,12,9,13,17,16]
BO5A=[77,52,74,101,100,37,86,101,70,20]
RS5A= [12,7,15,11,10,14,10,12,11,10]
BO6A=[100,100,20,78,101,100,70,101,100,100]
RS6A=[12,13,10,9,8,15,10,9,10,8]
BO7A=[101,100,100,31,100,13,100,100,100,100]
RS7A=[18,6,10,12,14,9,9,14,11,9]
BO8A=[90,100,102,100,100,12,100,100,100,100]
RS8A=[6,11,15,12,6,11,7,9,13,13]
BO9A=[100,100,100,102,100,100,100,100,100,100]
RS9A=[7,8,12,8,13,8,12,15,10,10]
BO10A=[100,100,100,101,100,100,64,101,100,100]
RS10A=[12,7,14,8,12,15,10,11,9,14]

BO1B=[42, 61, 39, 62, 26, 35, 29, 56, 69, 52]
RS1B=[36,35,30,29,48,33,29,33,33,35]
BO2B=[42, 35, 39, 59, 24, 22, 51, 87, 57, 27]
RS2B=[29,32,32,38,29,30,35,32,37,29]
BO3B=[96, 87, 60, 43, 46, 27, 37, 97, 92, 98]
RS3B=[40,29,28,37,26,31,33,25,30,32]
BO4B=[30, 51, 70, 25, 23, 30, 72, 35, 25, 38]
RS4B=[30,33,34,27,33,31,31,38,31,33]
BO5B=[7, 32, 40, 87, 42, 52, 62, 22, 44, 7]
RS5B=[31,32,28,29,33,31,33,29,28,28]
BO6B=[19, 29, 23, 59, 69, 94, 37, 24, 78, 80]
RS6B=[33,30,37,35,34,27,33,34,37,33]
BO7B=[31, 24, 43, 57, 36, 68, 35, 21, 26, 30]
RS7B=[31,34,30,34,24,32,36,40,44,28]
BO8B=[88, 22, 39, 88, 44, 52, 45, 27, 22, 37]
RS8B=[30,44,28,35,29,30,32,35,30,24]
BO9B=[30, 53, 96, 42, 76, 48, 35, 34, 22, 95]
RS9B=[28,26,32,33,16,31,36,33,29,39]
BO10B=[35, 95, 58, 64, 17, 50, 35, 14, 27, 24]
RS10B=[34,28,25,32,29,30,37,27,28,28]
"""
"""
Drone
BO1=[28,46,24,23,29,45,21,31,54,24]
RS1=[24,24,27,18,28,26,19,19,27,19]
BO2=[29,79,2,30,28,32,55,91,60,66]
RS2=[29,24,23,20,37,22,22,22,24,25]
BO3=[59,27,46,46,32,51,80,33,32,26]
RS3=[26,32,31,23,16,24,22,30,25,26]
BO4=[8,21,57,92,24,46,28,56,91,38]
RS4=[22,24,29,20,26,18,22,25,22,26]
BO5=[33,24,30,21,34,39,80,32,26,25]
RS5=[15,20,31,29,22,24,35,24,20,30]
BO6=[27,3,59,59,28,42,28,22,5,18]
RS6=[29,25,24,26,26,32,23,27,27,21]
BO7=[28,21,24,23,41,30,34,42,22,26]
RS7=[27,29,20,20,22,23,25,32,30,18]
BO8=[57,57,27,27,31,43,37,58,29,33]
RS8=[18,22,20,28,24,29,19,21,24,34]
BO9=[34,70,88,38,28,26,33,28,43,38]
RS9=[21,23,23,29,25,25,24,24,23,27]
BO10=[30,35,36,73,86,30,24,39,36,33]
RS10=[30,32,22,18,22,32,22,23,32,24]

BO1A=[7, 19, 27, 13, 10, 20, 16, 19, 17, 3]
BO2A=[15, 12, 12, 23, 19, 23, 37, 6, 14, 19]
BO3A=[11, 20, 15, 26, 23, 21, 19, 26, 16, 24]
BO4A=[19, 26, 11, 17, 30, 19, 20, 15, 25, 20]
BO5A=[18, 35, 19, 15, 24, 18, 27, 5, 15, 16]
BO6A=[15, 13, 23, 26, 17, 20, 28, 24, 31, 10]
BO7A=[18, 20, 20, 12, 27, 11, 22, 26, 16, 14]
BO8A=[20, 17, 17, 20, 19, 14, 15, 21, 11, 8]
BO9A=[29, 17, 24, 21, 22, 23, 21, 19, 15, 26]
BO10A=[20, 22, 16, 22, 2, 10, 14, 14, 21, 7]

Glob_min_Pos_BO1A=[ObsAr([ 0.47558662, -0.49722901,  0.99622949,  1.56941632,  0.45236235]),
 ObsAr([-0.4900094 , -0.10794036,  0.40208315,  0.83963164, -1.19515229]),
 ObsAr([ 0.27414521, -0.1993606 ,  0.90582835, -0.02303769, -0.72484121]),
 ObsAr([ 0.24410239, -0.00642285,  0.45122059,  0.96760512, -0.45600321]),
 ObsAr([-0.29509126, -0.46674986,  0.99128984,  1.66817149,  1.74259866]),
 ObsAr([-0.08848861,  0.01056975,  0.77124451, -0.24107198,  0.96732987]),
 ObsAr([ 0.07855307, -0.33541375,  0.87081818,  0.80299882, -1.13322124]),
 ObsAr([-0.39644866, -0.19839055,  0.76724359,  0.60786652, -0.4669241 ]),
 ObsAr([-0.24484028, -0.47078057,  0.30276167,  1.01991999,  0.44641938]),
 ObsAr([-0.29105982, -0.41508241,  0.99188993,  1.79501403,  1.62840466])]
Glob_min_Pos_BO2A=[ObsAr([-0.28138878, -0.38577292,  0.5622608 ,  1.37367269,  0.10442363]),
 ObsAr([ 0.39180664,  0.47996472,  0.54858728,  0.44600815, -1.23194979]),
 ObsAr([ 0.43788161, -0.22504611,  0.67708799,  1.04710504,  0.86028319]),
 ObsAr([-0.37136159,  0.37186825,  0.5515994 ,  0.83554723, -0.04010878]),
 ObsAr([-0.22591354, -0.08576498,  0.43686395,  0.46239903,  0.286649  ]),
 ObsAr([-0.27217697,  0.45298142,  0.21754732,  0.50562531,  1.37539483]),
 ObsAr([-0.30884758,  0.20939899,  0.47830926, -0.14042941, -0.48665874]),
 ObsAr([-0.05137493,  0.02743418,  0.99127683,  0.18663549,  0.81577353]),
 ObsAr([ 0.44803404, -0.24080331,  0.95711249,  0.00826501, -1.19392644]),
 ObsAr([ 0.10393086,  0.14365272,  0.51043933, -0.24789097, -0.66899275])]
Glob_min_Pos_BO3A=
Glob_min_Pos_BO4A=
Glob_min_Pos_BO5A=
Glob_min_Pos_BO6A=[ObsAr([ 0.11780361, -0.36725947,  0.99029367, -0.20553551, -0.49769126]),
 ObsAr([-0.07700451, -0.19568113,  0.44335346,  0.83919728,  1.01281309]),
 ObsAr([-0.21440738,  0.44343373,  0.81858154,  0.71878964,  1.73486219]),
 ObsAr([ 0.49859483, -0.32493598,  0.66346617,  0.85411298,  0.76661684]),
 ObsAr([-0.0817803 ,  0.28209662,  0.73761185,  0.53636881, -0.99729646]),
 ObsAr([ 0.35176924, -0.40193283,  0.61340351,  0.15102435, -1.4993708 ]),
 ObsAr([ 0.15289083, -0.38827792,  0.87657294,  0.19348703,  0.95084972]),
 ObsAr([-0.46198012,  0.23735332,  0.65857992,  0.74641351,  1.07140488]),
 ObsAr([-0.42060782,  0.4853613 ,  0.48994637,  1.73103027, -1.78783824]),
 ObsAr([ 0.48039261, -0.34335839,  0.88322813, -0.99884831,  0.39901158])]
Glob_min_Pos_BO7A=
Glob_min_Pos_BO8A=
Glob_min_Pos_BO9A=
Glob_min_Pos_BO10A=[-0.47518055  0.34191738  0.21899433  0.07346946  0.70128446]
[0.17198451 0.17820328 0.75442945 0.88876885 0.57577121]
[ 0.41044255 -0.06622879  0.8350225   0.92824749  0.77271331]
[ 0.29211388  0.07825001  0.65169246  0.56339785 -1.67967013]
[-0.47242776 -0.08050306  0.89183321  1.56162307  0.04425568]
[-0.00658882  0.00537687  0.61289485  0.83764402 -0.53442917]
[ 0.01860092 -0.15333266  0.58533895  1.19034344  0.53006365]
[0.08288423 0.49998091 0.47702544 1.12719697 1.60157968]
[-0.2493947   0.023952    0.25663929  0.319934   -0.34085197]
[-0.4969948  -0.4682062   0.9263736   1.78913843 -1.53717047]

BO1B=[84, 14, 77, 54, 76, 62, 52, 12, 8, 5]
BO2B=[85, 37, 64, 53, 90, 81, 82, 62, 65, 5]
BO3B=[79, 75, 20, 80, 83, 83, 81, 26, 69, 52]
BO4B=[23, 42, 50, 55, 59, 64, 71, 21, 10, 44]
BO5B=[1, 84, 73, 7, 52, 79, 67, 71, 84, 84]
BO6B=[86, 7, 76, 78, 74, 77, 84, 4, 6, 4]
BO7B=[86, 75, 34, 61, 83, 83, 3, 55, 74, 85]
BO8B=[7, 9, 80, 66, 67, 68, 58, 13, 82, 78]
BO9B=[69, 72, 81, 52, 48, 83, 62, 77, 62, 49]
BO10B=


Glob_min_Pos_BO1B=
Glob_min_Pos_BO2B=
Glob_min_Pos_BO3B=
Glob_min_Pos_BO4B=
Glob_min_Pos_BO5B=
Glob_min_Pos_BO6B=
Glob_min_Pos_BO7B=
Glob_min_Pos_BO8B=
Glob_min_Pos_BO9B=
Glob_min_Pos_BO10B=
"""

BO1_mean=np.mean(BO1)
BO2_mean=np.mean(BO2)
BO3_mean=np.mean(BO3)
BO4_mean=np.mean(BO4)
BO5_mean=np.mean(BO5)
BO6_mean=np.mean(BO6)
BO7_mean=np.mean(BO7)
BO8_mean=np.mean(BO8)
BO9_mean=np.mean(BO9)
BO10_mean=np.mean(BO10)
BO1_std=np.std(BO1)
BO2_std=np.std(BO2)
BO3_std=np.std(BO3)
BO4_std=np.std(BO4)
BO5_std=np.std(BO5)
BO6_std=np.std(BO6)
BO7_std=np.std(BO7)
BO8_std=np.std(BO8)
BO9_std=np.std(BO9)
BO10_std=np.std(BO10)

RS1_mean=np.mean(RS1)
RS2_mean=np.mean(RS2)
RS3_mean=np.mean(RS3)
RS4_mean=np.mean(RS4)
RS5_mean=np.mean(RS5)
RS6_mean=np.mean(RS6)
RS7_mean=np.mean(RS7)
RS8_mean=np.mean(RS8)
RS9_mean=np.mean(RS9)
RS10_mean=np.mean(RS10)
RS1_std=np.std(RS1)
RS2_std=np.std(RS2)
RS3_std=np.std(RS3)
RS4_std=np.std(RS4)
RS5_std=np.std(RS5)
RS6_std=np.std(RS6)
RS7_std=np.std(RS7)
RS8_std=np.std(RS8)
RS9_std=np.std(RS9)
RS10_std=np.std(RS10)


BO1A_mean=np.mean(BO1A)
BO2A_mean=np.mean(BO2A)
BO3A_mean=np.mean(BO3A)
BO4A_mean=np.mean(BO4A)
BO5A_mean=np.mean(BO5A)
BO6A_mean=np.mean(BO6A)
BO7A_mean=np.mean(BO7A)
BO8A_mean=np.mean(BO8A)
BO9A_mean=np.mean(BO9A)
BO10A_mean=np.mean(BO10A)
BO1A_std=np.std(BO1A)
BO2A_std=np.std(BO2A)
BO3A_std=np.std(BO3A)
BO4A_std=np.std(BO4A)
BO5A_std=np.std(BO5A)
BO6A_std=np.std(BO6A)
BO7A_std=np.std(BO7A)
BO8A_std=np.std(BO8A)
BO9A_std=np.std(BO9A)
BO10A_std=np.std(BO10A)

RS1A_mean=np.mean(RS1A)
RS2A_mean=np.mean(RS2A)
RS3A_mean=np.mean(RS3A)
RS4A_mean=np.mean(RS4A)
RS5A_mean=np.mean(RS5A)
RS6A_mean=np.mean(RS6A)
RS7A_mean=np.mean(RS7A)
RS8A_mean=np.mean(RS8A)
RS9A_mean=np.mean(RS9A)
RS10A_mean=np.mean(RS10A)
RS1A_std=np.std(RS1A)
RS2A_std=np.std(RS2A)
RS3A_std=np.std(RS3A)
RS4A_std=np.std(RS4A)
RS5A_std=np.std(RS5A)
RS6A_std=np.std(RS6A)
RS7A_std=np.std(RS7A)
RS8A_std=np.std(RS8A)
RS9A_std=np.std(RS9A)
RS10A_std=np.std(RS10A)


BO1B_mean=np.mean(BO1B)
BO2B_mean=np.mean(BO2B)
BO3B_mean=np.mean(BO3B)
BO4B_mean=np.mean(BO4B)
BO5B_mean=np.mean(BO5B)
BO6B_mean=np.mean(BO6B)
BO7B_mean=np.mean(BO7B)
BO8B_mean=np.mean(BO8B)
BO9B_mean=np.mean(BO9B)
BO10B_mean=np.mean(BO10B)
BO1B_std=np.std(BO1B)
BO2B_std=np.std(BO2B)
BO3B_std=np.std(BO3B)
BO4B_std=np.std(BO4B)
BO5B_std=np.std(BO5B)
BO6B_std=np.std(BO6B)
BO7B_std=np.std(BO7B)
BO8B_std=np.std(BO8B)
BO9B_std=np.std(BO9B)
BO10B_std=np.std(BO10B)

RS1B_mean=np.mean(RS1B)
RS2B_mean=np.mean(RS2B)
RS3B_mean=np.mean(RS3B)
RS4B_mean=np.mean(RS4B)
RS5B_mean=np.mean(RS5B)
RS6B_mean=np.mean(RS6B)
RS7B_mean=np.mean(RS7B)
RS8B_mean=np.mean(RS8B)
RS9B_mean=np.mean(RS9B)
RS10B_mean=np.mean(RS10B)
RS1B_std=np.std(RS1B)
RS2B_std=np.std(RS2B)
RS3B_std=np.std(RS3B)
RS4B_std=np.std(RS4B)
RS5B_std=np.std(RS5B)
RS6B_std=np.std(RS6B)
RS7B_std=np.std(RS7B)
RS8B_std=np.std(RS8B)
RS9B_std=np.std(RS9B)
RS10B_std=np.std(RS10B)

Experiment_num=['Exp.\n1','Exp.\n2','Exp.\n3','Exp.\n4','Exp.\n5','Exp.\n6','Exp.\n7','Exp.\n8','Exp.\n9','Exp.\n10']

#X_axis=np.arange(len(Experiment_num))
X_axis=np.array([0,2,4,6,8,10,12,14,16,18])
Mean_BO=[BO1_mean,BO2_mean,BO3_mean,BO4_mean,BO5_mean,BO6_mean,BO7_mean,BO8_mean,BO9_mean,BO10_mean]
Mean_BOA=[BO1A_mean,BO2A_mean,BO3A_mean,BO4A_mean,BO5A_mean,BO6A_mean,BO7A_mean,BO8A_mean,BO9A_mean,BO10A_mean]
Mean_BOB=[BO1B_mean,BO2B_mean,BO3B_mean,BO4B_mean,BO5B_mean,BO6B_mean,BO7B_mean,BO8B_mean,BO9B_mean,BO10B_mean]

Error_BO=[BO1_std,BO2_std,BO3_std,BO4_std,BO5_std,BO6_std,BO7_std,BO8_std,BO9_std,BO10_std]
Error_BOA=[BO1A_std,BO2A_std,BO3A_std,BO4A_std,BO5A_std,BO6A_std,BO7A_std,BO8A_std,BO9A_std,BO10A_std]
Error_BOB=[BO1B_std,BO2B_std,BO3B_std,BO4B_std,BO5B_std,BO6B_std,BO7B_std,BO8B_std,BO9B_std,BO10B_std]

Mean_RS=[RS1_mean,RS2_mean,RS3_mean,RS4_mean,RS5_mean,RS6_mean,RS7_mean,RS8_mean,RS9_mean,RS10_mean]
Mean_RSA=[RS1A_mean,RS2A_mean,RS3A_mean,RS4A_mean,RS5A_mean,RS6A_mean,RS7A_mean,RS8A_mean,RS9A_mean,RS10A_mean]
Mean_RSB=[RS1B_mean,RS2B_mean,RS3B_mean,RS4B_mean,RS5B_mean,RS6B_mean,RS7B_mean,RS8B_mean,RS9B_mean,RS10B_mean]

Error_RS=[RS1_std,RS2_std,RS3_std,RS4_std,RS5_std,RS6_std,RS7_std,RS8_std,RS9_std,RS10_std]
Error_RSA=[RS1A_std,RS2A_std,RS3A_std,RS4A_std,RS5A_std,RS6A_std,RS7A_std,RS8A_std,RS9A_std,RS10A_std]
Error_RSB=[RS1B_std,RS2B_std,RS3B_std,RS4B_std,RS5B_std,RS6B_std,RS7B_std,RS8B_std,RS9B_std,RS10B_std]

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='Times New Roman', fontsize=38)
plt.setp(ax.get_xmajorticklabels(), family='Times New Roman', fontsize=38)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Experiments', family='Times New Roman', fontsize=48, weight='bold', labelpad=5)
plt.ylabel('Number of \n Counterexamples', family='Times New Roman', fontsize=48, weight='bold', labelpad=10)

#Graph of BO only
plt.bar(X_axis-0.4,Mean_BO,0.2,yerr=Error_BO,label=r'BO_Both_Specs')
plt.bar(X_axis-0.2,Mean_BOA,0.2,yerr=Error_BOA,label=r'BO_Spec_1')
plt.bar(X_axis,Mean_BOB,0.2,yerr=Error_BOB,label=r'BO_Spec_2')

#Graph of RS only
plt.bar(X_axis-0.4,Mean_RS,0.2,yerr=Error_RS,label=r'RS_Both_Specs')
plt.bar(X_axis-0.2,Mean_RSA,0.2,yerr=Error_RSA,label=r'RS_Spec_1')
plt.bar(X_axis,Mean_RSB,0.2,yerr=Error_RSB,label=r'RS_Spec_2')

#BO and RS both safety specs
plt.bar(X_axis-0.2,Mean_BO,0.2,yerr=Error_BO,label=r'Bayesian Optimization ($\mu_{1}$ and $\mu_{2}$)')
plt.bar(X_axis,Mean_RS,0.2,yerr=Error_RS,label=r'Random Sampling')

#Plot all both BO, ind, and RS

plt.bar(X_axis-0.6,Mean_BO,0.3,yerr=Error_BO,label=r'Bayesian Optimization ($\mu_{1}$ and $\mu_{2}$)')
plt.bar(X_axis-0.3,Mean_BOA,0.3,yerr=Error_BOA,label=r'Bayesian Optimization (only $\mu_{1}$)')
plt.bar(X_axis+0.0,Mean_BOB,0.3,yerr=Error_BOB,label=r'Bayesian Optimization (only $\mu_{2}$)')
plt.bar(X_axis+0.3,Mean_RS,0.3,yerr=Error_RS,label=r'Random Sampling')

plt.xticks(X_axis,Experiment_num)

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='Times New Roman', fontsize=40)
plt.ylim(0,160)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)
ax.grid(False)





#Sample Plot of STL
x=np.linspace(0,10,100)
def f1(x):
    return np.sin(x)+x+x*np.sin(x)
def f2(x):
    return .5*x*np.cos(x)+5
fig = plt.figure(figsize=(15, 10))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=25)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=25)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time [s]', family='serif', fontsize=30, weight='bold', labelpad=5)
plt.ylabel('F(t)', family='serif', fontsize=30, weight='bold', labelpad=10)

plt.plot(x,f1(x),label=r'Unsafe Trajectory',color='r',linewidth=5)
plt.plot(x,f2(x),label=r'Safe Trajectory',color='g',linewidth=5)
plt.axhline(y=10,color='k',linestyle='dashed',linewidth=5,label=r'Safety Boundary ($\mu_{1}$)')
plt.axhspan(1.5,-5,xmin=.7,xmax=1,color='orange',label='Safety Boundary ($\mu_{2}$)')
plt.axhline(y=1.5,xmin=.7,xmax=1,color='k',linestyle='dashed',linewidth=5)

leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=25)

plt.ylim(-5,25)
plt.tight_layout(pad=0.5)
ax.grid(False)
