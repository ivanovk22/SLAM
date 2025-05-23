import numpy as np
from PlotMapSN import *
import matplotlib.pyplot as plt

d = np.load('data_sim_lidar_1.npz', allow_pickle=True)

Uf = d['Uf']  # measured forward velocity (odometry)
Ua = d['Ua']  # measured angular velocity (odometry)
Obstacles = d['Obstacles']
trueMap = d['trueMap']

plt.figure()
PlotMapSN(Obstacles)
plt.show()