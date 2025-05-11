import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import block_diag

plt.close('all')

data = np.load('data_point_land_1.npz', allow_pickle=True)

#print(data.files)

Meas = data['Meas']
Uf = data['Uf']
Ua = data['Ua']
Q = data['Q']
Qturn = data['Qturn']
R = data['R']
Nland = data['Nland']
Ts = data['Ts']
wturn = data['wturn']
Pose = data['Pose']
Landmarks = data['Landmarks']

