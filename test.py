import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import block_diag

plt.close('all')

d = np.load('data_point_land_1.npz', allow_pickle=True)
Meas = d['Meas']
Vtn = d['Uf']
Wtn = d['Ua']




#enviroenment
sizeenv = 27.