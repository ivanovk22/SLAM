from os import times

import numpy as np
# import PlotMapSN
import matplotlib.pyplot as plt
import scipy

def PlotMapSN(Obstacles):
    # function that plots map with polygonal obstacles
    for i in range(len(Obstacles)):
        plt.fill(Obstacles[i][:,0], Obstacles[i][:,1], facecolor='lightgrey', edgecolor='black')

data = np.load('data_sim_lidar_1.npz', allow_pickle=True)
# for k in data.keys():
#     print(k)
# exit()
Uf = data['Uf']  # measured forward velocity (odometry)
Ua = data['Ua']  # measured angular velocity (odometry)
Q = data['Q']
Qturn = data['Qturn']
R = data['R']
Ts = data['Ts']
Wturn = data['wturn']  # treshold Wturn
Pose = data['Pose']  # data to be used only for comparison (x(t), y(t), theta(t) of the robot)
ranges = data['noisyRangeData']
angles = data['angles']
Obstacles = data["Obstacles"]
trueMap = data["trueMap"]
prom = 0.10 #prominence

# print(ranges[:, 0])
print(Pose[0, 0])
print(Pose[0, 1])
print(Pose[0, 2])
# print(ranges.shape)
# print(len(angles))
# exit()
in_degrees = []
for a in range(len(angles)):
    in_degrees.append(np.degrees(angles[a]))

changed_0 = []
for c in range(len(ranges[:, 0])):
    if ranges[c][0] == -1.0:
        changed_0.append(10)
    else:
        changed_0.append(ranges[c][0])
# print(changed_0)
# exit()
plt.figure(figsize=(8, 8))
plt.plot(in_degrees, changed_0, label=r'ranges(0)$')
# plt.plot(angles, label=r'$\angles(0)$', color='red', linestyle='--')
# plt.xlim(0,181)

#landmark finder
def land_find(ranges,angles,prom):
    index_range = []
    new_ranges = np.zeros(181)
    print('dsfsfd',new_ranges.shape)
    print('ranges', ranges)
    peaks, prominence = scipy.signal.find_peaks(-ranges, prominence=prom)
    for c in range(len(angles)):
        if ranges[c] >= 0:
            index_range.append(1)
        else:
            index_range.append(0)
    print('index',index_range)
    print('peaks',peaks)
    new_peaks2 = []
    for p in peaks:
        print('p', p)
        if index_range[p] == 1:
            new_peaks2.append(int(p))

    print(new_peaks2, prominence)

for t in range(len(ranges[0])):
    land_find(ranges[:,t], angles, prom)
exit()
plt.figure()
PlotMapSN(Obstacles)
plt.plot(trueMap[:,0],trueMap[:,1],'r.')
plt.show()