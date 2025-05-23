import numpy as np
# import PlotMapSN
import matplotlib.pyplot as plt


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

# print(ranges[:, 0])
print(Pose[0, 0])
print(Pose[0, 1])
print(Pose[0, 2])
print(Pose[1, 0])
print(Pose[1, 1])

print('---------------')
# print(ranges.shape)
# print(len(angles))
# exit()
in_degrees = []
for a in range(len(angles)):
    in_degrees.append(np.degrees(angles[a]))
print(angles)
print(in_degrees)
# exit()
changed_0 = []
for c in range(len(ranges[:, 0])):
    if ranges[c][0] == -1.0:
        changed_0.append(10)
    else:
        changed_0.append(ranges[c][0])
radius = 10
theta = np.linspace(0, np.pi, 100)  # Angle from 0 to π (upper half)
print(changed_0)
# Circle coordinates
# x = radius * np.cos(theta)
# y = radius * np.sin(theta)
plt.figure(figsize=(8, 8))
#
plt.plot(in_degrees, changed_0, label=r'ranges(0)$')
plt.gca().invert_xaxis()




# Parameters
radius = 10
theta = np.linspace(0, np.pi, 100)  # Upper half-circle
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Rotation
angle_deg = -90
angle_rad = np.radians(angle_deg)
cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
x_rot = x * cos_a - y * sin_a
y_rot = x * sin_a + y * cos_a

# Translation (change center)
x_center, y_center = 17, 30
x_trans = x_rot + x_center
y_trans = y_rot + y_center

# Plot
plt.figure(figsize=(6,6))
plt.plot(x_trans, y_trans)
plt.gca().set_aspect('equal', adjustable='box')
PlotMapSN(Obstacles)
plt.plot(trueMap[:,0],trueMap[:,1],'r.')
plt.scatter(17, 30)
plt.title(f"Rotated Half Circle Centered at ({x_center}, {y_center})")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
