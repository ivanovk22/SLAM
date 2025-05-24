import numpy as np
# import PlotMapSN
import matplotlib.pyplot as plt
import scipy


def PlotMapSN(Obstacles):
    # function that plots map with polygonal obstacles
    for i in range(len(Obstacles)):
        plt.fill(Obstacles[i][:,0], Obstacles[i][:,1], facecolor='lightgrey', edgecolor='black')

def land_find(ranges,angles,prom, robot_x, robot_y, robot_theta):
    index_range = []
    peaks, prominence = scipy.signal.find_peaks(-ranges, prominence=prom)
    for c in range(len(angles)):
        if ranges[c] >= 0:
            index_range.append(1)
        else:
            index_range.append(0)
    new_peaks = []
    for p in peaks:
        if index_range[p] == 1:
            new_peaks.append(int(p))
    print(new_peaks)
    # plt.figure(figsize=(8, 8))
    # x_arr = []
    # y_arr = []
    # for l in range(new_peaks[m] - 3, new_peaks[m]+4):
    #     mp = ranges[l]
    #     ma = angles[l]
    #     lx = robot_x + mp * np.cos(robot_theta + ma)
    #     ly = robot_y + mp * np.sin(robot_theta + ma)
    #     x_arr.append(lx)
    #     y_arr.append(ly)
    #     if l == new_peaks[m]:
    #         print(f'min index:{l}, x = {lx}, y = {ly}')
    #         plt.scatter(lx, ly, c='blue', marker='o', s=50)
    #     else:
    #         print(f'index:{l}, x = {lx}, y = {ly}')
    # plt.plot(x_arr, y_arr)
    final_peaks = []
    for m in range(len(new_peaks)):
        previous_choice = 3
        take_previous = new_peaks[m] - previous_choice
        mp_prev = ranges[take_previous]
        ma_prev = angles[take_previous]
        lx_prev = robot_x + mp_prev * np.cos(robot_theta + ma_prev)
        ly_prev = robot_y + mp_prev * np.sin(robot_theta + ma_prev)
        take_next = new_peaks[m] + previous_choice
        mp_next = ranges[take_next]
        ma_next = angles[take_next]
        lx_next = robot_x + mp_next * np.cos(robot_theta + ma_next)
        ly_next = robot_y + ma_next * np.sin(robot_theta + ma_next)
        diff_x = abs(lx_prev - lx_next)
        diff_y = abs(ly_prev - ly_next)
        if diff_x > 0.5 or diff_y > 0.5:
            final_peaks.append(new_peaks[m])
        # print(f'index prev: {take_previous}, x prev = {lx_prev}, y prev = {ly_prev}')
        # print(f'index next {take_next}, x next = {lx_next}, y next = {ly_next}')
        print(f'diff x = {diff_x}, diff y = {diff_y}')
    new_ranges = []
    new_angles = []
    for index in final_peaks:
        new_ranges.append(ranges[index])
        new_angles.append(angles[index])
    print(new_ranges)
    print(new_angles)
    return new_peaks

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

for p in Pose:
    # if p[2] < 0:
    print(np.degrees(p[2]))
exit()
# print(ranges[:, 0])
t = 108
print(Pose[t, 0])
print(Pose[t, 1])
print(Pose[t, 2])


print('---------------')
# print(ranges.shape)
# print(len(angles))
# exit()
in_degrees = []
for a in range(len(angles)):
    in_degrees.append(np.degrees(angles[a]))

# exit()
changed = []
for c in range(len(ranges[:, t])):
    if ranges[c][t] == -1.0:
        changed.append(10)
    else:
        changed.append(ranges[c][t])
radius = 10
theta = np.linspace(0, np.pi, 100)  # Angle from 0 to π (upper half)

# Circle coordinates
# x = radius * np.cos(theta)
# y = radius * np.sin(theta)
peaks = land_find(ranges[:, t], angles, prom, Pose[t, 0], Pose[t, 1], Pose[t, 2])

plt.figure(figsize=(8, 8))
#
plt.plot(in_degrees, changed, label=r'ranges(0)$')
for x in peaks:
    plt.scatter(in_degrees[x], changed[x])


plt.gca().invert_xaxis()




# # Parameters
# radius = 10
# theta = np.linspace(0, np.pi, 100)  # Upper half-circle
# x = radius * np.cos(theta)
# y = radius * np.sin(theta)
#
# # Rotation
# angle_deg = -90 + np.degrees(Pose[t, 2])
# angle_rad = np.radians(angle_deg)
# cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
# x_rot = x * cos_a - y * sin_a
# y_rot = x * sin_a + y * cos_a

# # Translation (change center)
# x_center, y_center = Pose[t, 0], Pose[t, 1]
# x_trans = x_rot + x_center
# y_trans = y_rot + y_center
#
#
#
# # Plot
# plt.figure(figsize=(6,6))
# plt.plot(x_trans, y_trans)
# plt.gca().set_aspect('equal', adjustable='box')
# PlotMapSN(Obstacles)
# plt.plot(trueMap[181*(t-1):181*t,0],trueMap[181*(t-1):181*t,1],'r.')
# plt.scatter(Pose[t, 0], Pose[t, 1])
#
# plt.title(f"Rotated Half Circle Centered at ({x_center}, {y_center})")
# plt.grid(True)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

radius = 10
center_x, center_y = Pose[t, 0], Pose[t, 1]     # New center
rotation_deg = -90 + np.degrees(Pose[t, 2])             # Rotation angle in degrees
rotation_rad = np.radians(rotation_deg)

# === GENERATE RADIUS ANGLES (0° to 180°) ===
angles_deg = np.linspace(0, 180, 181)
angles_rad = np.radians(angles_deg)

# === PREPARE PLOT ===
plt.figure(figsize=(6, 6))

# === DRAW 181 ROTATED RADII ===
for angle in angles_rad:
    # Point on the circle (before rotation)
    x_tip = radius * np.cos(angle)
    y_tip = radius * np.sin(angle)

    # Rotate the tip
    x_rot = x_tip * np.cos(rotation_rad) - y_tip * np.sin(rotation_rad)
    y_rot = x_tip * np.sin(rotation_rad) + y_tip * np.cos(rotation_rad)

    # Translate to new center
    x_final = x_rot + center_x
    y_final = y_rot + center_y

    # Draw the radius from center to tip
    plt.plot([center_x, x_final], [center_y, y_final], color='red', linewidth=0.5)

# === DRAW THE HALF-CIRCLE ARC ===
arc_theta = np.linspace(0, np.pi, 200)
x_arc = radius * np.cos(arc_theta)
y_arc = radius * np.sin(arc_theta)

# Rotate the arc
x_arc_rot = x_arc * np.cos(rotation_rad) - y_arc * np.sin(rotation_rad)
y_arc_rot = x_arc * np.sin(rotation_rad) + y_arc * np.cos(rotation_rad)

# Translate arc to center
x_arc_final = x_arc_rot + center_x
y_arc_final = y_arc_rot + center_y

# Draw the arc
plt.plot(x_arc_final, y_arc_final, color='blue', linewidth=2, label="Rotated Half Circle")

# === PLOT SETTINGS ===
plt.gca().set_aspect('equal', adjustable='box')
PlotMapSN(Obstacles)
plt.plot(trueMap[181*(t-1):181*t,0],trueMap[181*(t-1):181*t,1],'r.')
plt.scatter(Pose[t, 0], Pose[t, 1])
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"181 Radii in Half-Circle\nRotated {rotation_deg}° at Center ({center_x}, {center_y})")
plt.legend()
plt.show()