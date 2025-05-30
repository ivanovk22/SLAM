import numpy as np
# import PlotMapSN
import matplotlib.pyplot as plt
import scipy


def PlotMapSN(Obstacles):
    # function that plots map with polygonal obstacles
    for i in range(len(Obstacles)):
        plt.fill(Obstacles[i][:,0], Obstacles[i][:,1], facecolor='lightgrey', edgecolor='black')

def land_find(ranges, angles, prom):

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
    print(len(new_peaks))

    final_peaks = []

    for m in range(len(new_peaks)):
        mp_current = ranges[new_peaks[m]]
        ma_current = angles[new_peaks[m]]
        previous_choice = next_choice = 9
        if new_peaks[m] + next_choice >= len(ranges) - 1:
            take_next = next_choice - (len(ranges) - new_peaks[m])
        else:
            take_next = new_peaks[m] + next_choice

        take_previous = new_peaks[m] - previous_choice
        print(f'curr: {new_peaks[m]}, prev: {take_previous}, next: {take_next}')
        mp_prev = ranges[take_previous]
        ma_prev = angles[take_previous]
        mp_next = ranges[take_next]
        ma_next = angles[take_next]

        # den = abs(ma_prev - ma_current) + 1e-10 #avoid dividing by zero
        den = previous_choice * np.pi / 180  # radians
        print(f'mp_current: {mp_current}, mp_prev: {mp_prev}, mp_next: {mp_next}, den: {den}')

        print(f'diff x: {abs(mp_current - mp_prev)}')
        print(f'diff y: {abs(mp_current - mp_next)}')
        diff_x = abs(mp_current - mp_prev) / den  # derivative
        diff_y = abs(mp_current - mp_next) / den  # derivative
        diff = abs(diff_x - diff_y)
        print(f'dif x = {diff_x}, dif y = {diff_y}, diff = {diff}')
        print('----------------------------------')
        if diff > 1:
            final_peaks.append(new_peaks[m])
    new_ranges = []
    new_angles = []
    for index in final_peaks:
        new_ranges.append(ranges[index])
        new_angles.append(angles[index])
    # return [new_ranges, new_angles]
    return new_peaks

data = np.load('complete_test.npz', allow_pickle=True)

Uf = data['Uf']  # measured forward velocity (odometry)
Ua = data['Ua']  # measured angular velocity (odometry)

Pose = data['PoseOdom']  # data to be used only for comparison (x(t), y(t), theta(t) of the robot)
ranges = data['noisyRangeData']
angles = data['angles']
prom = 0 #prominence


ran0 = ranges[:, 0]

for i in ran0:
    print(i)


t = 0

in_degrees = []
for a in range(len(angles)):
    in_degrees.append(np.degrees(angles[a]))

# exit()
changed = []
new_ranges = []
count_inf = 0
for c in range(len(ranges[:, t])):
    # print('r:', type(ranges[c][t]))
    if np.isinf(ranges[c][t]):
        changed.append(30)
        new_ranges.append(-1)
        count_inf += 1
    else:
        changed.append(ranges[c][t])
        new_ranges.append(ranges[c][t])


changed = np.array(changed)
new_ranges = np.array(new_ranges)
print(changed)
print("INF values num: ", count_inf)


peaks = land_find(new_ranges, angles, prom)
print(len(peaks))
plt.figure(figsize=(8, 8))
#
plt.plot(in_degrees, changed, label=r'ranges(0)$')
for x in peaks:
    plt.scatter(in_degrees[x], changed[x])


# plt.gca().invert_xaxis()




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

# radius = 10
# center_x, center_y = Pose[t, 0], Pose[t, 1]     # New center
# rotation_deg = -90 + np.degrees(Pose[t, 2])             # Rotation angle in degrees
# rotation_rad = np.radians(rotation_deg)
#
# # === GENERATE RADIUS ANGLES (0° to 180°) ===
# angles_deg = np.linspace(0, 360, 1080)
# angles_rad = np.radians(angles_deg)
#
# # === PREPARE PLOT ===
# plt.figure(figsize=(6, 6))
#
# # === DRAW 181 ROTATED RADII ===
# for angle in angles_rad:
#     # Point on the circle (before rotation)
#     x_tip = radius * np.cos(angle)
#     y_tip = radius * np.sin(angle)
#
#     # Rotate the tip
#     x_rot = x_tip * np.cos(rotation_rad) - y_tip * np.sin(rotation_rad)
#     y_rot = x_tip * np.sin(rotation_rad) + y_tip * np.cos(rotation_rad)
#
#     # Translate to new center
#     x_final = x_rot + center_x
#     y_final = y_rot + center_y
#
#     # Draw the radius from center to tip
#     plt.plot([center_x, x_final], [center_y, y_final], color='red', linewidth=0.5)
#
# # === DRAW THE HALF-CIRCLE ARC ===
# arc_theta = np.linspace(0, np.pi, 200)
# x_arc = radius * np.cos(arc_theta)
# y_arc = radius * np.sin(arc_theta)
#
# # Rotate the arc
# x_arc_rot = x_arc * np.cos(rotation_rad) - y_arc * np.sin(rotation_rad)
# y_arc_rot = x_arc * np.sin(rotation_rad) + y_arc * np.cos(rotation_rad)
#
# # Translate arc to center
# x_arc_final =  center_x
# y_arc_final =  center_y
#
# # Draw the arc
# plt.plot(x_arc_final, y_arc_final, color='blue', linewidth=2, label="Rotated Half Circle")
#
# # === PLOT SETTINGS ===
# plt.gca().set_aspect('equal', adjustable='box')
# # PlotMapSN(Obstacles)
# # plt.plot(trueMap[181*(t-1):181*t,0],trueMap[181*(t-1):181*t,1],'r.')
# plt.scatter(Pose[t, 0], Pose[t, 1])
# plt.grid(True)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title(f"181 Radii in Half-Circle\nRotated {rotation_deg}° at Center ({center_x}, {center_y})")
# plt.legend()
plt.show()