import numpy as np
import matplotlib.pyplot as plt
#from ruamel.yaml.timestamp import TimeStamp

from functions import *
from matplotlib.patches import Ellipse
import scipy
from scipy.linalg import block_diag
import sys

plt.close('all')

data = np.load('complete_test.npz', allow_pickle=True)
Uf = data['Uf']  # measured forward velocity (odometry)
Ua = data['Ua']  # measured angular velocity (odometry)
time_stamps = data['TimeStamp']
Pose = data['PoseOdom']  # data to be used only for comparison (x(t), y(t), theta(t) of the robot)
ranges = data['noisyRangeData']
angles = data['angles']
wheelvels_readings = data['wheelvels_readings']
wheelvels_times = data['wheelvels_times']
counter = 0


angle_res = angles[1]-angles[0] #Angle resolution of lidar
k = 3 #3 sigma limit
sigma_R_angle = (angle_res/2)/k # Error is within k*sigma => sigma = Error/k
Q = np.array([[1.0000000e-10, 0.0000000e+00],
              [0.0000000e+00, 3.0461742e-10]])

R = np.array([[1.00000000e-8, 0.00000000e+00],
              [0.00000000e+00, sigma_R_angle**2]])


# Q = np.array([[ 2.38468441e+00, -1.12586708e-19],
#               [-1.12586708e-19,  1.59612223e-34]])







def find_inf_intervals(column, pad=3):
    length = len(column)
    is_inf = np.isinf(column)
    intervals = []

    start = None
    for i, val in enumerate(is_inf):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i - 1
            padded_start = (start - pad) % length
            padded_end = (end + pad) % length
            intervals.append((padded_start, padded_end))
            start = None
    if start is not None:
        end = length - 1
        padded_start = (start - pad) % length
        padded_end = (end + pad) % length
        intervals.append((padded_start, padded_end))
    return intervals

def expand_wrapped_intervals(intervals, length=1080):
    indices = set()
    for start, end in intervals:
        if start <= end:
            # Normal case: just range from start to end
            indices.update(range(start, end + 1))
        else:
            # Wrapped case: from start to end of array, then 0 to end
            indices.update(range(start, length))
            indices.update(range(0, end + 1))
    return indices
# Iterate over all columns and print inf intervals
# max_dist = 0
# for col in range(ranges.shape[1]):
#     # print(f'-------------{col}----------------')
#     column = ranges[:, col]
#     inf_intervals = find_inf_intervals(column)
#     if inf_intervals:
#         indices = expand_wrapped_intervals(inf_intervals)
#         # for interval in inf_intervals:
#         #     dist = interval[1] - interval[0]
#         #     if dist > max_dist:
#         #         max_dist = dist
#         print(f"Column {col}: {inf_intervals}")
#         print(f'indices: {indices}')
#
#     if col == 3:
#         break
# print(max_dist)
# exit()
# decrease_data = 2
# Uf = Uf[::decrease_data]
# Ua = Ua[::decrease_data]
# time_stamps = time_stamps[::decrease_data]
# Pose = Pose[::decrease_data]
# ranges=ranges[:, ::decrease_data]

# for t in time_stamps:
#     print(t)
# print('Pose:', Pose.shape)
# print('ranges:', ranges.shape)
# print('angles:', angles.shape)
# exit()
# fl = ranges.flatten()
# new_arr = fl[np.isfinite(fl)]
#
# max_val = np.max(new_arr)

# for i in range(len(ranges[:, 0])):
#     print(f"i = {i}, range: {ranges[:, 0][i]}, angles: {np.degrees(angles[i])}")
#
# exit()
# fig = plt.figure()
# plt.plot(Pose[:, 0][:800], Pose[:, 1][:800], label=r'$x_1(t)$')
# plt.show()
# exit()

N = Uf.shape[0]  # Number of odometry measurements
n_upper = 3  # upper system order: x,y,theta

# upper_threshold = 50
# lower_threshold = 5
"""-------------------------------PROMINENCE HERE ----------------------------------"""
prom = (0.1, 0.8) #prominence

x0 = np.array([])  # initial states
x0 = append_to_array(x0, Pose[0, 0])
x0 = append_to_array(x0, Pose[0, 1])
x0 = append_to_array(x0, Pose[0, 2])

# upper covariance
lambda_ = 1e-6
P_upper = lambda_ * np.eye(n_upper)  # uncertainty of x,y,theta
# eta = 1000
P0 = P_upper
Xp = x0.copy()
Pp = P0.copy()
X_pred = np.empty((N, 0)) # as beginning 220x0
P_pred = np.empty((N, 0)) # as beginning 220x0
Q_straight = Q.copy()
landmarks_map = []
# for row in len(ranges[0])


def land_find(ranges, angles, prom):
    mask_finite = np.isfinite(ranges)
    ranges_new = ranges[mask_finite]
    angles_new = angles[mask_finite]

    peaks, prominence = scipy.signal.find_peaks(-ranges_new, prominence=prom)
    final_peaks = []
    for m in range(len(peaks)):
        mp_current = ranges_new[peaks[m]]
        ma_current = angles_new[peaks[m]]
        previous_choice = 1
        if peaks[m] + previous_choice >= len(ranges) - 1:
            previous_choice = len(ranges) - peaks[m] - 1
        elif peaks[m] - previous_choice < 0:
            previous_choice = peaks[m]


        take_previous = peaks[m] - previous_choice
        mp_prev = ranges[take_previous]
        ma_prev = angles[take_previous]
        take_next = peaks[m] + previous_choice
        mp_next = ranges[take_next]
        ma_next = angles[take_next]

        avg_range = (mp_prev + mp_next) / 2
        diff_range = ranges[peaks[m]] -avg_range
        #print('diffs', diff_range)
        if diff_range < -1 and diff_range > -50.0:
            final_peaks.append(peaks[m])
            print('passed diffS', diff_range)

    new_ranges = []
    new_angles = []
    for index in final_peaks:
        new_ranges.append(ranges_new[index])
        new_angles.append(angles_new[index])
    print(new_ranges)
    print(new_angles)
    return [new_ranges, new_angles]

"""
           ########### Initialize and correct for t = 0 ###########
"""

# new_ranges = np.empty((ranges.shape[0], ranges.shape[1]))
# left = 0


# METERS_THRESHOLD = 3
# for row in range(ranges.shape[0]):
#     for col in range(ranges.shape[1]):
#         if ranges[row][col] > METERS_THRESHOLD:
#             new_ranges[row][col] = np.inf
#         else:
#             new_ranges[row][col] = ranges[row][col]
#             left += 1
# print('left', left)

# ranges = new_ranges.copy()
range_land, angle_land = land_find(ranges[:, 0], angles, prom)

for l in range(len(range_land)):
    if range_land[l] not in landmarks_map:  # Initialize
        landmarks_map.append(len(landmarks_map))
        j = l
        mp = range_land[l]
        ma = angle_land[l]
        lx = Xp[0] + mp * np.cos(Xp[2] + ma)
        ly = Xp[1] + mp * np.sin(Xp[2] + ma)
        Xp = append_to_array(Xp, lx)
        Xp = append_to_array(Xp, ly)

# CORRECT

to_correct = []
for i in range(len(landmarks_map)):
    to_correct.append([range_land[i], angle_land[i], landmarks_map[i]])
to_correct = np.array(to_correct).T
Pp = extend_P(Pp, len(range_land))
if len(to_correct) > 0:
    Xp, Pp = correction(Xp, Pp, to_correct, R)
X_pred = update_data(X_pred, 0, Xp)
P_pred = update_P_pred(P_pred, 0, np.diag(Pp))

i = 1
ts_prev = time_stamps[0]
# print(f'Xp initial len: {len(Xp)}, Xp: {Xp}')
to_check = N
count_old = 0
while i < to_check:
    print(f'i = {i}, Xp = {len(Xp)}')

    """
            ########### Prediction ###########
    """
    X = Xp.copy()
    Ts = time_stamps[i] - ts_prev
    n_lower = len(X) - 3
    Xp = X + np.concatenate((Ts * np.array([Uf[i - 1] * np.cos(X[2]), Uf[i - 1] * np.sin(X[2]), Ua[i - 1]]), [0]*(len(X) - 3)))
    dUpper_dx = np.array([[1, 0, -Ts * Uf[i - 1] * np.sin(X[2])],
                          [0, 1, Ts * Uf[i - 1] * np.cos(X[2])],
                          [0, 0, 1]])
    dLower_dx = np.eye(n_lower)
    lower_zeros = np.zeros((n_lower, n_upper))
    upper_zeros = np.zeros((n_upper, n_lower))

    F = np.block([[dUpper_dx, upper_zeros],
                  [lower_zeros, dLower_dx]])

    dupper_dw = np.array([[-Ts * np.cos(X[2]), 0],
                          [-Ts * np.sin(X[2]), 0],
                          [0, -Ts]])
    dlower_dw = np.zeros((n_lower, 2))
    G = np.block([[dupper_dw],
                  [dlower_dw]])
    # Q = Qturn if abs(Ua[i - 1]) > Wturn else Q_straight
    Pp = F @ Pp @ F.T + G @ Q @ G.T


    """
        ########### Correction ###########
    """
    for_correction = []
    number_of_new_initializations = 0
    range_land, angle_land = land_find(ranges[:, i], angles, prom)

    # check the new measurements
    Xp_new = Xp.copy()
    upper_threshold = 50
    lower_threshold = 15
    for l in range(len(range_land)):
        # for each new measurements find the min d_jk
        range_m = range_land[l]
        angle_m = angle_land[l]
        d_min, index_of_landmark = max_likelihood(range_m, angle_m, Xp, Pp, R)
        print('d_min', d_min)
        # print(f'd_min: {d_min}')

        # HAVE TO ASSOCIATE EACH MEASUREMENT WITH A LANDMARK INDEX
        if d_min < lower_threshold:
            # print(f'OLD ONE, numer: {count_old}')
            # old one make a vector [range, angle, corresponding landmark index]
            for_correction.append([range_m, angle_m, index_of_landmark])

        if d_min > upper_threshold:

            # new one( initialize )
            landmarks_map.append(len(landmarks_map))
            number_of_new_initializations += 1
            # add new landmark in Xp_new

            lx = Xp_new[0] + range_m * np.cos(Xp_new[2] + angle_m)
            ly = Xp_new[1] + range_m * np.sin(Xp_new[2] + angle_m)
            Xp_new = append_to_array(Xp_new, lx)
            Xp_new = append_to_array(Xp_new, ly)
            for_correction.append([range_m, angle_m, len(landmarks_map) - 1])

        # if d_min > lower_threshold and d_min < upper_threshold:
        #     count_old += 1
        #     print('IN Between: ', count_old)


    Xp = Xp_new.copy()
    # print(f'Xp before correction: {Xp},')
    if len(for_correction) > 0:
        for_correction = np.array(for_correction).T
        # correct
        if number_of_new_initializations > 0:
            Pp = extend_P(Pp, number_of_new_initializations)

        Xp, Pp = correction(Xp, Pp, for_correction, R)

    # print(f'Xp after correction: {Xp},')

    X_pred = update_data(X_pred, i, Xp)
    P_pred = update_P_pred(P_pred, i, np.diag(Pp))
    ts_prev = time_stamps[i]
    # print('----------------------------')

    i += 1


pose_pred = X_pred[:, :n_upper][:to_check]
pose_true = Pose[:to_check]
# print('true pose:', pose_true)
# print('predicted pose', pose_pred)
# exit()
landmark_pred = Xp[3:]

# T = np.arange(0, N * Ts, Ts) # Time
T = time_stamps.copy() # Time
T = T[:to_check]

true_meas = []
pred_meas = []
for p in range(to_check):
    # print('p', p)
    ranges_t = ranges[:, p]

    for meas_idx in range(len(ranges_t)):
        if np.isinf(ranges_t[meas_idx]) == False:

            true_r_x = pose_true[p][0]
            true_r_y = pose_true[p][1]
            true_r_theta = pose_true[p][2]

            pred_r_x = pose_pred[p][0]
            pred_r_y = pose_pred[p][1]
            pred_r_theta = pose_pred[p][2]

            mp = ranges_t[meas_idx]
            ma = angles[meas_idx]


            true_l_x = true_r_x + mp * np.cos(true_r_theta + ma)
            true_l_y = true_r_y + mp * np.sin(true_r_theta + ma)
            true_meas.append([true_l_x, true_l_y])
            #
            pred_l_x = pred_r_x + mp * np.cos(pred_r_theta + ma)
            pred_l_y = pred_r_y + mp * np.sin(pred_r_theta + ma)
            pred_meas.append([pred_l_x, pred_l_y])


true_meas = np.array(true_meas)
pred_meas = np.array(pred_meas)
#
# print('true shape', true_meas.shape)
# print('pred shape', pred_meas.shape)












# Plots and comparisons
fig = plt.figure()
ax1 = plt.subplot(3, 1, 1)
ax1.plot(T, pose_true[:, 0], label=r'$x_1(t)$')
ax1.plot(T, pose_pred[:, 0], label=r'$\hat{x}_1(t)$', color='red', linestyle='--')
plt.xlabel("$t$")
plt.ylabel("$x_1(t)$")
plt.title("Velocity $x_1(t)$")
plt.legend(loc='upper right')

ax1 = plt.subplot(3, 1, 2)
ax1.plot(T, pose_true[:, 1], label=r'$x_2(t)$')
ax1.plot(T, pose_pred[:, 1], label=r'$\hat{x}_2(t)$', color='red', linestyle='--')
plt.xlabel("$t$")
plt.ylabel("$x_2(t)$")
plt.title("Current $x_2(t)$")
plt.legend(loc='upper right')

ax1 = plt.subplot(3, 1, 3)
ax1.plot(T, pose_true[:, 2], label=r'$\theta(t)$')
ax1.plot(T, pose_pred[:, 2], label=r'$\hat{\theta}(t)$', color='red', linestyle='--')
plt.xlabel("$t$ (s)")
plt.ylabel(r"$\theta(t)$ (rad)")
plt.title("Robot Orientation")
plt.legend(loc='upper right')
# plt.grid(True)
fig.tight_layout()


# Trajectory
# fig = plt.figure()
# ax1 = plt.subplot(2, 1, 1)
# ax1.plot(pose_true[:, 0], pose_true[:, 1], label=r'$x_1(t)$')
# ax1.plot(pose_pred[:, 0], pose_pred[:, 1], label=r'$\hat{x}_1(t)$', color='red', linestyle='--')
# plt.legend(loc='center')

# Landmark positions
landmark_pred = landmark_pred.reshape(-1, 2)
print('num landmarks:', len(landmark_pred))
print('shape', landmark_pred.shape)
# fig = plt.figure()
# # ax1 = plt.subplot(2, 1, 1)
# #
# plt.scatter(landmark_pred[:, 0], landmark_pred[:, 1], label='Estimated Landmarks (Final)',
#             color='green', marker='o', s=10)


#
# plt.plot(pose_true[:, 0], pose_true[:, 1], label=r'$x_1(t)$')
# plt.plot(pose_pred[:, 0], pose_pred[:, 1], label=r'$\hat{x}_1(t)$', color='red', linestyle='--')
# # plt.plot(trueMap[:,0],trueMap[:,1],'r.')
#
# plt.xlabel("X position (m)")
# plt.ylabel("Y position (m)")
# plt.title("Robot Trajectory and Landmarks")
# plt.legend()


fig = plt.figure()

plt.scatter(true_meas[:, 0], true_meas[:, 1], label='True Meas',
            color='green', marker='o', s=0.3)

plt.scatter(landmark_pred[:, 0], landmark_pred[:, 1], label='Estimated Landmarks (Final)',
            color='red', marker='o', s=5)

fig = plt.figure()
plt.scatter(pred_meas[:, 0], pred_meas[:, 1], label='Pred Meas',
            color='blue', marker='o', s=0.3)

plt.scatter(landmark_pred[:, 0], landmark_pred[:, 1], label='Estimated Landmarks (Final)',
            color='red', marker='o', s=5)


#Confidence intervals
# plt.figure(figsize=(6, 8))
# plt.subplot(311)
# plt.plot(T, pose_true[:, 0] - pose_pred[:, 0], 'r', T, 3 * np.sqrt(P_pred[:, 0]), 'b--', T, -3 * np.sqrt(P_pred[:, 0]),
#          'b--')
# plt.ylabel(r'$x(t)-\hat{x}(t)$')
# title = r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $x(t)$'
# plt.title(title)
# plt.axis([T[0], T[-1], -np.sqrt(P_pred[-1, 0]) * 20, np.sqrt(P_pred[-1, 0]) * 20])
# #
# plt.subplot(312)
# plt.plot(T, pose_true[:, 1] - pose_pred[:, 1], 'r', T, 3 * np.sqrt(P_pred[:, 1]), 'b--', T, -3 * np.sqrt(P_pred[:, 1]),
#          'b--')
# plt.ylabel(r'$y(t)-\hat{y}(t)$')
# title = r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $y(t)$'
# plt.title(title)
# plt.axis([T[0], T[-1], -np.sqrt(P_pred[-1, 1]) * 20, np.sqrt(P_pred[-1, 1]) * 20])
# #
# plt.subplot(313)
# plt.plot(T, 180 / np.pi * np.unwrap(pose_true[:, 2] - pose_pred[:, 2]), 'r', T, 3 * 180 / np.pi * np.sqrt(P_pred[:, 2]),
#          'b--', T,
#          -3 * 180 / np.pi * np.sqrt(P_pred[:, 2]), 'b--')
# plt.xlabel('$t$')
# plt.ylabel(r'$\theta(t)-\hat{\theta}(t)$')
# plt.title(r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $\theta(t)$')

# # Confidence ellipses
# r_x = 9.21  # 99% confidence ellipse
# # r_x = 20 # 99% confidence ellipse
# consistent = []
# radii_all = []
# fig, ax = plt.subplots(figsize=(8, 8))
#
# # sort the true landmarks corresponding to our indexes
#
# new_landmarks = np.zeros((18, 2))
# for l in range(len(landmarks_map)):
#     lx = landmark_pred[l][0]
#     ly = landmark_pred[l][1]
#     # find the one with the smallest error
#     for l2 in range(len(Landmarks)):
#         if abs(lx - Landmarks[l2][0]) < 0.5 and abs(ly - Landmarks[l2][1]) < 0.5:
#             new_landmarks[l] = [Landmarks[l2][0], Landmarks[l2][1]]
#             break
#
# Landmarks = new_landmarks.copy()
# for j in range(Nland): #for all landmarks
#     singular = False
#     idx = n_upper + 2 * j
#     L_true = Landmarks[j]
#     L_est = X_pred[-1][idx:idx + 2]
#     Pj = Pp[idx:idx + 2, idx:idx + 2]
#
#     error = (L_true - L_est).reshape(-1, 1)
#     VTV = error.T @ np.linalg.inv(Pj) @ error
#
#     consistent.append(VTV[0][0] < r_x)
#
#     eigenvals, eigenvecs = np.linalg.eig(Pj)
#
#     # Sort eigenvalues and eigenvectors so largest eigenvalue comes first (major axis)
#     order = eigenvals.argsort()[::-1]
#     eigenvals = eigenvals[order]
#     eigenvecs = eigenvecs[:, order]
#
#     # Calculate ellipse angle in degrees
#     angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]) * 180 / np.pi
#
#     # Calculate axes lengths (scaled by sqrt of chi-square quantile for confidence)
#     width, height = 2 * np.sqrt(eigenvals * r_x)  # factor 2 because width = 2*a, height = 2*b
#
#     # Create ellipse patch
#     ellipse = Ellipse(xy=L_est, width=width, height=height, angle=angle, edgecolor='blue', fc='None', lw=2)
#
#     ax.add_patch(ellipse)
#     # Plot landmark estimate as a point
#     ax.scatter(L_est[0], L_est[1], label='Estimated Landmarks (Final)',
#                color='red', marker='x')
#     if j == 0:
#         ax.scatter(Landmarks[:, 0], Landmarks[:, 1], label='True Landmarks', color='green', marker='o', s=40,
#                    facecolors='none')
#         ax.legend(['Confidence Ellipse', 'Estimated Landmarks', 'True Landmarks'])
# ax.set_xlabel('X position')
# ax.set_ylabel('Y position')
# ax.set_title('Landmark position estimates with 99% confidence ellipses')
# ax.axis('equal')
# plt.grid(True)
# # plt.show()
# all_good = True
# for lm in range(len(consistent)):
#     if consistent[lm] != np.True_:
#         print(f"Landmark {lm + 1} = NOT CONSISTENT!")
#         all_good = False
#
# if all_good:
#     print("All landmarks are consistent. All good!")
# else:
#     print("Error: One or more landmarks are inconsistent. :(")
# #
plt.show()