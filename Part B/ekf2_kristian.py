import numpy as np
import matplotlib.pyplot as plt
from functions import *
from matplotlib.patches import Ellipse

import scipy
from scipy.linalg import block_diag

plt.close('all')

data = np.load('data_point_land_2.npz', allow_pickle=True)

Meas = data['Meas']  # Landmark measurements
Uf = data['Uf']  # measured forward velocity (odometry)
Ua = data['Ua']  # measured angular velocity (odometry)
Q = data['Q']
Qturn = data['Qturn']
R = data['R']
Nland = data['Nland']  # number of Landmarks
Ts = data['Ts']
Wturn = data['wturn']  # treshold Wturn
Pose = data['Pose']  # data to be used only for comparison (x(t), y(t), theta(t) of the robot)
Landmarks = data[
    'Landmarks']  # data to be used only for comparison (ith row corresponds to i+1th of landmark locations.)

range_land = Meas.item()['range']  # landmark ranges
angle_land = Meas.item()['angle']  # landmark angles
index_land = Meas.item()['land']  # landmark indices

N = Uf.shape[0]  # Number of odometry measurements
n_upper = 3  # upper system order: x,y,theta
# upper_threshold = 14
upper_threshold = 13.8155
lower_threshold = 5.9915


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

"""
           ########### Initialize and correct for t = 0 ###########
"""

for l in range(len(range_land[0])):
    if range_land[0][l] not in landmarks_map:  # Initialize
        landmarks_map.append(len(landmarks_map))
        j = l
        mp = range_land[0][l]
        ma = angle_land[0][l]
        lx = Xp[0] + mp * np.cos(Xp[2] + ma)
        ly = Xp[1] + mp * np.sin(Xp[2] + ma)
        Xp = append_to_array(Xp, lx)
        Xp = append_to_array(Xp, ly)

# CORRECT

to_correct = []
for i in range(len(landmarks_map)):
    to_correct.append([range_land[0][i], angle_land[0][i], landmarks_map[i]])
to_correct = np.array(to_correct).T
Pp = extend_P(Pp, len(range_land[0]))

Xp, Pp = correction(Xp, Pp, to_correct, R)
X_pred = update_data(X_pred, 0, Xp)
P_pred = update_P_pred(P_pred, 0, np.diag(Pp))

i = 1

while i < N:
    """
            ########### Prediction ###########
    """
    X = Xp.copy()
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
    Q = Qturn if abs(Ua[i - 1]) > Wturn else Q_straight
    Pp = F @ Pp @ F.T + G @ Q @ G.T
    print(f'Prediction i:{i}, Xp: {Xp}')

    """
        ########### Correction ###########
    """
    for_correction = []
    number_of_new_initializations = 0
    # check the new measurements
    Xp_new = Xp.copy()
    for l in range(len(range_land[i])):
        # for each new measurements find the min d_jk
        range_m = range_land[i][l]
        angle_m = angle_land[i][l]
        d_min, index_of_landmark = max_likelihood(range_m, angle_m, Xp, Pp, R)


        # HAVE TO ASSOCIATE EACH MEASUREMENT WITH A LANDMARK INDEX

        if d_min < lower_threshold:
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

    Xp = Xp_new.copy()
    if len(for_correction) > 0:
        for_correction = np.array(for_correction).T
        # correct

        if number_of_new_initializations > 0:
            Pp = extend_P(Pp, number_of_new_initializations)

        Xp, Pp = correction(Xp, Pp, for_correction, R)


    X_pred = update_data(X_pred, i, Xp)
    P_pred = update_P_pred(P_pred, i, np.diag(Pp))
    i += 1

pose_pred = X_pred[:, :n_upper]
pose_true = Pose
landmark_pred = Xp[3:]

T = np.arange(0, N * Ts, Ts) # Time
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
fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax1.plot(pose_true[:, 0], pose_true[:, 1], label=r'$x_1(t)$')
ax1.plot(pose_pred[:, 0], pose_pred[:, 1], label=r'$\hat{x}_1(t)$', color='red', linestyle='--')
plt.legend(loc='center')

# Landmark positions
landmark_pred = landmark_pred.reshape(-1, 2)
fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)

plt.scatter(Landmarks[:, 0], Landmarks[:, 1], label='True Landmarks', color='red', marker='o', s=50, facecolors='none')

plt.scatter(landmark_pred[:, 0], landmark_pred[:, 1], label='Estimated Landmarks (Final)',
            color='green', marker='x', s=100)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Robot Trajectory and Landmarks")
plt.legend()

# Confidence intervals
plt.figure(figsize=(6, 12))
plt.subplot(311)
plt.plot(T, pose_true[:, 0] - pose_pred[:, 0], 'r', T, 3 * np.sqrt(P_pred[:, 0]), 'b--', T, -3 * np.sqrt(P_pred[:, 0]),
         'b--')
plt.ylabel(r'$x(t)-\hat{x}(t)$')
title = r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $x(t)$'
plt.title(title)
plt.axis([T[0], T[-1], -np.sqrt(P_pred[-1, 0]) * 20, np.sqrt(P_pred[-1, 0]) * 20])
plt.ylim(-0.4, 0.4)  # Set x-axis range

plt.subplot(312)
plt.plot(T, pose_true[:, 1] - pose_pred[:, 1], 'r', T, 3 * np.sqrt(P_pred[:, 1]), 'b--', T, -3 * np.sqrt(P_pred[:, 1]),
         'b--')
plt.ylabel(r'$y(t)-\hat{y}(t)$')
title = r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $y(t)$'
plt.title(title)
plt.axis([T[0], T[-1], -np.sqrt(P_pred[-1, 1]) * 20, np.sqrt(P_pred[-1, 1]) * 20])
plt.ylim(-0.5, 0.5)  # Set x-axis range

plt.subplot(313)
plt.plot(T, 180 / np.pi * np.unwrap(pose_true[:, 2] - pose_pred[:, 2]), 'r', T, 3 * 180 / np.pi * np.sqrt(P_pred[:, 2]),
         'b--', T,
         -3 * 180 / np.pi * np.sqrt(P_pred[:, 2]), 'b--')
plt.ylim(-3, 3)  # Set x-axis range

plt.xlabel('$t$')
plt.ylabel(r'$\theta(t)-\hat{\theta}(t)$')
plt.title(r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $\theta(t)$')

# Confidence ellipses
r_x = 9.21  # 99% confidence ellipse
# r_x = 20 # 99% confidence ellipse
consistent = []
radii_all = []
fig, ax = plt.subplots(figsize=(8, 8))

# sort the true landmarks corresponding to our indexes

new_landmarks = np.zeros((18, 2))
for l in range(len(landmarks_map)):
    lx = landmark_pred[l][0]
    ly = landmark_pred[l][1]
    # find the one with the smallest error
    for l2 in range(len(Landmarks)):
        if abs(lx - Landmarks[l2][0]) < 0.5 and abs(ly - Landmarks[l2][1]) < 0.5:
            new_landmarks[l] = [Landmarks[l2][0], Landmarks[l2][1]]
            break

Landmarks = new_landmarks.copy()
for j in range(Nland): #for all landmarks
    singular = False
    idx = n_upper + 2 * j
    L_true = Landmarks[j]
    L_est = X_pred[-1][idx:idx + 2]
    Pj = Pp[idx:idx + 2, idx:idx + 2]

    error = (L_true - L_est).reshape(-1, 1)
    VTV = error.T @ np.linalg.inv(Pj) @ error

    consistent.append(VTV[0][0] < r_x)

    eigenvals, eigenvecs = np.linalg.eig(Pj)

    # Sort eigenvalues and eigenvectors so largest eigenvalue comes first (major axis)
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    # Calculate ellipse angle in degrees
    angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]) * 180 / np.pi

    # Calculate axes lengths (scaled by sqrt of chi-square quantile for confidence)
    width, height = 2 * np.sqrt(eigenvals * r_x)  # factor 2 because width = 2*a, height = 2*b

    # Create ellipse patch
    ellipse = Ellipse(xy=L_est, width=width, height=height, angle=angle, edgecolor='blue', fc='None', lw=2)

    ax.add_patch(ellipse)
    # Plot landmark estimate as a point
    ax.scatter(L_est[0], L_est[1], label='Estimated Landmarks (Final)',
               color='red', marker='x')
    if j == 0:
        ax.scatter(Landmarks[:, 0], Landmarks[:, 1], label='True Landmarks', color='green', marker='o', s=40,
                   facecolors='none')
        ax.legend(['Confidence Ellipse', 'Estimated Landmarks', 'True Landmarks'])
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Landmark position estimates with 99% confidence ellipses')
ax.axis('equal')
plt.grid(True)
# plt.show()
all_good = True
for lm in range(len(consistent)):
    if consistent[lm] != np.True_:
        print(f"Landmark {lm + 1} = NOT CONSISTENT!")
        all_good = False

if all_good:
    print("All landmarks are consistent. All good!")
else:
    print("Error: One or more landmarks are inconsistent. :(")
#
plt.show()