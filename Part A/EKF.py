import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import scipy
from scipy.linalg import block_diag

plt.close('all')
save_fig = False

dataset = 'data_point_land_1.npz'
data = np.load(dataset, allow_pickle=True)

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
n_lower = Nland.item() * 2  # lower system order: 2 for every landmark (x,y)
n = n_upper + n_lower  # system order

x0 = np.zeros(n)  # initial states
x0[0] = Pose[0, 0]  # x(0)
x0[1] = Pose[0, 1]  # y(0)
x0[2] = Pose[0, 2]  # theta(0)
# upper covariance
lambda_ = 1e-6
P_upper = lambda_ * np.eye(n_upper)  # uncertainty of x,y,theta
upper_zeros = np.zeros((n_upper, n_lower))
# lower covariance
eta = 1000
P_lower = eta * np.eye(n_lower)  # uncertainty landmarks
lower_zeros = np.zeros((n_lower, n_upper))
# Initial covariance
P0 = np.block([[P_upper, upper_zeros],
               [lower_zeros, P_lower]])


def angle_wrap(angle):
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle


def innovation(M, xp):
    landmarks = M[:, 2]

    # take the predictions
    predicted_meas = np.zeros((len(landmarks), 2))

    for i in range(len(landmarks)):
        j = int(landmarks[i] - 1)

        lx = xp[n_upper + 2 * j]
        ly = xp[n_upper + 2 * j + 1]

        predicted_p = np.sqrt((lx - xp[0]) ** 2 + (ly - xp[1]) ** 2)
        predicted_alpha = angle_wrap(np.arctan2(ly - xp[1], lx - xp[0]) - xp[2])
        predicted_meas[i][0] = predicted_p
        predicted_meas[i][1] = predicted_alpha

    inn = M[:, :2] - predicted_meas
    return inn.reshape(-1, 1)


def correction(xp, pp, M):

    ranges = M[0]
    angles = M[1]
    indexes = M[2]
    H = np.zeros((len(indexes) * 2, n))
    for k in range(len(indexes)):
        # landmarks are from 1, so when we see landmark 1 it means that it will
        # be on the 0 position after x, y, theta - Lx1 = 3, Ly1 = 4
        j = int(indexes[k]) - 1
        lx = xp[n_upper + 2 * j]
        ly = xp[n_upper + 2 * j + 1]

        row_p = k * 2
        row_alpha = k * 2 + 1
        # derivatives of m_p
        p_den = np.sqrt((lx - xp[0]) ** 2 + (ly - xp[1]) ** 2)

        H[row_p][0] = (xp[0] - lx) / p_den  # dx
        H[row_p][1] = (xp[1] - ly) / p_den  # dy
        H[row_p][2] = 0  # dtheta
        H[row_p][n_upper + 2 * j] = (lx - xp[0]) / p_den  # dlx
        H[row_p][n_upper + 2 * j + 1] = (ly - xp[1]) / p_den  # dly

        # derivatives of m_alpha
        alpha_den = (lx - xp[0]) ** 2 + (ly - xp[1]) ** 2
        H[row_alpha][0] = (ly - xp[1]) / alpha_den  # dx
        H[row_alpha][1] = -(lx - xp[0]) / alpha_den  # dy
        H[row_alpha][2] = -1  # dtheta
        H[row_alpha][n_upper + 2 * j] = -(ly - xp[1]) / alpha_den  # dlx
        H[row_alpha][n_upper + 2 * j + 1] = (lx - xp[0]) / alpha_den  # dly
    R_new = np.kron(np.eye(len(indexes)), R)
    # Equations
    K = pp @ H.T @ np.linalg.inv(H @ pp @ H.T + R_new)
    P = pp @ (np.eye(n) - H.T @ K.T)
    measurement = np.array([ranges, angles, indexes]).T
    inn = innovation(measurement, xp)
    X = xp + (K @ inn).reshape(-1)


    xp = X.copy()
    pp = P.copy()
    return xp, pp


Xp = x0.copy()
Pp = P0.copy()
X_pred = np.empty((N, n))
X_odom_history = np.empty((N, 3))
P_pred = np.empty((N, n))
P_pred_full = np.zeros((N, n, n))

Q_straight = Q.copy()
checked_landmarks = []
Xp_lower = np.zeros(n_lower)
#Odometry trajectory

Xodometry = Xp[0:3].copy()
for o in range(0,N):
    Xod = Xodometry.copy()
    Xodometry = Xod + (Ts * np.array([Uf[o - 1] * np.cos(Xod[2]), Uf[o - 1] * np.sin(Xod[2]), Ua[o - 1]]))
    X_odom_history[o, :] = Xodometry

"""
           ########### Initialize and correct for t = 0 ###########
"""

for l in range(len(index_land[0])):
    if index_land[0][l] not in checked_landmarks:  # Initialize
        checked_landmarks.append(index_land[0][l])
        j = index_land[0][l] - 1
        mp = range_land[0][l]
        ma = angle_land[0][l]
        lx = Xp[0] + mp * np.cos(Xp[2] + ma)
        ly = Xp[1] + mp * np.sin(Xp[2] + ma)
        Xp[n_upper + 2 * j] = lx
        Xp[n_upper + 2 * j + 1] = ly

m = np.array([range_land[0], angle_land[0], index_land[0]])

Xp, Pp = correction(Xp, Pp, m)
X_pred[0, :] = Xp
P_pred[0, :] = np.diag(Pp)
P_pred_full[0, :] = Pp

i = 1
while i < N:
    """
            ########### Prediction ###########
    """
    X = Xp.copy()
    Xp = X + np.concatenate((Ts * np.array([Uf[i - 1] * np.cos(X[2]), Uf[i - 1] * np.sin(X[2]), Ua[i - 1]]), Xp_lower))

    dUpper_dx = np.array([[1, 0, -Ts * Uf[i - 1] * np.sin(X[2])],
                          [0, 1, Ts * Uf[i - 1] * np.cos(X[2])],
                          [0, 0, 1]])
    dLower_dx = np.eye(n_lower)
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
    """
        ########### Correction ###########
    """
    for l in range(len(index_land[i])):
        if index_land[i][l] not in checked_landmarks:  # Initialize
            checked_landmarks.append(index_land[i][l])
            j = index_land[i][l] - 1
            mp = range_land[i][l]
            ma = angle_land[i][l]
            lx = Xp[0] + mp * np.cos(Xp[2] + ma)
            ly = Xp[1] + mp * np.sin(Xp[2] + ma)
            Xp[n_upper + 2 * j] = lx
            Xp[n_upper + 2 * j + 1] = ly

    m = np.array([range_land[i], angle_land[i], index_land[i]])
    Xp, Pp = correction(Xp, Pp, m)

    X_pred[i, :] = Xp
    P_pred[i, :] = np.diag(Pp)
    P_pred_full[i, :] = Pp

    i += 1



pose_pred = X_pred[:, :n_upper]
pose_odom_pred = X_odom_history[:, :n_upper]
pose_true = Pose
landmark_pred = X_pred[N - 1, 3:]

print("\nFinal Estimated Robot Pose:", pose_pred[-1, :])
print("Final True Robot Pose:", pose_true[-1, :])
print("\nFinal Estimated Landmark Positions:\n", landmark_pred)
# print("True Landmark Positions:\n", Landmarks)
T = np.arange(0, N * Ts, Ts) # Time



# Plots and comparisons
fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(3, 1, 1)
ax1.plot(T, pose_true[:, 0], label=r'$x_1(t)$')
ax1.plot(T, pose_pred[:, 0], label=r'$\hat{x}_1(t)$', color='red', linestyle='--')
ax1.plot(T, pose_odom_pred[:, 0], label=r'$\hat{x}_{1,  odom}(t)$', color='black', linestyle=':')

plt.xlabel("$t$")
plt.ylabel("$x_1(t)$")
plt.title("Velocity $x_1(t)$")
plt.legend(loc='upper left')

ax1 = plt.subplot(3, 1, 2)
ax1.plot(T, pose_true[:, 1], label=r'$x_2(t)$')
ax1.plot(T, pose_pred[:, 1], label=r'$\hat{x}_2(t)$', color='red', linestyle='--')
ax1.plot(T, pose_odom_pred[:, 1], label=r'$\hat{x}_{2,odom}(t)$', color='black', linestyle=':')

plt.xlabel("$t$")
plt.ylabel("$x_2(t)$")
plt.title("Current $x_2(t)$")
plt.legend(loc='upper left')

ax1 = plt.subplot(3, 1, 3)
ax1.plot(T, pose_true[:, 2], label=r'$\theta(t)$')
ax1.plot(T, pose_pred[:, 2], label=r'$\hat{\theta}(t)$', color='red', linestyle='--')
ax1.plot(T, pose_odom_pred[:, 2], label=r'$\hat{x}_{theta,odom}(t)$', color='black', linestyle=':')

plt.xlabel("$t$ (s)")
plt.ylabel(r"$\theta(t)$ (rad)")
plt.title("Robot Orientation")
plt.subplots_adjust(hspace=0.6)
plt.legend(loc='upper left')
if save_fig:
    if int(dataset.split('_')[-1].split('.')[0]) == 1:
        plt.savefig(f'Figures_Part_A/Comparison_data_1.eps', format='eps')
    else:
        plt.savefig(f'Figures_Part_A/Comparison_data_2.eps', format='eps')

fig.tight_layout()

# Trajectory
fig = plt.figure()
# ax1 = plt.subplot(2, 1, 1)
plt.plot(pose_true[:, 0], pose_true[:, 1], label=r'$x_1(t)$')
plt.plot(pose_pred[:, 0], pose_pred[:, 1], label=r'$\hat{x}_1(t)$', color='red', linestyle='--')
plt.plot(pose_odom_pred[:, 0], pose_odom_pred[:, 1], label=r'$\hat{x}_{1,odom}(t)$', color='black', linestyle=':')
plt.title("Robot Trajectory")
plt.legend(loc='center')
if save_fig:
    if int(dataset.split('_')[-1].split('.')[0]) == 1:
        plt.savefig(f'Figures_Part_A/Robot_trajectory_data_1.eps', format='eps')
    else:
        plt.savefig(f'Figures_Part_A/Robot_trajectory_data_2.eps', format='eps')

# Landmark positions
landmark_pred = landmark_pred.reshape(-1, 2)
fig = plt.figure()
# ax1 = plt.subplot(2, 1, 2)

plt.scatter(Landmarks[:, 0], Landmarks[:, 1], label='True Landmarks', color='red', marker='o', s=50, facecolors='none')

plt.scatter(landmark_pred[:, 0], landmark_pred[:, 1], label='Estimated Landmarks (Final)',
            color='green', marker='x', s=100)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Robot Trajectory and Landmarks")
plt.legend()
# plt.grid(True)


# Confidence intervals
plt.figure(figsize=(6, 12))
plt.subplot(311)
plt.plot(T, pose_true[:, 0] - pose_pred[:, 0], 'r', T, 3 * np.sqrt(P_pred[:, 0]), 'b--', T, -3 * np.sqrt(P_pred[:, 0]),
         'b--')
plt.ylabel(r'$x(t)-\hat{x}(t)$')
title = r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $x(t)$'
plt.title(title)
plt.axis([T[0], T[-1], -np.sqrt(P_pred[-1, 0]) * 30, np.sqrt(P_pred[-1, 0]) * 30])
plt.subplot(312)
plt.plot(T, pose_true[:, 1] - pose_pred[:, 1], 'r', T, 3 * np.sqrt(P_pred[:, 1]), 'b--', T, -3 * np.sqrt(P_pred[:, 1]),
         'b--')
plt.ylabel(r'$y(t)-\hat{y}(t)$')
title = r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $y(t)$'
plt.title(title)
plt.axis([T[0], T[-1], -np.sqrt(P_pred[-1, 1]) * 20, np.sqrt(P_pred[-1, 1]) * 20])
plt.subplot(313)
plt.plot(T, 180 / np.pi * np.unwrap(pose_true[:, 2] - pose_pred[:, 2]), 'r', T, 3 * 180 / np.pi * np.sqrt(P_pred[:, 2]),
         'b--', T,
         -3 * 180 / np.pi * np.sqrt(P_pred[:, 2]), 'b--')
plt.xlabel('$t$')
plt.ylabel(r'$\theta(t)-\hat{\theta}(t)$')
plt.title(r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $\theta(t)$')

if save_fig:
    if int(dataset.split('_')[-1].split('.')[0]) == 1:
        plt.savefig(f'Figures_Part_A/Confidence_intervals_data_1.eps', format='eps')
    else:
        plt.savefig(f'Figures_Part_A/Confidence_intervals_data_2.eps', format='eps')

# Confidence ellipses

r_x = 9.21  # 99% confidence ellipse
consistent = []
radii_all = []
fig, ax = plt.subplots(figsize=(8, 8))

for j in range(Nland): #for all landmarks
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
if save_fig:
    if int(dataset.split('_')[-1].split('.')[0]) == 1:
        plt.savefig(f'Figures_Part_A/Confidence_ellipses_data_1.eps', format='eps')
    else:
        plt.savefig(f'Figures_Part_A/Confidence_ellipses_data_2.eps', format='eps')


plt.grid(True)
all_good = True
for lm in range(len(consistent)):
    if consistent[lm] == np.True_:
        print(f"Landmark {lm+1} = Consistent within confidence ellipse. ")
    else:
        print(f"Landmark {lm+1} = NOT CONSISTENT!")
        all_good = False

if all_good:
    print("All landmarks are consistent. All good!")
else:
    print("Error: One or more landmarks are inconsistent. :(")

plt.show()
