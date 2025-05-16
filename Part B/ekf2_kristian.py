import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import scipy
from scipy.linalg import block_diag

plt.close('all')

data = np.load('data_point_land_1.npz', allow_pickle=True)

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
# index_land = Meas.item()['land']  # landmark indices

N = Uf.shape[0]  # Number of odometry measurements
n_upper = 3  # upper system order: x,y,theta
upper_threshold = 13.8155
lower_threshold = 5.9915
# n_lower = Nland.item() * 2  # lower system order: 2 for every landmark (x,y)
# n = n_upper + n_lower  # system order
def append_to_array(arr, to_append):
    arr = arr.tolist()
    arr.append(to_append)
    return np.array(arr)



# x0 = np.zeros(n_upper)  # initial states
# x0[0] = Pose[0, 0]  # x(0)
# x0[1] = Pose[0, 1]  # y(0)
# x0[2] = Pose[0, 2]  # theta(0)

x0 = np.array([])  # initial states
x0 = append_to_array(x0, Pose[0, 0])
x0 = append_to_array(x0, Pose[0, 1])
x0 = append_to_array(x0, Pose[0, 2])



# upper covariance
lambda_ = 1e-6
P_upper = lambda_ * np.eye(n_upper)  # uncertainty of x,y,theta
# upper_zeros = np.zeros((2, 2))
# lower covariance
eta = 1000
# P_lower = eta * np.eye(n_lower)  # uncertainty landmarks
# lower_zeros = np.zeros((n_lower, n_upper))
# Initial covariance
# P0 = np.block([[P_upper, upper_zeros],
#                [lower_zeros, P_lower]])
P0 = P_upper
Xp = x0.copy()
Pp = P0.copy()
# AT THE END WE HAVE TO FILL ALL OF THE 220 ROWS WITH 0's DEPENDING ON THE LAST DIMENSION OF Xp
X_pred = np.empty((N, 3))
P_pred = np.empty((N, 3))
# P_pred_full = np.zeros((N, n, n))

Q_straight = Q.copy()
landmarks_map = []

def angle_wrap(angle):
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

def extend_P(p, num_new_landmarks):
    # extend P with the number of the new initialized landmarks
    diagonal = p.diagonal().tolist()
    previous = len(diagonal)
    diagonal.extend([eta]*num_new_landmarks*2)

    new_p = np.zeros((previous+2*num_new_landmarks, previous+2*num_new_landmarks))
    np.fill_diagonal(new_p, diagonal)
    return new_p

# def innovation(M, xp):
#     # landmarks = M[:, 2]
#
#     # take the predictions
#     predicted_meas = np.zeros((len(landmarks), 2))
#
#     for i in range(len(landmarks)):
#         j = int(landmarks[i])
#
#         lx = xp[n_upper + 2 * j]
#         ly = xp[n_upper + 2 * j + 1]
#
#         predicted_p = np.sqrt((lx - xp[0]) ** 2 + (ly - xp[1]) ** 2)
#         predicted_alpha = angle_wrap(np.arctan2(ly - xp[1], lx - xp[0]) - xp[2])
#         predicted_meas[i][0] = predicted_p
#         predicted_meas[i][1] = predicted_alpha
#
#     inn = M[:, :2] - predicted_meas
#     return inn.reshape(-1, 1)
def innovation(M, xp):
    landmarks = M[:, 2]
    # take the predictions
    predicted_meas = np.zeros((len(landmarks), 2))

    for i in range(len(landmarks)):
        j = int(landmarks[i])

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
    n_states = len(xp)
    H = np.zeros((len(ranges) * 2, n_states)) # (num_measurements*2(rho and theta))x(num_states)

    for k in range(len(ranges)):
        j = int(indexes[k])
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

    # print('H shape:', H.shape)
    R_new = np.kron(np.eye(len(ranges)), R)
    # Equations
    K = pp @ H.T @ np.linalg.inv(H @ pp @ H.T + R_new)
    P = pp @ (np.eye(n_states) - H.T @ K.T)
    measurement = np.array([ranges, angles, indexes]).T

    inn = innovation(measurement, xp)
    X = xp + (K @ inn).reshape(-1)


    xp = X.copy()
    pp = P.copy()
    return xp, pp




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
# m = np.array([range_land[0], angle_land[0]])
Pp = extend_P(Pp, len(range_land[0]))
Xp, Pp = correction(Xp, Pp, to_correct)


def max_likelihood(r, alpha, x, p):
    # print('------------ in maxlikelihood-------------')

    landmarks = x[3:]
    # print('landmarks', landmarks)
    # print('x:', x)
    # print('len x:', len(x))
    landmarks = landmarks.reshape(int(len(landmarks)/2), 2)
    distances = []
    robot_x = x[0]
    robot_y = x[1]
    robot_theta = x[2]


    # H = np.zeros((2, len(x)))
    for k in range(len(landmarks)):
        H = np.zeros((2, len(x)))
        l_x = landmarks[k][0]
        l_y = landmarks[k][1]
        # print(f"lx = {l_x}, ly = {l_y}")
        new_p = r -  np.sqrt((l_x - robot_x) ** 2 + (l_y - robot_y) ** 2)
        new_alpha = alpha - angle_wrap(np.arctan2(l_y - robot_y, l_x - robot_x)) + robot_theta
        delta = np.array([new_p, new_alpha]).reshape(-1, 1)
        p_den = np.sqrt((lx - robot_x) ** 2 + (l_y - robot_y) ** 2)
        H[0][0] = (robot_x - l_x) / p_den  # dx
        H[0][1] = (robot_y - l_y) / p_den  # dy
        H[0][2] = 0  # dtheta
        H[0][3+k*2] = (l_x - robot_x) / p_den  # dlx
        H[0][3+k*2+1] = (l_y - robot_y) / p_den  # dly

        alpha_den = (l_x - robot_x) ** 2 + (l_y - robot_y) ** 2

        H[1][0] = (l_y - robot_y) / alpha_den  # dx
        H[1][1] = -(l_x - robot_x) / alpha_den  # dy
        H[1][2] = -1  # dtheta
        H[1][3 + k * 2] = -(l_y - robot_y) / alpha_den  # dlx
        H[1][3 + k * 2 + 1] = (l_x - robot_x) / alpha_den  # dly
        # print('H shape:', H.shape)
        # print('p shape', p.shape)
        d = delta.T @ np.linalg.inv(H@p@H.T + R) @ delta
        distances.append(d[0][0])
    # find the minimum distance
    min_d = 10000000000000000
    min_d_index = -1
    for i in range(len(distances)):
        if distances[i] < min_d:
            min_d = distances[i]
            min_d_index = i
    # print(f"min_d = {min_d}")
    # print(f"min_d_index = {min_d_index}")
    return min_d, min_d_index


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
    """
        ########### Correction ###########
    """
    for_correction = []
    number_of_new_initializations = 0
    # check the new measurements
    Xp_new = Xp.copy()
    # print('---------before likelihood---------------')
    # print('ranges:', range_land[i])
    # print('Xp:', Xp)
    # print('Xp len:', len(Xp))
    for l in range(len(range_land[i])):
        # for each new measurements find the min d_jk
        range_m = range_land[i][l]
        angle_m = angle_land[i][l]
        # print('------------------before max liklihood-----------------')
        # print('range_m', range_m)
        # print('angle_m', angle_m)
        # print('Xp shape:', Xp.shape)
        # print('Xp:', Xp)
        # print('Pp:', Pp)
        # print('Pp shape:', Pp.shape)
        d_min, index_of_landmark = max_likelihood(range_m, angle_m, Xp, Pp)
        # HAVE TO ASSSOCIATE EACH MEASUREMENT WITH A LANDMARK INDEX
        if d_min < lower_threshold:
            # old one make a vector [range, angle, corresponding landmark index
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
        # print('for correction:', for_correction)
        # print('Xp:', Xp)
        for_correction = np.array(for_correction).T
        # correct
        # print('number_of_new_initializations:', number_of_new_initializations)
        # print('Xp shape:', Xp.shape)

        # print('Pp shape:', Pp.shape)

        Pp = extend_P(Pp, number_of_new_initializations)
        # print('pp shape:', Pp.shape)
        # print('Xp shape:', Xp.shape)
        Xp, Pp = correction(Xp, Pp, for_correction)

    # X_pred[i, :] = Xp
    # P_pred[i, :] = np.diag(Pp)
    # P_pred_full[i, :] = Pp

    i += 1
print('Xp final shape', Xp.shape)
# print('Xp final: ', Xp)
""" ---------------------------------------------------------------------------------------------- """
# take the first three columns for every row of X_pred

# pose_pred = X_pred[:, :n_upper]
# pose_true = Pose
#
# landmark_pred = X_pred[N - 1, 3:]

# print("\nFinal Estimated Robot Pose:", pose_pred[-1, :])
# print("Final True Robot Pose:", pose_true[-1, :])
# print("\nFinal Estimated Landmark Positions:\n", landmark_pred)
# print("True Landmark Positions:\n", Landmarks)

#
# T = np.arange(0, N * Ts, Ts) # Time
# # Plots and comparisons
# fig = plt.figure()
# ax1 = plt.subplot(3, 1, 1)
# ax1.plot(T, pose_true[:, 0], label=r'$x_1(t)$')
# ax1.plot(T, pose_pred[:, 0], label=r'$\hat{x}_1(t)$', color='red', linestyle='--')
# plt.xlabel("$t$")
# plt.ylabel("$x_1(t)$")
# plt.title("Velocity $x_1(t)$")
# plt.legend(loc='upper right')
#
# ax1 = plt.subplot(3, 1, 2)
# ax1.plot(T, pose_true[:, 1], label=r'$x_2(t)$')
# ax1.plot(T, pose_pred[:, 1], label=r'$\hat{x}_2(t)$', color='red', linestyle='--')
# plt.xlabel("$t$")
# plt.ylabel("$x_2(t)$")
# plt.title("Current $x_2(t)$")
# plt.legend(loc='upper right')
#
# ax1 = plt.subplot(3, 1, 3)
# ax1.plot(T, pose_true[:, 2], label=r'$\theta(t)$')
# ax1.plot(T, pose_pred[:, 2], label=r'$\hat{\theta}(t)$', color='red', linestyle='--')
# plt.xlabel("$t$ (s)")
# plt.ylabel(r"$\theta(t)$ (rad)")
# plt.title("Robot Orientation")
# plt.legend(loc='upper right')
# # plt.grid(True)
#
# fig.tight_layout()
#
# # Trajectory
# fig = plt.figure()
# ax1 = plt.subplot(2, 1, 1)
# ax1.plot(pose_true[:, 0], pose_true[:, 1], label=r'$x_1(t)$')
# ax1.plot(pose_pred[:, 0], pose_pred[:, 1], label=r'$\hat{x}_1(t)$', color='red', linestyle='--')
# plt.legend(loc='center')
#
# # Landmark positions
# landmark_pred = landmark_pred.reshape(-1, 2)
# fig = plt.figure()
# ax1 = plt.subplot(2, 1, 1)
#
# plt.scatter(Landmarks[:, 0], Landmarks[:, 1], label='True Landmarks', color='red', marker='o', s=50, facecolors='none')
#
# plt.scatter(landmark_pred[:, 0], landmark_pred[:, 1], label='Estimated Landmarks (Final)',
#             color='green', marker='x', s=100)
#
# plt.xlabel("X position (m)")
# plt.ylabel("Y position (m)")
# plt.title("Robot Trajectory and Landmarks")
# plt.legend()
# # plt.grid(True)
#
#
# # Confidence intervals
# plt.figure(figsize=(6, 12))
# plt.subplot(311)
# plt.plot(T, pose_true[:, 0] - pose_pred[:, 0], 'r', T, 3 * np.sqrt(P_pred[:, 0]), 'b--', T, -3 * np.sqrt(P_pred[:, 0]),
#          'b--')
# plt.ylabel(r'$x(t)-\hat{x}(t)$')
# title = r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $x(t)$'
# plt.title(title)
# plt.axis([T[0], T[-1], -np.sqrt(P_pred[-1, 0]) * 20, np.sqrt(P_pred[-1, 0]) * 20])
# plt.subplot(312)
# plt.plot(T, pose_true[:, 1] - pose_pred[:, 1], 'r', T, 3 * np.sqrt(P_pred[:, 1]), 'b--', T, -3 * np.sqrt(P_pred[:, 1]),
#          'b--')
# plt.ylabel(r'$y(t)-\hat{y}(t)$')
# title = r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $y(t)$'
# plt.title(title)
# plt.axis([T[0], T[-1], -np.sqrt(P_pred[-1, 1]) * 20, np.sqrt(P_pred[-1, 1]) * 20])
# plt.subplot(313)
# plt.plot(T, 180 / np.pi * np.unwrap(pose_true[:, 2] - pose_pred[:, 2]), 'r', T, 3 * 180 / np.pi * np.sqrt(P_pred[:, 2]),
#          'b--', T,
#          -3 * 180 / np.pi * np.sqrt(P_pred[:, 2]), 'b--')
# plt.xlabel('$t$')
# plt.ylabel(r'$\theta(t)-\hat{\theta}(t)$')
# plt.title(r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $\theta(t)$')
# # plt.axis([T[0], T[-1], -np.sqrt(P_pred[-1, 2]) * 20, np.sqrt(P_pred[-1, 2]) * 20])
#
#
# # Confidence ellipses
# r_x = 9.21  # 99% confidence ellipse
# consistent = []
# radii_all = []
# fig, ax = plt.subplots(figsize=(8, 8))
#
#
# for j in range(Nland): #for all landmarks
#     idx = n_upper + 2 * j
#     L_true = Landmarks[j]
#     L_est = X_pred[-1][idx:idx + 2]
#
#     Pj = P_pred_full[-1][idx:idx + 2, idx:idx + 2]
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
#     ax.plot(L_est[0], L_est[1], 'ro')
# ax.set_xlabel('X position')
# ax.set_ylabel('Y position')
# ax.set_title('Landmark position estimates with 99% confidence ellipses')
# ax.axis('equal')
# plt.grid(True)
# plt.show()
# all_good = True
# for lm in range(len(consistent)):
#     if consistent[lm] == np.True_:
#         print(f"Landmark {lm+1} = Consistent within confidence ellipse. ")
#     else:
#         print(f"Landmark {lm+1} = NOT CONSISTENT!")
#         all_good = False
#
# if all_good:
#     print("All landmarks are consistent. All good!")
# else:
#     print("Error: One or more landmarks are inconsistent. :(")
#
# plt.show()
