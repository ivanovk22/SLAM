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
index_land = Meas.item()['land']  # landmark indices

N = Uf.shape[0]  # Number of odometry measurements
n_upper = 3  # upper system order: x,y,theta
# upper_threshold = 13.8155
upper_threshold = 14
lower_threshold = 5.9915
# n_lower = Nland.item() * 2  # lower system order: 2 for every landmark (x,y)
# n = n_upper + n_lower  # system order
def append_to_array(arr, to_append):
    arr = arr.tolist()
    arr.append(to_append)
    return np.array(arr)


x0 = np.array([])  # initial states
x0 = append_to_array(x0, Pose[0, 0])
x0 = append_to_array(x0, Pose[0, 1])
x0 = append_to_array(x0, Pose[0, 2])



# upper covariance
lambda_ = 1e-6
P_upper = lambda_ * np.eye(n_upper)  # uncertainty of x,y,theta
eta = 1000
P0 = P_upper
Xp = x0.copy()
Pp = P0.copy()
X_pred = np.empty((N, n_upper)) # as beginning 220X3




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
    # P = pp @ (np.eye(n_states) - H.T @ K.T)
    # P = pp @ (np.eye(n_states) - H.T @ K.T)
    P = (np.eye(n_states) - K @ H) @ pp
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
Pp = extend_P(Pp, len(range_land[0]))
Xp, Pp = correction(Xp, Pp, to_correct)
X_pred[0, :] = Xp[:3]

# def max_likelihood(r, alpha, x, p):
#     landmarks = x[3:]
#     landmarks = landmarks.reshape(int(len(landmarks)/2), 2)
#     distances = []
#     robot_x = x[0]
#     robot_y = x[1]
#     robot_theta = x[2]
#
#     for k in range(len(landmarks)):
#         H = np.zeros((2, len(x)))
#         l_x = landmarks[k][0]
#         l_y = landmarks[k][1]
#         # print(f"lx = {l_x}, ly = {l_y}")
#         new_p = r -  np.sqrt((l_x - robot_x) ** 2 + (l_y - robot_y) ** 2)
#         new_alpha = angle_wrap(alpha - angle_wrap(np.arctan2(l_y - robot_y, l_x - robot_x) - robot_theta))
#         delta = np.array([new_p, new_alpha]).reshape(-1, 1)
#         p_den = np.sqrt((l_x - robot_x) ** 2 + (l_y - robot_y) ** 2)
#         H[0][0] = (robot_x - l_x) / p_den  # dx
#         H[0][1] = (robot_y - l_y) / p_den  # dy
#         H[0][2] = 0  # dtheta
#         H[0][3+k*2] = (l_x - robot_x) / p_den  # dlx
#         H[0][3+k*2+1] = (l_y - robot_y) / p_den  # dly
#
#         alpha_den = (l_x - robot_x) ** 2 + (l_y - robot_y) ** 2
#
#         H[1][0] = (l_y - robot_y) / alpha_den  # dx
#         H[1][1] = -(l_x - robot_x) / alpha_den  # dy
#         H[1][2] = -1  # dtheta
#         H[1][3 + k * 2] = -(l_y - robot_y) / alpha_den  # dlx
#         H[1][3 + k * 2 + 1] = (l_x - robot_x) / alpha_den  # dly
#         d = delta.T @ np.linalg.inv(H@p@H.T + R) @ delta
#         distances.append(d[0][0])
#     # find the minimum distance
#     min_d = 10000000000000000
#     min_d_index = -1
#     for i in range(len(distances)):
#         if distances[i] < min_d:
#             min_d = distances[i]
#             min_d_index = i
#
#     return min_d, min_d_index
def max_likelihood(r, alpha, x, p):
    landmarks = x[3:].reshape(-1, 2)
    robot_x, robot_y, robot_theta = x[0], x[1], x[2]
    distances = []

    for k, (l_x, l_y) in enumerate(landmarks):
        dx = l_x - robot_x
        dy = l_y - robot_y
        q = dx**2 + dy**2
        sqrt_q = np.sqrt(q)

        # Predicted measurement
        predicted_r = sqrt_q
        predicted_alpha = angle_wrap(np.arctan2(dy, dx) - robot_theta)

        # Innovation (measurement - prediction)
        delta_r = r - predicted_r
        delta_alpha = angle_wrap(alpha - predicted_alpha)
        delta = np.array([[delta_r], [delta_alpha]])

        # Jacobian H (2 x n)
        Hk = np.zeros((2, len(x)))
        Hk[0, 0] = -(dx) / sqrt_q
        Hk[0, 1] = -(dy) / sqrt_q
        Hk[0, 2] = 0
        Hk[0, 3 + 2*k]     =  dx / sqrt_q
        Hk[0, 3 + 2*k + 1] =  dy / sqrt_q

        Hk[1, 0] =  dy / q
        Hk[1, 1] = -dx / q
        Hk[1, 2] = -1
        Hk[1, 3 + 2*k]     = -dy / q
        Hk[1, 3 + 2*k + 1] =  dx / q

        # Mahalanobis distance
        S = Hk @ p @ Hk.T + R
        d = delta.T @ np.linalg.inv(S) @ delta
        distances.append(d[0, 0])

    min_d_index = np.argmin(distances)
    min_d = distances[min_d_index]
    return min_d, min_d_index

i = 1
while i < N:
    """
            ########### Prediction ###########
    """
    # print('\n\n')
    # print(f'----------------------i: {i}----------------------------')
    # print(f'ranges: {range_land[i]}')
    # print(f'angles: {angle_land[i]}')
    # print(f'indexes: {index_land[i]}')
    # print(f'Xp shape: {Xp.shape}')
    # print(f'Xp : {Xp}')
    # print(f'Pp shape: {Pp.shape}')
    # print(f'Pp : {Pp}')

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
    # print(f'Xp shape after prediction: {Xp.shape}')
    # print(f'Xp after prediction: {Xp}')
    # print(f'Pp shape after prediction: {Pp.shape}')
    # print(f'Pp after prediction: {Pp}')

    """
        ########### Correction ###########
    """
    for_correction = []
    number_of_new_initializations = 0
    # check the new measurements
    Xp_new = Xp.copy()
    # print(f'Xp new shape: {Xp_new.shape} Xp new : {Xp_new}')
    # print('----------------initialize measurements---------------')
    for l in range(len(range_land[i])):
        # print(f'---------------l = {l}---------------')
        # print(f'range[{i}][{l}] = {range_land[i][l]}')
        # print(f'angle[{i}][{l}] = {angle_land[i][l]}')
        # print(f'index from true data[{i}][{l}] = {index_land[i][l]}')
        # for each new measurements find the min d_jk
        range_m = range_land[i][l]
        angle_m = angle_land[i][l]
        d_min, index_of_landmark = max_likelihood(range_m, angle_m, Xp, Pp)
        # print(f'd_min = {d_min}, index_of_landmark = {index_of_landmark}')

        # HAVE TO ASSOCIATE EACH MEASUREMENT WITH A LANDMARK INDEX

        if d_min < lower_threshold:
            # old one make a vector [range, angle, corresponding landmark index]
            for_correction.append([range_m, angle_m, index_of_landmark])

        # for l in range(len(range_land[0])):
        #     if range_land[0][l] not in landmarks_map:  # Initialize
        #         landmarks_map.append(len(landmarks_map))
        #         j = l
        #         mp = range_land[0][l]
        #         ma = angle_land[0][l]
        #         lx = Xp[0] + mp * np.cos(Xp[2] + ma)
        #         ly = Xp[1] + mp * np.sin(Xp[2] + ma)
        #         Xp = append_to_array(Xp, lx)
        #         Xp = append_to_array(Xp, ly)

        if d_min > upper_threshold:
            # new one( initialize )
            landmarks_map.append(len(landmarks_map))
            number_of_new_initializations += 1
            # add new landmark in Xp_new
            # print('------------ in lower threshold-------')
            # print(f'range_m: {range_m}, angle_m: {angle_m}')
            # print(f'Xp_new[0]: {Xp_new[0]}, Xp_new[1]: {Xp_new[1]}, Xp_new[2]: {Xp_new[2]} ')
            lx = Xp_new[0] + range_m * np.cos(Xp_new[2] + angle_m)
            ly = Xp_new[1] + range_m * np.sin(Xp_new[2] + angle_m)
            # print(f'lx: {lx}, ly: {ly}')

            Xp_new = append_to_array(Xp_new, lx)
            Xp_new = append_to_array(Xp_new, ly)
            # print(f'---------Xp_new: {Xp_new}')
            for_correction.append([range_m, angle_m, len(landmarks_map) - 1])
    # print(f'Xp after initialization: {Xp}')
    # print(f'Xp_new after initialization: {Xp_new}')
    # print(f'for correction after initialization: {for_correction}')
    # print(f'lanmarks_map after initialization: {landmarks_map}')
    Xp = Xp_new.copy()
    # print(f'Xp after copy from Xp_new: {Xp}')

    if len(for_correction) > 0:
        for_correction = np.array(for_correction).T
        # correct
        Pp = extend_P(Pp, number_of_new_initializations)
        Xp, Pp = correction(Xp, Pp, for_correction)
        # print(f'Xp shape after correcction: {Xp.shape}')
        # print(f'Xp after correcction: {Xp}')
        # print(f'Pp shape after correcction: {Pp.shape}')
        # print(f'Pp after correcction: {Pp}')
    X_pred[i, :] = Xp[:3]

    i += 1

    # if i == 110:
    #     break
print('Xp:', Xp)
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
plt.show()


# Trajectory
fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax1.plot(pose_true[:, 0], pose_true[:, 1], label=r'$x_1(t)$')
ax1.plot(pose_pred[:, 0], pose_pred[:, 1], label=r'$\hat{x}_1(t)$', color='red', linestyle='--')
plt.legend(loc='center')
plt.show()

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
plt.show()