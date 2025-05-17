import numpy as np
import matplotlib.pyplot as plt
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
Nland = data['Nland']  # number of Landmarks (not to be used for part B)
Ts = data['Ts']
Wturn = data['wturn']  # treshold Wturn
Pose = data['Pose']  # data to be used only for comparison (x(t), y(t), theta(t) of the robot)
Landmarks = data[
    'Landmarks']  # data to be used only for comparison (ith row corresponds to i+1th of landmark locations.)

range_land = Meas.item()['range']  # landmark ranges
angle_land = Meas.item()['angle']  # landmark angles
index_land = Meas.item()['land']  # landmark indices !not to be used for part B!

N = Uf.shape[0]  # Number of odometry measurements

n_upper = 3  # upper system order: x,y,theta
#n_lower = Nland.item() * 2  # lower system order: 2 for every landmark (x,y)
#n = n_upper + n_lower  # system order
n = n_upper #initial states are only x,y,theta

x0 = np.zeros(n)  # initial states
x0[0] = Pose[0, 0]  # x(0)
x0[1] = Pose[0, 1]  # y(0)
x0[2] = Pose[0, 2]  # theta(0)
# upper covariance
lambda_ = 1e-6
P_upper = lambda_ * np.eye(n_upper)  # uncertainty of x,y,theta
#upper_zeros = np.zeros((n_upper, n_lower))
# lower covariance
eta = 1000
#P_lower = eta * np.eye(n_lower)  # uncertainty landmarks
#lower_zeros = np.zeros((n_lower, n_upper))
# Initial covariance
"""P0 = np.block([[P_upper, upper_zeros],
               [lower_zeros, P_lower]])"""
P0 = P_upper


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

def calc_H_R(xp, indices): # calculates the H and R matrices
    H = np.zeros((len(indices) * 2, xp.shape[0]))
    for k in range(len(indices)):
        # landmarks are from 1, so when we see landmark 1 it means that it will
        # be on the 0 position after x, y, theta - Lx1 = 3, Ly1 = 4
        j = int(indices[k]) - 1
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
    R_new = np.kron(np.eye(len(indices)), R)
    return H, R_new

def update_data(data, t, new_vector):
    n_rows, current_cols = data.shape
    v_len = len(new_vector)

    if v_len > current_cols:
        # Need to expand all rows to match the new vector length
        padding = np.zeros((n_rows, v_len - current_cols))
        data = np.hstack((data, padding))

    # Pad new_vector if it's shorter than current number of columns
    if v_len < data.shape[1]:
        new_vector = np.pad(new_vector, (0, data.shape[1] - v_len))

    data[t] = new_vector
    return data

def update_P_pred(data, t, new_vector):
    n_rows, current_cols = data.shape
    v_len = len(new_vector)

    if v_len > current_cols:
        # Need to expand all rows to match the new vector length
        padding = np.full((n_rows, v_len - current_cols), fill_value=eta)
        data = np.hstack((data, padding))

    # Pad new_vector if it's shorter than current number of columns
    if v_len < data.shape[1]:

        new_vector = np.pad(new_vector, (eta, data.shape[1] - v_len))

    data[t] = new_vector
    return data

def correction(xp, pp, M):
    ranges = M[0]
    angles = M[1]
    indices = M[2]
    H,R = calc_H_R(xp, indices)
    # Equations
    K = pp @ H.T @ np.linalg.inv(H @ pp @ H.T + R)
    P = pp @ (np.eye(xp.shape[0]) - H.T @ K.T)
    measurement = np.array([ranges, angles, indices]).T
    inn = innovation(measurement, xp)
    X = xp + (K @ inn).reshape(-1)


    xp = X.copy()
    pp = P.copy()
    return xp, pp

def calc_d(xp,pp,range_, angle_, index):
    #find delta
    mp = range_
    ma = angle_
    index = index-1
    l_x = xp[n_upper + 2 * index]
    l_y = xp[n_upper + 2 * index + 1]
    delta_ =np.array([ mp - np.sqrt((l_x-xp[0])**2+(l_y-xp[1])**2),
                      ma - (angle_wrap(np.atan2((l_y-xp[1]),(l_x-xp[0]))-xp[2]))])

    #calculate H, R
    H, R = calc_H_R(xp, index+1)
    #calculate d
    d = delta_.T @ np.linalg.inv(H @ pp @ H.T + R) @ delta_
    return d

Xp = x0.copy()
Pp = P0.copy()
X_pred = np.empty((N, n))
P_pred = np.empty((N, n))
P_pred_full = np.zeros((N, n, n))

Q_straight = Q.copy()

#Xp_lower = np.zeros(n_lower)

"""
           ########### Initialize and correct for t = 0 ###########
"""
X_pred = np.empty((N, 0)) # as beginning 220x0
P_pred = np.empty((N, 0)) # as beginning 220x0
checked_landmarks = []
indexes = []
for l in range(len(range_land[0])): #This loop initializes the first landmarks, increases xp and pp size accordingly
    indexes.append(l+1) #not so sure about this one. maybe only using checked_landmarks can be enough
    checked_landmarks.append(l)
    j = indexes[l]
    mp = range_land[0][l]
    ma = angle_land[0][l]
    lx = np.array([Xp[0] + mp * np.cos(Xp[2] + ma)])
    ly = np.array([Xp[1] + mp * np.sin(Xp[2] + ma)])
    Xp = np.concatenate((Xp,lx))
    Xp = np.concatenate((Xp,ly))
    Pp = np.block([[Pp, np.zeros((Pp.shape[0],2))],
                   [np.zeros((2,Pp.shape[0])), eta * np.eye(2)]])



m = np.array([range_land[0], angle_land[0], indexes])
Xp, Pp = correction(Xp, Pp, m)
X_pred = update_data(X_pred, 0, Xp)
P_pred = update_P_pred(P_pred, 0, np.diag(Pp))



""" to be done later
#Odometry trajectory
X_predod = np.empty((N, n))
Xodometry = Xp.copy()
for o in range(1,N):
    Xod = Xodometry.copy()
    Xodometry = Xod + np.concatenate((Ts * np.array([Uf[o - 1] * np.cos(Xod[2]), Uf[o - 1] * np.sin(Xod[2]), Ua[o - 1]]), Xp_lower))
    X_predod[o, :] = Xodometry
print(f"X_predod.shape = {X_predod.shape}")
"""

n_upper = 3
i = 1
lower_threshold = 5.9915
upper_threshold = 13.8155

while i < N:
    """
            ########### Prediction ###########
    """
    n_lower = len(Xp) - n_upper
    Xp_lower = np.zeros(n_lower)
    X = Xp.copy()
    Xp = X + np.concatenate((Ts * np.array([Uf[i - 1] * np.cos(X[2]), Uf[i - 1] * np.sin(X[2]), Ua[i - 1]]), Xp_lower))
    dUpper_dx = np.array([[1, 0, -Ts * Uf[i - 1] * np.sin(X[2])],
                          [0, 1, Ts * Uf[i - 1] * np.cos(X[2])],
                          [0, 0, 1]])
    upper_zeros = np.zeros((n_upper, n_lower))
    dLower_dx = np.eye(n_lower)
    lower_zeros = np.zeros((n_lower, n_upper))
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
    #maximum likelihood calculation
    indic = []
    Xnew = Xp.copy()
    Pnew = Pp.copy()
    Indexnew = indexes.copy()
    range_new = range_land[i].copy()
    angle_new = angle_land[i].copy()
    to_be_deleted = []
    for j in range(len(range_new)): #for every measurement
        distances = []
        range_ = range_new[j]
        angle_ = angle_new[j]

        for k in Indexnew: #for every landmark already in map+
            kk = np.array([k])

            d = calc_d(Xnew, Pnew, range_, angle_, kk)
            distances.append(d[0][0])
        min_d_index = np.argmin(distances)
        min_d = distances[min_d_index]

        if min_d < lower_threshold: # measurement j belongs at the kth index
            indic.append(indexes[min_d_index])

        elif min_d > upper_threshold: # meausrement j belongs to a new landmark
            next_index_value = indexes[-1] + 1
            indic.append(next_index_value)
            indexes.append(next_index_value)
            mp = range_new[j]
            ma = angle_new[j]
            lx = np.array([Xp[0] + mp * np.cos(Xnew[2] + ma)])
            ly = np.array([Xp[1] + mp * np.sin(Xnew[2] + ma)])
            Xp = np.concatenate((Xp, lx))
            Xp = np.concatenate((Xp, ly))
            Pp = np.block([[Pp, np.zeros((Pp.shape[0], 2))],
                           [np.zeros((2, Pp.shape[0])), eta * np.eye(2)]])

        else:
            to_be_deleted.append(j)
            """ print('range',range_land[i],'j',j)
            if len(range_land[i]) == 1:
                range_land[i] = np.delete(range_land[i], 0)
            elif len(range_land[i]) == 0:
                pass
            else:
                range_land[i] = np.delete(range_land[i], j)
            if len(angle_land[i]) == 1:
                angle_land[i] = np.delete(angle_land[i], 0)
            elif len(angle_land[i]) == 0:
                pass
            else:
                angle_land[i] = np.delete(angle_land[i], j)"""

    to_be_deleted = np.array(to_be_deleted)
    for dele in range(len(to_be_deleted)):
        range_land[i] = np.delete(range_land[i], to_be_deleted[dele])
        angle_land[i] = np.delete(angle_land[i], to_be_deleted[dele])
        to_be_deleted = to_be_deleted - 1

    m = np.array([range_land[i], angle_land[i], indic])
    Xp, Pp = correction(Xp, Pp, m)

    X_pred = update_data(X_pred, i, Xp)
    P_pred = update_P_pred(P_pred, i, np.diag(Pp))

    i = i + 1


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

# Confidence ellipses
r_x = 9.21  # 99% confidence ellipse
# r_x = 20 # 99% confidence ellipse
consistent = []
radii_all = []
fig, ax = plt.subplots(figsize=(8, 8))
# sort the true landmarks corresponding to our indexes

new_landmarks = np.zeros((18, 2))
for l in range(len(indexes)):
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

all_good = True
for lm in range(len(consistent)):
    if consistent[lm] != np.True_:
        print(f"Landmark {lm + 1} = NOT CONSISTENT!")
        all_good = False
if all_good:
    print("All landmarks are consistent. All good!")
else:
    print("Error: One or more landmarks are inconsistent. :(")

plt.show()
