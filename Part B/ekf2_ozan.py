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


"""X_pred = [[]]
P_pred = [[]]
X_pred = np.
P_pred[0, :] = np.diag(Pp)
P_pred_full[0, :] = Pp
"""
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
X_pred = np.empty((N, 3))
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
        print(min_d_index, min_d)
        print(angle_land[i].shape)

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
    print('to_be_deleted',to_be_deleted)
    print(range_land[i].shape)
    to_be_deleted = np.array(to_be_deleted)
    for dele in range(len(to_be_deleted)):
        print('sonaşama',range_land[i],to_be_deleted[dele])
        range_land[i] = np.delete(range_land[i], to_be_deleted[dele])
        angle_land[i] = np.delete(angle_land[i], to_be_deleted[dele])
        to_be_deleted = to_be_deleted - 1
    print('predicted' , indic)
    print(indexes)
    print('real     ',index_land[i])
    print('-'*30,i)
    m = np.array([range_land[i], angle_land[i], indic])
    Xp, Pp = correction(Xp, Pp, m)

    X_pred[i, :] = Xp[0:3]

    i = i + 1
print(X_pred)

pose_true = Pose

fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax1.plot(pose_true[:, 0], pose_true[:, 1], label=r'$x_1(t)$')
ax1.plot(X_pred[:, 0], X_pred[:, 1], label=r'$\hat{x}_1(t)$', color='red', linestyle='--')
plt.legend(loc='center')
plt.show()