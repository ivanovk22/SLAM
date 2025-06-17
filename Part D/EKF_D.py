import matplotlib.pyplot as plt
from functions_D import *
from matplotlib.patches import Ellipse
import scipy


plt.close('all')
save_fig = False
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

# angle_res = angles[1]-angles[0] #Angle resolution of lidar
angle_res = angles[3]-angles[0] #Angle resolution of lidar FOR EVERY 3TH DEGREE
# k = 3 #3 sigma limit
k = 9 #3 sigma limit FOR EVERY 3th DEG
sigma_R_angle = (angle_res/2)/k # Error is within k*sigma => sigma = Error/k
Q = np.array([[1.0000000e-04, 0.0000000e+00],
              [0.0000000e+00, 3.0461742e-06]])

R = np.array([[1.00000000e-8, 0.00000000e+00],
              [0.00000000e+00, sigma_R_angle]])


"""
    MAKE RANGES 360 (EVERY THIRD DEGREE)
"""
ranges = ranges[::3, :]
angles = angles[::3]


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


N = Uf.shape[0]  # Number of odometry measurements
n_upper = 3  # upper system order: x,y,theta

upper_threshold = 30
lower_threshold = 5
prom = (0.1, 0.6) #prominence


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

def land_find(ranges, angles, prom, time):
    previous_choice = next_choice = 3
    inf_intervals = find_inf_intervals(ranges, previous_choice)
    to_exclude = expand_wrapped_intervals(inf_intervals, previous_choice)
    index_range = []

    peaks, prominence = scipy.signal.find_peaks(-ranges, prominence=prom)

    in_degrees = []
    for a in range(len(angles)):
        in_degrees.append(np.degrees(angles[a]))

    for c in range(len(angles)):
        if ranges[c] >= 0 and np.isinf(ranges[c]) == False:
            index_range.append(1)
        else:
            index_range.append(0)
    new_peaks = []
    METERS_THRESHOLD = 30
    for p in peaks:
        if index_range[p] == 1 and p not in to_exclude and ranges[p] < METERS_THRESHOLD:
            new_peaks.append(int(p))


    final_peaks = []
    for m in range(len(new_peaks)):
        mp_current = ranges[new_peaks[m]]
        if new_peaks[m] + next_choice >= len(ranges) - 1:
            take_next = next_choice - (len(ranges) - new_peaks[m])
        else:
            take_next = new_peaks[m] + next_choice

        take_previous = new_peaks[m] - previous_choice
        mp_prev = ranges[take_previous]
        mp_next = ranges[take_next]

        avg_range = (mp_prev + mp_next) / 2
        diff_range = ranges[peaks[m]] - avg_range

        diff = ranges[take_previous] + ranges[take_next] - 2 * ranges[peaks[m]]  # 2nd derivative

        if diff > 10 and diff_range > -50:
            final_peaks.append(peaks[m])

    new_ranges = []
    new_angles = []
    for index in final_peaks:
        new_ranges.append(ranges[index])
        new_angles.append(angles[index])
    return [new_ranges, new_angles]


"""Odometry trajectory"""
ts_prev = time_stamps[0]
X_odom_history = np.empty((N, 3))
Xodometry = Xp.copy()
X_odom_history[0, :] = Xodometry

for o in range(1,N):
    Ts = time_stamps[o] - ts_prev
    Xod = Xodometry.copy()
    Xodometry = Xod + (Ts * np.array([Uf[o - 1] * np.cos(Xod[2]), Uf[o - 1] * np.sin(Xod[2]), Ua[o - 1]]))
    X_odom_history[o, :] = Xodometry
    ts_prev = time_stamps[o]

"""
           ########### Initialize and correct for t = 0 ###########
"""

range_land, angle_land = land_find(ranges[:, 0], angles, prom, 0)

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
    Pp = F @ Pp @ F.T + G @ Q @ G.T

    """
        ########### Correction ###########
    """
    for_correction = []
    number_of_new_initializations = 0
    range_land, angle_land = land_find(ranges[:, i], angles, prom, i)

    # check the new measurements
    Xp_new = Xp.copy()

    for l in range(len(range_land)):
        # for each new measurements find the min d_jk
        range_m = range_land[l]
        angle_m = angle_land[l]
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
    ts_prev = time_stamps[i]
    i += 1


pose_pred = X_pred[:, :n_upper][:to_check]
pose_true = Pose[:to_check]
pose_odom_pred = X_odom_history[:, :n_upper]



landmark_pred = Xp[3:].reshape(-1, 2)
print('Number of landmarks: ', len(landmark_pred))

T = time_stamps.copy() # Time
T = T[:to_check]

""" Plots and comparisons """
fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(3, 1, 1)
ax1.plot(T, pose_true[:, 0], label=r'$x_1(t)$')
ax1.plot(T, pose_pred[:, 0], label=r'$\hat{x}_1(t)$', color='red', linestyle='--')
ax1.plot(T, pose_odom_pred[:, 0], label=r'$\hat{x}_{1, odom}(t)$', color='black', linestyle=':')

plt.xlabel("$t$")
plt.ylabel("$x_1(t)$")
plt.title("Velocity $x_1(t)$")
plt.legend(loc='upper right')

ax1 = plt.subplot(3, 1, 2)
ax1.plot(T, pose_true[:, 1], label=r'$x_2(t)$')
ax1.plot(T, pose_pred[:, 1], label=r'$\hat{x}_2(t)$', color='red', linestyle='--')
ax1.plot(T, pose_odom_pred[:, 1], label=r'$\hat{x}_{2,odom}(t)$', color='black', linestyle=':')
plt.xlabel("$t$")
plt.ylabel("$x_2(t)$")
plt.title("Current $x_2(t)$")
plt.legend(loc='upper right')

ax1 = plt.subplot(3, 1, 3)
ax1.plot(T, pose_true[:, 2], label=r'$\theta(t)$')
ax1.plot(T, pose_pred[:, 2], label=r'$\hat{\theta}(t)$', color='red', linestyle='--')
ax1.plot(T, pose_odom_pred[:, 2], label=r'$\hat{\theta}_{odom}(t)$', color='black', linestyle=':')
plt.xlabel("$t$ (s)")
plt.ylabel(r"$\theta(t)$ (rad)")
plt.title("Robot Orientation")
plt.legend(loc='upper right')
if save_fig:
    plt.savefig(f'Figures_Part_D/Comparison_data.eps', format='eps')



""" Matrices for the lidar scans"""

true_meas = []
pred_meas = []
for p in range(to_check):
    ranges_t = ranges[:, p]
    for meas_idx in range(0, len(ranges_t)):
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

""" Lidar scans from the robot pose provided by the odometry """
plt.figure()
plt.scatter(true_meas[:, 0], true_meas[:, 1], label='Lidar scans',
            color='green', marker='o', s=0.3)
plt.legend()
plt.title(r'Lidar scans from the robot pose provided by the odometry')

if save_fig:
    plt.savefig(f'Figures_Part_D/Odom_Lidar_Scans.png', dpi=300)

""" Lidar scans from the estimated robot pose """
plt.figure()
plt.scatter(pred_meas[:, 0], pred_meas[:, 1], label='Lidar scans',
            color='blue', marker='o', s=0.3)
plt.legend()
plt.title(r'Lidar scans from the estimated robot pose')
if save_fig:
    plt.savefig(f'Figures_Part_D/Estimated_Lidar_Scans.png', dpi=300)

""" Estimated landmarks over Lidar scans from the estimated robot pose + trajectory"""
plt.figure()
plt.scatter(pred_meas[:, 0], pred_meas[:, 1], label='Lidar Scans',
            color='blue', marker='o', s=0.3)
plt.scatter(landmark_pred[:, 0], landmark_pred[:, 1], label='Estimated Landmarks',
            color='orange', marker='o', s=10)
plt.plot(pose_pred[:, 0], pose_pred[:, 1], label=r'Trajectory', color='red', linestyle='--')
plt.legend()
plt.title(r'Estimated landmarks over lidar scans(from robot pose)')
if save_fig:
    plt.savefig(f'Figures_Part_D/Landmarks_and_Lidar_Scans.png', dpi=300)



"""Confidence intervals"""
plt.figure(figsize=(6, 12))
plt.subplot(311)
plt.plot(T, pose_true[:, 0] - pose_pred[:, 0], 'r', T, 3 * np.sqrt(P_pred[:, 0]), 'b--', T, -3 * np.sqrt(P_pred[:, 0]),
         'b--')
plt.ylabel(r'$x(t)-\hat{x}(t)$')
title = r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $x(t)$'
plt.title(title)
plt.axis([T[0], T[-1], -np.sqrt(P_pred[-1, 0]) * 20, np.sqrt(P_pred[-1, 0]) * 20])
#
plt.subplot(312)
plt.plot(T, pose_true[:, 1] - pose_pred[:, 1], 'r', T, 3 * np.sqrt(P_pred[:, 1]), 'b--', T, -3 * np.sqrt(P_pred[:, 1]),
         'b--')
plt.ylabel(r'$y(t)-\hat{y}(t)$')
title = r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $y(t)$'
plt.title(title)
plt.axis([T[0], T[-1], -np.sqrt(P_pred[-1, 1]) * 20, np.sqrt(P_pred[-1, 1]) * 20])
#
plt.subplot(313)
plt.plot(T, 180 / np.pi * np.unwrap(pose_true[:, 2] - pose_pred[:, 2]), 'r', T, 3 * 180 / np.pi * np.sqrt(P_pred[:, 2]),
         'b--', T,
         -3 * 180 / np.pi * np.sqrt(P_pred[:, 2]), 'b--')
plt.xlabel('$t$')
plt.ylabel(r'$\theta(t)-\hat{\theta}(t)$')
plt.title(r'Estimation error (red) and $3\sigma$-confidence intervals (blue) for $\theta(t)$')
if save_fig:
    plt.savefig(f'Figures_Part_D/Confidence_intervals.eps', format='eps')

"""Confidence Ellipses"""
r_x = 9.21  # 99% confidence ellipse
consistent = []
radii_all = []
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(pred_meas[:, 0], pred_meas[:, 1], label='Lidar Scans', color='green', marker='o', s=0.3)


for j in range(len(landmark_pred)): #for all landmarks
    idx = n_upper + 2 * j
    L_est = X_pred[-1][idx:idx + 2]
    Pj = Pp[idx:idx + 2, idx:idx + 2]

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
    ax.scatter(L_est[0], L_est[1], label='Estimated Landmarks',
               color='red', marker='o', s=5)
    if j == 0:
        ax.legend(['Lidar Scans', 'Confidence Ellipse', 'Estimated Landmarks'])
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Landmark position estimates with 99% confidence ellipses')
plt.grid(True)
if save_fig:
    plt.savefig(f'Figures_Part_D/Confidence_ellipses.png', dpi=300)

if not save_fig:
    plt.show()
print('Finished execution')