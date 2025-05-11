import numpy as np
import matplotlib.pyplot as plt

# from scipy.linalg import block_diag # Not strictly needed if using np.block

plt.close('all')

data = np.load('data_point_land_1.npz', allow_pickle=True)

Meas = data['Meas']
Uf_all = data['Uf'].flatten()  # Ensure Uf is 1D array
Ua_all = data['Ua'].flatten()  # Ensure Ua is 1D array
Q_orig = data['Q']  # Renamed to avoid conflict
Qturn_orig = data['Qturn']  # Renamed
R_matrix = data['R']  # Renamed
Nland = data['Nland'].item()  # Get scalar value
Ts = data['Ts'].item()  # Get scalar value
Wturn = data['wturn'].item()  # Get scalar value
Pose_true = data['Pose']  # Renamed
Landmarks_true = data['Landmarks']  # Renamed

range_land_all = Meas.item()['range']
angle_land_all = Meas.item()['angle']
index_land_all = Meas.item()['land']


def angle_wrap(angle):
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle


n_upper = 3
n_lower = Nland * 2
n = n_upper + n_lower


def innovation(M_meas_data, xp_current):  # M_meas_data is array of [range, angle, index]
    predicted_meas = np.zeros((M_meas_data.shape[0], 2))
    for idx in range(M_meas_data.shape[0]):
        meas_range, meas_angle, landmark_idx_1based = M_meas_data[idx]
        j_lm = int(landmark_idx_1based - 1)  # 0-based landmark index

        lx_lm = xp_current[n_upper + 2 * j_lm]
        ly_lm = xp_current[n_upper + 2 * j_lm + 1]

        predicted_p = np.sqrt((lx_lm - xp_current[0]) ** 2 + (ly_lm - xp_current[1]) ** 2)
        predicted_alpha = angle_wrap(np.arctan2(ly_lm - xp_current[1], lx_lm - xp_current[0]) - xp_current[2])
        predicted_meas[idx, 0] = predicted_p
        predicted_meas[idx, 1] = predicted_alpha

    inn_val = M_meas_data[:, :2] - predicted_meas
    # Ensure angle residual is wrapped
    for r_idx in range(inn_val.shape[0]):
        inn_val[r_idx, 1] = angle_wrap(inn_val[r_idx, 1])
    return inn_val.reshape(-1, 1)  # Flatten to column vector


N = Uf_all.shape[0]  # Number of time steps

# --- Initialization (State at t=0) ---
x0 = np.zeros(n)
x0[0] = Pose_true[0, 0]
x0[1] = Pose_true[0, 1]
x0[2] = Pose_true[0, 2]

lambda_ = 1e-6  # Per document, initial pose known, so small cov
P_upper = lambda_ * np.eye(n_upper)
upper_zeros = np.zeros((n_upper, n_lower))
eta = 1000  # Per document, landmark cov large
P_lower = eta * np.eye(n_lower)
lower_zeros = np.zeros((n_lower, n_upper))
P0 = np.block([[P_upper, upper_zeros],
               [lower_zeros, P_lower]])

# Xp and Pp will represent Xp(t|t) and Pp(t|t) at the end of each iteration t
Xp = x0.copy()  # This is Xp(0|0) based on prior x0
Pp = P0.copy()  # This is Pp(0|0) based on prior P0

X_pred_history = np.empty((N, n))  # To store Xp(t|t) for each t
X_pred_history[0, :] = Xp  # Store initial state for t=0

Q_straight = Q_orig.copy()
checked_landmarks = []

print('Initial Xp(0|0): [x, y, theta] = ', Xp[:n_upper])
print('----------------')

# --- Main EKF Loop (t goes from 1 to N-1) ---
# In iteration t:
# 1. Predict Xp(t|t-1), Pp(t|t-1) from Xp(t-1|t-1), Pp(t-1|t-1) using Uf_all[t-1], Ua_all[t-1]
# 2. Correct Xp(t|t-1), Pp(t|t-1) to Xp(t|t), Pp(t|t) using Meas[t]

for t in range(1, N):
    # At the start of this iteration, Xp and Pp are Xp(t-1|t-1) and Pp(t-1|t-1)

    # --- 1. PREDICTION Step (from t-1 to t) ---
    Xp_prev_corrected = Xp.copy()  # This is Xp(t-1|t-1)
    Pp_prev_corrected = Pp.copy()  # This is Pp(t-1|t-1)
    theta_prev_corrected = Xp_prev_corrected[2]

    # Control inputs from time t-1
    Uf_input = Uf_all[t - 1]
    Ua_input = Ua_all[t - 1]

    # Predict state mean Xp_predicted_for_t = Xp(t|t-1)
    Xp_predicted_for_t = Xp_prev_corrected.copy()
    Xp_predicted_for_t[0] = Xp_prev_corrected[0] + Ts * Uf_input * np.cos(theta_prev_corrected)
    Xp_predicted_for_t[1] = Xp_prev_corrected[1] + Ts * Uf_input * np.sin(theta_prev_corrected)
    Xp_predicted_for_t[2] = theta_prev_corrected + Ts * Ua_input
    Xp_predicted_for_t[2] = angle_wrap(Xp_predicted_for_t[2])
    # Landmarks are static, so Xp_predicted_for_t[n_upper:] remains same as Xp_prev_corrected[n_upper:]

    # Calculate Jacobians F and G based on Xp_prev_corrected and Uf_input, Ua_input
    dUpper_dx = np.array([[1, 0, -Ts * Uf_input * np.sin(theta_prev_corrected)],
                          [0, 1, Ts * Uf_input * np.cos(theta_prev_corrected)],
                          [0, 0, 1]])
    dLower_dx = np.eye(n_lower)
    F = np.block([[dUpper_dx, upper_zeros],
                  [lower_zeros, dLower_dx]])

    # Corrected G (Jacobian of f w.r.t process noise w=[wf, wa]')
    dupper_dw = np.array([[-Ts * np.cos(theta_prev_corrected), 0],
                          [-Ts * np.sin(theta_prev_corrected), 0],
                          [0, -Ts]])
    dlower_dw = np.zeros((n_lower, 2))
    G = np.block([[dupper_dw],
                  [dlower_dw]])

    # Select Q based on Ua_input (angular velocity at t-1)
    current_Q = Q_straight if np.abs(Ua_input) <= Wturn else Qturn_orig

    # Predict covariance Pp_predicted_for_t = Pp(t|t-1)
    Pp_predicted_for_t = F @ Pp_prev_corrected @ F.T + G @ current_Q @ G.T

    # Now Xp_predicted_for_t and Pp_predicted_for_t are the *predicted* state/cov for current time t

    # --- 2. CORRECTION Step (at time t, using Meas[t]) ---
    # Use Xp_predicted_for_t and Pp_predicted_for_t as the basis for correction

    Xp_current_iter = Xp_predicted_for_t.copy()  # State to be updated by correction
    Pp_current_iter = Pp_predicted_for_t.copy()  # Covariance to be updated

    # Measurements for the current time step t
    current_ranges = range_land_all[t]
    current_angles = angle_land_all[t]
    current_indices = index_land_all[t]

    # --- Landmark Initialization & Correction Logic (using Xp_current_iter, Pp_current_iter and current_ranges/angles/indices) ---
    # This section needs to be adapted for sequential or your batch processing
    # For simplicity, let's assume your batch processing structure for "to_be_corrected"
    # and handle initialization first.

    # Separate new landmarks for initialization
    new_landmark_meas_data = []  # List to store [range, angle, index_1based] for new landmarks
    old_landmark_meas_data = []  # List to store [range, angle, index_1based] for old landmarks

    if len(current_indices) > 0:  # Check if there are any measurements at this time step
        for l_idx in range(len(current_indices)):
            landmark_id_1based = current_indices[l_idx]
            meas_r = current_ranges[l_idx]
            meas_a = current_angles[l_idx]

            if landmark_id_1based not in checked_landmarks:
                new_landmark_meas_data.append([meas_r, meas_a, landmark_id_1based])
                checked_landmarks.append(landmark_id_1based)  # Mark as seen for initialization
            else:
                old_landmark_meas_data.append([meas_r, meas_a, landmark_id_1based])

    # Initialize new landmarks
    for meas_data in new_landmark_meas_data:
        mp_init, ma_init, landmark_id_1based_init = meas_data
        j_init = int(landmark_id_1based_init - 1)  # 0-based index

        # Use robot pose from Xp_current_iter (which is Xp(t|t-1))
        lx_init = Xp_current_iter[0] + mp_init * np.cos(Xp_current_iter[2] + ma_init)
        ly_init = Xp_current_iter[1] + mp_init * np.sin(Xp_current_iter[2] + ma_init)
        Xp_current_iter[n_upper + 2 * j_init] = lx_init
        Xp_current_iter[n_upper + 2 * j_init + 1] = ly_init
        # Note: Covariance for these initialized landmarks is handled by the subsequent correction step
        # if they are also in old_landmark_meas_data (which they won't be in this pass if truly new)
        # or relies on the initial large Pp_current_iter for their parts.
        # A more robust initialization might also adjust parts of Pp_current_iter here,
        # or ensure these "new" landmarks are also processed by a correction step if possible.
        # For now, we just update the mean state.

    # Correction for already seen landmarks (using batch approach from your code)
    if old_landmark_meas_data:  # If there are landmarks to correct
        num_old_landmarks = len(old_landmark_meas_data)
        H = np.zeros((num_old_landmarks * 2, n))

        # Convert to numpy array for easier processing in innovation function
        measurement_for_correction = np.array(old_landmark_meas_data)

        for k_corr in range(num_old_landmarks):
            # meas_r_corr, meas_a_corr, landmark_id_1based_corr = old_landmark_meas_data[k_corr]
            # Use data from measurement_for_correction for consistency with innovation fn
            landmark_id_1based_corr = measurement_for_correction[k_corr, 2]
            j_corr = int(landmark_id_1based_corr - 1)

            lx_corr = Xp_current_iter[n_upper + 2 * j_corr]
            ly_corr = Xp_current_iter[n_upper + 2 * j_corr + 1]

            row_p = k_corr * 2
            row_alpha = k_corr * 2 + 1

            p_den_sq = (lx_corr - Xp_current_iter[0]) ** 2 + (ly_corr - Xp_current_iter[1]) ** 2
            p_den = np.sqrt(p_den_sq)

            if p_den > 1e-6:  # Avoid division by zero
                H[row_p, 0] = (Xp_current_iter[0] - lx_corr) / p_den
                H[row_p, 1] = (Xp_current_iter[1] - ly_corr) / p_den
                # H[row_p, 2] = 0 # dtheta is 0 for range
                H[row_p, n_upper + 2 * j_corr] = (lx_corr - Xp_current_iter[0]) / p_den
                H[row_p, n_upper + 2 * j_corr + 1] = (ly_corr - Xp_current_iter[1]) / p_den

                if p_den_sq > 1e-6:  # For alpha_den
                    H[row_alpha, 0] = (ly_corr - Xp_current_iter[1]) / p_den_sq
                    H[row_alpha, 1] = -(lx_corr - Xp_current_iter[0]) / p_den_sq
                    H[row_alpha, n_upper + 2 * j_corr] = -(ly_corr - Xp_current_iter[1]) / p_den_sq
                    H[row_alpha, n_upper + 2 * j_corr + 1] = (lx_corr - Xp_current_iter[0]) / p_den_sq
            H[row_alpha, 2] = -1  # dtheta for angle

        R_batch = np.kron(np.eye(num_old_landmarks), R_matrix)

        try:
            S_inv = np.linalg.inv(H @ Pp_current_iter @ H.T + R_batch)
            K = Pp_current_iter @ H.T @ S_inv

            # Pp_updated = Pp_current_iter - K @ H @ Pp_current_iter # Simpler form
            # Joseph form for P update (more stable)
            I_n = np.eye(n)
            Pp_updated = (I_n - K @ H) @ Pp_current_iter @ (I_n - K @ H).T + K @ R_batch @ K.T

            inn_val = innovation(measurement_for_correction, Xp_current_iter)

            Xp_updated = Xp_current_iter + (K @ inn_val).flatten()  # K @ inn_val might be (n,1)
            Xp_updated[2] = angle_wrap(Xp_updated[2])

            Xp_current_iter = Xp_updated
            Pp_current_iter = Pp_updated
        except np.linalg.LinAlgError:
            print(f"Warning: Matrix inversion failed in correction at time {t}. Skipping correction.")

    # After all corrections (init and updates), Xp_current_iter and Pp_current_iter are Xp(t|t), Pp(t|t)
    Xp = Xp_current_iter
    Pp = Pp_current_iter

    X_pred_history[t, :] = Xp  # Store corrected state Xp(t|t)

    if t % 100 == 0 or t == N - 1:  # Print progress
        print(f"Processed time t={t}: Xp[:3]={Xp[:n_upper].T}")

# --- Plotting (using your existing plotting code) ---
pose_pred = X_pred_history[:, :n_upper]
pose_true = Pose_true  # Assuming Pose_true is shaped (N, 3)

# Ensure T_plot matches the number of stored estimates
T_plot = np.arange(0, N * Ts, Ts)[:N]  # Match length of pose_pred/pose_true

fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(3, 1, 1)
ax1.plot(T_plot, pose_true[:, 0], label=r'True $x(t)$')
ax1.plot(T_plot, pose_pred[:, 0], label=r'Estimated $\hat{x}(t)$', linestyle='--')
plt.xlabel("$t$ (s)")
plt.ylabel("$x$-position (m)")
plt.title("Robot X Position")
plt.legend()
plt.grid(True)

ax1 = plt.subplot(3, 1, 2)
ax1.plot(T_plot, pose_true[:, 1], label=r'True $y(t)$')
ax1.plot(T_plot, pose_pred[:, 1], label=r'Estimated $\hat{y}(t)$', linestyle='--')
plt.xlabel("$t$ (s)")
plt.ylabel("$y$-position (m)")
plt.title("Robot Y Position")
plt.legend()
plt.grid(True)

ax1 = plt.subplot(3, 1, 3)
ax1.plot(T_plot, angle_wrap(pose_true[:, 2]), label=r'True $\theta(t)$')  # Wrap true theta for comparison
ax1.plot(T_plot, pose_pred[:, 2], label=r'Estimated $\hat{\theta}(t)$', linestyle='--')
plt.xlabel("$t$ (s)")
plt.ylabel(r"Orientation $\theta(t)$ (rad)")
plt.title("Robot Orientation")
plt.legend()
plt.grid(True)

fig.tight_layout()
plt.show()

fig_traj = plt.figure(figsize=(8, 8))
plt.plot(pose_true[:, 0], pose_true[:, 1], label='True Trajectory', color='blue')
plt.plot(pose_pred[:, 0], pose_pred[:, 1], label='Estimated Trajectory (EKF)', color='red', linestyle='--')
plt.scatter(Landmarks_true[:, 0], Landmarks_true[:, 1], label='True Landmarks', color='green', marker='x', s=100)

# Plot final estimated landmark positions
final_landmark_estimates_flat = X_pred_history[-1, n_upper:]
final_landmark_estimates = final_landmark_estimates_flat.reshape(-1, 2)
plt.scatter(final_landmark_estimates[:, 0], final_landmark_estimates[:, 1], label='Estimated Landmarks (Final)',
            color='purple', marker='o', s=50, facecolors='none')

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Robot Trajectory and Landmarks")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

print("\nFinal Estimated Robot Pose:", pose_pred[-1, :])
print("Final True Robot Pose:", pose_true[-1, :])
print("\nFinal Estimated Landmark Positions:\n", final_landmark_estimates)
print("True Landmark Positions:\n", Landmarks_true)


