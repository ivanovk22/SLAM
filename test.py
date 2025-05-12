import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

data = np.load('data_point_land_1.npz', allow_pickle=True)

Meas_dict = data['Meas']
Uf_all = data['Uf'].flatten()
Ua_all = data['Ua'].flatten()
Q_orig = data['Q']
Qturn_orig = data['Qturn']
R_matrix = data['R'] # Changed name back for consistency with prior correction step
Nland = data['Nland'].item()
Ts = data['Ts'].item()
Wturn = data['wturn'].item()
Pose_true = data['Pose']
Landmarks_true = data['Landmarks']

range_land_all = Meas_dict.item()['range']
angle_land_all = Meas_dict.item()['angle']
index_land_all = Meas_dict.item()['land']

def angle_wrap(angle):
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

n_upper = 3
n_lower = Nland * 2
n = n_upper + n_lower

N_timesteps = Uf_all.shape[0]

# --- Initialization (State PRIOR to t=0 correction) ---
x0 = np.zeros(n)
x0[0] = Pose_true[0, 0]
x0[1] = Pose_true[0, 1]
x0[2] = Pose_true[0, 2]

lambda_ = 1e-6
P_upper = lambda_ * np.eye(n_upper)
upper_zeros = np.zeros((n_upper, n_lower))
eta = 1000
P_lower = eta * np.eye(n_lower)
lower_zeros = np.zeros((n_lower, n_upper))
P0 = np.block([[P_upper, upper_zeros], [lower_zeros, P_lower]])

# Xp and Pp start as Xp(0|-1) and Pp(0|-1)
Xp = x0.copy()
Pp = P0.copy()

X_pred_history = np.empty((N_timesteps, n))
# P_pred_history = np.empty((N_timesteps, n, n)) # Optional: Store covariance history

Q_straight = Q_orig.copy()
checked_landmarks = [] # Keep track of initialized landmarks

print(f'Initial Xp(0|-1): [x, y, theta] = {Xp[:n_upper]}')
print('----------------')

# --- Initial Correction Step for t=0 using Meas[0] ---
t0 = 0 # Explicitly using index 0
Xp_for_correction_iter_t0 = Xp.copy() # Start with Xp(0|-1)
Pp_for_correction_iter_t0 = Pp.copy() # Start with Pp(0|-1)

current_ranges_at_t0 = range_land_all[t0]
current_angles_at_t0 = angle_land_all[t0]
current_indices_at_t0 = index_land_all[t0] # 1-based

measurements_to_process_seq_t0 = []
if len(current_indices_at_t0) > 0:
    for l_idx_loop_t0 in range(len(current_indices_at_t0)):
        measurements_to_process_seq_t0.append({
            'range': current_ranges_at_t0[l_idx_loop_t0],
            'angle': current_angles_at_t0[l_idx_loop_t0],
            'id_1based': current_indices_at_t0[l_idx_loop_t0]
        })

print(f"Performing initial correction at t=0 with {len(measurements_to_process_seq_t0)} measurements.")

for meas_data_seq_t0 in measurements_to_process_seq_t0:
    mj_range_item_t0 = meas_data_seq_t0['range']
    mj_angle_item_t0 = meas_data_seq_t0['angle']
    landmark_id_1based_item_t0 = meas_data_seq_t0['id_1based']
    j_lm_0based_item_t0 = int(landmark_id_1based_item_t0 - 1)

    x_robot_iter_t0, y_robot_iter_t0, theta_robot_iter_t0 = Xp_for_correction_iter_t0[0:n_upper].flatten()

    if landmark_id_1based_item_t0 not in checked_landmarks:
        # print(f"Time {t0}: Initializing landmark {landmark_id_1based_item_t0}")
        lx_init_item_t0 = x_robot_iter_t0 + mj_range_item_t0 * np.cos(theta_robot_iter_t0 + mj_angle_item_t0)
        ly_init_item_t0 = y_robot_iter_t0 + mj_range_item_t0 * np.sin(theta_robot_iter_t0 + mj_angle_item_t0)
        Xp_for_correction_iter_t0[n_upper + 2*j_lm_0based_item_t0] = lx_init_item_t0
        Xp_for_correction_iter_t0[n_upper + 2*j_lm_0based_item_t0 + 1] = ly_init_item_t0
        checked_landmarks.append(landmark_id_1based_item_t0)

    Lj_x_iter_t0 = Xp_for_correction_iter_t0[n_upper + 2*j_lm_0based_item_t0]
    Lj_y_iter_t0 = Xp_for_correction_iter_t0[n_upper + 2*j_lm_0based_item_t0 + 1]

    dx_iter_t0 = Lj_x_iter_t0 - x_robot_iter_t0
    dy_iter_t0 = Lj_y_iter_t0 - y_robot_iter_t0
    d_sq_iter_t0 = dx_iter_t0**2 + dy_iter_t0**2

    if d_sq_iter_t0 < 1e-9: continue

    d_pred_iter_t0 = np.sqrt(d_sq_iter_t0)
    h_angle_pred_iter_t0 = angle_wrap(np.arctan2(dy_iter_t0, dx_iter_t0) - theta_robot_iter_t0)
    h_vec_iter_t0 = np.array([[d_pred_iter_t0], [h_angle_pred_iter_t0]])
    y_meas_iter_t0 = np.array([[mj_range_item_t0], [mj_angle_item_t0]])
    y_tilde_iter_t0 = y_meas_iter_t0 - h_vec_iter_t0
    y_tilde_iter_t0[1, 0] = angle_wrap(y_tilde_iter_t0[1, 0])

    H_item_t0 = np.zeros((2, n))
    idx_Ljx_item_t0 = n_upper + 2*j_lm_0based_item_t0
    idx_Ljy_item_t0 = n_upper + 2*j_lm_0based_item_t0 + 1
    H_item_t0[0, 0] = -dx_iter_t0 / d_pred_iter_t0
    H_item_t0[0, 1] = -dy_iter_t0 / d_pred_iter_t0
    H_item_t0[0, idx_Ljx_item_t0] = dx_iter_t0 / d_pred_iter_t0
    H_item_t0[0, idx_Ljy_item_t0] = dy_iter_t0 / d_pred_iter_t0
    H_item_t0[1, 0] =  dy_iter_t0 / d_sq_iter_t0
    H_item_t0[1, 1] = -dx_iter_t0 / d_sq_iter_t0
    H_item_t0[1, 2] = -1
    H_item_t0[1, idx_Ljx_item_t0] = -dy_iter_t0 / d_sq_iter_t0
    H_item_t0[1, idx_Ljy_item_t0] =  dx_iter_t0 / d_sq_iter_t0

    S_item_t0 = H_item_t0 @ Pp_for_correction_iter_t0 @ H_item_t0.T + R_matrix
    try:
        K_item_t0 = Pp_for_correction_iter_t0 @ H_item_t0.T @ np.linalg.inv(S_item_t0)
        Xp_for_correction_iter_t0 = Xp_for_correction_iter_t0 + (K_item_t0 @ y_tilde_iter_t0).flatten()
        Xp_for_correction_iter_t0[2] = Xp_for_correction_iter_t0[2]
        I_n_identity_t0 = np.eye(n)
        Pp_for_correction_iter_t0 = (I_n_identity_t0 - K_item_t0 @ H_item_t0) @ Pp_for_correction_iter_t0 @ \
                                    (I_n_identity_t0 - K_item_t0 @ H_item_t0).T + \
                                    K_item_t0 @ R_matrix @ K_item_t0.T
    except np.linalg.LinAlgError:
        # print(f"Warning: Time {t0}, Landmark {landmark_id_1based_item_t0}: S matrix inversion failed.")
        continue

# After processing all t=0 measurements, update Xp and Pp to Xp(0|0), Pp(0|0)
Xp = Xp_for_correction_iter_t0
Pp = Pp_for_correction_iter_t0

# Store the corrected state for t=0
X_pred_history[0, :] = Xp
# P_pred_history[0, :, :] = Pp # Optional

print(f'Corrected Xp(0|0): [x, y, theta] = {Xp[:n_upper]}')
print('----------------')

# --- Main EKF Loop (t goes from 1 to N_timesteps-1) ---
# Now this loop starts with the corrected Xp(0|0), Pp(0|0)
for t_loop_idx in range(1, N_timesteps):
    # At the start of this iteration, Xp and Pp are Xp(t-1|t-1) and Pp(t-1|t-1)

    # --- 1. PREDICTION Step (from t-1 to t) ---
    Xp_at_t_minus_1 = Xp.copy()
    Pp_at_t_minus_1 = Pp.copy()
    theta_at_t_minus_1 = Xp_at_t_minus_1[2]
    Uf_input_prev = Uf_all[t_loop_idx - 1] # Use t_loop_idx-1 for input
    Ua_input_prev = Ua_all[t_loop_idx - 1]

    Xp_predicted_for_current_t = Xp_at_t_minus_1.copy()
    Xp_predicted_for_current_t[0] = Xp_at_t_minus_1[0] + Ts * Uf_input_prev * np.cos(theta_at_t_minus_1)
    Xp_predicted_for_current_t[1] = Xp_at_t_minus_1[1] + Ts * Uf_input_prev * np.sin(theta_at_t_minus_1)
    Xp_predicted_for_current_t[2] = theta_at_t_minus_1 + Ts * Ua_input_prev
    Xp_predicted_for_current_t[2] = Xp_predicted_for_current_t[2]

    dUpper_dx = np.array([[1, 0, -Ts * Uf_input_prev * np.sin(theta_at_t_minus_1)],
                          [0, 1, Ts * Uf_input_prev * np.cos(theta_at_t_minus_1)],
                          [0, 0, 1]])
    dLower_dx = np.eye(n_lower)
    F = np.block([[dUpper_dx, upper_zeros], [lower_zeros, dLower_dx]]) # Your F

    # Make sure G is correct (corrected signs from previous discussions)
    dupper_dw = np.array([[-Ts * np.cos(theta_at_t_minus_1), 0],
                          [-Ts * np.sin(theta_at_t_minus_1), 0],
                          [0,                               -Ts]])
    dlower_dw = np.zeros((n_lower, 2))
    G = np.block([[dupper_dw], [dlower_dw]]) # Your G

    current_Q_process_noise = Q_straight if np.abs(Ua_input_prev) <= Wturn else Qturn_orig # Your Q

    Pp_predicted_for_current_t = F @ Pp_at_t_minus_1 @ F.T + G @ current_Q_process_noise @ G.T

    # --- 2. CORRECTION Step (at current time t_loop_idx, using Meas[t_loop_idx]) ---
    Xp_for_correction_iter = Xp_predicted_for_current_t.copy()
    Pp_for_correction_iter = Pp_predicted_for_current_t.copy()

    current_ranges_at_t = range_land_all[t_loop_idx]
    current_angles_at_t = angle_land_all[t_loop_idx]
    current_indices_at_t = index_land_all[t_loop_idx] # 1-based

    measurements_to_process_seq = []
    if len(current_indices_at_t) > 0:
        for l_iter_idx in range(len(current_indices_at_t)):
            measurements_to_process_seq.append({
                'range': current_ranges_at_t[l_iter_idx],
                'angle': current_angles_at_t[l_iter_idx],
                'id_1based': current_indices_at_t[l_iter_idx]
            })

    for meas_data_seq in measurements_to_process_seq:
        mj_range_item = meas_data_seq['range']
        mj_angle_item = meas_data_seq['angle']
        landmark_id_1based_item = meas_data_seq['id_1based']
        j_lm_0based_item = int(landmark_id_1based_item - 1)

        x_robot_iter, y_robot_iter, theta_robot_iter = Xp_for_correction_iter[0:n_upper].flatten()

        if landmark_id_1based_item not in checked_landmarks:
            lx_init_item = x_robot_iter + mj_range_item * np.cos(theta_robot_iter + mj_angle_item)
            ly_init_item = y_robot_iter + mj_range_item * np.sin(theta_robot_iter + mj_angle_item)
            Xp_for_correction_iter[n_upper + 2*j_lm_0based_item] = lx_init_item
            Xp_for_correction_iter[n_upper + 2*j_lm_0based_item + 1] = ly_init_item
            checked_landmarks.append(landmark_id_1based_item)

        Lj_x_iter = Xp_for_correction_iter[n_upper + 2*j_lm_0based_item]
        Lj_y_iter = Xp_for_correction_iter[n_upper + 2*j_lm_0based_item + 1]

        dx_iter = Lj_x_iter - x_robot_iter
        dy_iter = Lj_y_iter - y_robot_iter
        d_sq_iter = dx_iter**2 + dy_iter**2

        if d_sq_iter < 1e-9: continue

        d_pred_iter = np.sqrt(d_sq_iter)
        h_angle_pred_iter = angle_wrap(np.arctan2(dy_iter, dx_iter) - theta_robot_iter)
        h_vec_iter = np.array([[d_pred_iter], [h_angle_pred_iter]])
        y_meas_iter = np.array([[mj_range_item], [mj_angle_item]])
        y_tilde_iter = y_meas_iter - h_vec_iter
        y_tilde_iter[1, 0] = y_tilde_iter[1, 0]

        H_item = np.zeros((2, n)) # Your H
        idx_Ljx_item = n_upper + 2*j_lm_0based_item
        idx_Ljy_item = n_upper + 2*j_lm_0based_item + 1
        H_item[0, 0] = -dx_iter / d_pred_iter
        H_item[0, 1] = -dy_iter / d_pred_iter
        H_item[0, idx_Ljx_item] = dx_iter / d_pred_iter
        H_item[0, idx_Ljy_item] = dy_iter / d_pred_iter
        H_item[1, 0] =  dy_iter / d_sq_iter
        H_item[1, 1] = -dx_iter / d_sq_iter
        H_item[1, 2] = -1
        H_item[1, idx_Ljx_item] = -dy_iter / d_sq_iter
        H_item[1, idx_Ljy_item] =  dx_iter / d_sq_iter

        S_item = H_item @ Pp_for_correction_iter @ H_item.T + R_matrix # Your R

        try:
            K_item = Pp_for_correction_iter @ H_item.T @ np.linalg.inv(S_item) # Your K
            Xp_for_correction_iter = Xp_for_correction_iter + (K_item @ y_tilde_iter).flatten()
            Xp_for_correction_iter[2] = Xp_for_correction_iter[2]
            I_n_identity = np.eye(n) # Your I_n
            Pp_for_correction_iter = (I_n_identity - K_item @ H_item) @ Pp_for_correction_iter @ \
                                     (I_n_identity - K_item @ H_item).T + \
                                     K_item @ R_matrix @ K_item.T
        except np.linalg.LinAlgError:
            continue

    Xp = Xp_for_correction_iter
    Pp = Pp_for_correction_iter
    X_pred_history[t_loop_idx, :] = Xp # Your X_pred history

    if t_loop_idx % 100 == 0 or t_loop_idx == N_timesteps - 1:
        print(f"Processed time t={t_loop_idx}: Xp[:3]={Xp[:n_upper]}")


# --- Plotting (Using your variable names) ---
pose_pred = X_pred_history[:, :n_upper]
pose_true = Pose_true

num_plot_points = min(pose_pred.shape[0], pose_true.shape[0])
time_axis_plot = np.arange(0, num_plot_points * Ts, Ts)[:num_plot_points]

fig_timeseries, axs_timeseries = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig_timeseries.suptitle('Robot Pose Components vs. Time', fontsize=16)
axs_timeseries[0].plot(time_axis_plot, pose_true[:num_plot_points, 0], 'b-', label='True $x$')
axs_timeseries[0].plot(time_axis_plot, pose_pred[:num_plot_points, 0], 'r--', label='Estimated $\hat{x}$')
axs_timeseries[0].set_ylabel('$x$-position (m)')
axs_timeseries[0].legend()
axs_timeseries[0].grid(True)
axs_timeseries[1].plot(time_axis_plot, pose_true[:num_plot_points, 1], 'b-', label='True $y$')
axs_timeseries[1].plot(time_axis_plot, pose_pred[:num_plot_points, 1], 'r--', label='Estimated $\hat{y}$')
axs_timeseries[1].set_ylabel('$y$-position (m)')
axs_timeseries[1].legend()
axs_timeseries[1].grid(True)
axs_timeseries[2].plot(time_axis_plot, pose_true[:num_plot_points, 2], 'b-', label='True $\\theta$')
axs_timeseries[2].plot(time_axis_plot, pose_pred[:num_plot_points, 2], 'r--', label='Estimated $\hat{\\theta}$')
axs_timeseries[2].set_ylabel('Orientation $\\theta$ (rad)')
axs_timeseries[2].set_xlabel('Time (s)')
axs_timeseries[2].legend()
axs_timeseries[2].grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

fig_trajectory = plt.figure(figsize=(10, 8))
plt.plot(pose_true[:num_plot_points, 0], pose_true[:num_plot_points, 1], 'b-', label='True Trajectory')
plt.plot(pose_pred[:num_plot_points, 0], pose_pred[:num_plot_points, 1], 'r--', label='Estimated Trajectory (EKF)')
plt.scatter(Landmarks_true[:, 0], Landmarks_true[:, 1], color='green', marker='x', s=100, label='True Landmarks')
final_landmark_state_flat = X_pred_history[-1, n_upper:]
if len(final_landmark_state_flat) > 0 :
    final_estimated_landmarks = final_landmark_state_flat.reshape(-1, 2)
    plt.scatter(final_estimated_landmarks[:, 0], final_estimated_landmarks[:, 1],
                edgecolor='purple', facecolor='none', marker='o', s=70, label='Estimated Landmarks (Final)')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Robot Trajectory and Landmark Map')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

print("\nEKF processing complete.")
print("\nFinal Estimated Robot Pose (x, y, theta):", pose_pred[-1, :])
print("Final True Robot Pose (x, y, theta):", pose_true[-1, :])
if len(final_landmark_state_flat) > 0:
    print("\nFinal Estimated Landmark Positions:\n", final_estimated_landmarks)
print("True Landmark Positions:\n", Landmarks_true)