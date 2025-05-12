# Simplified and complete EKF-SLAM with full correction at initialization
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('data_point_land_1.npz', allow_pickle=True)
Meas = data['Meas'].item()
Uf, Ua = data['Uf'].flatten(), data['Ua'].flatten()
Q, Qturn, R = data['Q'], data['Qturn'], data['R']
Nland, Ts, Wturn = int(data['Nland']), float(data['Ts']), float(data['wturn'])
Pose, Landmarks = data['Pose'], data['Landmarks']

# Parameters
N = Uf.shape[0]
n_upper, n_lower = 3, 2 * Nland
n = n_upper + n_lower
x0 = np.zeros(n)
x0[:3] = Pose[0]
P0 = np.diag([1e-6] * n_upper + [1e3] * n_lower)

# Helpers
def angle_wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def correct(x, P, r, a, j):
    j = int(j) - 1
    lx, ly = x[n_upper + 2*j:n_upper + 2*j+2]
    dx, dy = lx - x[0], ly - x[1]
    d2 = dx**2 + dy**2
    if d2 < 1e-9: return x, P
    r_hat = np.sqrt(d2)
    a_hat = angle_wrap(np.arctan2(dy, dx) - x[2])
    y = np.array([[r], [a]])
    h = np.array([[r_hat], [a_hat]])
    inn = y - h
    H = np.zeros((2, n))
    H[0, 0], H[0, 1] = -dx / r_hat, -dy / r_hat
    H[0, n_upper + 2*j], H[0, n_upper + 2*j + 1] = dx / r_hat, dy / r_hat
    H[1, 0], H[1, 1], H[1, 2] = dy / d2, -dx / d2, -1
    H[1, n_upper + 2*j], H[1, n_upper + 2*j + 1] = -dy / d2, dx / d2
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + (K @ inn).flatten()
    I = np.eye(n)
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    return x, P

# Initialization
Xp, Pp = x0.copy(), P0.copy()
X_hist = np.zeros((N, n))
seen = set()

for r, a, j in zip(Meas['range'][0], Meas['angle'][0], Meas['land'][0]):
    j = int(j)
    if j not in seen:
        lx = Xp[0] + r * np.cos(Xp[2] + a)
        ly = Xp[1] + r * np.sin(Xp[2] + a)
        Xp[n_upper + 2*(j-1):n_upper + 2*(j-1)+2] = [lx, ly]
        seen.add(j)
    Xp, Pp = correct(Xp, Pp, r, a, j)
X_hist[0] = Xp

# EKF loop
dZ = np.zeros((n_lower, 2))
for t in range(1, N):
    u_f, u_a = Uf[t-1], Ua[t-1]
    theta = Xp[2]
    dx, dy, dtheta = Ts * u_f * np.cos(theta), Ts * u_f * np.sin(theta), Ts * u_a
    Xp[:3] += [dx, dy, dtheta]

    F = np.eye(n)
    F[0, 2], F[1, 2] = -Ts * u_f * np.sin(theta), Ts * u_f * np.cos(theta)
    G = np.zeros((n, 2))
    G[0, 0], G[1, 0], G[2, 1] = Ts * np.cos(theta), Ts * np.sin(theta), Ts
    Qk = Qturn if abs(u_a) > Wturn else Q
    Pp = F @ Pp @ F.T + G @ Qk @ G.T

    for r, a, j in zip(Meas['range'][t], Meas['angle'][t], Meas['land'][t]):
        j = int(j)
        if j not in seen:
            lx = Xp[0] + r * np.cos(Xp[2] + a)
            ly = Xp[1] + r * np.sin(Xp[2] + a)
            Xp[n_upper + 2*(j-1):n_upper + 2*(j-1)+2] = [lx, ly]
            seen.add(j)
        Xp, Pp = correct(Xp, Pp, r, a, j)

    X_hist[t] = Xp

# Plotting
T = np.arange(0, N*Ts, Ts)
pose_true, pose_est = Pose, X_hist[:, :3]

plt.figure()
plt.subplot(3,1,1)
plt.plot(T, pose_true[:,0], label='True x')
plt.plot(T, pose_est[:,0], '--', label='Est x')
plt.legend(); plt.ylabel('x (m)')
plt.subplot(3,1,2)
plt.plot(T, pose_true[:,1], label='True y')
plt.plot(T, pose_est[:,1], '--', label='Est y')
plt.legend(); plt.ylabel('y (m)')
plt.subplot(3,1,3)
plt.plot(T, pose_true[:,2], label='True theta')
plt.plot(T, pose_est[:,2], '--', label='Est theta')
plt.legend(); plt.ylabel('theta (rad)'); plt.xlabel('Time (s)')
plt.tight_layout(); plt.show()

# Trajectory and landmarks
plt.figure()
plt.plot(pose_true[:,0], pose_true[:,1], label='True Traj')
plt.plot(pose_est[:,0], pose_est[:,1], '--', label='Est Traj')
plt.scatter(Landmarks[:,0], Landmarks[:,1], c='g', marker='x', label='True LM')
LM_est = X_hist[-1, n_upper:].reshape(-1,2)
plt.scatter(LM_est[:,0], LM_est[:,1], edgecolor='purple', facecolor='none', label='Est LM')
plt.legend(); plt.title('Trajectory and Landmarks'); plt.axis('equal'); plt.grid(); plt.show()
