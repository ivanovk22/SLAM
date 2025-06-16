import numpy as np
eta = 100000
n_upper = 3

def angle_wrap(angle):
    # angle = (angle + np.pi) % (2 * np.pi) - np.pi
    angle = (angle + np.pi/2) % (2 * np.pi) - np.pi/2

    return angle

def append_to_array(arr, to_append):
    arr = arr.tolist()
    arr.append(to_append)
    return np.array(arr)

def extend_P(p, num_new_landmarks):

    P_new = np.block([[p, np.zeros((p.shape[0], 2*num_new_landmarks))],
                   [np.zeros((2*num_new_landmarks, p.shape[0])), eta * np.eye(2*num_new_landmarks)]])


    return P_new

def innovation(M, xp):
    landmarks = M[:, 2]
    # print(f'M: {M}')
    # print(f'Landmarks: {landmarks}')

    # take the predictions
    predicted_meas = np.zeros((len(landmarks), 2))
    for i in range(len(landmarks)):
        j = int(landmarks[i])

        lx = xp[n_upper + 2 * j]
        ly = xp[n_upper + 2 * j + 1]
        # print(f'lx = {lx}, ly = {ly}')
        predicted_p = np.sqrt((lx - xp[0]) ** 2 + (ly - xp[1]) ** 2)
        predicted_alpha = angle_wrap(np.arctan2(ly - xp[1], lx - xp[0]) - xp[2])
        # print(f'predicted_p = {predicted_p}, predicted_alpha = {predicted_alpha}')
        predicted_meas[i][0] = predicted_p
        predicted_meas[i][1] = predicted_alpha
    inn = M[:, :2] - predicted_meas
    return inn.reshape(-1, 1)

def max_likelihood(r, alpha, x, p, Rmat):
    landmarks = x[3:]
    landmarks = landmarks.reshape(int(len(landmarks)/2), 2)
    distances = []
    robot_x = x[0]
    robot_y = x[1]
    robot_theta = x[2]

    for k in range(len(landmarks)):
        H = np.zeros((2, len(x)))
        l_x = landmarks[k][0]
        l_y = landmarks[k][1]
        new_p = r -  np.sqrt((l_x - robot_x) ** 2 + (l_y - robot_y) ** 2)
        new_alpha = angle_wrap(alpha - angle_wrap(np.arctan2(l_y - robot_y, l_x - robot_x) - robot_theta))

        delta = np.array([new_p, new_alpha]).reshape(-1, 1)
        p_den = np.sqrt((l_x - robot_x) ** 2 + (l_y - robot_y) ** 2)

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
        d = delta.T @ np.linalg.inv(H@p@H.T + Rmat) @ delta
        distances.append(d[0][0])
    # find the minimum distance
    min_d = 10000000000000000
    min_d_index = -1
    for i in range(len(distances)):
        if distances[i] < min_d:
            min_d = distances[i]
            min_d_index = i

    return min_d, min_d_index

def correction(xp, pp, M, Rmat):
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

    R_new = np.kron(np.eye(len(ranges)), Rmat)
    # Equations
    K = pp @ H.T @ np.linalg.inv(H @ pp @ H.T + R_new)
    # print('K:', K)

    P = (np.eye(n_states) - K @ H) @ pp
    measurement = np.array([ranges, angles, indexes]).T
    inn = innovation(measurement, xp)
    # print('inn:', inn)
    # print(f'shapes: K:{K.shape}, inn:{inn.shape}')
    # print(f'K@inn: {(K@inn).reshape(-1)}')
    # print(f'xp: {xp}')
    # print(f'X: {xp + (K @ inn).reshape(-1)}')
    X = xp + (K @ inn).reshape(-1)
    xp = X.copy()

    pp = P.copy()
    return xp, pp


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


def pad_to_max_size(matrices, pad_value=1000):
    max_dim = max(m.shape[0] for m in matrices)  # find max size

    padded = []
    for m in matrices:
        dim = m.shape[0]
        if dim == max_dim:
            padded.append(m)
        else:
            # create a max_dim x max_dim matrix filled with pad_value
            big_m = np.full((max_dim, max_dim), pad_value)
            # copy m into the top-left corner
            big_m[:dim, :dim] = m
            padded.append(big_m)

    return np.array(padded)