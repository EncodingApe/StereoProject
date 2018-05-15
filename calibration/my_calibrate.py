import numpy as np
import calibrate
import cv2


def estimate_H(obj_points, img_points):
    """
    Estimate the homography matrix for each picture.
    :param obj_points: 3-D object points
    :param img_points: 2-D image points
    :return:
    """

    # turn the 3-D object coordinate into 2-D homogeneous coordinates
    obj_points[:, 2] = 1

    n = obj_points.shape[0]
    P = np.zeros((2 * n, 9))

    for i in range(n):
        P[2 * i, :3] = obj_points[i]
        P[2 * i, 6:] = -img_points[i, 0] * obj_points[i]
        P[2 * i + 1, 3:6] = obj_points[i]
        P[2 * i + 1, 6:] = -img_points[i, 1] * obj_points[i]

    eig_values, eig_vectors = np.linalg.eig(np.dot(P.T, P))

    min_val_index = np.argmin(eig_values)
    # min_val = eig_values[min_val_index]
    min_vector = eig_vectors[:, min_val_index]

    H = np.reshape(min_vector, (3, 3))
    
    # 因为单应矩阵是在齐次坐标系下的, 所以可以将任意一个尺度因子乘上它, 为了保持和OpenCV的旋转矩阵和平移矩阵同方向
    # 将单应矩阵乘上-1
    return -H


def v(i, j, H):
    ret = np.zeros((6, 1))
    ret[0, 0] = H[0, i] * H[0, j]
    ret[1, 0] = H[0, i] * H[1, j] + H[1, i] * H[0, j]
    ret[2, 0] = H[1, i] * H[1, j]
    ret[3, 0] = H[2, i] * H[0, j] + H[0, i] * H[2, j]
    ret[4, 0] = H[2, i] * H[1, j] + H[1, i] * H[2, j]
    ret[5, 0] = H[2, i] * H[2, j]

    return ret


def estimate_intrinsic(h_lst):
    """
    Estimate the intrinsic matrix K.
    :param h_lst: The list of homography matrix.
    :return:
    """

    n = len(h_lst)

    if n < 3:
        print("Not enough homography matrix")
        return

    V = np.ones((2 * n, 6))

    for i in range(n):
        V[2 * i] = v(0, 1, h_lst[i]).T
        V[2 * i + 1] = (v(0, 0, h_lst[i]) - v(1, 1, h_lst[i])).T

    eig_values, eig_vectors = np.linalg.eig(np.dot(V.T, V))

    min_val_index = np.argmin(eig_values)
    b = eig_vectors[:, min_val_index]

    B = np.zeros((3, 3))
    B[0, 0] = b[0]
    B[1, 0] = B[0, 1] = b[1]
    B[1, 1] = b[2]
    B[2, 0] = B[0, 2] = b[3]
    B[2, 1] = B[1, 2] = b[4]
    B[2, 2] = b[5]

    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
    lambda_ = B[2, 2] - (B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    alpha = np.sqrt(lambda_ / B[0, 0])
    beta = np.sqrt(lambda_ * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] ** 2))
    gamma = -B[0, 1] * (alpha ** 2) * beta / lambda_
    u0 = gamma * v0 / alpha - B[0, 2] * (alpha ** 2) / lambda_

    K = np.zeros((3, 3))
    K[0, 0] = alpha
    K[0, 1] = gamma
    K[0, 2] = u0
    K[1, 1] = beta
    K[1, 2] = v0
    K[2, 2] = 1

    return K


def estimate_extrinsic(K, H):
    R = np.zeros((3, 3))

    lambda_ = np.sum(np.array(np.mat(K).I * np.mat(H[:, 0]).T).squeeze() ** 2) ** 0.5
    R[:, 0] = np.array(np.mat(K).I * np.mat(H[:, 0]).T).squeeze() / lambda_
    R[:, 1] = np.array(np.mat(K).I * np.mat(H[:, 1]).T).squeeze() / lambda_
    r1_x = np.zeros((3, 3))
    r1_x[0, 1] = R[:, 0][2]
    r1_x[0, 2] = R[:, 0][1]
    r1_x[1, 2] = R[:, 0][0]
    r1_x[1, 0] = -r1_x[0, 1]
    r1_x[2, 0] = -r1_x[0, 2]
    r1_x[2, 1] = -r1_x[1, 2]
    R[:, 2] = np.dot(r1_x, R[:, 1])

    t = np.array(np.mat(K).I * np.mat(H[:, 2]).T).squeeze() / lambda_

    return R, t


def estimate_ideal_point(obj_points, K, R, t):
    M = np.mat(K) * np.mat(np.c_[R, t])

    n = obj_points.shape[0]
    obj_points_3D = np.zeros((n, 4))
    obj_points_3D[:, :2] = obj_points[:, :2]
    obj_points_3D[:, 3] = obj_points[:, 2]

    ideal_pixel_coordinate = M * np.mat(obj_points_3D).T
    ideal_pixel_coordinate[0] = ideal_pixel_coordinate[0] / ideal_pixel_coordinate[2]
    ideal_pixel_coordinate[1] = ideal_pixel_coordinate[1] / ideal_pixel_coordinate[2]

    ideal_image_coordinate = np.mat(np.c_[R, t]) * np.mat(obj_points_3D).T
    ideal_image_coordinate[0] = ideal_image_coordinate[0] / ideal_image_coordinate[2]
    ideal_image_coordinate[1] = ideal_image_coordinate[1] / ideal_image_coordinate[2]

    return ideal_pixel_coordinate[:2, :].T, ideal_image_coordinate[:2, :].T


def estimate_radial_distortion(image_points, ideal_points, ideal_image_points, u0, v0):
    """
    :param image_points: distorted pixel coordinate
    :param ideal_points: undistorted pixel coordinate
    :param ideal_image_points: undistorted image coordinate
    :param u0:
    :param v0:
    :return:
    """

    # m: the number of point in each image
    # n: the number of image
    m, n = image_points[0].shape[0], len(image_points)
    D = np.zeros((2 * m * n, 2))
    d = np.zeros((2 * m * n, 1))

    for i in range(n):
        for j in range(m):
            r2 = ideal_image_points[i][j, 0] ** 2 + ideal_image_points[i][j, 1] ** 2
            D[2 * (m * i + j), 0] = (ideal_points[i][j, 0] - u0) * r2
            D[2 * (m * i + j), 1] = (ideal_points[i][j, 0] - u0) * (r2 ** 2)
            d[2 * (m * i + j)] = image_points[i][j, 0] - ideal_points[i][j, 0]
            D[2 * (m * i + j) + 1, 0] = (ideal_points[i][j, 1] - v0) * r2
            D[2 * (m * i + j) + 1, 1] = (ideal_points[i][j, 1] - v0) * (r2 ** 2)
            d[2 * (m * i + j) + 1] = image_points[i][j, 1] - ideal_points[i][j, 1]

    D = np.mat(D)
    d = np.mat(d)

    k = (D.T * D).I * D.T * d

    return np.array(k)


def my_calibrate():
    """
    :return: Parameters of camera
    """
    object_points, image_points, image_size = calibrate.load_point_pairs()

    h_lst = []
    for i in range(len(object_points)):
        H = estimate_H(object_points[i], image_points[i])
        h_lst.append(H)

    K = estimate_intrinsic(h_lst)

    R_lst = []
    t_lst = []
    for i in range(len(h_lst)):
        R, t = estimate_extrinsic(K, h_lst[i])
        R_lst.append(R)
        t_lst.append(t)

    ideal_points = []
    ideal_image_points = []
    for i in range(len(object_points)):
        ideal_pixel, ideal_image = estimate_ideal_point(object_points[i], K, R_lst[i], t_lst[i])
        ideal_points.append(ideal_pixel)
        ideal_image_points.append(ideal_image)

    k = estimate_radial_distortion(image_points, ideal_points, ideal_image_points, K[0, 2], K[1, 2])

    return K, k, R_lst, t_lst


if __name__ == '__main__':
    K, k, R_lst, t_lst = my_calibrate()
    _, cv_K, cv_k, cv_R, cv_t = calibrate.calibrate()

    print("The intrinsic matrix estimated with OpenCV API is \n{}\n".format(cv_K))
    print("The intrinsic matrix estimated with my implementation is \n{}\n".format(K))
    print("The RMSE between two matrix is {}\n".format(np.sqrt(np.sum((cv_K-K) ** 2))))

    print('-' * 60)

    print("[R | t] of first calibration plate estimated with OpenCV API is \n{}\n".format
          (np.c_[cv2.Rodrigues(cv_R[0])[0], cv_t[0]]))
    print("[R | t] of first calibration plate estimated with my implementation is \n{}\n".format
          (np.c_[R_lst[0], t_lst[0]]))
    rmse = np.sqrt(np.sum(np.c_[cv2.Rodrigues(cv_R[0])[0], cv_t[0]] - np.c_[R_lst[0], t_lst[0]]) ** 2)
    print("The RMSE between two matrix is {}\n".format(rmse))

    print('-' * 60)
    print("The distortion parameters k1, k2 estimated with OpenCV API is \n{}\n".format(cv_k[0][:2]))
    print("The distortion parameters k1, k2 estimated with my implementation is \n{}\n".format(k.flatten()))
    print("The RMSE between two vector is {}\n".format(np.sqrt(np.sum(cv_k[0][:2] - k.flatten()) ** 2)))
