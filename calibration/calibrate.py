import cv2
import os
import numpy as np


def load_point_pairs():
    """
    Load the 3-D object points and corresponding 2-D image points from sample picture.
    :return: (lst1, lst2, image_size).
            # lst1 consists of the 3-D object points of every image.
            # lst2 consists of the 2-D image points of every image.
    """

    # output the image for further distortion
    output_path = './output/'

    # the board is 9*6, that is, a board contains 9*6 squares
    board_size = (9, 6)

    # the size of the square
    square_size = 1.0

    # 3-D point
    point = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    point[:, :2] = np.indices((9, 6)).T.reshape(-1, 2)
    point = point * square_size

    corners_lst = []
    object_lst = []
    image_size = (0, 0)

    # load image and generate corner list and object list
    file_dir = './left'
    for file in os.listdir(file_dir):
        im = cv2.imread(file_dir + '/' + file)
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        if image_size == (0, 0):
            image_size = im_gray.shape[: 2]

        ret, corners = cv2.findChessboardCorners(im_gray, board_size)

        # threshold to locate the corners much preciser
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(im_gray, corners, (5, 5), (-1, -1), criteria)

        if not ret:
            print("Error")
            continue

        corners_lst.append(corners.squeeze())
        object_lst.append(point)
        cv2.drawChessboardCorners(im, (9, 6), corners, True)
        cv2.imwrite(output_path + file[:-4] + '_distorted.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))

    return object_lst, corners_lst, image_size


def calibrate():
    object_points, img_points, image_size = load_point_pairs()

    # 返回内参矩阵, 形变系数, 旋转矩阵, 平移矩阵
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, image_size, None,
                                                                        None)

    return rms, camera_matrix, dist_coeffs, rvecs, tvecs


if __name__ == '__main__':
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate()

