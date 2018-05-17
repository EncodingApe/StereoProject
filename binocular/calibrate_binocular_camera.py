import cv2
import os
import numpy as np

np.set_printoptions(suppress=True)

org_image_left = None
org_image_right = None


def calibrate_each_camera(input_dir1, input_dir2, chosen_image = 1):
    pattern_size = (9, 6)
    square_size = 1.0

    obj_point = np.zeros((np.prod(pattern_size), 3), np.float32)
    obj_point[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    obj_point = obj_point * square_size

    image1_lst = []
    image2_lst = []
    objects_lst = []
    image_size = (0, 0)

    lst1 = sorted(os.listdir(input_dir1), key=lambda x: int(x[-6: -4]))
    lst2 = sorted(os.listdir(input_dir2), key=lambda x: int(x[-6: -4]))

    i = 0

    # 保证读取的图片对应
    for files in zip(lst1, lst2):
        i = i + 1

        file1, file2 = files
        im1 = cv2.imread(input_dir1 + '/' + file1)
        im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)

        _, corners1 = cv2.findChessboardCorners(im1, pattern_size)

        # Threshold
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(im1, corners1, (5, 5), (-1, -1), criteria)

        corners1 = corners1.squeeze()
        image1_lst.append(corners1)

        im2 = cv2.imread(input_dir2 + '/' + file2)
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

        _, corners2 = cv2.findChessboardCorners(im2, pattern_size)

        cv2.cornerSubPix(im2, corners2, (5, 5), (-1, -1), criteria)
        corners2 = corners2.squeeze()
        image2_lst.append(corners2)

        if image_size == (0, 0):
            image_size = im1.shape[:2]

        if i == chosen_image:
            global org_image_left
            org_image_left = im1

            global org_image_right
            org_image_right = im2

        objects_lst.append(obj_point)

    flags = (cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST +
             cv2.CALIB_SAME_FOCAL_LENGTH)
    return cv2.stereoCalibrate(objects_lst, image1_lst, image2_lst, None, None, None, None, image_size,
                               flags=flags), image_size


if __name__ == '__main__':
    ret, image_size = calibrate_each_camera('./left', './right')
    _, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = ret

    print("cameraMatrix1 = \n{}\n".format(cameraMatrix1))
    print("cameraMatrix2 = \n{}\n".format(cameraMatrix2))
    print("distCoeffs1 = \n{}\n".format(distCoeffs1))
    print("distCoeffs2 = \n{}\n".format(distCoeffs2))
    print("R = \n{}\n".format(R))
    print("T = \n{}\n".format(T))

    # # R1, R2 用来表示将原来的图像旋转到修正后的图像 3*3
    # # P1, P2 表示新的将空间点投影到修正后的图像上的投影矩阵 3*4
    # R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2,
    #                                                                   distCoeffs2, image_size, R, T, alpha=0)
    #
    # # left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, image_size, cv2.CV_16SC2)
    # # right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, image_size, cv2.CV_16SC2)
    #
    # # cv2.imshow('original-left', org_image_left)
    # # cv2.imshow('original-right', org_image_right)
    # #
    # # left = cv2.remap(org_image_left, left_map1, left_map2, cv2.INTER_LINEAR)
    # # cv2.imshow('rectified-left', left)
    # #
    # # right = cv2.remap(org_image_right, right_map1, right_map2, cv2.INTER_LINEAR)
    # # cv2.imshow('rectified-right', right)
    # # cv2.waitKey()
    #
    # print("The rectified profect matrix of left camera is \n{}".format(P1))
    # print("The rectified profect matrix of right camera is \n{}".format(P2))
    #
    # cl = -np.mat(P1[:, :3]).I * np.mat(P1[:, 3]).T
    # cr = -np.mat(P2[:, :3]).I * np.mat(P2[:, 3]).T
    #
    # print("The optical center of left camera is \n{}".format(cl))
    # print("The optical center of right camera is \n{}".format(cr))
    # print("So the translation matrix [b; 0; 0] is \n{}".format(cr - cl))
    #
    # print(np.c_[np.squeeze(cl.flatten()), [1]])
    # print(np.dot(P2, np.c_[np.squeeze(cl.flatten()), [1]].reshape(4, 1)))
