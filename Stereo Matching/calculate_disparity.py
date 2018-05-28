import cv2
import numpy as np


def calculate_disp(img1, img2):
    disp = stereo.compute(img1, img2).astype(np.float32) / 16.0

    return disp


def init_StereoSGBM(min_disp, num_disp, window_size):
    global stereo
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )


if __name__ == '__main__':
    # the left-image and right-image used to generate the disparity map.
    img1 = cv2.imread('./left/left05.jpg')
    img2 = cv2.imread('./right/right05.jpg')

    window_size = 3
    min_disp = 16
    # 表示最大的视差减去最小的视差 即视差的取值有多少个
    num_disp = 96 - min_disp

    init_StereoSGBM(min_disp, num_disp, window_size)

    ret = calculate_disp(img1, img2)

    cv2.imshow('left-image', img1)
    cv2.imshow('disparity-image', (ret - min_disp) / num_disp)
    while cv2.waitKey() != ord('q'):
        pass
