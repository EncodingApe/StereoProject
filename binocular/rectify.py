import calibrate_binocular_camera
import cv2


def get_rectify_parameter(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size, R, T):
    # R1, R2 用来表示将原来的图像旋转到修正后的图像 3*3
    # P1, P2 表示新的将空间点投影到修正后的图像上的投影矩阵 3*4
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2,
                                                                      distCoeffs2, image_size, R, T, alpha=0)

    return R1, R2, P1, P2, Q


def rectify_image(org_image, cameraMatrix, distCoeffs, R, P, image_size):
    map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, P, image_size, cv2.CV_16SC2)

    rectify_image = cv2.remap(org_image, map1, map2, cv2.INTER_LINEAR)

    return rectify_image


if __name__ == '__main__':
    # 将哪一张图片显示出来证明相机已经经过修正
    chosen_image_to_show = 10

    _, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, image_size = \
        calibrate_binocular_camera.calibrate_each_camera('./left', './right', chosen_image_to_show)

    R1, R2, P1, P2, Q = \
        get_rectify_parameter(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size, R, T)

    rectified_image = rectify_image(calibrate_binocular_camera.org_image_left, cameraMatrix1, distCoeffs1, R1, P1,
                                    image_size)

    cv2.imshow('No. {} original_left'.format(chosen_image_to_show), calibrate_binocular_camera.org_image_left)
    cv2.imshow('No. {} rectified_left'.format(chosen_image_to_show), rectified_image)

    rectified_image = rectify_image(calibrate_binocular_camera.org_image_right, cameraMatrix2, distCoeffs2, R2, P2,
                                    image_size)

    cv2.imshow('No. {} original_right'.format(chosen_image_to_show), calibrate_binocular_camera.org_image_right)
    cv2.imshow('No. {} rectified_right'.format(chosen_image_to_show), rectified_image)

    while cv2.waitKey() & 0xFF != ord('q'):
        pass

    cv2.destroyAllWindows()

