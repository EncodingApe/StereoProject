import rectify
import calibrate_binocular_camera
import numpy as np

if __name__ == '__main__':
    _, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, image_size = \
        calibrate_binocular_camera.calibrate_each_camera('./left', './right')

    R1, R2, P1, P2, Q = \
        rectify.get_rectify_parameter(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, image_size, R, T)

    print("The rectified profect matrix of left camera is \n{}\n".format(P1))
    print("The rectified profect matrix of right camera is \n{}\n".format(P2))

    cl = -np.mat(P1[:, :3]).I * np.mat(P1[:, 3]).T
    cr = -np.mat(P2[:, :3]).I * np.mat(P2[:, 3]).T

    print("The optical center of left camera is \n{}\n".format(cl))
    print("The optical center of right camera is \n{}\n".format(cr))
    print("So the translation matrix [b; 0; 0] is \n{}\n".format(cr - cl))
