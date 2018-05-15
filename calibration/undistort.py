import cv2
import calibrate
import os


def undistort(distort_im, camera_matrix, dist_coeffs):
        h, w = distort_im.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

        dst = cv2.undistort(distort_im, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # crop and save the image
        # without cropping the edge would be black
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        return dst


if __name__ == '__main__':
    # calibrate first
    _, camera_matrix, dist_coeffs, _, _ = calibrate.calibrate()

    # used the image saved in calibrate.calibrate()
    # undistorted images would be saved in output_path.
    input_path = './output/'
    output_path = './output/'
    for file in os.listdir(input_path):
        distort_im = cv2.imread(input_path + '/' + file)
        distort_im = cv2.cvtColor(distort_im, cv2.COLOR_RGB2GRAY)

        undistort_im = undistort(distort_im, camera_matrix, dist_coeffs)

        outfile = output_path + file[:6] + '_undistorted.jpg'
        print('Undistorted image written to: {}'.format(outfile))
        cv2.imwrite(outfile, undistort_im)
