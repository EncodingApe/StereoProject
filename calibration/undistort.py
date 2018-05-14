import cv2
import calibrate
import os

# calibrate first
rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate.calibrate()

output_path = './output/'
for file in os.listdir(output_path):
    distort_im = cv2.imread(output_path + '/' + file)
    distort_im = cv2.cvtColor(distort_im, cv2.COLOR_RGB2GRAY)

    h, w = distort_im.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    dst = cv2.undistort(distort_im, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # crop and save the image
    # without cropping the edge would be black
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    outfile = output_path + file[:6] + '_undistorted.jpg'
    print('Undistorted image written to: {}'.format(outfile))
    cv2.imwrite(outfile, dst)
