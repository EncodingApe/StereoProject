# Stereo-Project
There are three python files in this project. 

## calibrate.py
This is implemented with OpenCV API. There are two functions in this file.

- ***load_point_pairs() -> object_lst, corners_lst, image_size***

This is implemented to detect the key points (i.e. corners) in the chessboard in each given image and generate the corresponding 3-D points in the world coordinate. 

In this function, **cv2.findChessboardCorners()** is used to detect the corners of chessboard and **cv2.cornerSubPix()** is used to locate corners more precisely. And then, the deteced corners and their corresponding 3-D points of each image are stored in a list. 

What's more, **cv2.drawChessboardCorners()** is used to draw the corners in each image and those images are saved in the output path for further usage.

return parameters:
  1. object_lst: 3-D points in the world coordinate of each key point in each image.
  2. corners_lst: 2-D points in the pixel coordinate of each key point in each image.
  3. image_size: The size of the image.
  
---
  
- ***calibrate() -> rms, camera_matrix, dist_coeffs, rvecs, tvecs***

In this function, the 3-D points, corresponding 2-D points in each image and the size of the image are obtained by the **load_point_pairs()**.

Then **cv2.calibrateCamera()** is called to calibrate the camera based on the parameters obtained above. And the returns of this function are used to be the returns of **calibrate()**

return parameters:

  1. rms: The root-mean-square error of the calibration.
  2. camera_matrix: The intrinsic matrix of the camera.
  3. dist_coeffs: The distortion coefficients. 
  4. rvecs: A 3*1 vector. A rotation matrix could be obtained by calling cv2.Rodrigues().
  5. tvecs: The translation matrix.

---

## undistort.py


## my_calibrate.py
In this file, I basically implement the Zhang’s method for camera calibration according to his paper. There're six mainly functions to fulfill this method.

- ***estimate_H(obj_points, img_points) -> H***

This function is used to estimate the homography matrix between the calibration plate and corresponding image of each plate based on coordinates of the points in the original plane and corresponding pixel coordinates of the points in the image.

given parameters:
  1. obj_points: Coordinates of the 3-D points in the original plane.
  2. img_points: Corresponding pixel coordinates of the points in the image.
  
return parameters:
  1. H: The estimated homography matrix.
  
---

- ***estimate_intrinsic(h_lst) -> K***

This funtion is used to estimated the intrinsic matrix of camera based on a list of homograhpy matrix between different calibration plates and their corresponding images. There are at least 3 homography matrix to estimate the intrinsic matrix.

given parameters:
  1. h_lst: List of homograhpy matrix.
  
return parameters:
  1. K: The estimated intrinsic matrix.
  
---
  
- ***estimate_extrinsic(K, H) -> R, t***
 
This function is used to estimate the rotation matrix and the translation matrix between a calibration plate and its corresponding images. 
 
given parameters:
  1. K: The intrinsic matrix.
  2. H: The homography of a calibration plate. 
  
return parameters:
  1. R: The rotation matrix of specific calibration plate.
  2. t: The translation matrix of specific calibration plate.
  
---
  
- ***estimate_ideal_point(obj_points, K, R, t) -> ideal_pixel, ideal_image***

 This function is used to estimate the ideal point in both continuous image coordinate and discontinuous pixel coordinate for estimate the distortion parameters in the following step.
 
given parameters:
  1. obj_points: Coordinates of the 3-D points in the original plane.
  2. K: The intrinsic matrix.
  3. R: The rotation matrix of specific calibration plate.
  4. t: The translation matrix of specific calibration plate.
  
return parameters:
  1. ideal_pixel: The undistorted point in the discontinuous pixel coordinate.
  2. ideal_image: The undistorted point in the continuous image coordinate.
  
---
  
- ***estimate_radial_distortion(image_points, ideal_points, ideal_image_points, u0, v0) -> k***

This function is implemented to estimate the distortion parameters with the given parameters.

given parameters:
  1. image_points: Distorted discontinuous pixel coordinate.
  2. ideal_points: Undistorted discontinuous pixel coordinate.
  3. ideal_image_points: Undistorted continuous image coordinate.
  4. u0: The x-axis coordinate of the principal point.
  5. v0: The y-axis coordinate of the principal point
  
return parameters:
  1. k: 2*1 vector consists of k1, k2 in the radial distortion coefficients.
  
---

- ***my_calibrate() -> K, k, R_lst, t_lst***

This is the main function while calibrating a camera. In this function, the function **load_point_pairs()** implemented in **calibrate.py** with OpenCV API is used to generate the corresponding key point pairs. This step is implemented with OpenCV, but it could also be implemented by the **Harris Corner Detection**. It could be achieved in the future.

Obtaining the corresponding point pairs, a series of functions implemented above are called to estimate the parameters of the camera. The intrinsic matrix, the rotation matrix and translation matrix of each calibration plate and the distortion coefficients are estimated one by one.

return parameters:
  1. K: The intrinsic matrix.
  2. k: 2*1 vector consists of k1, k2 in the radial distortion coefficients.
  3. R_lst: List consists of the rotation matrix of each calibration plate.
  4. t_lst: List consists of the translation matrix of each calibration plate.
