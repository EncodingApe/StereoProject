# StereoProject

## calibrate folder

The implementation of calibration and undistortion

- calibrate.py (The implementation of calibration with OpenCV API.)

- my_calibrate.py (My implementation of calibration.)

- undistort.py (Undistort images.)

## binocular folder

The implementation of binocular stereo calibration and the rectification of the cameras.

- calibrate_binocular_camera.py (The implementation of binocular stereo calibration.)

- rectify.py (The implementation of rectification of the cameras.)

- estimate_baseline.py (Estimate the baseline between two rectified cameras.)

## SGM folder

The CPU&GPU implementation of Semi-global Matching.

- sgm_cpu.cpp (The cpu implementation by https://github.com/reisub/SemiGlobal-Matching.)

- sgm_gpu.cu (The gpu implementation.)

## Stereo Matching forlder

The implementation of stereo matching (i.e. the disparity computation).

- calculate_disparity.py (Implementation of stereo matching and disparity map would be obtained).
