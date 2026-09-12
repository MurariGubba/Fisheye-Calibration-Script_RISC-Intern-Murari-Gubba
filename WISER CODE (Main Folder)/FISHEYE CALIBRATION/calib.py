import cv2
import numpy as np
import glob
import os

# ================= USER CONFIGURATION =================
CHECKERBOARD = (15, 10)  # Internal corners (columns, rows)
SQUARE_SIZE_MM = 26.0    # Square width/height in millimeters

LEFT_PATTERN = 'left_*.jpg'
RIGHT_PATTERN = 'right_*.jpg'
# ======================================================

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# Prepare 3D object points (1, N, 3) in float64
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), dtype=np.float64)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []
imgpoints_l = []
imgpoints_r = []

left_images = sorted(glob.glob(LEFT_PATTERN))
right_images = sorted(glob.glob(RIGHT_PATTERN))

if len(left_images) == 0 or len(left_images) != len(right_images):
    raise ValueError(f"Found {len(left_images)} left and {len(right_images)} right images. Check pattern.")

print(f"Found {len(left_images)} image pairs. Extracting corners...")
valid_pairs = 0
first_valid_l = None
first_valid_r = None

for path_l, path_r in zip(left_images, right_images):
    img_l = cv2.imread(path_l)
    img_r = cv2.imread(path_r)

    if img_l is None or img_r is None:
        continue

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, cb_flags)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, cb_flags)

    if ret_l and ret_r:
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (3, 3), (-1, -1), subpix_criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (3, 3), (-1, -1), subpix_criteria)

        # Detect and fix upside-down corner ordering between Left and Right
        vec_l = corners_l[-1, 0] - corners_l[0, 0]
        vec_r = corners_r[-1, 0] - corners_r[0, 0]
        if np.dot(vec_l, vec_r) < 0:
            corners_r = np.flip(corners_r, axis=0)

        objpoints.append(objp.astype(np.float64))
        imgpoints_l.append(corners_l.astype(np.float64).reshape(1, -1, 2))
        imgpoints_r.append(corners_r.astype(np.float64).reshape(1, -1, 2))

        if first_valid_l is None:
            first_valid_l = gray_l.copy()
            first_valid_r = gray_r.copy()

        valid_pairs += 1
        print(f"✓ Pair {valid_pairs}: {os.path.basename(path_l)} & {os.path.basename(path_r)}")

if valid_pairs < 4:
    raise ValueError(f"Only {valid_pairs} valid pairs detected. Need at least 4 valid pairs.")

h, w = gray_l.shape[:2]
image_size = (w, h)

print(f"\n1. Calibrating individual camera intrinsics across {valid_pairs} pairs...")
K1 = np.zeros((3, 3), dtype=np.float64)
K2 = np.zeros((3, 3), dtype=np.float64)
D1 = np.zeros((4, 1), dtype=np.float64)
D2 = np.zeros((4, 1), dtype=np.float64)

calib_flags_single = (
    getattr(cv2.fisheye, 'CALIB_RECOMPUTE_EXTRINSIC', 2) +
    getattr(cv2.fisheye, 'CALIB_FIX_SKEW', 8)
)

rms_l, K1, D1, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpoints_l, image_size, K1, D1,
    flags=calib_flags_single,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
)
print(f"   Left Camera Intrinsic RMS  : {rms_l:.4f} pixels")

rms_r, K2, D2, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpoints_r, image_size, K2, D2,
    flags=calib_flags_single,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
)
print(f"   Right Camera Intrinsic RMS : {rms_r:.4f} pixels")

print("\n2. Calibrating Fisheye Stereo Extrinsics...")
calib_flags_stereo = (
    getattr(cv2.fisheye, 'CALIB_FIX_INTRINSIC', 256) +
    getattr(cv2.fisheye, 'CALIB_FIX_SKEW', 8)
)

calib_res = cv2.fisheye.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K1, D1, K2, D2, image_size,
    flags=calib_flags_stereo,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
)

rms, K1, D1, K2, D2, R, T = calib_res[:7]

print("\n" + "=" * 45)
print("          STEREO CALIBRATION SUCCESS        ")
print("=" * 45)
print(f"Stereo RMS Re-projection Error : {rms:.4f} pixels")
print(f"Calculated Baseline (T_x)      : {abs(T[0][0]):.2f} mm")
print("=" * 45)

print("\n3. Calculating Fisheye Rectification Maps...")
zero_disparity_flag = getattr(cv2.fisheye, 'CALIB_ZERO_DISPARITY', 1024)

R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
    K1, D1, K2, D2, image_size, R, T,
    flags=zero_disparity_flag,
    balance=0.0, fov_scale=1.0
)

# Save stereo calibration data
np.savez("stereo_calib.npz", K1=K1, D1=D1, K2=K2, D2=D2, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, img_shape=image_size)
print("✓ Saved parameters to 'stereo_calib.npz'")

print("\n4. Generating Prototype Disparity & Depth Map...")
map1_l, map2_l = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
map1_r, map2_r = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

rect_l = cv2.remap(first_valid_l, map1_l, map2_l, cv2.INTER_LINEAR)
rect_r = cv2.remap(first_valid_r, map1_r, map2_r, cv2.INTER_LINEAR)

# StereoSGBM Matcher Configuration
num_disp = 16 * 6
block_size = 11
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 1 * block_size ** 2,
    P2=32 * 1 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disparity = stereo.compute(rect_l, rect_r).astype(np.float32) / 16.0

# Normalize disparity for visual demo preview
disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# Stack rectify left image and color depth map side-by-side
preview = np.hstack((cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR), disp_color))
cv2.imwrite("prototype_depth_map.png", preview)

print("✓ Saved depth visual preview to 'prototype_depth_map.png'")
print("\nPipeline Complete!")