import cv2
import numpy as np

# ================= USER CONFIGURATION =================
LEFT_IMAGE_PATH = 'left_test.jpg'  # Replace with your test image name
RIGHT_IMAGE_PATH = 'right_test.jpg'  # Replace with your test image name
CALIB_FILE = 'stereo_calib.npz'
# ======================================================

# 1. Load Calibration Parameters
print(f"Loading calibration from '{CALIB_FILE}'...")
calib_data = np.load(CALIB_FILE)
K1, D1 = calib_data["K1"], calib_data["D1"]
K2, D2 = calib_data["K2"], calib_data["D2"]
R1, R2 = calib_data["R1"], calib_data["R2"]
P1, P2 = calib_data["P1"], calib_data["P2"]
Q = calib_data["Q"]
img_shape = tuple(calib_data["img_shape"])  # (width, height)

# 2. Load Input Images
img_l = cv2.imread(LEFT_IMAGE_PATH)
img_r = cv2.imread(RIGHT_IMAGE_PATH)

if img_l is None or img_r is None:
    raise FileNotFoundError(f"Could not load images '{LEFT_IMAGE_PATH}' or '{RIGHT_IMAGE_PATH}'.")

# Resize images if they do not match calibration resolution
h_img, w_img = img_l.shape[:2]
if (w_img, h_img) != img_shape:
    print(f"Resizing images from ({w_img}x{h_img}) to calibrated size {img_shape}...")
    img_l = cv2.resize(img_l, img_shape)
    img_r = cv2.resize(img_r, img_shape)

gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

# 3. Compute Fisheye Rectification Maps & Remap Images
map1_l, map2_l = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, img_shape, cv2.CV_16SC2)
map1_r, map2_r = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2, img_shape, cv2.CV_16SC2)

rect_l = cv2.remap(gray_l, map1_l, map2_l, cv2.INTER_LINEAR)
rect_r = cv2.remap(gray_r, map1_r, map2_r, cv2.INTER_LINEAR)

# Global variables for interactive display
show_lines = False
depth_meters_global = None
display_canvas_global = None


def nothing(x):
    pass


# Create GUI Window and Trackbars
WIN_NAME = "Stereo Depth Inspector & Tuner"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

cv2.createTrackbar("Num Disparities /16", WIN_NAME, 12, 20, nothing)  # 12 * 16 = 192
cv2.createTrackbar("Block Size", WIN_NAME, 7, 15, nothing)  # 7*2 + 1 = 15
cv2.createTrackbar("Min Disparity", WIN_NAME, 0, 50, nothing)
cv2.createTrackbar("Uniqueness Ratio", WIN_NAME, 10, 30, nothing)
cv2.createTrackbar("Speckle Window", WIN_NAME, 100, 300, nothing)


def click_to_measure(event, x, y, flags, param):
    global depth_meters_global, display_canvas_global
    if event == cv2.EVENT_LBUTTONDOWN and depth_meters_global is not None:
        col = x if x < img_shape[0] else x - img_shape[0]
        row = y

        if row < depth_meters_global.shape[0] and col < depth_meters_global.shape[1]:
            d_m = depth_meters_global[row, col]
            d_cm = d_m * 100.0

            canvas = display_canvas_global.copy()
            if d_m > 0:
                text = f"Depth: {d_m:.2f} m ({d_cm:.1f} cm)"
                print(f"[{x}, {y}] -> {text}")
                cv2.circle(canvas, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(canvas, text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                print(f"[{x}, {y}] -> Invalid depth point (unmapped texture/shadow region)")
                cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(canvas, "Invalid Point", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(WIN_NAME, canvas)


cv2.setMouseCallback(WIN_NAME, click_to_measure)

print("\n" + "=" * 60)
print("  TUNING INSTRUCTIONS:")
print("  - Adjust trackbars at the top of window to tune depth quality live.")
print("  - Click anywhere on the image to measure depth.")
print("  - Press 'L' to toggle red epipolar alignment lines.")
print("  - Press 'ESC' or 'Q' to quit.")
print("=" * 60)

while True:
    # Read Trackbar Values
    num_disp_factor = max(1, cv2.getTrackbarPos("Num Disparities /16", WIN_NAME))
    num_disp = num_disp_factor * 16

    block_size_factor = cv2.getTrackbarPos("Block Size", WIN_NAME)
    block_size = block_size_factor * 2 + 1  # Must be odd
    if block_size < 5:
        block_size = 5

    min_disp = cv2.getTrackbarPos("Min Disparity", WIN_NAME)
    uniq_ratio = cv2.getTrackbarPos("Uniqueness Ratio", WIN_NAME)
    speckle_win = cv2.getTrackbarPos("Speckle Window", WIN_NAME)

    # Initialize StereoSGBM Matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 1 * block_size ** 2,
        P2=32 * 1 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=uniq_ratio,
        speckleWindowSize=speckle_win,
        speckleRange=16,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute Disparity Map
    raw_disparity = stereo.compute(rect_l, rect_r).astype(np.float32) / 16.0

    # Calculate 3D Depth
    points_3d = cv2.reprojectImageTo3D(raw_disparity, Q)
    depth_mm = points_3d[:, :, 2]
    depth_meters = depth_mm / 1000.0
    depth_meters[raw_disparity <= min_disp] = 0
    depth_meters[depth_meters > 15.0] = 0
    depth_meters_global = depth_meters

    # Prepare Visualization
    disp_vis = cv2.normalize(raw_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    rect_l_bgr = cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR)

    # Toggle Epipolar Alignment Lines
    if show_lines:
        for line_y in range(0, img_shape[1], 40):
            cv2.line(rect_l_bgr, (0, line_y), (img_shape[0], line_y), (0, 0, 255), 1)
            cv2.line(depth_color, (0, line_y), (img_shape[0], line_y), (0, 0, 255), 1)

    display_canvas = np.hstack((rect_l_bgr, depth_color))
    display_canvas_global = display_canvas

    cv2.imshow(WIN_NAME, display_canvas)

    key = cv2.waitKey(100) & 0xFF
    if key in [27, ord('q'), ord('Q')]:  # ESC or Q
        break
    elif key in [ord('l'), ord('L')]:
        show_lines = not show_lines

cv2.destroyAllWindows()