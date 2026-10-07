# WISER RISC LAB MURARI GUBBA G11
 RISC Lab Intern for ROV Camera Systems, from 2D to 3D translation to be used with a grabber arm and camera system.

ROV Fisheye Stereo Depth Mapping Pipeline
A robust OpenCV stereo vision pipeline for dual-fisheye underwater camera systems, handling calibration, distortion correction, stereo rectification, and real-time metric depth estimation.

Pipeline Overview
[Raw Images] ➔ [Independent Fisheye Calib] ➔ [Scanline Rectification] ➔ [StereoSGBM Matching] ➔ [3D Depth (Meters)]

How It Works
1. Calibration & Corner Vector Alignment

-Sub-Pixel Extraction: Detects 15*10 internal checkerboard corners with `cv2.cornerSubPix` precision.
-Symmetry Check: Calculates corner-to-corner vectors to automatically fix inverted checkerboard corner indexing between views.
-Two-Step Fisheye Calibration: Solves left and right intrinsic parameters independently before computing extrinsics. This prevents C++ matrix divergence errors in wide-angle fisheye lens models.

2. Stereo Rectification
-Epipolar Constraint: Computes projection and rotation matrices via `cv2.fisheye.stereoRectify` to force left and right image rows onto identical horizontal scanlines.
-Remapping: Generates lookup maps using `cv2.fisheye.initUndistortRectifyMap` to remove fisheye barrel distortion in real time.

3. Disparity & Metric Depth Estimation

-Semi-Global Block Matching (StereoSGBM): Evaluates local texture similarities along horizontal scanlines to compute pixel displacement.
-3D Metric Reprojection: Converts disparity directly into metric distance using camera focal length and baseline distance.
-Interactive Inspection: Reprojects 2D disparity to 3D space using perspective matrix $Q$, outputting metric measurements in meters and centimeters.

Quick Start

```bash
# 1. Run calibration pipeline across image pairs (Outputs stereo_calib.npz)
python calib.py

# 2. Launch interactive live-tuning and depth measurement inspector
python depth_analyze.py

```
Keyboard Controls (Depth Inspector)

`L` — Toggle red epipolar scanline alignment.
`Left Mouse Click` — Inspect metric distance at any point.


* **`ESC` / `Q**` — Exit inspector.
