# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""
scene.py — Point cloud loading and frame-to-3D projection utilities.

Matches the coordinate conventions used in Uni3C/cam_render_video.py:
  - OpenCV camera convention (X right, Y down, Z forward)
  - Intrinsics: K with focal = max(H, W) * focal_multiplier
  - Source camera placed at (0, 0, -radius) with upward tilt = start_elevation degrees
"""

import math
import numpy as np
import cv2


def load_video_frames(
    video_path: str, max_frames: int = 200, target_fps: float = None
) -> tuple[np.ndarray, float]:
    """
    Load video frames from an mp4 file.

    Returns:
        frames: uint8 [T, H, W, 3] RGB array
        effective_fps: float — the FPS of the returned frames (after any downsampling)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_fps is not None and target_fps < src_fps:
        step = max(1, round(src_fps / target_fps))
    else:
        step = 1

    effective_fps = src_fps / step

    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1

    cap.release()
    if not frames:
        raise ValueError(f"No frames could be read from: {video_path}")
    return np.stack(frames, axis=0), effective_fps


def load_depth_maps(depth_path: str) -> np.ndarray:
    """
    Load metric depth maps from a .npz file.

    The file should have a 'depths' key with shape [T, H, W] or [T, 1, H, W].

    Returns:
        depths: float32 [T, H, W] metric depth in metres
    """
    data = np.load(depth_path)
    if "depths" not in data:
        raise KeyError(f"'depths' key not found in {depth_path}. Available keys: {list(data.keys())}")
    depths = data["depths"].astype(np.float32)
    if depths.ndim == 4:          # [T, 1, H, W] → [T, H, W]
        depths = depths[:, 0]
    if depths.ndim != 3:
        raise ValueError(f"Expected depth shape [T, H, W] or [T, 1, H, W], got {depths.shape}")

    valid = depths[np.isfinite(depths) & (depths > 0)]
    if valid.size > 0:
        median_d = float(np.median(valid))
        if median_d < 0.01 or median_d > 100.0:
            print(
                f"[scene] WARNING: median depth = {median_d:.4f} m — "
                f"outside typical metric range [0.01, 100]. "
                f"Check that depth values are in metres."
            )
    return depths


def compute_intrinsics(H: int, W: int, focal_multiplier: float = 1.0) -> np.ndarray:
    """
    Compute the camera intrinsic matrix K.

    Matches cam_render_video.py: focal = max(H, W) * focal_multiplier,
    principal point at image centre.

    Returns:
        K: float32 [3, 3]
    """
    focal = max(H, W) * focal_multiplier
    K = np.array([
        [focal, 0.0,   W / 2.0],
        [0.0,   focal, H / 2.0],
        [0.0,   0.0,   1.0],
    ], dtype=np.float32)
    return K


def unproject_frame(
    frame_rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    subsample: int = 1,
    K_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lift a single RGB frame + depth map into a coloured 3-D point cloud
    in the source-camera coordinate frame (OpenCV convention).

    Args:
        frame_rgb:  uint8  [H, W, 3]
        depth:      float32 [H, W]  metric depth
        K:          float32 [3, 3]  camera intrinsics
        subsample:  int — keep every n-th pixel (1 = full resolution)
        K_inv:      float32 [3, 3]  precomputed inverse of K (avoids redundant inv per frame)

    Returns:
        points_xyz: float32 [N, 3]  3-D positions in camera space
        colors_rgb: uint8   [N, 3]  corresponding RGB colours
    """
    H, W = depth.shape
    if K_inv is None:
        K_inv = np.linalg.inv(K)

    ys = np.arange(0, H, subsample)
    xs = np.arange(0, W, subsample)
    xv, yv = np.meshgrid(xs, ys)       # [h', w']
    ones = np.ones_like(xv)

    pixel_hom = np.stack([xv, yv, ones], axis=-1).reshape(-1, 3).T   # [3, N]
    d_flat = depth[yv, xv].reshape(-1)                                 # [N]

    points_xyz = (K_inv @ pixel_hom * d_flat).T.astype(np.float32)    # [N, 3]
    colors_rgb = frame_rgb[yv, xv].reshape(-1, 3)                      # [N, 3]

    # Remove points at zero/invalid depth
    valid = np.isfinite(d_flat) & (d_flat > 0)
    return points_xyz[valid], colors_rgb[valid]


def transform_points_to_world(points_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    Transform points from camera space to world space using c2w matrix.

    Args:
        points_cam: float32 [N, 3]
        c2w:        float32 [4, 4]  camera-to-world

    Returns:
        points_world: float32 [N, 3]
    """
    N = points_cam.shape[0]
    hom = np.concatenate([points_cam, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, 4]
    points_world = (c2w @ hom.T).T[:, :3]
    return points_world.astype(np.float32)


def get_source_camera(depth_avg: float, start_elevation: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the source (reference) camera pose that aligns with the first video frame.

    Replicates set_initial_camera() from Uni3C/src/utils.py:
      - Camera placed at (0, 0, -radius) in world space
      - Tilted upward by start_elevation degrees (rotation around X axis)

    Returns:
        w2c_0: float32 [4, 4]  world-to-camera
        c2w_0: float32 [4, 4]  camera-to-world
    """
    radius = depth_avg
    elev_rad = math.radians(start_elevation)

    # Base c2w: camera at (0, 0, -radius), looking toward origin
    c2w_0 = np.eye(4, dtype=np.float64)
    c2w_0[2, 3] = -radius

    # Elevation rotation (pitch) — rotate up (negative angle around X)
    cos_e = math.cos(-elev_rad)
    sin_e = math.sin(-elev_rad)
    R_elevation = np.array([
        [1, 0,     0,     0],
        [0, cos_e, -sin_e, 0],
        [0, sin_e,  cos_e, 0],
        [0, 0,     0,     1],
    ], dtype=np.float64)

    c2w_0 = R_elevation @ c2w_0
    w2c_0 = np.linalg.inv(c2w_0)

    return w2c_0.astype(np.float32), c2w_0.astype(np.float32)
