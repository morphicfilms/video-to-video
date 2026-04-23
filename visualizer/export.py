# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""
export.py — Export camera path data to cam_info.json.

Output format matches what cam_render_video.py consumes:
{
    "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],   # 3×3 K matrix
    "extrinsic": [<w2c_4x4_source>, <w2c_4x4_frame_0>, ..., <w2c_4x4_frame_N-1>],
    "height": H,
    "width":  W
}

extrinsic[0] = source camera (aligns with first video frame).
extrinsic[1:] = target camera trajectory (one per output frame).
All matrices are world-to-camera, OpenCV convention, stored as nested lists.
"""

import json
import logging
import os
import numpy as np

from pipeline_spec import is_valid_wan_frame_count, wan_consumed_frames

logger = logging.getLogger(__name__)


def export_cam_info(
    w2c_source: np.ndarray,
    w2c_targets: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    output_path: str,
) -> None:
    """
    Serialise camera information to a JSON file.

    Args:
        w2c_source:  float [4, 4]  world-to-camera of the source/reference frame.
        w2c_targets: float [N, 4, 4]  world-to-camera for each output frame.
        K:           float [3, 3]  camera intrinsic matrix.
        H:           int  image height in pixels.
        W:           int  image width in pixels.
        output_path: str  path to write the .json file (directories created if needed).
    """
    n_target = len(w2c_targets)
    wan_out = wan_consumed_frames(n_target)
    if not is_valid_wan_frame_count(wan_out):
        logger.warning(
            f"Target frame count {n_target} yields {wan_out} WAN frames "
            f"(not 4k+1). WAN will silently truncate to {((wan_out - 1) // 4) * 4 + 1} frames."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    extrinsics = [w2c_source.tolist()] + [m.tolist() for m in w2c_targets]

    cam_info = {
        "intrinsic": K.tolist(),
        "extrinsic": extrinsics,
        "height": int(H),
        "width":  int(W),
        "k_resolution": [int(H), int(W)],
    }

    with open(output_path, "w") as f:
        json.dump(cam_info, f, indent=2)

    print(f"[export] Wrote {1 + n_target} cameras (1 source + {n_target} target) → {output_path}")
    print(f"[export] WAN will use {wan_out} of {n_target} target frames.")


def load_cam_info(json_path: str) -> dict:
    """
    Load a previously exported cam_info.json.

    Returns a dict with numpy arrays:
        {
            "intrinsic": float32 [3, 3],
            "extrinsic": float32 [N+1, 4, 4],
            "height": int,
            "width":  int,
        }
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    result = {
        "intrinsic": np.array(raw["intrinsic"], dtype=np.float32),
        "extrinsic": np.array(raw["extrinsic"], dtype=np.float32),
        "height":    int(raw["height"]),
        "width":     int(raw["width"]),
    }
    if "k_resolution" in raw:
        result["k_resolution"] = tuple(int(x) for x in raw["k_resolution"])
    return result
