# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""
interpolation.py — Camera path interpolation.

Given a list of c2w keyframe matrices (OpenCV convention),
produces a smooth interpolated path of N frames using:
  - Quaternion SLERP (scipy) for rotation
  - Cubic spline (scipy) for translation

Supports multiple easing/transition modes between keyframes.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline


# ── Easing mode names (used for the UI dropdown) ──────────────────────────────
EASING_MODES: tuple[str, ...] = (
    "Linear",       # Constant velocity — robotic/mechanical
    "Ease In/Out",  # Slow→fast→slow per segment — natural, comfortable
    "Cinematic",    # Heavy slow→fast→slow — dramatic, emphasis on keyframes
    "Ease In",      # Starts slow, arrives fast — accelerating camera
    "Ease Out",     # Launches fast, settles gently — decelerating camera
    "Overshoot",    # Flies ~10% past each waypoint then snaps back — lively
    "Stepped",      # Dwells at keyframes, quick whip-pan between — editorial
)


def _ease_fn(mode: str):
    """Return a vectorised easing function t ∈ [0, 1] → value for the given mode."""
    if mode == "Ease In/Out":
        # Smoothstep: 3t² − 2t³
        return lambda t: t * t * (3.0 - 2.0 * t)
    elif mode == "Cinematic":
        # Smootherstep (Ken Perlin): 6t⁵ − 15t⁴ + 10t³
        return lambda t: t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    elif mode == "Ease In":
        # Cubic ease-in
        return lambda t: t ** 3
    elif mode == "Ease Out":
        # Cubic ease-out
        return lambda t: 1.0 - (1.0 - t) ** 3
    elif mode == "Overshoot":
        # Back ease-out: overshoots target by ~10%, then snaps back
        c1, c3 = 1.70158, 2.70158
        return lambda t: 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2
    elif mode == "Stepped":
        # Dwell at each keyframe for the outer 20%; quick move through the middle 60%
        def _stepped(t: np.ndarray) -> np.ndarray:
            s = np.clip((np.asarray(t) - 0.2) / 0.6, 0.0, 1.0)
            return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)
        return _stepped
    else:
        # "Linear" or unknown: identity
        return lambda t: np.asarray(t, dtype=float)


def _apply_per_segment_easing(
    out_times: np.ndarray,
    kf_times: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    Remap *out_times* so that each keyframe-to-keyframe segment gets its own
    easing envelope.

    For Overshoot mode the returned values may slightly exceed [0, 1]; callers
    that cannot handle extrapolation (e.g. scipy Slerp) should clamp them.
    """
    if mode == "Linear":
        return out_times  # fast path

    ease = _ease_fn(mode)
    eased = out_times.copy()

    # searchsorted(..., side="right") − 1  →  left-keyframe index of each segment
    seg_idx = np.clip(
        np.searchsorted(kf_times, out_times, side="right") - 1,
        0, len(kf_times) - 2,
    )

    for i in range(len(kf_times) - 1):
        mask = seg_idx == i
        if not np.any(mask):
            continue
        t0, t1 = kf_times[i], kf_times[i + 1]
        dt = t1 - t0
        s = np.clip((out_times[mask] - t0) / dt, 0.0, 1.0)  # normalise → [0, 1]
        eased[mask] = t0 + ease(s) * dt                      # scale back; may exceed t1

    return eased


def interpolate_camera_path(
    c2w_keyframes: list[np.ndarray],
    n_frames: int,
    easing: str = "Linear",
) -> np.ndarray:
    """
    Interpolate a smooth camera path between keyframe c2w matrices.

    Args:
        c2w_keyframes: list of K float32 [4, 4] camera-to-world matrices (OpenCV).
                       K must be >= 2.
        n_frames:      number of output frames (including endpoints).
        easing:        transition style — one of EASING_MODES (default "Linear").

    Returns:
        c2ws: float32 [n_frames, 4, 4] interpolated camera-to-world matrices.

    Raises:
        ValueError: if fewer than 2 keyframes are provided.
    """
    K = len(c2w_keyframes)
    if K < 2:
        raise ValueError(f"At least 2 keyframes required, got {K}.")

    if K == 1:
        # Degenerate — return the single keyframe repeated
        return np.stack([c2w_keyframes[0]] * n_frames, axis=0)

    # Normalised keyframe times in [0, 1]
    kf_times = np.linspace(0.0, 1.0, K)
    out_times = np.linspace(0.0, 1.0, n_frames)

    # Apply per-segment easing
    eased_times = _apply_per_segment_easing(out_times, kf_times, easing)

    # ── Rotation: cubic spline on quaternions (same method as translation) ────
    # Interpolate the 4 quaternion components with the same CubicSpline /
    # not-a-knot BC used for translation, then normalise.  This keeps rotation
    # and translation on identical interpolation footing so easing applies
    # equally to both — no more endpoint velocity mismatch or lag/snap.
    quats = Rotation.from_matrix(
        np.stack([c2w[:3, :3] for c2w in c2w_keyframes], axis=0)
    ).as_quat()  # [K, 4]  xyzw

    # Ensure quaternion sign consistency (avoid flips across keyframes).
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    rot_times = np.clip(eased_times, kf_times[0], kf_times[-1])
    if K == 2:
        t = rot_times[:, None]
        q_interp = quats[0] * (1.0 - t) + quats[1] * t
    else:
        q_interp = CubicSpline(kf_times, quats)(rot_times)  # [n_frames, 4]

    # Normalise to unit quaternion before converting to matrix.
    q_interp /= np.linalg.norm(q_interp, axis=1, keepdims=True)
    interp_rots = Rotation.from_quat(q_interp).as_matrix()  # [n_frames, 3, 3]

    # ── Translation: cubic spline ────────────────────────────────────────────
    translations = np.stack([c2w[:3, 3] for c2w in c2w_keyframes], axis=0)  # [K, 3]

    if K == 2:
        # CubicSpline needs at least 3 knots; fall back to linear for 2 keyframes.
        t_interp = (
            translations[0] * (1.0 - eased_times[:, None])
            + translations[1] * eased_times[:, None]
        )
    else:
        # "not-a-knot" (default) — no artificial zero-velocity constraint at
        # endpoints.  The easing function already provides smooth acceleration
        # through time remapping, and clamped BC would desync translation from
        # RotationSpline (which has its own natural endpoint conditions).
        cs = CubicSpline(kf_times, translations)
        t_interp = cs(eased_times)   # [n_frames, 3]

    # ── Assemble output c2w matrices ─────────────────────────────────────────
    c2ws = np.eye(4, dtype=np.float32)[None].repeat(n_frames, axis=0)
    c2ws[:, :3, :3] = interp_rots.astype(np.float32)
    c2ws[:, :3, 3] = t_interp.astype(np.float32)

    return c2ws


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Invert a c2w matrix to get the world-to-camera matrix."""
    return np.linalg.inv(c2w).astype(np.float32)


# ── Coordinate system conversions ─────────────────────────────────────────────
# Viser uses OpenGL convention: Y-up, camera looks -Z.
# Pipeline uses OpenCV convention: Y-down, camera looks +Z.
# The transform between them is a 180° rotation around the X axis.

_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
_OPENCV_TO_OPENGL = _OPENGL_TO_OPENCV  # Its own inverse (diagonal ±1)


def viser_c2w_to_opencv(c2w_viser: np.ndarray) -> np.ndarray:
    """
    Convert a c2w matrix from Viser's OpenGL convention to OpenCV convention.

    Viser reports camera poses in OpenGL (Y-up, -Z forward).
    Our pipeline uses OpenCV (Y-down, +Z forward).
    """
    return (c2w_viser @ _OPENGL_TO_OPENCV).astype(np.float32)


def opencv_c2w_to_viser(c2w_opencv: np.ndarray) -> np.ndarray:
    """
    Convert a c2w matrix from OpenCV convention to Viser's OpenGL convention.
    Used when displaying pipeline cameras inside Viser.
    """
    return (c2w_opencv @ _OPENCV_TO_OPENGL).astype(np.float32)
