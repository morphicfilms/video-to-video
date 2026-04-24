# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""
pipeline_spec.py — Shared pipeline constraints for the Reshoot-Anything pipeline.

The WAN 2.2 video diffusion model consumes a fixed number of frames determined by:
    frame_num = (total_input // 4) * 4 + 1 - 4
This means the model silently drops trailing frames unless the input frame count
is chosen carefully. The utilities here encode this constraint so the visualizer,
renderer, and inference script all agree on frame counts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def wan_consumed_frames(render_count: int) -> int:
    """How many frames WAN actually uses from *render_count* input frames.

    WAN formula (wan/video2video.py:414):
        frame_num = (total // 4) * 4 + 1 - 4

    Examples:
        81 -> 77,  85 -> 81,  49 -> 49,  53 -> 49
    """
    return (render_count // 4) * 4 + 1 - 4


def render_frames_for_wan_output(desired_wan_frames: int) -> int:
    """Minimum render frames needed so WAN produces exactly *desired_wan_frames*.

    *desired_wan_frames* must satisfy (n - 1) % 4 == 0  (i.e. 4k+1 form).
    Returns the smallest render count r such that wan_consumed_frames(r) == desired_wan_frames.
    """
    if desired_wan_frames < 1:
        raise ValueError(f"desired_wan_frames must be >= 1, got {desired_wan_frames}")
    if (desired_wan_frames - 1) % 4 != 0:
        raise ValueError(
            f"desired_wan_frames must be 4k+1 (e.g. 5, 9, 13, ... 77, 81), "
            f"got {desired_wan_frames}"
        )
    # Invert: we need r such that (r // 4) * 4 - 3 == desired_wan_frames
    # => (r // 4) * 4 == desired_wan_frames + 3
    # Smallest r: desired_wan_frames + 3  (already a multiple of 4 since desired is 4k+1)
    r = desired_wan_frames + 3
    assert wan_consumed_frames(r) == desired_wan_frames, (
        f"round-trip failed: render_frames={r}, consumed={wan_consumed_frames(r)}, "
        f"expected={desired_wan_frames}"
    )
    return r


def snap_to_valid_wan_output(n: int) -> int:
    """Snap *n* down to the nearest valid WAN output frame count (4k+1 form).

    Examples:
        81 -> 81,  80 -> 77,  82 -> 81,  50 -> 49
    """
    if n < 5:
        return 5
    return ((n - 1) // 4) * 4 + 1


def is_valid_wan_frame_count(n: int) -> bool:
    """Check if *n* is a valid WAN output frame count (4k+1 form, >= 5)."""
    return n >= 5 and (n - 1) % 4 == 0


# ── Condition pack validation ────────────────────────────────────────────────

CONDITION_PACK_FILES = (
    "render.mp4",
    "render_mask.mp4",
    "input.mp4",
    "reference.png",
    "cam_info.json",
)


def _video_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def validate_condition_pack(output_dir: str) -> list[str]:
    """Validate a rendered condition pack for WAN inference.

    Returns a list of warning/error strings. Empty list = all OK.
    """
    issues: list[str] = []
    d = Path(output_dir)

    for fname in CONDITION_PACK_FILES:
        p = d / fname
        if not p.exists():
            issues.append(f"Missing: {fname}")
        elif p.stat().st_size == 0:
            issues.append(f"Empty file: {fname}")

    if issues:
        return issues

    render_n = _video_frame_count(str(d / "render.mp4"))
    mask_n = _video_frame_count(str(d / "render_mask.mp4"))
    input_n = _video_frame_count(str(d / "input.mp4"))

    if render_n != mask_n:
        issues.append(f"Frame count mismatch: render.mp4={render_n}, render_mask.mp4={mask_n}")
    if render_n != input_n:
        issues.append(f"Frame count mismatch: render.mp4={render_n}, input.mp4={input_n}")

    wan_frames = wan_consumed_frames(render_n)
    if wan_frames < 5:
        issues.append(
            f"WAN will only use {wan_frames} frames from {render_n} rendered — "
            f"need at least 8 rendered frames for a usable output."
        )
    else:
        dropped = render_n - wan_frames
        if dropped > 0:
            optimal = render_frames_for_wan_output(wan_frames)
            if render_n != optimal:
                issues.append(
                    f"WAN will use {wan_frames} of {render_n} frames "
                    f"(dropping {dropped}). Optimal render count: {optimal}."
                )

    pink_path = d / "render_pink.mp4"
    if pink_path.exists():
        pink_n = _video_frame_count(str(pink_path))
        if pink_n != render_n:
            issues.append(f"Frame count mismatch: render_pink.mp4={pink_n}, render.mp4={render_n}")

    return issues
