# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""
pipeline_spec.py — Shared pipeline constraints for the Reshoot-Anything pipeline.

The WAN 2.2 video diffusion model consumes a fixed number of frames determined by:
    frame_num = (total_input // 4) * 4 + 1 - 4
This means the model silently drops trailing frames unless the input frame count
is chosen carefully. The utilities here encode this constraint so the visualizer,
renderer, and inference script all agree on frame counts.
"""

import logging

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
