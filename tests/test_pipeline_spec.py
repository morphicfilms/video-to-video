# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_spec import (
    is_valid_wan_frame_count,
    render_frames_for_wan_output,
    snap_to_valid_wan_output,
    wan_consumed_frames,
)


class TestWanConsumedFrames:
    def test_known_values(self):
        assert wan_consumed_frames(84) == 81
        assert wan_consumed_frames(81) == 77
        assert wan_consumed_frames(52) == 49
        assert wan_consumed_frames(49) == 45
        assert wan_consumed_frames(8) == 5

    def test_monotonic(self):
        prev = 0
        for r in range(5, 200):
            c = wan_consumed_frames(r)
            assert c >= prev or c == prev
            prev = c


class TestRenderFramesForWanOutput:
    def test_known_values(self):
        assert render_frames_for_wan_output(81) == 84
        assert render_frames_for_wan_output(77) == 80
        assert render_frames_for_wan_output(49) == 52
        assert render_frames_for_wan_output(5) == 8

    def test_round_trip(self):
        for desired in range(5, 300, 4):
            r = render_frames_for_wan_output(desired)
            assert wan_consumed_frames(r) == desired, f"failed for desired={desired}"

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            render_frames_for_wan_output(80)
        with pytest.raises(ValueError):
            render_frames_for_wan_output(0)
        with pytest.raises(ValueError):
            render_frames_for_wan_output(6)


class TestSnapToValidWanOutput:
    def test_already_valid(self):
        for n in [5, 9, 13, 77, 81, 101]:
            assert snap_to_valid_wan_output(n) == n

    def test_snaps_down(self):
        assert snap_to_valid_wan_output(80) == 77
        assert snap_to_valid_wan_output(82) == 81
        assert snap_to_valid_wan_output(83) == 81
        assert snap_to_valid_wan_output(50) == 49

    def test_minimum(self):
        assert snap_to_valid_wan_output(1) == 5
        assert snap_to_valid_wan_output(4) == 5


class TestIsValidWanFrameCount:
    def test_valid(self):
        for n in [5, 9, 13, 49, 77, 81, 101, 201]:
            assert is_valid_wan_frame_count(n), f"{n} should be valid"

    def test_invalid(self):
        for n in [0, 1, 3, 4, 6, 7, 8, 10, 80]:
            assert not is_valid_wan_frame_count(n), f"{n} should be invalid"
