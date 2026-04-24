# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""Comprehensive tests for pipeline_spec frame count utilities."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline_spec import (
    is_valid_wan_frame_count,
    max_wan_frames_for_source,
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

    def test_formula_matches_wan_source(self):
        for total in range(4, 300):
            expected = (total // 4) * 4 + 1 - 4
            assert wan_consumed_frames(total) == expected


class TestRenderFramesForWanOutput:
    def test_known_values(self):
        assert render_frames_for_wan_output(81) == 84
        assert render_frames_for_wan_output(77) == 80
        assert render_frames_for_wan_output(49) == 52
        assert render_frames_for_wan_output(5) == 8

    def test_round_trip_all_valid(self):
        for desired in range(5, 500, 4):
            r = render_frames_for_wan_output(desired)
            assert wan_consumed_frames(r) == desired

    def test_rejects_non_4k_plus_1(self):
        for bad in [6, 7, 8, 10, 80, 82, 100]:
            with pytest.raises(ValueError, match="4k\\+1"):
                render_frames_for_wan_output(bad)

    def test_rejects_zero_and_negative(self):
        with pytest.raises(ValueError):
            render_frames_for_wan_output(0)
        with pytest.raises(ValueError):
            render_frames_for_wan_output(-1)


class TestSnapToValidWanOutput:
    def test_already_valid_unchanged(self):
        for n in [5, 9, 13, 77, 81, 101, 201, 497]:
            assert snap_to_valid_wan_output(n) == n

    def test_snaps_down(self):
        assert snap_to_valid_wan_output(80) == 77
        assert snap_to_valid_wan_output(82) == 81
        assert snap_to_valid_wan_output(83) == 81
        assert snap_to_valid_wan_output(50) == 49

    def test_minimum_clamp(self):
        for n in range(0, 5):
            assert snap_to_valid_wan_output(n) == 5

    def test_result_always_valid(self):
        for n in range(1, 500):
            snapped = snap_to_valid_wan_output(n)
            assert is_valid_wan_frame_count(snapped)
            assert snapped <= n or snapped == 5


class TestIsValidWanFrameCount:
    def test_valid_sequence(self):
        for k in range(1, 100):
            assert is_valid_wan_frame_count(4 * k + 1)

    def test_invalid(self):
        for n in [0, 1, 2, 3, 4, 6, 7, 8, 10, 80]:
            assert not is_valid_wan_frame_count(n)

    def test_boundary(self):
        assert is_valid_wan_frame_count(5)
        assert not is_valid_wan_frame_count(4)


class TestMaxWanFramesForSource:
    def test_render_fits_within_source(self):
        for n_src in range(8, 300):
            max_wan = max_wan_frames_for_source(n_src)
            render_needed = render_frames_for_wan_output(max_wan)
            assert render_needed <= n_src, f"n_src={n_src}: render={render_needed} > source"

    def test_result_is_valid(self):
        for n_src in range(8, 300):
            max_wan = max_wan_frames_for_source(n_src)
            assert is_valid_wan_frame_count(max_wan), f"n_src={n_src}: {max_wan} not 4k+1"

    def test_known_values(self):
        assert max_wan_frames_for_source(30) == 25
        assert max_wan_frames_for_source(100) == 97
        assert max_wan_frames_for_source(84) == 81


class TestDefaultWorkflow:
    """Test the typical user workflow: pick desired WAN output, compute render frames."""

    def test_default_81(self):
        desired = 81
        snapped = snap_to_valid_wan_output(desired)
        assert snapped == 81
        render = render_frames_for_wan_output(snapped)
        assert render == 84
        consumed = wan_consumed_frames(render)
        assert consumed == 81

    def test_user_picks_round_number(self):
        desired = 100
        snapped = snap_to_valid_wan_output(desired)
        assert snapped == 97
        render = render_frames_for_wan_output(snapped)
        consumed = wan_consumed_frames(render)
        assert consumed == 97
