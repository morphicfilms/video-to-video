# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""Tests for pipeline_spec.py condition pack validation."""
import pytest
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
from pipeline_spec import validate_condition_pack, CONDITION_PACK_FILES


def _write_video(path: str, n_frames: int = 20, h: int = 64, w: int = 96):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 15.0, (w, h))
    for _ in range(n_frames):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()


def _make_valid_pack(d: str, n_frames: int = 20):
    _write_video(os.path.join(d, "render.mp4"), n_frames)
    _write_video(os.path.join(d, "render_mask.mp4"), n_frames)
    _write_video(os.path.join(d, "input.mp4"), n_frames)
    cv2.imwrite(os.path.join(d, "reference.png"), np.zeros((64, 96, 3), dtype=np.uint8))
    cam_info = {
        "intrinsic": np.eye(3).tolist(),
        "extrinsic": [np.eye(4).tolist()] * (n_frames + 1),
        "height": 64, "width": 96,
    }
    with open(os.path.join(d, "cam_info.json"), "w") as f:
        json.dump(cam_info, f)


class TestValidateConditionPack:
    def test_valid_pack(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_pack(d, n_frames=84)
            issues = validate_condition_pack(d)
        assert issues == []

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_pack(d)
            os.remove(os.path.join(d, "render.mp4"))
            issues = validate_condition_pack(d)
        assert any("Missing" in i and "render.mp4" in i for i in issues)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_pack(d)
            open(os.path.join(d, "render.mp4"), "w").close()
            issues = validate_condition_pack(d)
        assert any("Empty" in i for i in issues)

    def test_frame_count_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_pack(d, n_frames=20)
            _write_video(os.path.join(d, "render_mask.mp4"), n_frames=15)
            issues = validate_condition_pack(d)
        assert any("mismatch" in i.lower() for i in issues)

    def test_suboptimal_frame_count_warns(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_pack(d, n_frames=22)
            issues = validate_condition_pack(d)
        assert any("dropping" in i.lower() or "optimal" in i.lower() for i in issues)

    def test_optimal_frame_count_no_warning(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_pack(d, n_frames=84)
            issues = validate_condition_pack(d)
        assert issues == []

    def test_too_few_frames_warns(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_pack(d, n_frames=4)
            issues = validate_condition_pack(d)
        assert any("at least" in i.lower() for i in issues)

    def test_pink_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            _make_valid_pack(d, n_frames=84)
            _write_video(os.path.join(d, "render_pink.mp4"), n_frames=50)
            issues = validate_condition_pack(d)
        assert any("render_pink" in i for i in issues)
