# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""Tests for visualizer/export.py — cam_info.json serialization and loading."""
import pytest
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from visualizer.export import export_cam_info, load_cam_info


class TestExportCamInfo:
    def _make_inputs(self, n_targets=10):
        K = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
        w2c_source = np.eye(4, dtype=np.float32)
        w2c_targets = np.stack([np.eye(4, dtype=np.float32)] * n_targets)
        return w2c_source, w2c_targets, K

    def test_creates_file(self):
        w2c_src, w2c_tgt, K = self._make_inputs()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "cam_info.json")
            export_cam_info(w2c_src, w2c_tgt, K, 480, 640, path)
            assert os.path.exists(path)

    def test_json_structure(self):
        w2c_src, w2c_tgt, K = self._make_inputs(5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cam_info.json")
            export_cam_info(w2c_src, w2c_tgt, K, 480, 640, path)
            with open(path) as f:
                data = json.load(f)
        assert "intrinsic" in data
        assert "extrinsic" in data
        assert "height" in data
        assert "width" in data
        assert "k_resolution" in data
        assert data["height"] == 480
        assert data["width"] == 640
        assert data["k_resolution"] == [480, 640]
        assert len(data["extrinsic"]) == 6  # 1 source + 5 targets

    def test_k_resolution_matches_hw(self):
        w2c_src, w2c_tgt, K = self._make_inputs(3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cam_info.json")
            export_cam_info(w2c_src, w2c_tgt, K, 360, 640, path)
            with open(path) as f:
                data = json.load(f)
        assert data["k_resolution"] == [360, 640]


class TestLoadCamInfo:
    def test_round_trip(self):
        K = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
        w2c_src = np.eye(4, dtype=np.float32)
        w2c_src[0, 3] = 1.5
        w2c_tgt = np.stack([np.eye(4, dtype=np.float32)] * 8)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cam_info.json")
            export_cam_info(w2c_src, w2c_tgt, K, 480, 640, path)
            loaded = load_cam_info(path)
        np.testing.assert_allclose(loaded["intrinsic"], K, atol=1e-6)
        assert loaded["extrinsic"].shape == (9, 4, 4)
        np.testing.assert_allclose(loaded["extrinsic"][0], w2c_src, atol=1e-6)
        assert loaded["height"] == 480
        assert loaded["width"] == 640
        assert loaded["k_resolution"] == (480, 640)

    def test_backward_compat_no_k_resolution(self):
        data = {
            "intrinsic": np.eye(3).tolist(),
            "extrinsic": [np.eye(4).tolist(), np.eye(4).tolist()],
            "height": 480,
            "width": 640,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            loaded = load_cam_info(path)
            assert "k_resolution" not in loaded
        finally:
            os.unlink(path)
