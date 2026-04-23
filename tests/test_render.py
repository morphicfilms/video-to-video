# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""Tests for render_from_cam_info.py — intrinsic scaling and cam_info loading."""
import pytest
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


class TestScaleIntrinsics:
    @pytest.fixture(autouse=True)
    def _import(self):
        from render_from_cam_info import _scale_intrinsics
        self.scale = _scale_intrinsics

    def test_identity_when_same_resolution(self):
        K = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
        K2 = self.scale(K, (480, 640), (480, 640))
        np.testing.assert_allclose(K2, K)

    def test_scales_focal_and_principal(self):
        K = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
        K2 = self.scale(K, (480, 640), (960, 1280))
        assert K2[0, 0] == pytest.approx(1280.0)
        assert K2[1, 1] == pytest.approx(1280.0)
        assert K2[0, 2] == pytest.approx(640.0)
        assert K2[1, 2] == pytest.approx(480.0)

    def test_half_resolution(self):
        K = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
        K2 = self.scale(K, (480, 640), (240, 320))
        assert K2[0, 0] == pytest.approx(320.0)
        assert K2[0, 2] == pytest.approx(160.0)

    def test_does_not_modify_input(self):
        K = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
        K_orig = K.copy()
        self.scale(K, (480, 640), (960, 1280))
        np.testing.assert_array_equal(K, K_orig)


class TestLoadCamInfo:
    @pytest.fixture(autouse=True)
    def _import(self):
        from render_from_cam_info import _load_cam_info
        self.load = _load_cam_info

    def _write_cam_info(self, tmpdir, k_resolution=None, n_targets=5):
        data = {
            "intrinsic": np.eye(3).tolist(),
            "extrinsic": [np.eye(4).tolist() for _ in range(1 + n_targets)],
            "height": 480,
            "width": 640,
        }
        if k_resolution is not None:
            data["k_resolution"] = k_resolution
        path = os.path.join(tmpdir, "cam_info.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_basic_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write_cam_info(d)
            K, extr, H, W = self.load(path)
        assert K.shape == (3, 3)
        assert extr.shape == (6, 4, 4)
        assert H == 480
        assert W == 640

    def test_k_resolution_mismatch_warns(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            path = self._write_cam_info(d, k_resolution=[360, 640])
            self.load(path)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_k_resolution_match_no_warning(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            path = self._write_cam_info(d, k_resolution=[480, 640])
            self.load(path)
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_no_k_resolution_no_warning(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            path = self._write_cam_info(d)
            self.load(path)
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_rejects_bad_intrinsic(self):
        data = {
            "intrinsic": [[1, 0], [0, 1]],
            "extrinsic": [np.eye(4).tolist(), np.eye(4).tolist()],
            "height": 480, "width": 640,
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cam_info.json")
            with open(path, "w") as f:
                json.dump(data, f)
            with pytest.raises(ValueError, match="3,3"):
                self.load(path)

    def test_rejects_too_few_extrinsics(self):
        data = {
            "intrinsic": np.eye(3).tolist(),
            "extrinsic": [np.eye(4).tolist()],
            "height": 480, "width": 640,
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cam_info.json")
            with open(path, "w") as f:
                json.dump(data, f)
            with pytest.raises(ValueError, match="N>=2"):
                self.load(path)
