# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""Tests for visualizer/interpolation.py — camera path interpolation and coordinate conversions."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from visualizer.interpolation import (
    EASING_MODES,
    c2w_to_w2c,
    interpolate_camera_path,
    viser_c2w_to_opencv,
    opencv_c2w_to_viser,
)


class TestInterpolateCameraPath:
    def _identity_keyframes(self, n=3):
        kfs = []
        for i in range(n):
            c2w = np.eye(4, dtype=np.float32)
            c2w[0, 3] = float(i)
            kfs.append(c2w)
        return kfs

    def test_output_shape(self):
        kfs = self._identity_keyframes(3)
        path = interpolate_camera_path(kfs, n_frames=50)
        assert path.shape == (50, 4, 4)
        assert path.dtype == np.float32

    def test_endpoints_match_keyframes(self):
        kfs = self._identity_keyframes(3)
        path = interpolate_camera_path(kfs, n_frames=50)
        np.testing.assert_allclose(path[0], kfs[0], atol=1e-4)
        np.testing.assert_allclose(path[-1], kfs[-1], atol=1e-4)

    def test_two_keyframes(self):
        kfs = self._identity_keyframes(2)
        path = interpolate_camera_path(kfs, n_frames=21)
        assert path.shape == (21, 4, 4)
        np.testing.assert_allclose(path[0, 0, 3], 0.0, atol=1e-5)
        np.testing.assert_allclose(path[-1, 0, 3], 1.0, atol=1e-5)

    def test_rejects_single_keyframe(self):
        with pytest.raises(ValueError, match="At least 2"):
            interpolate_camera_path([np.eye(4)], n_frames=10)

    def test_all_easing_modes(self):
        kfs = self._identity_keyframes(4)
        for mode in EASING_MODES:
            path = interpolate_camera_path(kfs, n_frames=30, easing=mode)
            assert path.shape == (30, 4, 4), f"Failed for mode={mode}"
            np.testing.assert_allclose(path[0], kfs[0], atol=1e-3, err_msg=f"mode={mode}")
            np.testing.assert_allclose(path[-1], kfs[-1], atol=1e-3, err_msg=f"mode={mode}")

    def test_rotation_interpolation(self):
        kf0 = np.eye(4, dtype=np.float32)
        kf1 = np.eye(4, dtype=np.float32)
        angle = np.pi / 4
        kf1[:3, :3] = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        path = interpolate_camera_path([kf0, kf1], n_frames=11)
        for i in range(11):
            R = path[i, :3, :3]
            det = np.linalg.det(R)
            assert det == pytest.approx(1.0, abs=1e-4), f"frame {i}: det={det}"


class TestC2wToW2c:
    def test_inverse(self):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = [1, 2, 3]
        w2c = c2w_to_w2c(c2w)
        product = w2c @ c2w
        np.testing.assert_allclose(product, np.eye(4), atol=1e-5)

    def test_dtype(self):
        assert c2w_to_w2c(np.eye(4, dtype=np.float64)).dtype == np.float32


class TestCoordinateConversions:
    def test_round_trip(self):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = [1, 2, 3]
        c2w_gl = opencv_c2w_to_viser(c2w)
        c2w_back = viser_c2w_to_opencv(c2w_gl)
        np.testing.assert_allclose(c2w_back, c2w, atol=1e-6)

    def test_y_z_flip(self):
        c2w_cv = np.eye(4, dtype=np.float32)
        c2w_gl = opencv_c2w_to_viser(c2w_cv)
        assert c2w_gl[1, 1] == pytest.approx(-1.0)
        assert c2w_gl[2, 2] == pytest.approx(-1.0)
        assert c2w_gl[0, 0] == pytest.approx(1.0)
