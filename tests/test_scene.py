# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""Tests for visualizer/scene.py — point cloud, intrinsics, and camera utilities."""
import pytest
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from visualizer.scene import (
    compute_intrinsics,
    get_source_camera,
    load_depth_maps,
    transform_points_to_world,
    unproject_frame,
)


class TestComputeIntrinsics:
    def test_shape(self):
        K = compute_intrinsics(480, 640)
        assert K.shape == (3, 3)
        assert K.dtype == np.float32

    def test_focal_from_max_dim(self):
        K = compute_intrinsics(480, 640, focal_multiplier=1.0)
        assert K[0, 0] == pytest.approx(640.0)
        assert K[1, 1] == pytest.approx(640.0)

    def test_principal_point_centered(self):
        K = compute_intrinsics(480, 640)
        assert K[0, 2] == pytest.approx(320.0)
        assert K[1, 2] == pytest.approx(240.0)

    def test_focal_multiplier(self):
        K = compute_intrinsics(480, 640, focal_multiplier=0.5)
        assert K[0, 0] == pytest.approx(320.0)

    def test_square_image(self):
        K = compute_intrinsics(512, 512)
        assert K[0, 0] == K[1, 1]
        assert K[0, 2] == pytest.approx(256.0)


class TestUnprojectFrame:
    def _make_data(self, H=60, W=80):
        frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        depth = np.random.uniform(0.5, 10.0, (H, W)).astype(np.float32)
        K = compute_intrinsics(H, W)
        return frame, depth, K

    def test_output_shape(self):
        frame, depth, K = self._make_data()
        pts, cols = unproject_frame(frame, depth, K, subsample=1)
        assert pts.shape[1] == 3
        assert cols.shape[1] == 3
        assert pts.shape[0] == cols.shape[0]

    def test_subsample_reduces_points(self):
        frame, depth, K = self._make_data()
        pts_full, _ = unproject_frame(frame, depth, K, subsample=1)
        pts_sub, _ = unproject_frame(frame, depth, K, subsample=4)
        assert pts_sub.shape[0] < pts_full.shape[0]

    def test_precomputed_K_inv_matches(self):
        frame, depth, K = self._make_data()
        K_inv = np.linalg.inv(K)
        pts_auto, cols_auto = unproject_frame(frame, depth, K, subsample=2)
        pts_pre, cols_pre = unproject_frame(frame, depth, K, subsample=2, K_inv=K_inv)
        np.testing.assert_allclose(pts_auto, pts_pre, atol=1e-5)
        np.testing.assert_array_equal(cols_auto, cols_pre)

    def test_invalid_depth_filtered(self):
        frame, depth, K = self._make_data(H=10, W=10)
        depth[0, :5] = 0.0
        depth[1, :3] = np.nan
        depth[2, :2] = -1.0
        pts, cols = unproject_frame(frame, depth, K)
        total_pixels = 10 * 10
        invalid_count = 5 + 3 + 2
        assert pts.shape[0] == total_pixels - invalid_count

    def test_dtype(self):
        frame, depth, K = self._make_data()
        pts, cols = unproject_frame(frame, depth, K)
        assert pts.dtype == np.float32


class TestTransformPointsToWorld:
    def test_identity(self):
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        result = transform_points_to_world(pts, c2w)
        np.testing.assert_allclose(result, pts, atol=1e-6)

    def test_translation(self):
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = [10, 20, 30]
        result = transform_points_to_world(pts, c2w)
        np.testing.assert_allclose(result, [[10, 20, 30]], atol=1e-6)

    def test_output_dtype(self):
        pts = np.array([[1, 2, 3]], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float64)
        result = transform_points_to_world(pts, c2w)
        assert result.dtype == np.float32


class TestGetSourceCamera:
    def test_returns_pair(self):
        w2c, c2w = get_source_camera(5.0)
        assert w2c.shape == (4, 4)
        assert c2w.shape == (4, 4)

    def test_inverse_relationship(self):
        w2c, c2w = get_source_camera(5.0, start_elevation=10.0)
        product = w2c @ c2w
        np.testing.assert_allclose(product, np.eye(4), atol=1e-5)

    def test_camera_at_negative_z(self):
        _, c2w = get_source_camera(5.0, start_elevation=0.0)
        cam_pos = c2w[:3, 3]
        assert cam_pos[2] == pytest.approx(-5.0, abs=1e-5)

    def test_dtype(self):
        w2c, c2w = get_source_camera(3.0)
        assert w2c.dtype == np.float32
        assert c2w.dtype == np.float32


class TestLoadDepthMaps:
    def test_loads_3d(self):
        depths = np.random.uniform(1, 10, (10, 60, 80)).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            np.savez(f.name, depths=depths)
            loaded = load_depth_maps(f.name)
        assert loaded.shape == (10, 60, 80)
        np.testing.assert_allclose(loaded, depths)

    def test_loads_4d_squeeze(self):
        depths = np.random.uniform(1, 10, (10, 1, 60, 80)).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            np.savez(f.name, depths=depths)
            loaded = load_depth_maps(f.name)
        assert loaded.shape == (10, 60, 80)

    def test_missing_key_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            np.savez(f.name, wrong_key=np.zeros((5, 10, 10)))
            with pytest.raises(KeyError, match="depths"):
                load_depth_maps(f.name)

    def test_wrong_ndim_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            np.savez(f.name, depths=np.zeros((10, 10)))
            with pytest.raises(ValueError, match="shape"):
                load_depth_maps(f.name)

    def test_depth_range_warning(self, capsys):
        depths = np.full((5, 10, 10), 0.001, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            np.savez(f.name, depths=depths)
            load_depth_maps(f.name)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "metric range" in captured.out

    def test_normal_range_no_warning(self, capsys):
        depths = np.full((5, 10, 10), 5.0, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            np.savez(f.name, depths=depths)
            load_depth_maps(f.name)
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out
