"""
render_pointcloud.py — PyTorch3D-based GPU point-cloud rasterizer.

Adapted from Uni3C src/pointcloud.py (point_rendering_2 variant).
Requires: torch, pytorch3d (source build), kornia, einops.

PyTorch3D has no Linux PyPI wheels and must be built from source:
  uv pip install --no-build-isolation \
      git+https://github.com/facebookresearch/pytorch3d.git
"""

from __future__ import annotations

import contextlib
import os
import sys

import einops
import kornia
import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    PerspectiveCameras,
    PointsRenderer,
    PointsRasterizer,
)
from pytorch3d.ops import knn_points
from pytorch3d.structures import Pointclouds


# ── Utility ────────────────────────────────────────────────────────────────────

def _points_padding(points: torch.Tensor) -> torch.Tensor:
    """Append a column of ones (homogeneous coords): [N, D] → [N, D+1]."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Context manager that silences C-level stdout/stderr (pytorch3d is noisy)."""
    with open(os.devnull, "w") as devnull:
        old_out = os.dup(sys.stdout.fileno())
        old_err = os.dup(sys.stderr.fileno())
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_out, sys.stdout.fileno())
            os.dup2(old_err, sys.stderr.fileno())
            os.close(old_out)
            os.close(old_err)


# ── Boundary / Sobel mask ──────────────────────────────────────────────────────

def _get_boundaries_mask(disparity: torch.Tensor, sobel_threshold: float = 0.3) -> torch.Tensor:
    """Return True where depth edges exist (suppress points on occlusion boundaries)."""
    def _sobel(disp: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
        grad = kornia.filters.spatial_gradient(disp, mode="sobel", normalized=False)
        mag = torch.sqrt(grad[:, :, 0] ** 2 + grad[:, :, 1] ** 2)
        return torch.exp(-beta * mag).detach()

    normed = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
    return _sobel(normed, beta=10.0) < sobel_threshold


# ── Statistical outlier removal ───────────────────────────────────────────────

def _statistical_outlier_mask(
    points: torch.Tensor,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> torch.Tensor:
    """Return True for outlier points (mean KNN distance > μ + std_ratio × σ)."""
    K = nb_neighbors + 1  # +1 because knn_points includes self
    pts = points.unsqueeze(0)  # [1, N, 3]
    dists, _, _ = knn_points(pts, pts, K=K)  # dists: [1, N, K] (squared)
    dists = dists[0, :, 1:]  # drop self-distance, [N, nb_neighbors]
    mean_dists = torch.sqrt(dists).mean(dim=1)  # [N]
    mu = mean_dists.mean()
    sigma = mean_dists.std()
    return mean_dists > (mu + std_ratio * sigma)


# ── Custom renderer ────────────────────────────────────────────────────────────

class _PointsZbufRenderer(PointsRenderer):
    """
    PointsRenderer variant that returns (rgb_image, zbuf) instead of just rgb.
    Uses distance-based weights: w = 1 - dist² / r².
    """

    def forward(self, point_clouds: Pointclouds, **kwargs):
        fragments = self.rasterizer(point_clouds, **kwargs)
        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = torch.clamp(1.0 - dists2 / (r * r), 0.0, 1.0)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )
        return images.permute(0, 2, 3, 1), fragments.zbuf  # [F,H,W,C], [F,H,W,K]


# ── Public API ─────────────────────────────────────────────────────────────────

def point_rendering_2(
    K: torch.Tensor,
    w2cs: torch.Tensor,
    depth: torch.Tensor,
    image: torch.Tensor,
    raster_settings,
    device: str,
    background_color: list[float] | None = None,
    sobel_threshold: float = 0.35,
    contract: float = 8.0,
    sam_mask=None,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
):
    """
    Render novel views of a single source frame via point-cloud rasterization.

    This is the "v2" variant: it unprojects using w2cs[0] (source camera) and
    renders only to the remaining cameras w2cs[1:] (target cameras), returning
    nframe-1 output images.

    Args:
        K:                [F, 3, 3] intrinsics (F = 1 source + N targets).
        w2cs:             [F, 4, 4] world-to-camera matrices (OpenCV convention).
        depth:            [1, 1, H, W] metric depth of the source frame.
        image:            [1, 3, H, W] source RGB in [-1, 1].
        raster_settings:  PointsRasterizationSettings instance.
        device:           Target torch device string (e.g. "cuda").
        background_color: RGB background in [-1, 1] range; default black.
        sobel_threshold:  Boundary suppression threshold (0 = keep all points).
        contract:         Depth contraction factor for far points.
        sam_mask:         Optional [1, 1, H, W] bool mask (True = keep point).

    Returns:
        render_rgbs:  [F-1, 3, H, W] float32 rendered colors in [-1, 1].
        render_masks: [F-1, 1, H, W] float32 hole mask (1 = missing, 0 = filled).
    """
    if background_color is None:
        background_color = [0.0, 0.0, 0.0]

    nframe = w2cs.shape[0]
    _, _, h, w = image.shape

    depth = depth.to(device)
    K = K.to(device)
    w2cs = w2cs.to(device)
    image = image.to(device)
    c2ws = w2cs.inverse()

    n_out = nframe - 1  # number of target cameras

    if depth.max() == 0:
        render_rgbs = torch.zeros((n_out, 3, h, w), device=device, dtype=torch.float32)
        render_masks = torch.ones((n_out, 1, h, w), device=device, dtype=torch.float32)
        return render_rgbs, render_masks

    # Depth contraction: map far points toward a finite range.
    mid_depth = torch.median(depth.reshape(-1)) * contract
    far = depth > mid_depth
    depth[far] = (2 * mid_depth) - (mid_depth ** 2 / (depth[far] + 1e-6))

    point_depth = einops.rearrange(depth[0], "c h w -> (h w) c")  # [HW, 1]

    # Boundary mask from disparity (suppress depth-edge points).
    disp = 1.0 / (depth + 1e-7)
    boundary_mask = _get_boundaries_mask(disp, sobel_threshold=sobel_threshold)

    # Build pixel grid [HW, 2] and unproject to world 3D.
    xs = torch.arange(w, dtype=torch.float32, device=device) + 0.5
    ys = torch.arange(h, dtype=torch.float32, device=device) + 0.5
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), -1)  # [W, H, 2]
    points_2d = einops.rearrange(grid, "w h c -> (h w) c")         # [HW, 2]
    points_3d = (
        c2ws[0]
        @ _points_padding(
            (K[0].inverse() @ _points_padding(points_2d).T).T * point_depth
        ).T
    ).T[:, :3]  # [HW, 3] in world coords

    colors = einops.rearrange(image[0], "c h w -> (h w) c")  # [HW, 3]

    # Apply boundary mask.
    bm = boundary_mask.reshape(-1)
    if sam_mask is not None:
        bm[sam_mask.reshape(-1) == True] = True
    keep = bm == False
    points_3d = points_3d[keep]
    colors = colors[keep]

    # Statistical outlier removal.
    if std_ratio > 0 and points_3d.shape[0] > nb_neighbors + 1:
        outlier = _statistical_outlier_mask(points_3d, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        points_3d = points_3d[~outlier]
        colors = colors[~outlier]

    if points_3d.shape[0] <= 8:
        render_rgbs = torch.zeros((n_out, 3, h, w), device=device, dtype=torch.float32)
        render_masks = torch.ones((n_out, 1, h, w), device=device, dtype=torch.float32)
        return render_rgbs, render_masks

    point_cloud = Pointclouds(
        points=[points_3d.to(device)],
        features=[colors.to(device)],
    ).extend(n_out)

    # Convert OpenCV → OpenGL camera convention for pytorch3d.
    c2ws = c2ws.clone()
    c2ws[:, :, 0] = -c2ws[:, :, 0]
    c2ws[:, :, 1] = -c2ws[:, :, 1]
    w2cs = c2ws.inverse()

    # Set up target cameras (skip source at index 0).
    focal_length = torch.stack([K[1:, 0, 0], K[1:, 1, 1]], dim=1)
    principal_point = torch.stack([K[1:, 0, 2], K[1:, 1, 2]], dim=1)
    image_shapes = torch.tensor([[h, w]], device=device).repeat(n_out, 1)
    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=c2ws[1:, :3, :3],
        T=w2cs[1:, :3, 3],
        in_ndc=False,
        image_size=image_shapes,
        device=device,
    )

    renderer = _PointsZbufRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )

    try:
        with _suppress_stdout_stderr():
            render_rgbs, zbuf = renderer(point_cloud)  # [N, H, W, 3], [N, H, W, K]
    except Exception as exc:
        raise RuntimeError(f"pytorch3d rasterization failed: {exc}") from exc

    render_masks = (zbuf[..., 0:1] == -1).float()               # [N, H, W, 1]
    render_rgbs = einops.rearrange(render_rgbs, "f h w c -> f c h w")   # [N, 3, H, W]
    render_masks = einops.rearrange(render_masks, "f h w c -> f c h w") # [N, 1, H, W]

    return render_rgbs, render_masks
