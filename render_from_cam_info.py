#!/usr/bin/env python3
"""
render_from_cam_info.py — Novel-view point-cloud renderer + condition-pack exporter.

Inputs:
  - RGB video (.mp4)
  - Metric depth maps (.npz with key='depths')
  - cam_info.json exported from the visualizer (source + target camera poses)

Outputs (in --output-dir):
  - input.mp4         (trimmed/resampled source video used for rendering)
  - render.mp4        (novel-view render on black background)
  - render_mask.mp4   (white = hole / missing pixel, black = covered pixel)
  - render_<bg>.mp4   (e.g. render_pink.mp4) for extra backgrounds
  - cam_info.json     (camera params at render resolution — condition for next stage)
  - reference.png     (first source frame — reference image for generation stage)

The output directory is a self-contained condition package for the generation
pipeline (cam_control.py / PCDControllerPipeline).  Pass it as --render_path and
pass reference.png as --reference_image to the generation script.

Backends:
  numpy     — Pure NumPy z-buffer rasterizer; no GPU/PyTorch3D required.
  gpu_point — Uni3C PyTorch3D soft point rasterizer (higher quality).
              Requires: torch, pytorch3d (source build), kornia, einops.
              See README for pytorch3d installation instructions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def _load_video_frames(
    video_path: str,
    max_frames: int = -1,
    target_fps: float | None = None,
) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    if target_fps is not None and target_fps > 0 and target_fps < src_fps:
        step = max(1, int(round(src_fps / target_fps)))
        out_fps = float(target_fps)
    else:
        step = 1
        out_fps = src_fps

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            if max_frames != -1 and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    return np.stack(frames, axis=0), out_fps


def _load_depths(depth_path: str) -> np.ndarray:
    data = np.load(depth_path)
    if "depths" not in data:
        raise KeyError(f"'depths' key not found in {depth_path}; keys={list(data.keys())}")
    depths = data["depths"].astype(np.float32)
    if depths.ndim == 4 and depths.shape[1] == 1:
        depths = depths[:, 0]
    if depths.ndim != 3:
        raise ValueError(f"Expected depths shape [T,H,W] or [T,1,H,W], got {depths.shape}")
    return depths


def _load_cam_info(cam_info_path: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    with open(cam_info_path, "r") as f:
        raw = json.load(f)
    K = np.array(raw["intrinsic"], dtype=np.float32)
    extr = np.array(raw["extrinsic"], dtype=np.float32)
    H = int(raw["height"])
    W = int(raw["width"])
    if K.shape != (3, 3):
        raise ValueError(f"cam_info intrinsic must be [3,3], got {K.shape}")
    if extr.ndim != 3 or extr.shape[1:] != (4, 4) or extr.shape[0] < 2:
        raise ValueError(
            f"cam_info extrinsic must be [N>=2,4,4] (source + targets), got {extr.shape}"
        )
    return K, extr, H, W


def _scale_intrinsics(K: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    if (src_h, src_w) == (dst_h, dst_w):
        return K.astype(np.float32, copy=True)
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    K2 = K.astype(np.float32, copy=True)
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def _write_video_rgb(path: str, frames_rgb: list[np.ndarray], fps: float) -> None:
    if not frames_rgb:
        raise ValueError(f"No frames to write: {path}")
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"imageio is required to write videos ({e}). "
            "Install with `uv sync --group render` or `uv pip install 'imageio[ffmpeg]'`."
        ) from e

    writer = None
    try:
        writer = imageio.get_writer(
            path,
            fps=float(fps) if fps and fps > 0 else 30.0,
            codec="libx264",
            format="FFMPEG",
            macro_block_size=None,
        )
        for rgb in frames_rgb:
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"Expected RGB frame [H,W,3], got {rgb.shape}")
            writer.append_data(np.ascontiguousarray(rgb))
    except Exception as e:
        raise RuntimeError(
            f"Failed to write video with imageio: {path}. "
            f"{e.__class__.__name__}: {e}. "
            "Install FFmpeg plugin with `uv sync --group render` or `uv pip install 'imageio[ffmpeg]'`."
        ) from e
    finally:
        if writer is not None:
            writer.close()


def _parse_backgrounds(spec: str) -> dict[str, np.ndarray]:
    presets: dict[str, tuple[int, int, int]] = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (127, 127, 127),
        "pink": (255, 65, 174),  # matches reference script
        "green": (0, 255, 0),
    }
    out: dict[str, np.ndarray] = {}
    for item in [s.strip().lower() for s in spec.split(",") if s.strip()]:
        if item not in presets:
            raise ValueError(f"Unknown background '{item}'. Choices: {sorted(presets)}")
        out[item] = np.array(presets[item], dtype=np.uint8)
    if "black" not in out:
        out = {"black": np.array((0, 0, 0), dtype=np.uint8), **out}
    return out


def _prepare_sample_grid(H: int, W: int, K_src: np.ndarray, subsample: int) -> dict[str, np.ndarray]:
    ys = np.arange(0, H, subsample, dtype=np.int32)
    xs = np.arange(0, W, subsample, dtype=np.int32)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    ones = np.ones_like(xv, dtype=np.float32)
    pix = np.stack([xv.astype(np.float32), yv.astype(np.float32), ones], axis=-1).reshape(-1, 3)
    K_inv = np.linalg.inv(K_src).astype(np.float32)
    rays = (K_inv @ pix.T).T.astype(np.float32)  # [N,3]
    return {
        "xv": xv.reshape(-1),
        "yv": yv.reshape(-1),
        "rays": rays,
    }


def _render_frame_pointcloud(
    rgb_src: np.ndarray,
    depth_src: np.ndarray,
    grid: dict[str, np.ndarray],
    K_tgt: np.ndarray,
    w2c_src: np.ndarray,
    w2c_tgt: np.ndarray,
    background_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Render a single frame with a simple z-buffer point rasterizer.

    Returns:
      render_rgb [H,W,3]
      hole_mask  [H,W] uint8 (255 = hole / missing, 0 = covered)
    """
    H, W = depth_src.shape
    out = np.empty((H, W, 3), dtype=np.uint8)
    out[...] = background_rgb[None, None, :]
    hole_mask = np.full((H, W), 255, dtype=np.uint8)

    x_src = grid["xv"]
    y_src = grid["yv"]
    rays = grid["rays"]  # [N,3]
    d = depth_src[y_src, x_src].astype(np.float32)
    valid = np.isfinite(d) & (d > 1e-6)
    if not np.any(valid):
        return out, hole_mask

    rays = rays[valid]
    d = d[valid]
    colors = rgb_src[y_src[valid], x_src[valid]]

    # Source camera -> world.
    c2w_src = np.linalg.inv(w2c_src).astype(np.float32)
    R_sw = c2w_src[:3, :3]
    t_sw = c2w_src[:3, 3]
    pts_src = rays * d[:, None]                             # [N,3]
    pts_world = pts_src @ R_sw.T + t_sw[None, :]            # [N,3]

    # World -> target camera.
    R_tw = w2c_tgt[:3, :3].astype(np.float32)
    t_tw = w2c_tgt[:3, 3].astype(np.float32)
    pts_tgt = pts_world @ R_tw.T + t_tw[None, :]
    z = pts_tgt[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-5)
    if not np.any(valid_z):
        return out, hole_mask

    pts_tgt = pts_tgt[valid_z]
    colors = colors[valid_z]
    z = z[valid_z]

    fx, fy = float(K_tgt[0, 0]), float(K_tgt[1, 1])
    cx, cy = float(K_tgt[0, 2]), float(K_tgt[1, 2])
    u = fx * (pts_tgt[:, 0] / z) + cx
    v = fy * (pts_tgt[:, 1] / z) + cy

    xi = np.rint(u).astype(np.int32)
    yi = np.rint(v).astype(np.int32)
    inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & np.isfinite(u) & np.isfinite(v)
    if not np.any(inside):
        return out, hole_mask

    xi = xi[inside]
    yi = yi[inside]
    z = z[inside]
    colors = colors[inside]

    # Z-buffer via sort: draw far -> near so near points overwrite.
    flat_idx = yi * W + xi
    order = np.argsort(z, kind="mergesort")[::-1]
    flat_idx = flat_idx[order]
    colors = colors[order]

    flat_rgb = out.reshape(-1, 3)
    flat_mask = hole_mask.reshape(-1)
    flat_rgb[flat_idx] = colors
    flat_mask[flat_idx] = 0
    return out, hole_mask


def _resize_video_if_needed(frames: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    H_t, W_t = target_hw
    if frames.shape[1:3] == (H_t, W_t):
        return frames
    out = []
    for fr in frames:
        out.append(cv2.cvtColor(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), (W_t, H_t)), cv2.COLOR_BGR2RGB))
    return np.stack(out, axis=0)


def _render_frame_gpu(
    rgb_src: np.ndarray,
    depth_src: np.ndarray,
    K_tgt: np.ndarray,
    w2c_src: np.ndarray,
    w2c_tgt: np.ndarray,
    *,
    device: str,
    point_rendering_2,
    points_rasterization_settings_cls,
    radius: float,
    points_per_pixel: int,
    sobel_threshold: float,
    background_rgb: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Render one frame using the PyTorch3D point rasterizer (GPU).

    Returns:
      render_rgb [H,W,3] uint8
      hole_mask  [H,W] uint8 (255 = hole, 0 = covered)
    """
    import torch

    H, W = depth_src.shape
    rgb_t = (
        torch.from_numpy(rgb_src.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )  # [1,3,H,W], 0..1
    depth_t = torch.from_numpy(depth_src.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    K_pair = torch.from_numpy(np.stack([K_tgt, K_tgt], axis=0).astype(np.float32))      # [2,3,3]
    w2c_pair = torch.from_numpy(np.stack([w2c_src, w2c_tgt], axis=0).astype(np.float32)) # [2,4,4]

    bg_norm = (background_rgb.astype(np.float32) / 127.5) - 1.0
    bg_norm_list = [float(bg_norm[0]), float(bg_norm[1]), float(bg_norm[2])]

    raster_settings = points_rasterization_settings_cls(
        image_size=(H, W),
        radius=float(radius),
        points_per_pixel=int(points_per_pixel),
    )

    autocast_ctx = (
        torch.amp.autocast("cuda", enabled=False)
        if str(device).startswith("cuda")
        else _nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        render_t, mask_t = point_rendering_2(
            K=K_pair.float(),
            w2cs=w2c_pair.float(),
            depth=depth_t.float(),
            image=(rgb_t.float() * 2.0) - 1.0,
            raster_settings=raster_settings,
            device=device,
            background_color=bg_norm_list,
            sobel_threshold=float(sobel_threshold),
            nb_neighbors=int(nb_neighbors),
            std_ratio=float(std_ratio),
        )

    # point_rendering_2 returns only target renders when passed [source,target].
    # Expected shapes: render_t [1,3,H,W], mask_t [1,1,H,W]
    render_np = render_t[0].detach().float().cpu().permute(1, 2, 0).numpy()  # [-1,1]
    render_np = np.clip((render_np + 1.0) * 0.5, 0.0, 1.0)
    render_rgb = (render_np * 255.0 + 0.5).astype(np.uint8)

    hole = mask_t[0, 0].detach().float().cpu().numpy()
    hole_mask = (hole > 0.5).astype(np.uint8) * 255
    return render_rgb, hole_mask


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Minimal standalone point-cloud renderer from video + depths + cam_info.json"
    )
    p.add_argument("--video", required=True, help="Input source video (.mp4)")
    p.add_argument("--depth", required=True, help="Depth .npz with key='depths' [T,H,W]")
    p.add_argument("--cam-info", dest="cam_info", required=True, help="cam_info.json from visualizer export")
    p.add_argument("--output-dir", required=True, help="Output directory for render videos")
    p.add_argument("--max-frames", type=int, default=-1, help="Limit loaded source frames (-1 = all)")
    p.add_argument("--target-fps", type=float, default=None, help="Downsample source video before rendering")
    p.add_argument("--fps", type=float, default=0.0, help="Output video FPS (0 = source FPS)")
    p.add_argument("--subsample", type=int, default=1, help="Use every Nth source pixel for rendering")
    p.add_argument(
        "--backend",
        type=str,
        default="gpu_point",
        choices=["gpu_point", "numpy"],
        help="Rendering backend: Uni3C GPU point rasterizer or minimal NumPy z-buffer",
    )
    p.add_argument("--device", type=str, default="cuda", help="Torch device for gpu_point backend")
    p.add_argument("--radius", type=float, default=0.008, help="Point raster radius (gpu_point backend)")
    p.add_argument(
        "--points-per-pixel",
        type=int,
        default=8,
        help="Points per pixel in rasterization (gpu_point backend)",
    )
    p.add_argument(
        "--sobel-threshold",
        type=float,
        default=0.35,
        help="Boundary suppression threshold passed to Uni3C point renderer",
    )
    p.add_argument(
        "--nb-neighbors",
        type=int,
        default=20,
        help="KNN neighbors for statistical outlier removal (gpu_point backend)",
    )
    p.add_argument(
        "--std-ratio",
        type=float,
        default=1.0,
        help="Std ratio threshold for statistical outlier removal (0 = disabled)",
    )
    p.add_argument(
        "--backgrounds",
        type=str,
        default="black,pink",
        help="Comma-separated backgrounds to export (choices: black,pink,white,gray,green)",
    )
    return p


def render_assets_from_paths(
    *,
    video: str,
    depth: str,
    cam_info: str,
    output_dir: str,
    max_frames: int = -1,
    target_fps: float | None = None,
    fps: float = 0.0,
    subsample: int = 1,
    backend: str = "gpu_point",
    device: str = "cuda",
    radius: float = 0.008,
    points_per_pixel: int = 8,
    sobel_threshold: float = 0.35,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    backgrounds: str = "black,pink",
) -> dict[str, str]:
    if subsample < 1:
        raise ValueError("--subsample must be >= 1")
    if backend not in {"gpu_point", "numpy"}:
        raise ValueError(f"Invalid backend: {backend}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] Loading video: {video}")
    frames_rgb, src_fps = _load_video_frames(video, max_frames=max_frames, target_fps=target_fps)
    print(f"[render] Video frames: {frames_rgb.shape}  fps={src_fps:.2f}")

    print(f"[render] Loading depths: {depth}")
    depths = _load_depths(depth)
    print(f"[render] Depths: {depths.shape}  range=({float(np.nanmin(depths)):.3f}, {float(np.nanmax(depths)):.3f})")

    print(f"[render] Loading cam info: {cam_info}")
    K_cam, extrinsics, H_cam, W_cam = _load_cam_info(cam_info)
    w2c_src = extrinsics[0]
    w2c_targets = extrinsics[1:]

    # Depth resolution is the rendering resolution. Resize RGB if needed.
    H_d, W_d = depths.shape[1:]
    if frames_rgb.shape[1:3] != (H_d, W_d):
        print(f"[render] Resizing video frames from {frames_rgb.shape[1:3]} -> {(H_d, W_d)} to match depths")
        frames_rgb = _resize_video_if_needed(frames_rgb, (H_d, W_d))

    K = _scale_intrinsics(K_cam, (H_cam, W_cam), (H_d, W_d))
    if (H_cam, W_cam) != (H_d, W_d):
        print(f"[render] Scaled intrinsics from cam_info size {(H_cam, W_cam)} -> depth size {(H_d, W_d)}")

    n_src = min(frames_rgb.shape[0], depths.shape[0])
    if n_src <= 0:
        raise ValueError("No overlapping source frames between video and depth.")
    frames_rgb = frames_rgb[:n_src]
    depths = depths[:n_src]

    n_out = len(w2c_targets)
    if n_out <= 0:
        raise ValueError("cam_info.json contains no target cameras (need extrinsic[1:]).")

    # Map each output camera to a source frame/depth index.
    if n_out == 1:
        src_indices = np.array([0], dtype=np.int32)
    else:
        src_indices = np.rint(np.linspace(0, n_src - 1, n_out)).astype(np.int32)
    print(f"[render] Source frames={n_src}, target cameras={n_out}, mapped source indices range={src_indices.min()}..{src_indices.max()}")

    fps_out = float(fps) if fps and fps > 0 else float(src_fps)
    bgs = _parse_backgrounds(backgrounds)

    # Prepare output buffers.
    render_black_frames: list[np.ndarray] = []
    render_mask_frames: list[np.ndarray] = []
    extra_bg_frames: dict[str, list[np.ndarray]] = {k: [] for k in bgs if k != "black"}

    use_gpu_point = backend == "gpu_point"
    grid = None
    point_rendering_2 = None
    PointsRasterizationSettings = None
    if use_gpu_point:
        print(f"[render] Using GPU point rasterizer backend on device={device}")
        try:
            from pytorch3d.renderer import PointsRasterizationSettings as _PRS  # type: ignore
            from render_pointcloud import point_rendering_2 as _pr2
        except Exception as e:
            raise RuntimeError(
                "Failed to import GPU renderer backend. "
                "The gpu_point backend requires: torch, pytorch3d, kornia, einops. "
                "PyTorch3D has no Linux PyPI wheels and must be built from source:\n"
                "  uv pip install --no-build-isolation "
                "git+https://github.com/facebookresearch/pytorch3d.git\n"
                f"Original error: {e.__class__.__name__}: {e}"
            ) from e
        point_rendering_2 = _pr2
        PointsRasterizationSettings = _PRS
    else:
        print("[render] Using minimal NumPy z-buffer backend")
        grid = _prepare_sample_grid(H_d, W_d, K, subsample)
        print(f"[render] NumPy backend subsample={subsample}")
    bg_black = bgs["black"]

    print(f"[render] Rendering {n_out} frames (subsample={subsample}) ...")
    for j in range(n_out):
        src_i = int(src_indices[j])
        if use_gpu_point:
            render_rgb, hole_mask = _render_frame_gpu(
                rgb_src=frames_rgb[src_i],
                depth_src=depths[src_i],
                K_tgt=K,
                w2c_src=w2c_src,
                w2c_tgt=w2c_targets[j],
                device=device,
                point_rendering_2=point_rendering_2,
                points_rasterization_settings_cls=PointsRasterizationSettings,
                radius=radius,
                points_per_pixel=points_per_pixel,
                sobel_threshold=sobel_threshold,
                background_rgb=bg_black,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio,
            )
            render_black_frames.append(render_rgb)
            render_mask_frames.append(np.repeat(hole_mask[..., None], 3, axis=2))

            # Match reference behavior more closely: re-render for each background color.
            for name, bg in bgs.items():
                if name == "black":
                    continue
                render_bg, _ = _render_frame_gpu(
                    rgb_src=frames_rgb[src_i],
                    depth_src=depths[src_i],
                    K_tgt=K,
                    w2c_src=w2c_src,
                    w2c_tgt=w2c_targets[j],
                    device=device,
                    point_rendering_2=point_rendering_2,
                    points_rasterization_settings_cls=PointsRasterizationSettings,
                    radius=radius,
                    points_per_pixel=points_per_pixel,
                    sobel_threshold=sobel_threshold,
                    background_rgb=bg,
                    nb_neighbors=nb_neighbors,
                    std_ratio=std_ratio,
                )
                extra_bg_frames[name].append(render_bg)
        else:
            render_rgb, hole_mask = _render_frame_pointcloud(
                rgb_src=frames_rgb[src_i],
                depth_src=depths[src_i],
                grid=grid,
                K_tgt=K,
                w2c_src=w2c_src,
                w2c_tgt=w2c_targets[j],
                background_rgb=bg_black,
            )
            render_black_frames.append(render_rgb)
            render_mask_frames.append(np.repeat(hole_mask[..., None], 3, axis=2))

            holes = hole_mask > 0
            for name, bg in bgs.items():
                if name == "black":
                    continue
                comp = render_rgb.copy()
                comp[holes] = bg
                extra_bg_frames[name].append(comp)

        if (j + 1) % 10 == 0 or j == n_out - 1:
            print(f"  {j+1}/{n_out}", end="\r", flush=True)
    print()

    # Also export the trimmed source video used for rendering.
    input_out_frames = [frames_rgb[int(i)] for i in src_indices]
    _write_video_rgb(str(out_dir / "input.mp4"), input_out_frames, fps_out)
    _write_video_rgb(str(out_dir / "render.mp4"), render_black_frames, fps_out)
    _write_video_rgb(str(out_dir / "render_mask.mp4"), render_mask_frames, fps_out)
    print(f"[render] Wrote {out_dir / 'input.mp4'}")
    print(f"[render] Wrote {out_dir / 'render.mp4'}")
    print(f"[render] Wrote {out_dir / 'render_mask.mp4'}  (white = hole / missing)")

    for name, frames in extra_bg_frames.items():
        path = out_dir / f"render_{name}.mp4"
        _write_video_rgb(str(path), frames, fps_out)
        print(f"[render] Wrote {path}")

    # ── Condition package for the generation stage ─────────────────────────────
    # cam_control.py / PCDControllerPipeline (dataset.py) reads render.mp4,
    # render_mask.mp4, and cam_info.json from the same render_path directory.
    # Save cam_info at the actual render resolution so the output dir is
    # self-contained and can be passed directly as --render_path.
    cam_info_out = {
        "intrinsic": K.tolist(),           # scaled to render resolution H_d × W_d
        "extrinsic": extrinsics.tolist(),  # full [N, 4, 4] — source + all targets
        "height": int(H_d),
        "width": int(W_d),
    }
    cam_info_dst = out_dir / "cam_info.json"
    with open(str(cam_info_dst), "w") as f:
        json.dump(cam_info_out, f, indent=2)
    print(f"[render] Wrote {cam_info_dst}  (condition: camera params at render res)")

    # Save first source frame as reference.png — the high-quality reference image
    # that the generation pipeline uses to condition the first frame.
    ref_frame = frames_rgb[0]  # [H_d, W_d, 3] uint8 RGB at render resolution
    ref_dst = out_dir / "reference.png"
    cv2.imwrite(str(ref_dst), cv2.cvtColor(ref_frame, cv2.COLOR_RGB2BGR))
    print(f"[render] Wrote {ref_dst}  (condition: reference image for generation)")

    outputs = {
        "input": str(out_dir / "input.mp4"),
        "render": str(out_dir / "render.mp4"),
        "render_mask": str(out_dir / "render_mask.mp4"),
        "cam_info": str(cam_info_dst),
        "reference": str(ref_dst),
    }
    for name in extra_bg_frames:
        outputs[f"render_{name}"] = str(out_dir / f"render_{name}.mp4")
    return outputs


def main() -> None:
    args = _build_parser().parse_args()
    render_assets_from_paths(
        video=args.video,
        depth=args.depth,
        cam_info=args.cam_info,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        target_fps=args.target_fps,
        fps=args.fps,
        subsample=args.subsample,
        backend=args.backend,
        device=args.device,
        radius=args.radius,
        points_per_pixel=args.points_per_pixel,
        sobel_threshold=args.sobel_threshold,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        backgrounds=args.backgrounds,
    )


if __name__ == "__main__":
    main()
