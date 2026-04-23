#!/usr/bin/env python3
# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""
estimate_depth.py — Standalone video metric depth estimation.

Supports two methods (--method):

  depthcrafter (default)
    Uses DepthCrafter (Tencent) to produce temporally-consistent metric video
    depth in a single pass.  No additional alignment step needed.
    Requires: DepthCrafter repo cloned at --dc-dir.

  gc_moge
    1. Run GeometryCrafter → temporally-consistent relative point maps [T, H, W, 3]
    2. Run MoGe v2 on frame 0 → metric depth anchor [H, W]
    3. Align all GC frames to MoGe metric scale (scale-only fit in inverse-depth space)
    Requires: GeometryCrafter repo at --gc-dir, MoGe pip-installed.

Output: depths.npz  (key='depths', shape [T,H,W], float32)
Compatible with the V2V visualizer:
    python -m visualizer.app --video input.mp4 --depth depths.npz

Usage:
    python estimate_depth.py --video input.mp4 --output depths.npz [options]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


# ── MoGe wrapper ──────────────────────────────────────────────────────────────

class _MoGeWrapper(torch.nn.Module):
    """
    Thin wrapper around a MoGe model (v1 or v2) that provides two interfaces:

    1. forward_image(images [B,3,H,W]) → (points [B,H,W,3], masks [B,H,W])
       This is the interface GeometryCrafter expects for its prior_model.

    2. infer(image [1,3,H,W], **kwargs) → dict
       Direct access to the underlying MoGe model for metric depth estimation.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._moge = model

    @torch.no_grad()
    def forward_image(self, image: torch.Tensor, **kwargs):
        """Called by GeometryCrafter internally to compute per-frame priors."""
        out = self._moge.infer(image, resolution_level=9, apply_mask=False, **kwargs)
        return out["points"], out["mask"]   # [B,H,W,3], [B,H,W]

    @torch.no_grad()
    def infer(self, image: torch.Tensor, **kwargs) -> dict:
        """Direct MoGe inference (for metric depth of frame 0)."""
        return self._moge.infer(image, **kwargs)


def _load_moge(gc_dir: str, cache_dir: str, device: str) -> _MoGeWrapper:
    """
    Load MoGe, preferring v2 (pip) then v1 (pip) then GC's bundled third_party.

    Priority:
      1. moge.model.v2  — pip install git+https://github.com/microsoft/MoGe.git
      2. moge.model.v1  — same pip package, older checkpoint
      3. GC third_party — bundled inside the cloned GeometryCrafter repo
    """
    # 1. MoGe v2 (pip-installed, preferred)
    # Use the user's own HF cache so we don't need write access to gc_dir
    user_hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    try:
        from moge.model.v2 import MoGeModel  # type: ignore
        model_id = "Ruicheng/moge-2-vitl-normal"
        print(f"[depth] Loading MoGe v2 ({model_id}) …")
        moge = MoGeModel.from_pretrained(model_id, cache_dir=user_hf_cache).eval()
        return _MoGeWrapper(moge).requires_grad_(False).to(device, dtype=torch.float32)
    except Exception as e:
        print(f"[depth] MoGe v2 unavailable ({e.__class__.__name__}: {e})")

    # 2. MoGe v1 (pip-installed)
    try:
        from moge.model.v1 import MoGeModel  # type: ignore
        model_id = "Ruicheng/moge-vitl"
        print(f"[depth] Loading MoGe v1 ({model_id}) …")
        moge = MoGeModel.from_pretrained(model_id, cache_dir=cache_dir).eval()
        return _MoGeWrapper(moge).requires_grad_(False).to(device, dtype=torch.float32)
    except Exception as e:
        print(f"[depth] MoGe v1 (pip) unavailable ({e.__class__.__name__}: {e})")

    # 3. GC's bundled MoGe via third_party (fallback — requires GC clone)
    print("[depth] Falling back to MoGe bundled in GC third_party …")
    sys.path.insert(0, gc_dir)
    sys.path.insert(0, os.path.join(gc_dir, "third_party", "moge"))
    from third_party import MoGe as _GCMoGe  # type: ignore

    # GC's MoGe already has forward_image(); we wrap it for unified .infer() access
    class _GCMoGeAdaptor(torch.nn.Module):
        def __init__(self, gc_moge):
            super().__init__()
            self._gc = gc_moge

        @torch.no_grad()
        def forward_image(self, image, **kwargs):
            return self._gc.forward_image(image, **kwargs)

        @torch.no_grad()
        def infer(self, image, **kwargs):
            out = self._gc.model.infer(image, **kwargs)
            return out

    gc_moge = _GCMoGe(cache_dir=cache_dir).requires_grad_(False).to(device, dtype=torch.float32)
    return _GCMoGeAdaptor(gc_moge)


# ── Alignment helper ──────────────────────────────────────────────────────────

def _align_inv_depth_to_depth(
    source_inv_depth: torch.Tensor,
    target_depth: torch.Tensor,
    align_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit scale-only in inverse-depth space such that:
        1 / (source_inv_depth * scale)  ≈  target_depth

    Uses 10th–90th percentile outlier rejection, then least-squares scale fit.
    Returns (aligned_depth [H,W], scale).
    """
    target_inv = 1.0 / target_depth.clamp(min=1e-4)
    valid = (source_inv_depth > 1e-3) & (target_depth > 1e-3)
    if align_mask is not None:
        valid = valid & align_mask.bool()

    if valid.sum() < 20:
        scale = torch.tensor(1.0, device=source_inv_depth.device)
        return target_depth.clone(), scale

    q = torch.tensor([0.1, 0.9], device=source_inv_depth.device)
    s_lo, s_hi = torch.quantile(source_inv_depth[valid], q)
    t_lo, t_hi = torch.quantile(target_inv[valid], q)
    valid = (
        valid
        & (source_inv_depth > s_lo) & (source_inv_depth < s_hi)
        & (target_inv > t_lo) & (target_inv < t_hi)
    )

    if valid.sum() < 20:
        scale = torch.tensor(1.0, device=source_inv_depth.device)
        return target_depth.clone(), scale

    src = source_inv_depth[valid]
    tgt = target_inv[valid]
    # scale-only: scale = (src · tgt) / (src · src)
    scale = (src * tgt).sum() / (src * src).sum()

    aligned_inv = (source_inv_depth * scale).clamp(min=1e-3)
    return aligned_inv.reciprocal(), scale


# ── Video loading ─────────────────────────────────────────────────────────────

def _load_video(
    video_path: str,
    max_frames: int,
    max_res: int,
) -> tuple[np.ndarray, int, int, int, int, float]:
    """
    Load video as float32 [T, H, W, 3] in [0, 1].

    Resizes so the longest side ≤ max_res and both dims are divisible by 64
    (required by GeometryCrafter).

    Returns (frames, H, W, orig_H, orig_W, fps) where H/W are the inference
    dimensions and orig_H/orig_W are the native video dimensions.
    """
    def _target_hw(oh: int, ow: int) -> tuple[int, int]:
        scale = min(1.0, max_res / max(oh, ow))
        h = round(oh * scale / 64) * 64
        w = round(ow * scale / 64) * 64
        return h, w

    # decord is faster and used by GC itself
    try:
        from decord import VideoReader, cpu  # type: ignore
        vr  = VideoReader(video_path, ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        oh, ow = vr.get_batch([0]).shape[1:3]
        h, w = _target_hw(oh, ow)
        n = len(vr) if max_frames == -1 else min(len(vr), max_frames)
        vr = VideoReader(video_path, ctx=cpu(0), width=w, height=h)
        frames = vr.get_batch(list(range(n))).asnumpy().astype(np.float32) / 255.0
        return frames, h, w, oh, ow, fps
    except ImportError:
        pass

    # OpenCV fallback
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    oh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h, w = _target_hw(oh, ow)

    frames_list: list[np.ndarray] = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(cv2.resize(bgr, (w, h)), cv2.COLOR_BGR2RGB)
        frames_list.append(rgb.astype(np.float32) / 255.0)
        if max_frames != -1 and len(frames_list) >= max_frames:
            break
    cap.release()

    if not frames_list:
        raise RuntimeError(f"No frames read from {video_path}")
    return np.stack(frames_list), h, w, oh, ow, fps


# ── Visualization ───────────────────────────────────────────────────────────────

def _save_depth_video(
    frames: np.ndarray,
    depths: np.ndarray,
    fps: float,
    video_path: str,
    depth_path: str,
) -> None:
    """
    Save a side‑by‑side RGB + depth visualization video next to the .npz.

    Args:
        frames: float32 [T, H, W, 3] in [0, 1] RGB
        depths: float32 [T, H, W] metric depth in metres
        fps:    frames per second for output video
        video_path: original input video path (for logging only)
        depth_path: path to the saved .npz depth file
    """
    if depths.ndim != 3 or frames.ndim != 4:
        print("[depth] Skipping depth video (unexpected shapes).")
        return

    T_f, H_f, W_f, _ = frames.shape
    T_d, H_d, W_d = depths.shape
    if T_f != T_d or H_f != H_d or W_f != W_d:
        print(
            f"[depth] Skipping depth video (shape mismatch: "
            f"frames={frames.shape}, depths={depths.shape})."
        )
        return

    valid = np.isfinite(depths) & (depths > 0.0)
    if not np.any(valid):
        print("[depth] Skipping depth video (no valid depth values).")
        return

    # Use inverse depth (disparity-like) for visualization because metric depth often
    # has low perceptual contrast over mid/far ranges.
    inv_depths = np.zeros_like(depths, dtype=np.float32)
    inv_depths[valid] = 1.0 / np.maximum(depths[valid], 1e-6)

    try:
        vmin, vmax = np.percentile(inv_depths[valid], [2.0, 98.0])
    except Exception:
        vmin, vmax = float(inv_depths[valid].min()), float(inv_depths[valid].max())

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        print("[depth] Skipping depth video (invalid depth range).")
        return

    base = os.path.splitext(depth_path)[0]
    vis_path = base + "_vis.mp4"

    fps_out = fps if fps and fps > 0 else 30.0
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as e:
        print(f"[depth] Skipping depth video (imageio unavailable: {e}).")
        return

    writer = None

    try:
        writer = imageio.get_writer(
            vis_path,
            fps=fps_out,
            codec="libx264",
            format="FFMPEG",
            macro_block_size=None,  # keep exact frame size (e.g. 448x1664)
        )
    except Exception as e:
        print(f"[depth] Failed to open imageio writer for {vis_path}: {e}")
        print("[depth] Install video backend support, e.g.:")
        print("[depth]   uv sync --group depth")
        print("[depth]   # or: uv pip install 'imageio[ffmpeg]'")
        return

    try:
        for t in range(T_d):
            depth = depths[t]
            valid_t = np.isfinite(depth) & (depth > 0.0)
            inv = np.zeros_like(depth, dtype=np.float32)
            inv[valid_t] = 1.0 / np.maximum(depth[valid_t], 1e-6)

            depth_norm = np.zeros_like(depth, dtype=np.float32)
            if np.any(valid_t):
                depth_norm[valid_t] = (inv[valid_t] - vmin) / (vmax - vmin)
            depth_norm = np.clip(depth_norm, 0.0, 1.0)
            depth_img = (depth_norm * 255.0).astype(np.uint8)

            depth_color_bgr = cv2.applyColorMap(depth_img, cv2.COLORMAP_TURBO)
            depth_color_bgr[~valid_t] = 0  # invalid depth pixels shown as black
            depth_color = cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)

            rgb = np.clip(frames[t] * 255.0, 0, 255).astype(np.uint8)
            vis = np.concatenate([rgb, depth_color], axis=1)
            writer.append_data(np.ascontiguousarray(vis))
    finally:
        if writer is not None:
            writer.close()

    print(f"[depth] Saved depth visualization video → {vis_path}")


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_gc_pipeline(gc_dir: str, cache_dir: str, device: str):
    """Load the GeometryCrafter UNet, PointMap VAE, and diffusion pipeline."""
    sys.path.insert(0, gc_dir)

    from geometrycrafter import (                         # type: ignore
        GeometryCrafterDiffPipeline,
        PMapAutoencoderKLTemporalDecoder,
        UNetSpatioTemporalConditionModelVid2vid,
    )

    print("[depth] Loading UNet …")
    unet = (
        UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
            "TencentARC/GeometryCrafter",
            subfolder="unet_diff",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        )
        .requires_grad_(False)
        .to(device, dtype=torch.float16)
    )

    print("[depth] Loading Point Map VAE …")
    point_map_vae = (
        PMapAutoencoderKLTemporalDecoder.from_pretrained(
            "TencentARC/GeometryCrafter",
            subfolder="point_map_vae",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            cache_dir=cache_dir,
        )
        .requires_grad_(False)
        .to(device, dtype=torch.float32)
    )

    print("[depth] Loading GeometryCrafter diffusion pipeline …")
    pipe = GeometryCrafterDiffPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=cache_dir,
    ).to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[depth] xformers memory-efficient attention enabled.")
    except Exception:
        pass
    pipe.enable_attention_slicing()

    return pipe, point_map_vae


# ── DepthCrafter ──────────────────────────────────────────────────────────────

def _load_depthcrafter(dc_dir: str, device: str):
    """Load the DepthCrafter pipeline (cloned repo, not pip-installed)."""
    sys.path.insert(0, dc_dir)

    from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline  # type: ignore
    from depthcrafter.unet import (                                  # type: ignore
        DiffusersUNetSpatioTemporalConditionModelDepthCrafter,
    )

    print("[depth] Loading DepthCrafter UNet …")
    unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
        "tencent/DepthCrafter",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    print("[depth] Loading DepthCrafter pipeline …")
    pipe = DepthCrafterPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[depth] xformers memory-efficient attention enabled.")
    except Exception:
        pass
    pipe.enable_attention_slicing()

    return pipe


def _run_depthcrafter(
    pipe,
    frames: np.ndarray,
    *,
    steps: int = 5,
    guidance_scale: float = 1.0,
    window_size: int = 110,
    overlap: int = 25,
    near: float = 0.0001,
    far: float = 10000.0,
) -> np.ndarray:
    """
    Run DepthCrafter on float32 [T, H, W, 3] frames in [0, 1].

    Returns metric depths as float32 [T, H, W].
    """
    from diffusers.training_utils import set_seed  # type: ignore
    set_seed(42)

    H, W = frames.shape[1], frames.shape[2]
    with torch.inference_mode():
        res = pipe(
            frames,
            height=H,
            width=W,
            output_type="np",
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            window_size=window_size,
            overlap=overlap,
            track_time=True,
        ).frames[0]

    # Convert three-channel output to single channel
    res = res.sum(-1) / res.shape[-1]
    # Normalize to [0, 1] then convert to metric depth
    ori = (res - res.min()) / (res.max() - res.min() + 1e-8)
    depths = torch.from_numpy(ori.copy()).float()
    depths *= 3900  # empirical scale compatible with depth_pro output
    depths[depths < 1e-5] = 1e-5
    depths = 10000.0 / depths
    depths = depths.clamp(near, far)
    return depths.numpy()  # [T, H, W]


# ── Main ──────────────────────────────────────────────────────────────────────

def _save_and_report(
    frames: np.ndarray,
    depths: np.ndarray,
    fps: float,
    args: argparse.Namespace,
    orig_H: int,
    orig_W: int,
    H: int,
    W: int,
) -> None:
    """Resize if needed, save .npz, write visualization video."""
    T = depths.shape[0]

    if (H, W) != (orig_H, orig_W):
        depths = np.stack([
            cv2.resize(d, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
            for d in depths
        ])

    valid_depths = depths[depths > 0]
    if valid_depths.size:
        print(f"[depth] Depth range: {valid_depths.min():.3f} – {valid_depths.max():.3f} m")
    else:
        print("[depth] Depth range: no valid depths")

    out = args.output
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, depths=depths)
    print(f"[depth] Saved {T}×{orig_H}×{orig_W} depths → {out}")

    # Resize frames for visualization if needed
    vis_frames = frames
    if frames.shape[1:3] != (orig_H, orig_W):
        vis_frames = np.stack([
            cv2.resize(f, (orig_W, orig_H)) for f in frames
        ])
    _save_depth_video(vis_frames, depths, fps, args.video, out)
    print()
    print("Launch the visualizer:")
    print(f"  python -m visualizer.app --video {args.video} --depth {out}")


def run(args: argparse.Namespace) -> None:
    method = getattr(args, "method", "depthcrafter")
    device = args.device

    # ── 1. Load video ─────────────────────────────────────────────────────────
    print(f"[depth] Loading video: {args.video}")
    frames, H, W, orig_H, orig_W, fps = _load_video(args.video, args.max_frames, args.max_res)
    T = len(frames)
    print(f"[depth] {T} frames  {H}×{W}  fps={fps:.1f}")
    if (H, W) != (orig_H, orig_W):
        print(f"[depth] Inference at {H}×{W} (native {orig_H}×{orig_W}); depths will be resized back")

    if method == "depthcrafter":
        dc_dir = getattr(args, "dc_dir", None)
        if not dc_dir:
            raise ValueError("--dc-dir is required for depthcrafter method")

        pipe = _load_depthcrafter(dc_dir, device)

        window_size = min(args.window_size, T)
        overlap = args.overlap if window_size < T else 0

        print(f"[depth] Running DepthCrafter on {T} frames …")
        depths = _run_depthcrafter(
            pipe,
            frames,
            steps=args.steps,
            guidance_scale=1.0,
            window_size=window_size,
            overlap=overlap,
        )

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        _save_and_report(frames, depths, fps, args, orig_H, orig_W, H, W)

    elif method == "gc_moge":
        cache_dir = args.cache_dir or str(Path(args.gc_dir) / "workspace" / "cache")

        # ── 2. Load MoGe (shared between GC prior and metric depth) ──────────
        moge = _load_moge(args.gc_dir, cache_dir, device)

        # ── 3. Load GeometryCrafter ──────────────────────────────────────────
        pipe, point_map_vae = _load_gc_pipeline(args.gc_dir, cache_dir, device)

        # ── 4. GeometryCrafter → relative point maps ─────────────────────────
        print(f"[depth] Running GeometryCrafter on {T} frames …")
        frames_tensor = (
            torch.from_numpy(frames).float().permute(0, 3, 1, 2).to(device)
        )  # [T, 3, H, W]

        window_size = min(args.window_size, T)
        overlap     = args.overlap if window_size < T else 0

        with torch.inference_mode():
            point_maps, valid_masks = pipe(
                frames_tensor,
                point_map_vae,
                moge,
                height=H,
                width=W,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                window_size=window_size,
                decode_chunk_size=8,
                overlap=overlap,
                force_projection=True,
                force_fixed_focal=True,
            )

        gc_depth_z    = point_maps[:, :, :, 2].cpu().float()
        gc_valid_mask = valid_masks.cpu().bool()
        gc_valid_mask = gc_valid_mask & torch.isfinite(gc_depth_z) & (gc_depth_z > 1e-4)

        gc_inv_depth = torch.zeros_like(gc_depth_z)
        gc_inv_depth[gc_valid_mask] = 1.0 / gc_depth_z[gc_valid_mask]

        del point_maps, valid_masks, frames_tensor, pipe, point_map_vae
        gc.collect()
        torch.cuda.empty_cache()

        # ── 5. MoGe metric depth on frame 0 ──────────────────────────────────
        print("[depth] Running MoGe on frame 0 for metric scale anchor …")
        frame0 = (
            torch.from_numpy(frames[0]).float()
            .permute(2, 0, 1).unsqueeze(0)
            .to(device)
        )

        moge_kwargs: dict = {}
        if args.focal is not None:
            fov_x_deg = float(np.degrees(2.0 * np.arctan(W / (2.0 * args.focal))))
            moge_kwargs["fov_x"] = fov_x_deg
            print(f"[depth] focal={args.focal:.1f}px → fov_x={fov_x_deg:.1f}°")

        moge_out    = moge.infer(frame0, **moge_kwargs)
        moge_depth0 = moge_out["depth"].squeeze(0).cpu().float()
        moge_mask0  = moge_out["mask"].squeeze(0).cpu().bool()

        del moge, frame0
        gc.collect()
        torch.cuda.empty_cache()

        # ── 6. Align GC inv-depth to MoGe metric depth ──────────────────────
        print("[depth] Aligning video depth to MoGe metric scale …")
        align_mask = gc_valid_mask[0] & moge_mask0
        _, scale = _align_inv_depth_to_depth(gc_inv_depth[0], moge_depth0, align_mask)
        print(f"[depth] Alignment: scale={scale:.5f}")

        metric_depths: list[np.ndarray] = []
        for i in range(T):
            depth_i = torch.zeros_like(gc_inv_depth[i])
            valid_i = gc_valid_mask[i]
            if valid_i.any():
                aligned_inv = (gc_inv_depth[i][valid_i] * scale).clamp(min=1e-3)
                depth_i[valid_i] = aligned_inv.reciprocal()
            metric_depths.append(depth_i.numpy())

        depths = np.stack(metric_depths, axis=0).astype(np.float32)

        _save_and_report(frames, depths, fps, args, orig_H, orig_W, H, W)

    else:
        raise ValueError(f"Unknown depth method: {method}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Video metric depth estimation (DepthCrafter or GeometryCrafter+MoGe).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",       required=True,
                   help="Input video path")
    p.add_argument("--output",      default="depths.npz",
                   help="Output .npz path")
    p.add_argument("--method",      default="depthcrafter",
                   choices=["depthcrafter", "gc_moge"],
                   help="Depth estimation method")
    _default_dc_dir = str(Path(__file__).resolve().parent / ".deps" / "DepthCrafter")
    p.add_argument("--dc-dir",      dest="dc_dir",
                   default=_default_dc_dir,
                   help="DepthCrafter cloned repo root (for depthcrafter method)")
    _default_gc_dir = str(Path(__file__).resolve().parent / ".deps" / "GeometryCrafter")
    p.add_argument("--gc-dir",      dest="gc_dir",
                   default=_default_gc_dir,
                   help="GeometryCrafter cloned repo root (for gc_moge method)")
    p.add_argument("--cache-dir",   dest="cache_dir", default=None,
                   help="Model weight cache (default: <gc-dir>/workspace/cache)")
    p.add_argument("--max-res",     dest="max_res", default=1024, type=int,
                   help="Longest-side cap before inference")
    p.add_argument("--max-frames",  dest="max_frames", default=-1, type=int,
                   help="Max frames to process (-1 = all)")
    p.add_argument("--steps",       default=5, type=int,
                   help="Diffusion denoising steps")
    p.add_argument("--window-size", dest="window_size", default=110, type=int,
                   help="Sliding window size in frames")
    p.add_argument("--overlap",     default=25, type=int,
                   help="Sliding window overlap in frames")
    p.add_argument("--focal",       default=None, type=float,
                   help="Camera focal length in pixels (gc_moge: MoGe auto-estimates if omitted)")
    p.add_argument("--device",      default="cuda",
                   help="PyTorch device")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    if not Path(args.video).exists():
        print(f"[error] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    run(args)
