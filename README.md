# Reshoot-Anything

> *Reshoot-Anything: A Self-Supervised Model for In-the-Wild Video Reshooting*  
> [Avinash Paliwal](http://avinashpaliwal.com/),
> [Adithya Iyer](https://adithyaiyer1999.github.io/),
> Shivin Yadav,
> Muhammad Ali Afridi,
> Midhun Harikumar
> *Morphic Inc.*

[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2604.21776)
[![Project Page](https://img.shields.io/badge/Reshoot--Anything-Website-blue?logo=googlechrome&logoColor=blue)](https://adithyaiyer1999.github.io/reshoot-anything/)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://arxiv.org/abs/2604.21776)

<p align="center">
  <img src="assets/banner.png?raw=true" alt="Reshoot-Anything teaser" width="100%">
</p>

### Example Result

<table>
<tr>
<td align="center"><b>Source Video</b></td>
<td align="center"><b>Reshot Video</b></td>
</tr>
<tr>
<td><img src="assets/woman_og.gif" width="400" alt="Source video"></td>
<td><img src="assets/woman_01.gif" width="400" alt="Reshot video"></td>
</tr>
</table>

## Overview

Reshoot-Anything reshoots dynamic monocular videos under novel camera trajectories.
We train a self-supervised video diffusion model on pseudo multi-view triplets
extracted from monocular footage, forcing it to learn implicit 4D spatiotemporal
priors. At inference, a 4D point-cloud anchor derived from the source video
provides geometric conditioning for the new trajectory.

This repo hosts the inference pipeline (training code and pretrained weights
to follow). It is four stages:

- **`estimate_depth.py`** — Per-frame metric depth (GeometryCrafter + MoGe v2, or DepthCrafter).
- **`visualizer/app.py`** (via `run_visualizer.sh`) — 4D point-cloud visualizer for placing
  camera keyframes and exporting a trajectory as `cam_info.json`.
- **`render_from_cam_info.py`** — Forward-warps the source video along the trajectory
  into an anchor condition pack (render, hole mask, reference frame).
- **`inference_wan22_v2v_local.py`** (via `run_wan22_inference.sh`) — WAN 2.2 fine-tune
  that synthesizes the reshot video from source + anchor.

### Implementation Status

- [x] Metric depth estimation (GeometryCrafter + MoGe v2, DepthCrafter)
- [x] 4D camera trajectory visualizer (Viser)
- [x] Novel-view point-cloud rendering (Uni3C PyTorch3D backend + NumPy fallback)
- [x] Reshoot-Anything inference pipeline (WAN 2.2 LoRA fine-tune)
- [ ] Pretrained LoRA weights release
- [ ] Self-supervised training pipeline release

## Setup

Requirements: **Python 3.10**, **CUDA toolkit** (matching your PyTorch build, e.g. CUDA 12.1
for `torch>=2.4`), and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/morphicfilms/video-to-video.git
cd video-to-video

uv venv --python 3.10
source .venv/bin/activate

./install.sh
```

`install.sh` runs `uv sync`, builds **PyTorch3D** from source (~10–30 min; no
Linux wheels), builds **Flash Attention 2** from source, and clones
**DepthCrafter** and **GeometryCrafter** into `.deps/`. Skip groups with
`--no-depth` / `--no-generation` as needed.

## Usage

### 1. Depth Estimation

Estimate metric depth for the input video.

```bash
python estimate_depth.py \
  --video input.mp4 \
  --output depths.npz \
  --method gc_moge              # or: depthcrafter
```

Output: `depths.npz` (key `depths`, shape `[T,H,W]`, float32).

### 2. Camera Trajectory (4D Visualizer)

Place camera keyframes in the reconstructed 4D point cloud and export a trajectory.

```bash
./run_visualizer.sh --video input.mp4 --depth depths.npz
```

Open `http://localhost:8080` in your browser. For a remote server, tunnel
both ports: `ssh -L 8080:localhost:8080 -L 8081:localhost:8081 user@server`.

**Controls:**
- Orbit / pan / zoom with the mouse
- Scrub through frames with the **Frame** slider
- **Add Camera at Current View** — place a keyframe
- Drag the yellow gizmos to reposition/reorient keyframes
- **Preview Path** — visualise the interpolated trajectory
- **Export cam_info.json** when satisfied

### 3. Render Anchor Conditioning

Forward-warp the source along the exported trajectory into an anchor condition pack.

```bash
python render_from_cam_info.py \
  --video input.mp4 \
  --depth depths.npz \
  --cam-info cam_info.json \
  --output-dir render_outputs/my_scene \
  --backend gpu_point             # or: numpy  (no PyTorch3D required)
```

Outputs a self-contained condition pack:

- `input.mp4` — trimmed / resampled source video
- `render.mp4` — novel-view anchor render on black background
- `render_mask.mp4` — white = hole, black = covered
- `render_pink.mp4` — anchor render on pink background (alternative conditioning)
- `reference.png` — first source frame
- `cam_info.json` — camera params at render resolution

### 4. Reshoot Inference

Run the Reshoot-Anything WAN 2.2 fine-tune on the anchor pack. `run_wan22_inference.sh`
is a reference launcher; edit `VIDEO_PATH`, `MASK_PATH`, `REF_PATH`, `CAPTION`, and
the LoRA paths at the top to point at your condition pack, then:

Download the Wan2.2 I2V base weights:

```bash
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B
```

Download the Reshoot-Anything LoRA weights:

```bash
huggingface-cli download morphic/reshoot-anything --local-dir ./reshoot-anything-weights
```

```bash
HIGH_NOISE_LORA_WEIGHTS=./reshoot-anything-weights/jan06_scaling_80k_ckpt1400.safetensors \
LOW_NOISE_LORA_WEIGHTS=./reshoot-anything-weights/dec23_v2v_lownoise_black_lora_512_ckpt1000.safetensors \
./run_wan22_inference.sh
```

The base WAN 2.2 checkpoint is auto-downloaded from HuggingFace
(`Wan-AI/Wan2.2-I2V-A14B`) if `CKPT_DIR` is unset. Pretrained LoRA weights
release is pending (see Implementation Status above); once published, the
launcher will accept an `hf://<org>/<repo>/<filename>` URI for
`HIGH_NOISE_LORA_WEIGHTS` / `LOW_NOISE_LORA_WEIGHTS` in place of a local path.

The launcher assumes **8 GPUs**. For different GPU counts, edit `NUM_GPUS`,
`CUDA_VISIBLE_DEVICES`, and `--ulysses_size` in the script.

### Optional: Prompt Extension

Both `inference_wan22_v2v_local.py` and `generate.py` support LLM-based prompt
expansion via `--use_prompt_extend`:

- `--prompt_extend_method local_qwen` (default) — ~14 GB VRAM; model auto-downloaded
- `--prompt_extend_method dashscope` — Alibaba DashScope API; requires `DASH_API_KEY`
  (get one at https://dashscope.console.aliyun.com/)

## Acknowledgements

This codebase builds upon several excellent open-source projects:

- **[WAN 2.2](https://github.com/Wan-Video/Wan2.2)** — Alibaba Wan Team's video diffusion model (the base we fine-tune)
- **[Uni3C](https://github.com/ewrfcas/Uni3C)** — unified 3D-enhanced camera control; source of the point-rasterization backend
- **[GeometryCrafter](https://github.com/TencentARC/GeometryCrafter)** — temporally-consistent point-map estimation
- **[DepthCrafter](https://github.com/Tencent/DepthCrafter)** — consistent video depth estimation
- **[MoGe](https://github.com/microsoft/MoGe)** — monocular geometry estimation (metric-depth anchor)
- **[Viser](https://github.com/nerfstudio-project/viser)** — the browser-based 3D viewer used by the visualizer
- **[PyTorch3D](https://github.com/facebookresearch/pytorch3d)** — GPU point rasterization

We thank the authors for making their code publicly available.

## Citation

```bibtex
@inproceedings{paliwal2026reshoot,
    title     = {Reshoot-Anything: A Self-Supervised Model for In-the-Wild Video Reshooting},
    author    = {Paliwal, Avinash and Iyer, Adithya and Yadav, Shivin and Afridi, Muhammad Ali and Harikumar, Midhun},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
    year      = {2026}
}
```
