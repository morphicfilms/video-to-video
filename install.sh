#!/usr/bin/env bash
# install.sh — One-time project setup.
#
# Run this once after cloning to install all dependencies, including PyTorch3D
# (which has no Linux PyPI wheels and must be compiled from source after torch).
#
# Usage:
#   ./install.sh                    # core + render + depth + generation groups
#   ./install.sh --no-depth         # skip depth-estimation dependencies
#   ./install.sh --no-generation    # skip WAN video generation dependencies
#
# Requirements:
#   - uv  (https://docs.astral.sh/uv/getting-started/installation/)
#   - CUDA toolkit matching your PyTorch build (for PyTorch3D / flash-attn compilation)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_DEPTH=0
SKIP_GENERATION=0
for arg in "$@"; do
    [[ "$arg" == "--no-depth" ]] && SKIP_DEPTH=1
    [[ "$arg" == "--no-generation" ]] && SKIP_GENERATION=1
done

echo "=== V2V — installing dependencies ==="
echo ""

# ── Sync all groups in a single call so uv doesn't remove packages ────────────
# Running `uv sync --group render` followed by `uv sync --group depth` would
# cause each sync to uninstall packages belonging only to the other group.
# Combining them into one call installs everything and keeps it all in place.
UV_SYNC_CMD="uv sync --group render"
[[ "$SKIP_DEPTH" -eq 0 ]] && UV_SYNC_CMD="$UV_SYNC_CMD --group depth"
[[ "$SKIP_GENERATION" -eq 0 ]] && UV_SYNC_CMD="$UV_SYNC_CMD --group generation"

echo "▸ Running: $UV_SYNC_CMD"
eval "$UV_SYNC_CMD"
echo ""

# ── setuptools — required by pytorch3d's setup.py (not in venv by default) ───
echo "▸ Installing setuptools into project venv..."
uv pip install --python .venv/bin/python setuptools
echo ""

# ── PyTorch3D — must be built after torch is installed ────────────────────────
# No PyPI wheels exist for Linux; build from source with --no-build-isolation
# so that setup.py can find the torch already installed above.
echo "▸ Building PyTorch3D from source (this can take 10–30 minutes)..."
uv pip install --python .venv/bin/python --no-build-isolation \
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git"
echo ""

# ── Flash Attention 2 — required by WAN generation ───────────────────────────
# Must be compiled from source after torch is installed.
if [[ "$SKIP_GENERATION" -eq 0 ]]; then
    echo "▸ Building Flash Attention 2 from source (this can take a while)..."
    uv pip install --python .venv/bin/python packaging ninja
    # Limit parallel compilation jobs to avoid OOM during build
    MAX_JOBS=4 uv pip install --python .venv/bin/python \
        "flash-attn>=2.7" --no-build-isolation
    echo ""
fi

# ── DepthCrafter — clone repo (default depth estimation method) ──────────────
# DepthCrafter is not pip-installable; we clone it and add it to sys.path at runtime.
if [[ "$SKIP_DEPTH" -eq 0 ]]; then
    DC_DIR="${SCRIPT_DIR}/.deps/DepthCrafter"
    if [[ ! -d "$DC_DIR/.git" ]]; then
        echo "▸ Cloning DepthCrafter into .deps/DepthCrafter ..."
        git clone https://github.com/Tencent/DepthCrafter "$DC_DIR"
    else
        echo "▸ DepthCrafter already cloned — skipping."
    fi
    echo ""
fi

# ── GeometryCrafter — clone repo (alternative gc_moge depth method) ──────────
# GC is not pip-installable; we clone it and add it to sys.path at runtime.
# MoGe (GC's geometry prior) is already installed via the depth group above.
if [[ "$SKIP_DEPTH" -eq 0 ]]; then
    GC_DIR="${SCRIPT_DIR}/.deps/GeometryCrafter"
    if [[ ! -d "$GC_DIR/.git" ]]; then
        echo "▸ Cloning GeometryCrafter into .deps/GeometryCrafter ..."
        git clone --recursive \
            https://github.com/TencentARC/GeometryCrafter "$GC_DIR"
    else
        echo "▸ GeometryCrafter already cloned — skipping."
    fi
    echo ""
fi

echo "=== Setup complete ==="
echo "Run ./run_visualizer.sh to launch the visualizer."
