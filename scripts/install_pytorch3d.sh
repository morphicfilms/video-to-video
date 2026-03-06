#!/usr/bin/env bash
set -euo pipefail

# Install PyTorch3D into the current uv-managed environment.
# Usage:
#   ./scripts/install_pytorch3d.sh /path/to/Uni3C/pytorch3d   # local source checkout (preferred)
#   ./scripts/install_pytorch3d.sh                            # upstream source tag

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed or not on PATH." >&2
    exit 1
fi

if [[ ! -f "pyproject.toml" ]]; then
    echo "Error: run this from the repository root (pyproject.toml not found)." >&2
    exit 1
fi

if [[ "${VIRTUAL_ENV:-}" == "" ]]; then
    echo "Warning: no active virtual environment detected." >&2
    echo "Activate the repo uv environment first (e.g. 'source .venv/bin/activate')." >&2
fi

if [[ $# -ge 1 && -n "${1:-}" ]]; then
    SRC_PATH="$1"
    if [[ ! -d "$SRC_PATH" ]]; then
        echo "Error: PyTorch3D source path not found: $SRC_PATH" >&2
        exit 1
    fi
    echo "Installing PyTorch3D from local source: $SRC_PATH"
    uv pip install --no-build-isolation -e "$SRC_PATH"
else
    echo "Installing PyTorch3D from upstream source tag v0.7.9"
    uv pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
fi

echo "PyTorch3D install step completed."
