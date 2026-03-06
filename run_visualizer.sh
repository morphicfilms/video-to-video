#!/usr/bin/env bash
# run_visualizer.sh — V2V 3D Camera Path Visualizer launcher.
#
# Usage:
#   ./run_visualizer.sh
#       No arguments: opens a browser upload page (port 8081) where you select a
#       video; the clip is preprocessed client-side then depth is estimated.
#
#   ./run_visualizer.sh --example
#       Skips the UI and opens the visualizer directly with the cached example.
#
#   ./run_visualizer.sh --video path/to/video.mp4
#       Estimates depth for the given video, then launches the visualizer.
#
#   ./run_visualizer.sh --video path/to/video.mp4 --depth path/to/depths.npz
#       Opens the visualizer directly — no depth estimation needed.
#
# All extra flags are forwarded to the underlying app.
# Remote server: ssh -L 8080:localhost:8080 -L 8081:localhost:8081 user@server
#   then open http://localhost:8081/upload to upload a video (depth is estimated server-side)
#   the browser will redirect automatically to http://localhost:8080 (3D viewer) when ready

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse our own flags before forwarding the rest ───────────────────────────
USE_EXAMPLE=0
HAS_VIDEO=0
HAS_DEPTH=0
PASSTHROUGH=()
for arg in "$@"; do
    case "$arg" in
        --example) USE_EXAMPLE=1 ;;
        --video)   HAS_VIDEO=1; PASSTHROUGH+=("$arg") ;;
        --depth)   HAS_DEPTH=1; PASSTHROUGH+=("$arg") ;;
        *)         PASSTHROUGH+=("$arg") ;;
    esac
done

echo "Starting V2V 3D Camera Path Visualizer ..."
echo "Open http://localhost:8080 in your browser."
echo ""

if [[ $USE_EXAMPLE -eq 1 ]]; then
    # ── --example: open cached video + depth directly ─────────────────────────
    DEFAULT_DEPTH="${SCRIPT_DIR}/.cache/visualizer_depths/input_558a29e90dea_depths.npz"
    DEFAULT_VIDEO="${SCRIPT_DIR}/.cache/visualizer_depths/input_558a29e90dea_input.mp4"
    if [[ ! -f "$DEFAULT_VIDEO" ]]; then
        DEFAULT_VIDEO="${SCRIPT_DIR}/input.mp4"
    fi
    if [[ ! -f "$DEFAULT_VIDEO" ]]; then
        echo "Error: cached example video not found. Run run_visualizer_autodepth to generate it." >&2
        exit 1
    fi
    uv run python -m visualizer.app \
        --video "$DEFAULT_VIDEO" --depth "$DEFAULT_DEPTH" "${PASSTHROUGH[@]}"

elif [[ $HAS_VIDEO -eq 1 && $HAS_DEPTH -eq 1 ]]; then
    # ── Both provided: open visualizer directly ───────────────────────────────
    uv run python -m visualizer.app "${PASSTHROUGH[@]}"

else
    # ── Video only, or no args: go through autodepth (UI launcher or CLI) ─────
    # app_autodepth handles all three sub-cases:
    #   - no --video → opens browser UI for upload / remote-path selection
    #   - --video only → estimates depth then launches visualizer
    uv run --group depth python -m visualizer.app_autodepth "${PASSTHROUGH[@]}"
fi
