#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

SAVE_FOLDER="outputs"
# LoRA weights: pass either a local path to a `.safetensors` file, or an
# `hf://<org>/<repo>/<filename>` URI to auto-download from HuggingFace, e.g.
#   HIGH_NOISE_LORA_WEIGHTS="hf://morphic/reshoot-anything/lora_high_noise.safetensors"
# When unset, inference runs without any LoRA (base WAN 2.2 only).
# Reshoot-Anything LoRA release is pending — see Implementation Status in README.
# TODO: Update with public HuggingFace URIs once weights are released, e.g.
#   HIGH_NOISE_LORA_WEIGHTS="hf://morphic/reshoot-anything/lora_high_noise.safetensors"
HIGH_NOISE_LORA_WEIGHTS="${HIGH_NOISE_LORA_WEIGHTS:-}"
LOW_NOISE_LORA_WEIGHTS="${LOW_NOISE_LORA_WEIGHTS:-}"

# Base model checkpoint directory.
# If not set, the model will be downloaded from HuggingFace (Wan-AI/Wan2.2-I2V-A14B).
CKPT_DIR="${CKPT_DIR:-}"

# Base folder for relative paths (folder containing the JSON file)
BASE_FOLDER="render_outputs"

# Single test case inputs
VIDEO_PATH="render.mp4"
MASK_PATH="render_mask.mp4"
REF_PATH="input.mp4"
MASK_PINK_PATH="render_pink.mp4"
CAPTION="Cowboys riding on horses on a road in a dry region"
VIDEO_ID="test.mp4"

torchrun --master-port=29501 --nproc_per_node=$NUM_GPUS inference_wan22_v2v_local.py \
    --task=i2v-A14B \
    --size=832*480 \
    ${CKPT_DIR:+--ckpt_dir="$CKPT_DIR"} \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size=8 \
    --sample_steps=40 \
    --save_folder="$SAVE_FOLDER" \
    ${HIGH_NOISE_LORA_WEIGHTS:+--high_noise_lora_weights="$HIGH_NOISE_LORA_WEIGHTS"} \
    --lora_alpha=512 \
    --lora_rank=512 \
    --sample_solver=unipc \
    --sample_shift=5 \
    ${LOW_NOISE_LORA_WEIGHTS:+--low_noise_lora_weights="$LOW_NOISE_LORA_WEIGHTS"} \
    --base_folder="$BASE_FOLDER" \
    --video_path="$VIDEO_PATH" \
    --mask_path="$MASK_PATH" \
    --ref_path="$REF_PATH" \
    --mask_pink_path="$MASK_PINK_PATH" \
    --caption="$CAPTION" \
    --video_id="$VIDEO_ID"
