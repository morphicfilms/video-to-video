#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
NCCL_DEBUG=INFO

SAVE_FOLDER="ablations/feb26_scaling_80k_ckpt1400"
HIGH_NOISE_LORA_WEIGHTS="/data/adi_temp/MorphicVideo/ckpts/jan06_scaling_80k_ckpt1400.safetensors"
LOW_NOISE_LORA_WEIGHTS="/data/adi_temp/MorphicVideo/ckpts/dec23_v2v_lownoise_black_lora_512_ckpt1000.safetensors"

torchrun --master-port=29501 --nproc_per_node=$NUM_GPUS eval_wan22_v2v_local.py \
    --task=i2v-A14B \
    --size=832*480 \
    --ckpt_dir=/data/adi_temp/MorphicVideo/Wan2.2-I2V-A14B \
    --json_eval_path="/data/adi_temp/MorphicVideo/assets/synthetic_video_eval/new_v2v_eval_pink.json" \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size=8 \
    --sample_steps=40 \
    --save_folder="$SAVE_FOLDER" \
    --high_noise_lora_weights="$HIGH_NOISE_LORA_WEIGHTS" \
    --lora_alpha=512 \
    --lora_rank=512 \
    --sample_solver=unipc \
    --sample_shift=5 \
    --low_noise_lora_weights="$LOW_NOISE_LORA_WEIGHTS" \
