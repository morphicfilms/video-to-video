# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
import json

warnings.filterwarnings("ignore")

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool

EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # Validate that all required inputs are provided
    assert args.video_path is not None, "Please specify the video_path."
    assert args.mask_path is not None, "Please specify the mask_path."
    assert args.ref_path is not None, "Please specify the ref_path."
    assert args.mask_pink_path is not None, "Please specify the mask_pink_path."
    assert args.caption is not None, "Please specify the caption."
    assert args.video_id is not None, "Please specify the video_id."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = (
        args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    )
    # Size check
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a single test case using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.",
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.",
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.",
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=None,
        help="The folder to save the generated video to.",
    )
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.",
    )
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.",
    )
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.",
    )
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.",
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++", "euler"],
        help="The solver used to sample.",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.",
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.",
    )
    # high noise
    parser.add_argument(
        "--high_noise_dit_weights",
        type=str,
        default=None,
        help="The high noise dit weights to use.",
    )
    parser.add_argument(
        "--low_noise_dit_weights",
        type=str,
        default=None,
        help="The low noise dit weights to use.",
    )
    # lora
    parser.add_argument(
        "--high_noise_lora_weights",
        type=str,
        default=None,
        help="The high noise lora weights to use.",
    )
    parser.add_argument(
        "--low_noise_lora_weights",
        type=str,
        default=None,
        help="The low noise lora weights to use.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=512,
        help="The rank of the lora weights to use.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=512,
        help="The alpha of the lora weights to use.",
    )
    parser.add_argument(
        "--use_pink_video_path",
        action="store_true",
        default=False,
        help="Whether to use the pink video path.",
    )
    parser.add_argument(
        "--drop_reference_video",
        action="store_true",
        default=False,
        help="Whether to drop the reference video.",
    )
    # Single test case inputs
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to the anchor video.",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Path to the mask video.",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default=None,
        help="Path to the reference video.",
    )
    parser.add_argument(
        "--mask_pink_path",
        type=str,
        default=None,
        help="Path to the pink mask video.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Caption/prompt for the video generation.",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        default=None,
        help="Video ID for naming the output file.",
    )
    parser.add_argument(
        "--base_folder",
        type=str,
        default=None,
        help="Base folder path to prepend to relative paths (e.g., folder containing the JSON file).",
    )
    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), (
            f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        )
        assert not (args.ulysses_size > 1), (
            f"sequence parallel are not supported in non-distributed environments."
        )

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, (
            f"The number of ulysses_size should be equal to the world size."
        )
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, (
            f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
        )

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # Process paths - prepend base_folder if provided and path is relative
    def resolve_path(path):
        if path is None:
            return None
        if args.base_folder and not os.path.isabs(path):
            return os.path.join(args.base_folder, path)
        return path

    video_path = resolve_path(args.video_path)
    mask_path = resolve_path(args.mask_path)
    ref_path = resolve_path(args.ref_path)
    video_path_pink = resolve_path(args.mask_pink_path)
    caption = args.caption
    video_id = args.video_id

    if "t2v" in args.task:
        raise NotImplementedError("Text to video generation is not supported yet.")
    elif "ti2v" in args.task:
        raise NotImplementedError("Text to image generation is not supported yet.")
    else:
        logging.info("Creating WanI2V pipeline.")

        # set config.low_noise_checkpoint to args.low_noise_dit_weights
        cfg.low_noise_checkpoint = args.low_noise_dit_weights
        wan_v2v = wan.WanV2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            dit_weights_high_noise=args.high_noise_dit_weights,
            dit_weights_low_noise=args.low_noise_dit_weights,
            high_noise_lora_weights_path=args.high_noise_lora_weights,
            low_noise_lora_weights_path=args.low_noise_lora_weights,
            lora_alpha=args.lora_alpha,
            lora_rank=args.lora_rank,
        )

        if rank == 0:
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)

        logging.info(
            f"Generating video for {video_path} and {mask_path} and {ref_path}, caption: {caption}"
        )

        # Use video_id for output filename, or generate a default name
        if video_id:
            # Remove .mp4 extension if present, we'll add it back
            output_name = video_id.replace(".mp4", "")
            video_mp4_path = os.path.join(args.save_folder, f"{output_name}.mp4")
        else:
            video_mp4_path = os.path.join(args.save_folder, "output.mp4")

        if os.path.exists(video_mp4_path):
            logging.info(f"Video already exists: {video_mp4_path}")
            return

        video_euler = wan_v2v.generate(
            input_prompt=caption,
            max_area=MAX_AREA_CONFIGS[args.size],
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            anchor_video_path=video_path,
            anchor_video_path_pink=video_path_pink,
            mask_video_path=mask_path,
            reference_video_path=ref_path,
            use_pink_video_path=args.use_pink_video_path,
            drop_reference_video=args.drop_reference_video,
        )

        if rank == 0:
            save_video(
                tensor=video_euler[None],
                save_file=video_mp4_path,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            logging.info(f"Video saved to: {video_mp4_path}")

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
