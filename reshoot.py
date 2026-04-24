#!/usr/bin/env python3
# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""
reshoot.py — Unified CLI for the Reshoot-Anything pipeline.

Subcommands:
    reshoot.py depth      Estimate metric depth for a video
    reshoot.py visualize  Launch the 4D camera trajectory visualizer
    reshoot.py render     Render anchor condition pack from a trajectory
    reshoot.py infer      Launch WAN 2.2 reshoot inference
    reshoot.py validate   Validate a rendered condition pack
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cmd_depth(args: list[str]) -> int:
    return subprocess.call(
        [sys.executable, str(ROOT / "estimate_depth.py")] + args,
        cwd=str(ROOT),
    )


def cmd_visualize(args: list[str]) -> int:
    return subprocess.call(
        [sys.executable, "-m", "visualizer.app"] + args,
        cwd=str(ROOT),
    )


def cmd_render(args: list[str]) -> int:
    return subprocess.call(
        [sys.executable, str(ROOT / "render_from_cam_info.py")] + args,
        cwd=str(ROOT),
    )


def cmd_infer(args: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--condition-pack", required=True)
    parser.add_argument("--caption", default="")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--save-folder", default="outputs")
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--high-noise-lora", default=None)
    parser.add_argument("--low-noise-lora", default=None)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--sample-steps", type=int, default=40)
    known, extra = parser.parse_known_args(args)

    pack = Path(known.condition_pack)
    if not pack.exists():
        print(f"Error: condition pack dir does not exist: {pack}")
        return 1

    from pipeline_spec import validate_condition_pack
    issues = validate_condition_pack(str(pack))
    if any("Missing" in i for i in issues):
        print(f"Condition pack incomplete:")
        for i in issues:
            print(f"  - {i}")
        return 1
    for i in issues:
        print(f"[warning] {i}")

    n_gpus = known.num_gpus
    cmd = [
        "torchrun",
        "--master-port=29501",
        f"--nproc_per_node={n_gpus}",
        str(ROOT / "inference_wan22_v2v_local.py"),
        "--task=i2v-A14B",
        f"--size={known.size}",
        "--dit_fsdp", "--t5_fsdp",
        f"--ulysses_size={n_gpus}",
        f"--sample_steps={known.sample_steps}",
        f"--save_folder={known.save_folder}",
        "--sample_solver=unipc", "--sample_shift=5",
        "--lora_alpha=512", "--lora_rank=512",
        f"--base_folder={pack}",
        "--video_path=render.mp4",
        "--mask_path=render_mask.mp4",
        "--ref_path=input.mp4",
        "--mask_pink_path=render_pink.mp4",
        f"--caption={known.caption}",
        "--video_id=output.mp4",
    ]
    if known.ckpt_dir:
        cmd.append(f"--ckpt_dir={known.ckpt_dir}")
    if known.high_noise_lora:
        cmd.append(f"--high_noise_lora_weights={known.high_noise_lora}")
    if known.low_noise_lora:
        cmd.append(f"--low_noise_lora_weights={known.low_noise_lora}")
    cmd.extend(extra)

    print(f"[reshoot] Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def cmd_validate(args: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition_pack", help="Path to condition pack directory")
    parsed = parser.parse_args(args)

    from pipeline_spec import validate_condition_pack
    issues = validate_condition_pack(parsed.condition_pack)
    if not issues:
        print("Condition pack OK.")
        return 0
    for i in issues:
        print(f"  - {i}")
    return 1


COMMANDS = {
    "depth": ("Estimate metric depth for a video", cmd_depth),
    "visualize": ("Launch the 4D camera trajectory visualizer", cmd_visualize),
    "render": ("Render anchor condition pack from a trajectory", cmd_render),
    "infer": ("Launch WAN 2.2 reshoot inference", cmd_infer),
    "validate": ("Validate a rendered condition pack", cmd_validate),
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: reshoot.py <command> [args...]\n")
        print("Reshoot-Anything unified CLI\n")
        print("Commands:")
        for name, (desc, _) in COMMANDS.items():
            print(f"  {name:12s} {desc}")
        print(f"\nRun 'reshoot.py <command> --help' for command-specific help.")
        return 0

    cmd_name = sys.argv[1]
    if cmd_name not in COMMANDS:
        print(f"Unknown command: {cmd_name}")
        print(f"Available: {', '.join(COMMANDS)}")
        return 1

    _, handler = COMMANDS[cmd_name]
    return handler(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
