# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from safetensors.torch import load_file as safetensors_load_file
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
import torchvision

torch.cuda.synchronize()
from tqdm import tqdm
from peft import LoraConfig
from peft import inject_adapter_in_model
from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .t5_optimized import T5FastLoader
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from einops import rearrange


def process_mask_v2v(loaded_tensor):
    mask = 1 - loaded_tensor.bool().float()
    mask = mask[:1]
    lat_h, lat_w = mask.shape[-2] // 8, mask.shape[-1] // 8
    msk = torch.nn.functional.interpolate(mask, size=(lat_h, lat_w), mode="bilinear")
    msk = msk.bool()
    msk = msk.to(torch.bfloat16)
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
    )
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    return msk


def load_lora_weights_from_safetensors(lora_path):
    """
    Load LoRA weights from a local safetensors file or a HuggingFace Hub URI,
    and rename keys from ".weight" to ".default.weight".

    Accepts either:
      - a local filesystem path to a `.safetensors` file, or
      - an `hf://<repo_id>/<filename>` URI (e.g. `hf://morphic/reshoot-anything/lora_high_noise.safetensors`),
        which is resolved via `huggingface_hub.hf_hub_download`.
    """
    if lora_path.startswith("hf://"):
        repo_and_file = lora_path[len("hf://"):]
        repo_id, _, filename = repo_and_file.rpartition("/")
        if not repo_id or not filename:
            raise ValueError(
                f"Invalid HuggingFace LoRA URI: {lora_path!r}. "
                "Expected format: hf://<org>/<repo>/<filename.safetensors>"
            )
        from huggingface_hub import hf_hub_download
        logging.info(f"Downloading LoRA weights from HuggingFace: {repo_id}/{filename}")
        lora_path = hf_hub_download(repo_id=repo_id, filename=filename)
    elif not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights file not found: {lora_path}")

    # Load the entire state_dict
    lora_weights = safetensors_load_file(lora_path)

    # For each key: rename ".weight" with ".default.weight"
    # Save memory by modifying state_dict in-place instead of creating a new dict
    keys_to_modify = [
        k for k in lora_weights if (k.endswith(".weight") and "lora" in k)
    ]
    for k in keys_to_modify:
        v = lora_weights.pop(k)
        new_k = k[: -len(".weight")] + ".default.weight"
        lora_weights[new_k] = v

    # print all weight keys in the lora_weights which have "patch" in the key
    for k in lora_weights:
        if "patch" in k:
            print(f"Key {k} has patch in the key")

    return lora_weights


class WanV2V:
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        dit_weights_high_noise=None,
        dit_weights_low_noise=None,
        low_noise_lora_weights_path=None,
        high_noise_lora_weights_path=None,
        lora_rank=512,
        lora_alpha=512,
    ):
        r"""
        Initialize the WanI2V image-to-video generation pipeline.

        This class sets up all model components required for image-to-video generation, including
        the text encoder (T5), VAE, and DiT-based video transformer. It supports various distributed
        and memory-efficient training/inference strategies such as FSDP, sequence parallelism, and LoRA.

        Args:
            config (EasyDict):
                Configuration object containing model hyperparameters and paths (see config.py).
            checkpoint_dir (str):
                Path to directory containing all model checkpoints and tokenizer files.
            device_id (int, optional, default=0):
                CUDA device index to use for model weights and computation.
            rank (int, optional, default=0):
                Distributed process rank (used for multi-GPU/distributed inference).
            t5_fsdp (bool, optional, default=False):
                If True, enable Fully Sharded Data Parallel (FSDP) for the T5 text encoder.
            dit_fsdp (bool, optional, default=False):
                If True, enable FSDP for the DiT video transformer.
            use_sp (bool, optional, default=False):
                If True, enable sequence parallelism for distributed attention layers.
            t5_cpu (bool, optional, default=False):
                If True, keep the T5 text encoder on CPU (only if t5_fsdp is False).
            init_on_cpu (bool, optional, default=True):
                If True, initialize the DiT model on CPU (only if FSDP/USP are not used).
            convert_model_dtype (bool, optional, default=False):
                If True, convert DiT model parameters to config.param_dtype (only if FSDP is not used).
            dit_weights_high_noise (str, optional, default=None):
                Path to DiT weights for the high-noise model (for interpolation).
            dit_weights_low_noise (str, optional, default=None):
                Path to DiT weights for the low-noise model (for interpolation).
            high_noise_lora_weights_path (str, optional, default=None):
                Path to LoRA weights for the high-noise model (for interpolation).
            low_noise_lora_weights_path (str, optional, default=None):
                Path to LoRA weights for the low-noise model (for interpolation).
            lora_rank (int, optional, default=32):
                LoRA rank parameter (for LoRA finetuning).
            lora_alpha (int, optional, default=16):
                LoRA alpha parameter (for LoRA finetuning).

        Notes:
            - If any of t5_fsdp, dit_fsdp, or use_sp are True, models are initialized directly on GPU.
            - LoRA weights can be loaded for both high-noise and low-noise models if provided.
            - This class is used for both interpolation and I2V tasks (see eval_wan22.py).
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        import time

        t5_start = time.time()
        self.text_encoder = T5FastLoader.load(
            config=config,
            checkpoint_dir=checkpoint_dir,
            shard_fn=shard_fn if t5_fsdp else None,
            device=self.device,
            method="empty_init",  # Use the fastest method
            use_safetensors=True,
        )
        logging.info(f"T5 loaded in {time.time() - t5_start:.2f}s")

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,
        )

        LOW_NOISE_SUBFOLDER = "low_noise_model"
        if (
            dit_weights_low_noise is not None
            and ".safetensors" not in dit_weights_low_noise
        ):
            LOW_NOISE_SUBFOLDER = dit_weights_low_noise
        logging.info(
            f"Creating WanModel from {checkpoint_dir} subfolder {LOW_NOISE_SUBFOLDER}"
        )
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=LOW_NOISE_SUBFOLDER
        )

        if (
            dit_weights_low_noise is not None
            and ".safetensors" in dit_weights_low_noise
        ):
            logging.info(f"Loading DiT weights from {dit_weights_low_noise}")
            self.low_noise_model.load_state_dict(
                state_dict=safetensors_load_file(dit_weights_low_noise),
                strict=True,
            )

        if low_noise_lora_weights_path is not None:
            logging.info(f"Loading LoRA weights from {low_noise_lora_weights_path}")
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights=True,
                target_modules=["k", "q", "v", "o", "ffn.0", "ffn.2"],
            )
            self.low_noise_model = inject_adapter_in_model(
                transformer_lora_config, self.low_noise_model
            )

            lora_weights = load_lora_weights_from_safetensors(
                low_noise_lora_weights_path
            )
            missing_keys, unexpected_keys = self.low_noise_model.load_state_dict(
                state_dict=lora_weights,
                strict=False,
            )
            if unexpected_keys:
                logging.info(
                    f"Unexpected keys: {unexpected_keys[:10]}..."
                )  # Show first 10
            else:
                logging.info(f"Loaded LoRA weights from {low_noise_lora_weights_path}")

        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype,
        )

        HIGH_NOISE_SUBFOLDER = "high_noise_model"
        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=HIGH_NOISE_SUBFOLDER
        )
        if dit_weights_high_noise is not None:
            logging.info(f"Loading DiT weights from {dit_weights_high_noise}")
            self.high_noise_model.load_state_dict(
                state_dict=safetensors_load_file(dit_weights_high_noise),
                strict=True,
            )

        if high_noise_lora_weights_path is not None:
            logging.info(f"Loading LoRA weights from {high_noise_lora_weights_path}")
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights=True,
                target_modules=["k", "q", "v", "o", "ffn.0", "ffn.2"],
            )
            self.high_noise_model = inject_adapter_in_model(
                transformer_lora_config, self.high_noise_model
            )

            lora_weights = load_lora_weights_from_safetensors(
                high_noise_lora_weights_path
            )
            missing_keys, unexpected_keys = self.high_noise_model.load_state_dict(
                state_dict=lora_weights,
                strict=False,
            )
            if unexpected_keys:
                logging.info(
                    f"Unexpected keys: {unexpected_keys[:10]}..."
                )  # Show first 10
            else:
                logging.info(f"Loaded LoRA weights from {high_noise_lora_weights_path}")
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype,
        )
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn, convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn
                )
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = "high_noise_model"
            offload_model_name = "low_noise_model"
        else:
            required_model_name = "low_noise_model"
            offload_model_name = "high_noise_model"
        if offload_model or self.init_on_cpu:
            if (
                next(getattr(self, offload_model_name).parameters()).device.type
                == "cuda"
            ):
                getattr(self, offload_model_name).to("cpu")
            if (
                next(getattr(self, required_model_name).parameters()).device.type
                == "cpu"
            ):
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    def generate(
        self,
        input_prompt,
        anchor_video_path,
        anchor_video_path_pink,
        mask_video_path,
        max_area=720 * 1280,
        # frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        reference_video_path=None,
        use_pink_video_path=False,
        drop_reference_video=False,
    ):
        # preprocess
        guide_scale = (
            (guide_scale, guide_scale)
            if isinstance(guide_scale, float)
            else guide_scale
        )
        torchvision_anchor, _, metadata = torchvision.io.read_video(
            anchor_video_path, output_format="TCHW"
        )
        torchvision_anchor_pink, _, metadata_pink = torchvision.io.read_video(
            anchor_video_path_pink, output_format="TCHW"
        )
        frame_num = torchvision_anchor.shape[0] // 4 * 4 + 1 - 4
        anchor_video = torchvision_anchor[:frame_num]
        anchor_video_pink = torchvision_anchor_pink[:frame_num]
        torchvision_ref, _, metadata = torchvision.io.read_video(
            reference_video_path, output_format="TCHW"
        )
        reference_video = torchvision_ref[:frame_num]

        mask_video = torchvision.io.read_video(mask_video_path, output_format="TCHW")[0]
        mask_video = mask_video[:frame_num]

        F = frame_num
        h, w = torchvision_anchor.shape[-2:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // self.vae_stride[1]
            // self.patch_size[1]
            * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // self.vae_stride[2]
            // self.patch_size[2]
            * self.patch_size[2]
        )
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        # resize the anchor_video and reference_video to the shape
        anchor_video = TF.resize(anchor_video, (h, w))
        anchor_video_pink = TF.resize(anchor_video_pink, (h, w))
        reference_video = TF.resize(reference_video, (h, w))
        mask_video = TF.resize(mask_video, (h, w))
        mask_video = rearrange(mask_video, "t c h w -> c t h w")

        max_seq_len = (
            ((F - 1) // self.vae_stride[0] + 1)
            * lat_h
            * lat_w
            // (self.patch_size[1] * self.patch_size[2])
        )
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )
        if mask_video_path is not None:
            mask_video = mask_video.float() / 127.5 - 1.0
            msk = process_mask_v2v(mask_video).to(self.device)
        else:
            msk = torch.zeros(1, F, lat_h, lat_w, device=self.device)
            msk[:, 0] = 1  # Set first frame to 1

            msk = torch.concat(
                [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                dim=1,
            )
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        anchor_video = rearrange(anchor_video, "t c h w -> c t h w")
        anchor_video_pink = rearrange(anchor_video_pink, "t c h w -> c t h w")
        reference_video = rearrange(reference_video, "t c h w -> c t h w")

        anchor_video = anchor_video.to(torch.uint8)
        anchor_video_pink = anchor_video_pink.to(torch.uint8)
        reference_video = reference_video.to(torch.uint8)
        anchor_video = anchor_video.float() / 127.5 - 1.0
        anchor_video_pink = anchor_video_pink.float() / 127.5 - 1.0
        reference_video = reference_video.float() / 127.5 - 1.0

        y = self.vae.encode(anchor_video[None].to(self.device))[0].to(torch.bfloat16)
        y_pink = self.vae.encode(anchor_video_pink[None].to(self.device))[0].to(
            torch.bfloat16
        )
        y2 = self.vae.encode(reference_video[None].to(self.device))[0].to(
            torch.bfloat16
        )
        y = torch.concat([msk, y])
        y_pink = torch.concat([msk, y_pink])
        y2 = torch.concat([torch.ones_like(msk), y2])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, "no_sync", noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, "no_sync", noop_no_sync)

        # evaluation mode
        with (
            torch.amp.autocast("cuda", dtype=self.param_dtype),
            torch.no_grad(),
            no_sync_low_noise(),
            no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift
                )
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler, device=self.device, sigmas=sampling_sigmas
                )
            elif sample_solver == "euler":
                sample_scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler, device=self.device, sigmas=sampling_sigmas
                )
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_f = {
                "context": [context[0]],
                "seq_len": max_seq_len,
                "y": [y],
                "reference_latent": [y2],
            }
            arg_f_pink = {
                "context": [context[0]],
                "seq_len": max_seq_len,
                "y": [y_pink],
                "reference_latent": [y2],
            }
            arg_f_null = {
                "context": context_null,
                "seq_len": max_seq_len,
                "y": [y],
                "reference_latent": [y2],
            }
            arg_f_null_pink = {
                "context": context_null,
                "seq_len": max_seq_len,
                "y": [y_pink],
                "reference_latent": [y2],
            }

            if offload_model:
                torch.cuda.empty_cache()

            arg = arg_f_pink
            arg_null = arg_f_null_pink

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)
                model = self._prepare_model_for_timestep(t, boundary, offload_model)

                if drop_reference_video and t.item() > boundary:
                    arg["reference_latent"] = None
                    arg_null["reference_latent"] = None

                if (t.item() <= boundary) and (not use_pink_video_path):
                    arg = arg_f
                    arg_null = arg_f_null

                sample_guide_scale = (
                    guide_scale[1] if t.item() >= boundary else guide_scale[0]
                )

                noise_pred_cond = model(latent_model_input, t=timestep, **arg)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
