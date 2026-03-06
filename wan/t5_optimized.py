# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Optimized T5 loading with various techniques

import imp
import logging
import os
import time
from typing import Optional, Dict, Any
import gc

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from .modules.t5 import T5EncoderModel, HuggingfaceTokenizer, umt5_xxl


class OptimizedT5Loader:
    """Optimized T5 loader with multiple loading strategies"""

    @staticmethod
    def convert_to_safetensors(
        checkpoint_path: str, output_path: Optional[str] = None
    ) -> str:
        """Convert a PyTorch checkpoint to SafeTensors format for faster loading"""
        if output_path is None:
            output_path = checkpoint_path.replace(".pth", ".safetensors")
            output_path = output_path.replace(".pt", ".safetensors")

        if os.path.exists(output_path):
            logging.info(f"SafeTensors checkpoint already exists: {output_path}")
            return output_path

        logging.info(
            f"Converting checkpoint to SafeTensors: {checkpoint_path} -> {output_path}"
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True)

        # Convert any non-contiguous tensors
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                state_dict[k] = v.contiguous()

        save_file(state_dict, output_path)
        logging.info(f"Saved SafeTensors checkpoint: {output_path}")

        # Clean up memory
        del state_dict
        gc.collect()

        return output_path

    @staticmethod
    def load_t5_with_empty_init(
        config, checkpoint_dir, shard_fn=None, device="cuda", use_safetensors=True
    ):
        """Load T5 using empty weight initialization for faster loading"""

        checkpoint_path = os.path.join(checkpoint_dir, config.t5_checkpoint)

        # Convert to safetensors if requested and not already converted
        if use_safetensors and not checkpoint_path.endswith(".safetensors"):
            safetensors_path = checkpoint_path.replace(".pth", ".safetensors").replace(
                ".pt", ".safetensors"
            )
            if os.path.exists(safetensors_path):
                checkpoint_path = safetensors_path
                logging.info(
                    f"Using existing SafeTensors checkpoint: {checkpoint_path}"
                )
            else:
                try:
                    checkpoint_path = OptimizedT5Loader.convert_to_safetensors(
                        checkpoint_path
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to convert to SafeTensors: {e}. Using original checkpoint."
                    )

        start_time = time.time()

        # Initialize model with empty weights (no memory allocation)
        with init_empty_weights():
            model = umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=config.t5_dtype,
                device="meta",  # Meta device - no memory allocation
            )

        # Load and dispatch weights directly to target device
        if checkpoint_path.endswith(".safetensors"):
            # For safetensors, we can load directly
            state_dict = load_file(checkpoint_path, device=str(device))
            model = model.to_empty(device=device)
            model.load_state_dict(state_dict, assign=True)
        else:
            # For regular checkpoints, use accelerate's optimized loading
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint_path,
                device_map={"": device},
                no_split_module_classes=["T5Block"],
            )

        model = model.eval().requires_grad_(False)
        start_time = time.time()
        # Apply FSDP if needed
        if shard_fn is not None:
            model = shard_fn(model, sync_module_states=False)
        print("sharding t5 time: ", time.time() - start_time)
        # Create T5EncoderModel wrapper
        text_encoder = type(
            "T5EncoderModel",
            (),
            {
                "model": model,
                "tokenizer": HuggingfaceTokenizer(
                    name=os.path.join(checkpoint_dir, config.t5_tokenizer),
                    seq_len=config.text_len,
                    clean="whitespace",
                ),
                "device": device,
                "__call__": lambda self, texts, device: self._encode(texts, device),
            },
        )()

        # Add the encode method
        def _encode(self, texts, device):
            ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            context = self.model(ids, mask)
            return [u[:v] for u, v in zip(context, seq_lens)]

        text_encoder._encode = _encode.__get__(text_encoder)

        loading_time = time.time() - start_time
        logging.info(f"T5 loaded with empty_init in {loading_time:.2f}s")

        return text_encoder

    @staticmethod
    def load_t5_direct_gpu(config, checkpoint_dir, shard_fn=None, device="cuda"):
        """Load T5 directly to GPU to avoid CPU->GPU transfer"""

        checkpoint_path = os.path.join(checkpoint_dir, config.t5_checkpoint)
        start_time = time.time()

        # Create model directly on target device
        model = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=config.t5_dtype,
                device=device,
            )
            .eval()
            .requires_grad_(False)
        )

        # Load checkpoint with direct GPU mapping
        logging.info(f"Loading T5 directly to {device}: {checkpoint_path}")

        if checkpoint_path.endswith(".safetensors"):
            state_dict = load_file(checkpoint_path, device=str(device))
        else:
            # Load with map_location directly to GPU
            state_dict = torch.load(checkpoint_path, map_location=device, mmap=True)

        model.load_state_dict(state_dict, assign=True)
        del state_dict
        gc.collect()

        if shard_fn is not None:
            model = shard_fn(model, sync_module_states=False)

        # Create wrapper
        text_encoder = type(
            "T5EncoderModel",
            (),
            {
                "model": model,
                "tokenizer": HuggingfaceTokenizer(
                    name=os.path.join(checkpoint_dir, config.t5_tokenizer),
                    seq_len=config.text_len,
                    clean="whitespace",
                ),
                "device": device,
                "__call__": lambda self, texts, device: self._encode(texts, device),
            },
        )()

        def _encode(self, texts, device):
            ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            context = self.model(ids, mask)
            return [u[:v] for u, v in zip(context, seq_lens)]

        text_encoder._encode = _encode.__get__(text_encoder)

        loading_time = time.time() - start_time
        logging.info(f"T5 loaded directly to GPU in {loading_time:.2f}s")

        return text_encoder


class T5FastLoader:
    """Main interface for fast T5 loading"""

    @staticmethod
    def load(
        config,
        checkpoint_dir,
        shard_fn=None,
        device="cuda",
        method="empty_init",  # 'empty_init', 'direct_gpu', or 'standard'
        use_safetensors=True,
    ):
        """
        Load T5 with specified optimization method

        Args:
            method: Loading method to use
                - 'empty_init': Initialize with empty weights then load (recommended)
                - 'direct_gpu': Load directly to GPU
                - 'standard': Standard loading (fallback)
        """

        if method == "empty_init":
            try:
                return OptimizedT5Loader.load_t5_with_empty_init(
                    config, checkpoint_dir, shard_fn, device, use_safetensors
                )
            except Exception as e:
                logging.warning(
                    f"Empty init loading failed: {e}. Falling back to direct GPU."
                )
                method = "direct_gpu"

        if method == "direct_gpu":
            try:
                return OptimizedT5Loader.load_t5_direct_gpu(
                    config, checkpoint_dir, shard_fn, device
                )
            except Exception as e:
                logging.warning(
                    f"Direct GPU loading failed: {e}. Falling back to standard."
                )
                method = "standard"

        # Fallback to standard loading
        logging.info("Using standard T5 loading")
        return T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn,
        )
