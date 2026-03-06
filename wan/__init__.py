# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from . import configs, distributed, modules
from .image2video import WanI2V
from .text2video import WanT2V
from .video2video import WanV2V

# Lazy imports for pipelines with heavier dependencies (e.g. librosa)
def __getattr__(name):
    if name == "WanS2V":
        from .speech2video import WanS2V
        return WanS2V
    elif name == "WanTI2V":
        from .textimage2video import WanTI2V
        return WanTI2V
    elif name == "WanAnimate":
        from .animate import WanAnimate
        return WanAnimate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
