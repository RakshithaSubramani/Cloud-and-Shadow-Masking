from __future__ import annotations

import platform
import sys

import torch


def env_check(device: str = "cuda") -> None:
    print(f"Python exe: {sys.executable}")
    print(f"Python platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
