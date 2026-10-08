"""Device selection, model loading, seeding and environment reporting."""

from __future__ import annotations

import os
import platform
import random
from typing import Optional, Tuple

import torch

DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU."""
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name not in DTYPES:
        raise ValueError(f"unknown dtype '{name}', choose from {sorted(DTYPES)}")
    dtype = DTYPES[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 matmuls are slow or unsupported on CPU; use float32 or bfloat16")
    if device.type == "mps" and dtype == torch.bfloat16 and not _mps_supports_bf16():
        raise ValueError("this MPS backend does not support bfloat16; use float16 or float32")
    return dtype


def _mps_supports_bf16() -> bool:
    try:
        torch.zeros(1, dtype=torch.bfloat16, device="mps")
        return True
    except Exception:  # noqa: BLE001 - backend-specific failure types
        return False


def synchronize(device: torch.device) -> None:
    """Wait for queued kernels so wall-clock timings are meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _transformers_version() -> Tuple[int, int]:
    import transformers

    major, minor = transformers.__version__.split(".")[:2]
    return int(major), int("".join(c for c in minor if c.isdigit()) or 0)


def load_model_and_tokenizer(name_or_path: str, device: torch.device, dtype: torch.dtype):
    """Load a causal LM + tokenizer from the Hub or a local directory, in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_kw = "dtype" if _transformers_version() >= (4, 56) else "torch_dtype"
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **{dtype_kw: dtype})
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer


def environment_info(device: Optional[torch.device] = None) -> dict:
    import transformers

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu_threads": torch.get_num_threads(),
        "hf_offline": os.environ.get("HF_HUB_OFFLINE", "0"),
    }
    if device is not None:
        info["device"] = str(device)
        if device.type == "cuda":
            info["gpu"] = torch.cuda.get_device_name(device)
            info["cuda"] = torch.version.cuda
    return info
