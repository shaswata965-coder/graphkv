"""Tiny random models for offline tests and the installation self-check."""

from __future__ import annotations

import torch

def _available(model_types):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    return tuple(t for t in model_types if t in CONFIG_MAPPING)


TINY_CONFIGS = _available(("llama", "mistral", "qwen2", "qwen3"))  # Qwen3 needs transformers >= 4.51


def tiny_model(model_type: str = "llama", seed: int = 0, dtype: torch.dtype = torch.float32, **overrides) -> torch.nn.Module:
    """A 2-layer decoder with grouped-query attention and RoPE, random weights, eval mode."""
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    if model_type == "qwen3":
        kwargs["head_dim"] = 16
    if model_type == "mistral":
        kwargs["sliding_window"] = None
    kwargs.update(overrides)
    config = AutoConfig.for_model(model_type, **kwargs)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config)
    return model.to(dtype).eval()
