"""Offline parity checks against the Hugging Face reference implementation.

Builds tiny randomly initialised models (no download needed) and verifies that
the adapters in :mod:`graphkv.models` reproduce the Hugging Face logits, both
for a single full forward pass and for token-by-token decoding.
"""

from __future__ import annotations

import torch

from .models import make_adapter

TINY_CONFIGS = {
    "gpt2": dict(n_layer=2, n_head=4, n_embd=64, n_positions=128, vocab_size=97, bos_token_id=0, eos_token_id=0),
    "llama": dict(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=97,
        max_position_embeddings=256,
    ),
    "mistral": dict(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=97,
        max_position_embeddings=256,
        sliding_window=None,
    ),
    "qwen2": dict(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=97,
        max_position_embeddings=256,
    ),
}


def tiny_model(model_type: str, seed: int = 0, **overrides) -> torch.nn.Module:
    """A small random model of the given architecture, in float32 eval mode.

    Uses Hugging Face's eager attention, the reference implementation (some
    older SDPA code paths ignore rarely used options such as GPT-2's
    ``scale_attn_by_inverse_layer_idx``).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs = dict(TINY_CONFIGS[model_type])
    kwargs.update(overrides)
    config = AutoConfig.for_model(model_type, **kwargs)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    return model.float().eval()


@torch.inference_mode()
def parity_error(model: torch.nn.Module, seq_len: int = 24, prefill: int = 9, seed: int = 0) -> dict:
    """Max absolute logit difference between the adapter and Hugging Face.

    Returns errors for (a) one full forward pass and (b) prefilling ``prefill``
    tokens then decoding the rest one token at a time.
    """
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (1, seq_len), generator=gen)
    reference = model(input_ids=ids).logits.float()
    adapter = make_adapter(model)

    full = adapter(ids, adapter.new_cache()).float()

    cache = adapter.new_cache()
    steps = [adapter(ids[:, :prefill], cache)]
    for t in range(prefill, seq_len):
        steps.append(adapter(ids[:, t : t + 1], cache))
    incremental = torch.cat(steps, dim=1).float()

    return {
        "full": float((full - reference).abs().max()),
        "incremental": float((incremental - reference).abs().max()),
        "scale": float(reference.abs().max()),
    }


def run_all(atol: float = 1e-4) -> bool:
    ok = True
    for model_type in TINY_CONFIGS:
        try:
            err = parity_error(tiny_model(model_type))
        except Exception as exc:  # noqa: BLE001 - report every architecture
            print(f"  {model_type:8s} ERROR: {exc}")
            ok = False
            continue
        passed = err["full"] < atol and err["incremental"] < atol
        ok &= passed
        print(f"  {model_type:8s} max|dlogit| full={err['full']:.2e} incremental={err['incremental']:.2e} -> {'OK' if passed else 'FAIL'}")
    return ok
