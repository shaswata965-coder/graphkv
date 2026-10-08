"""Check that this machine can run the experiments.

Prints library versions and the selected device, then runs offline parity
checks (tiny random models, nothing is downloaded) that compare the
compression-aware forward pass with Hugging Face's own implementation on the
selected device. Add ``--online`` to also check Hub and dataset access.

    python -m graphkv.envcheck
    python -m graphkv.envcheck --device mps --online
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import List, Optional

import torch

from .cache import GraphKVConfig
from .engine import GraphKVEngine
from .models import make_adapter
from .selftest import TINY_CONFIGS, parity_error, tiny_model
from .utils import environment_info, resolve_device


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--online", action="store_true", help="also check Hugging Face Hub / dataset access")
    args = p.parse_args(argv)
    warnings.filterwarnings("ignore")

    device = resolve_device(args.device)
    print("environment:")
    for k, v in environment_info(device).items():
        print(f"  {k:15s} {v}")

    ok = True
    print(f"\nparity with Hugging Face on {device} (float32, tiny random models):")
    for model_type in TINY_CONFIGS:
        try:
            model = tiny_model(model_type).to(device)
            err = parity_error(model)
            passed = err["full"] < 1e-3 and err["incremental"] < 1e-3
            print(f"  {model_type:8s} full={err['full']:.1e} incremental={err['incremental']:.1e}  {'OK' if passed else 'FAIL'}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            passed = False
            print(f"  {model_type:8s} ERROR {type(exc).__name__}: {exc}")
        ok &= passed

    print("\ncompressed decoding smoke test:")
    try:
        model = tiny_model("llama").to(device)
        ids = torch.randint(0, 97, (64,), generator=torch.Generator().manual_seed(0)).tolist()
        for granularity in ("head", "token"):
            res = GraphKVEngine(make_adapter(model), GraphKVConfig(epsilon=0.6, interval=8, granularity=granularity)).generate(ids, 40)
            print(f"  {granularity:5s} 40 tokens, {len(res.events)} compressions, final saving {100 * res.final_compression:.1f}%  OK")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  ERROR {type(exc).__name__}: {exc}")

    if args.online:
        print("\nonline access:")
        try:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained("gpt2")
            print("  Hugging Face Hub (gpt2 tokenizer)  OK")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  Hub ERROR: {exc}")
        try:
            from .data import load_prompts

            print(f"  WikiText-2 test split  OK ({len(load_prompts(num_prompts=100))} prompts)")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  WikiText ERROR: {exc}")

    print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
