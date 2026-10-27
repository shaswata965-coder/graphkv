"""Quick self-check on the actual device (e.g. the Colab GPU) with tiny random models, ~1 minute.

    python -m graphkv_neurips.gpucheck            # uses cuda if available

The unit tests run on CPU; this repeats the key equivalences where the experiments will run:
the decoder against Hugging Face (full pass and token-by-token generation), the Graph-KV
context against one forward pass with the graph-structured block mask, moving cached keys,
and the official top-m and Full layouts against their sequential-reading equivalents.
Every line must say ``ok``.
"""

from __future__ import annotations

import argparse
import sys

import torch

from .graph import GraphPlan, encode_graph
from .methods import Method, Prompt, decode, prepare
from .model import RopeDecoder
from .testing import TINY_CONFIGS, tiny_model


def _ids(n, seed):
    return torch.randint(3, 256, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _hf_keys(model, ids, positions, allow, device):
    mask = torch.where(allow, 0.0, torch.finfo(torch.float32).min)[None, None].to(device)
    with torch.no_grad():
        out = model(torch.as_tensor([ids], device=device), position_ids=torch.as_tensor([positions], device=device), attention_mask=mask, use_cache=True)
    cache = out.past_key_values
    return [cache.layers[l].keys if hasattr(cache, "layers") else cache.key_cache[l] for l in range(model.config.num_hidden_layers)]


def run(device: str, tol: float = 1e-3) -> bool:
    ok = True

    def report(name: str, err: float, unit: str = "max abs difference") -> None:
        nonlocal ok
        passed = err < tol
        ok &= passed
        print(f"{'ok  ' if passed else 'FAIL'} {name}: {unit} {err:.3g}", flush=True)

    for mt in TINY_CONFIGS:
        model = tiny_model(mt).to(device)
        dec = RopeDecoder(model)
        ids = torch.as_tensor([_ids(24, 0)], device=device)
        with torch.no_grad():
            ref = model(ids).logits
            ours, _, _ = dec.forward(ids, torch.arange(24).unsqueeze(0), logits="all")
        report(f"{mt}: logits vs Hugging Face", float((ours - ref).abs().max()))

    model = tiny_model("llama").to(device)
    dec = RopeDecoder(model)
    with torch.no_grad():
        prompt_ids = torch.as_tensor([_ids(10, 1)], device=device)
        hf = model.generate(prompt_ids, max_new_tokens=12, do_sample=False, pad_token_id=0)[0, 10:].tolist()
    out, _, _ = decode(dec, prepare(dec, Prompt([], [], [], prompt_ids[0].tolist()), Method("sequential")), max_new_tokens=12)
    report("greedy decoding vs Hugging Face generate", float(sum(a != b for a, b in zip(out, hf)) + abs(len(out) - len(hf))), "differing tokens")

    d2, d3, d1 = _ids(7, 4), _ids(5, 5), _ids(6, 6)
    allow = torch.zeros(18, 18, dtype=torch.bool)
    allow[:7, :7] = torch.ones(7, 7, dtype=torch.bool).tril()
    allow[7:12, 7:12] = torch.ones(5, 5, dtype=torch.bool).tril()
    allow[12:, :12] = True
    allow[12:, 12:] = torch.ones(6, 6, dtype=torch.bool).tril()
    positions = list(range(7)) + list(range(5)) + list(range(7, 13))
    ref_keys = _hf_keys(model, d2 + d3 + d1, positions, allow, device)
    enc = encode_graph(dec, [], [d2, d3, d1], GraphPlan.from_edges(3, [(0, 2), (1, 2)]))
    report("Graph-KV context vs graph-structured block mask (Fig. 1)", max(float((a - b).abs().max()) for a, b in zip(enc.segment.keys, ref_keys)))

    with torch.no_grad():
        _, at0 = dec.encode(torch.as_tensor([d1]), torch.arange(6).unsqueeze(0))
        _, at50 = dec.encode(torch.as_tensor([d1]), 50 + torch.arange(6).unsqueeze(0))
    moved = dec.move(at0, 50 + torch.arange(6))
    report("moving cached keys to new positions", max(float((a - b).abs().max()) for a, b in zip(moved.keys, at50.keys)))

    # official Full layout: a target sees all chunks laid out in a row, then itself
    chunks, prefix = [_ids(4, 7), _ids(6, 8), _ids(5, 9)], _ids(3, 10)
    full = encode_graph(dec, prefix, chunks, GraphPlan.rag_full(3), layout="sequential")
    n = 15 + 6
    allow = torch.zeros(n, n, dtype=torch.bool)
    cursor = 0
    for c in chunks:
        allow[cursor : cursor + len(c), cursor : cursor + len(c)] = torch.ones(len(c), len(c), dtype=torch.bool).tril()
        cursor += len(c)
    allow[15:, :15] = True
    allow[15:, 15:] = torch.ones(6, 6, dtype=torch.bool).tril()
    ref_keys = _hf_keys(model, chunks[0] + chunks[1] + chunks[2] + chunks[1], list(range(n)), allow, device)
    target = dec.move(full.segment.slice(3 + 15 + 4, 3 + 15 + 10), torch.arange(15, 21))
    report("official Graph-KV-Full target vs reading all chunks then the target", max(float((a - b[:, :, 15:]).abs().max()) for a, b in zip(target.keys, ref_keys)))
    return ok


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    print(f"device: {args.device}" + (f" ({torch.cuda.get_device_name()})" if args.device.startswith("cuda") else ""))
    good = run(args.device)
    print("ALL CHECKS PASSED" if good else "SOME CHECKS FAILED - do not run the paid experiments; report the output")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
