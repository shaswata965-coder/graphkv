"""Graph-KV and the baselines of Sec. 4 behind one interface, plus greedy decoding.

Every method receives the same three parts (Appendix A.2): a *prefix*
(instruction), the *text chunks* and the *question*, already tokenized.

=============  ===================================================================
``sequential`` prefix + chunks + question as one causal sequence (the official
               code tokenizes the joined string, ``Prompt.joined``).
``pcw``        Parallel Context Windows: prefix and chunks encoded alone, all
               windows share positions ``[0, L)``; the question starts at the
               longest window's length. Windows are encoded after a BOS token
               whose KV is kept once (``pcw_wrapper.py``).
``ape``        Adaptive Parallel Encoding: the prefix is encoded first and every
               chunk attends to it from the shared range ``[P, P + L)``; question
               and answer use APE's temperature/scale attention (T = S = 0.9).
``block``      Block-Attention / Block-RAG: blocks encoded alone, then placed at
               consecutive positions (``block_generate_server.py``).
``graphkv``    Graph-KV with a :class:`~graphkv_neurips.graph.GraphPlan` and a
               position layout.
=============  ===================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch

from .graph import GraphPlan, encode_graph
from .model import APEConfig, KVBuffer, RopeDecoder, Segment

METHODS = ("sequential", "pcw", "ape", "block", "graphkv")


@dataclass
class Prompt:
    prefix: List[int]
    chunks: List[List[int]]
    query: List[int]
    joined: Optional[List[int]] = None  # whole prompt tokenized at once (sequential baseline)
    order: Optional[List[int]] = None  # chunk indices by ascending relevance (default: chunk order)

    def relevance_order(self) -> List[int]:
        return list(range(len(self.chunks))) if self.order is None else list(self.order)


@dataclass
class Method:
    name: str
    plan: Optional[Callable[[Prompt], GraphPlan]] = None  # graphkv: prompt -> plan
    layout: str = "paper"
    ape_temperature: float = 0.9
    ape_scale: float = 0.9
    bos_id: Optional[int] = None  # pcw: token every window is encoded after
    label: str = ""

    def __post_init__(self) -> None:
        if self.name not in METHODS:
            raise ValueError(f"unknown method '{self.name}' (choose from {', '.join(METHODS)})")
        if self.name == "graphkv" and self.plan is None:
            raise ValueError("graphkv needs a plan")
        if not self.label:
            self.label = self.name


@dataclass
class Prepared:
    """KV context that the question attends to."""

    segment: Optional[Segment]
    query_ids: List[int]
    query_start: int
    ape: Optional[APEConfig] = None
    attention_pairs: int = 0
    position_span: int = 0  # largest position index used by the prompt + 1 (context window consumption)


@dataclass
class Generation:
    token_ids: List[int]
    prepare_s: float  # building the KV context (chunk prefill)
    first_token_s: float  # question prefill up to the first answer token
    decode_s: float
    context_tokens: int
    position_span: int
    attention_pairs: int
    stats: dict = field(default_factory=dict)

    @property
    def ttft_s(self) -> float:
        return self.prepare_s + self.first_token_s


def _causal_pairs(n: int) -> int:
    return n * (n + 1) // 2


@torch.no_grad()
def prepare(decoder: RopeDecoder, prompt: Prompt, method: Method, *, pad_id: int = 0, level0_cache=None) -> Prepared:
    """Build the KV context of ``method``; ``level0_cache`` (chunk -> independent encoding) is re-used by Graph-KV."""
    name = method.name
    if name == "sequential":
        ids = prompt.joined if prompt.joined is not None else prompt.prefix + [t for c in prompt.chunks for t in c] + prompt.query
        return Prepared(None, list(ids), 0, attention_pairs=_causal_pairs(len(ids)), position_span=len(ids))

    if name == "block":
        blocks = [b for b in [prompt.prefix] + prompt.chunks if b]
        starts, cursor = [], 0
        for b in blocks:
            starts.append(cursor)
            cursor += len(b)
        segs = decoder.encode_batch(blocks, starts, pad_id=pad_id)
        pairs = sum(_causal_pairs(len(b)) for b in blocks)
        return Prepared(Segment.cat(segs), list(prompt.query), cursor, attention_pairs=pairs, position_span=cursor + len(prompt.query))

    if name == "pcw":
        windows = [b for b in [prompt.prefix] + prompt.chunks if b]
        bos = method.bos_id
        segs: List[Segment] = []
        keep_bos = True
        longest = 0
        if bos is not None:
            encoded = []
            for w in windows:
                has_bos = len(w) > 0 and w[0] == bos
                encoded.append(list(w) if has_bos else [bos] + list(w))
            raw = decoder.encode_batch(encoded, [0] * len(encoded), pad_id=pad_id)
            for seg in raw:
                # keep one BOS entry in total (pcw_wrapper.combine_past_key_values)
                segs.append(seg if keep_bos else seg.slice(1, len(seg)))
                keep_bos = False
            longest = max(len(w) for w in encoded)
        else:
            segs = decoder.encode_batch(windows, [0] * len(windows), pad_id=pad_id)
            longest = max(len(w) for w in windows)
        pairs = sum(_causal_pairs(len(s)) for s in segs)
        return Prepared(Segment.cat(segs), list(prompt.query), longest, attention_pairs=pairs, position_span=longest + len(prompt.query))

    if name == "ape":
        p = len(prompt.prefix)
        if p == 0:
            raise ValueError("APE needs a non-empty prefix")
        _, prefix = decoder.encode(torch.as_tensor([prompt.prefix]), torch.arange(p).unsqueeze(0))
        segs = decoder.encode_batch(prompt.chunks, [p] * len(prompt.chunks), prefix, pad_id=pad_id)
        ctx = Segment.cat([prefix] + segs)
        mask = torch.zeros(len(ctx), dtype=torch.bool)
        mask[p:] = True
        start = p + max(len(c) for c in prompt.chunks)
        pairs = _causal_pairs(p) + sum(len(c) * p + _causal_pairs(len(c)) for c in prompt.chunks)
        ape = APEConfig(mask, method.ape_temperature, method.ape_scale)
        query = list(prompt.query)
        if method.bos_id is not None and (not query or query[0] != method.bos_id):
            query = [method.bos_id] + query
        return Prepared(ctx, query, start, ape=ape, attention_pairs=pairs, position_span=start + len(query))

    plan = method.plan(prompt)
    cache = level0_cache if method.layout != "sequential" else None
    enc = encode_graph(decoder, prompt.prefix, prompt.chunks, plan, method.layout, pad_id=pad_id, level0_cache=cache)
    return Prepared(enc.segment, list(prompt.query), enc.query_start, attention_pairs=enc.attention_pairs, position_span=enc.query_start + len(prompt.query))


@torch.no_grad()
def decode(
    decoder: RopeDecoder,
    prepared: Prepared,
    *,
    max_new_tokens: int = 256,
    stop_ids: Sequence[int] = (),
    chunk_size: Optional[int] = None,
) -> tuple:
    """Greedy decoding on top of a prepared context. Returns ``(token_ids, first_token_s, decode_s)``."""
    seg = prepared.segment
    q = prepared.query_ids
    cache = KVBuffer(decoder, (len(seg) if seg is not None else 0) + len(q) + max_new_tokens + 1, seg)
    t0 = time.perf_counter()
    step = chunk_size or len(q)
    logits = None
    for start in range(0, len(q), step):
        ids = torch.as_tensor([q[start : start + step]])
        pos = prepared.query_start + start + torch.arange(ids.shape[1]).unsqueeze(0)
        logits, _, _ = decoder.forward(ids, pos, cache=cache, ape=prepared.ape)
    first = int(logits[0].argmax())
    t1 = time.perf_counter()
    out: List[int] = []
    stop = set(int(s) for s in stop_ids)
    nxt, position = first, prepared.query_start + len(q)
    while len(out) < max_new_tokens:
        if nxt in stop:
            break
        out.append(nxt)
        if len(out) == max_new_tokens:
            break
        logits, _, _ = decoder.forward(torch.as_tensor([[nxt]]), torch.as_tensor([[position]]), cache=cache, ape=prepared.ape)
        position += 1
        nxt = int(logits[0].argmax())
    return out, t1 - t0, time.perf_counter() - t1


@torch.no_grad()
def answer_nll(decoder: RopeDecoder, prepared: Prepared, answer_ids: Sequence[int], *, chunk_size: Optional[int] = None) -> float:
    """Teacher-forced negative log-likelihood (nats per token) of ``answer_ids`` after the question.

    One forward pass instead of up to 256 decoding steps: a cheap, deterministic
    proxy for comparing conditions on the same examples (lower is better).
    """
    if not answer_ids:
        raise ValueError("empty answer")
    seg = prepared.segment
    q, a = list(prepared.query_ids), list(answer_ids)
    cache = KVBuffer(decoder, (len(seg) if seg is not None else 0) + len(q) + len(a) + 1, seg)
    ids = q + a
    step = chunk_size or len(ids)
    tail = len(a) + 1  # logits needed for the last question token and every answer token but the last
    out = []
    for start in range(0, len(ids), step):
        piece = ids[start : start + step]
        pos = prepared.query_start + start + torch.arange(len(piece)).unsqueeze(0)
        keep = max(0, min(len(piece), start + len(piece) - (len(ids) - tail)))
        logits, _, _ = decoder.forward(torch.as_tensor([piece]), pos, cache=cache, ape=prepared.ape, logits=keep if keep else "none")
        if keep:
            out.append(logits[0])
    logits = torch.cat(out, dim=0)[:-1].float()  # predictions for a[0], ..., a[-1]
    target = torch.as_tensor(a, device=logits.device)
    return float(torch.nn.functional.cross_entropy(logits, target))


def generate(
    decoder: RopeDecoder,
    prompt: Prompt,
    method: Method,
    *,
    max_new_tokens: int = 256,
    stop_ids: Sequence[int] = (),
    pad_id: int = 0,
) -> Generation:
    t0 = time.perf_counter()
    prepared = prepare(decoder, prompt, method, pad_id=pad_id)
    prepare_s = time.perf_counter() - t0
    tokens, first_s, decode_s = decode(decoder, prepared, max_new_tokens=max_new_tokens, stop_ids=stop_ids)
    return Generation(
        token_ids=tokens,
        prepare_s=prepare_s,
        first_token_s=first_s,
        decode_s=decode_s,
        context_tokens=len(prepared.segment) if prepared.segment is not None else 0,
        position_span=prepared.position_span,
        attention_pairs=prepared.attention_pairs,
    )


def parse_method(spec: str, *, bos_id: Optional[int] = None) -> Method:
    """Parse a method name.

    ``sequential | pcw | ape | block | parallel | graphkv-top<m> | noedge-top<m> |
    graphkv-full | graphkv-chain<a>-<b>... | graphkv-star``, optionally followed by

    * ``@official``: the official RAG variant (``pcw.py``), i.e. official top-m
      targets and sequential source positions;
    * ``@global``: shared ranges at multiples of the longest chunk overall
      (literal Sec. 3.1) instead of after the longest source;
    * ``:targets`` / ``:sources``: which representations the query sees
      (default: both). ``graphkv-top3:targets`` drops the sources after the
      targets were updated; ``noedge-top3:targets`` shows the same chunks at the
      same positions but encoded without edges (the control).
    """
    spec_body, _, view = spec.partition(":")
    base, _, variant = spec_body.partition("@")
    view = view or "all"
    if variant not in ("", "official", "global"):
        raise ValueError(f"unknown variant '{variant}'")
    if view not in ("all", "targets", "sources"):
        raise ValueError(f"unknown view '{view}'")
    if view != "all" and (variant == "official" or not base.startswith(("graphkv-top", "noedge-top", "graphkv-full"))):
        raise ValueError("views are available for graphkv-top<m>, noedge-top<m> and graphkv-full (paper variant)")
    layout = {"official": "sequential", "global": "paper-global"}.get(variant, "paper")
    plan_variant = "official" if variant == "official" else "paper"
    if variant and not base.startswith(("graphkv", "noedge")):
        raise ValueError(f"'@{variant}' applies to Graph-KV methods only")
    if base in ("sequential", "block"):
        return Method(base, label=spec)
    if base == "ape":
        # APE's code tokenizes the question with special tokens, i.e. with a BOS in front (demo_ape.py, pred_parallel9.py)
        return Method("ape", bos_id=bos_id, label=spec)
    if base == "pcw":
        return Method("pcw", bos_id=bos_id, label=spec)
    if base == "parallel":
        return Method("graphkv", plan=lambda p: GraphPlan.parallel(len(p.chunks)), label=spec)
    if base == "graphkv-full":
        return Method("graphkv", plan=lambda p: GraphPlan.rag_full(len(p.chunks), view=view), layout=layout, label=spec)
    if base.startswith(("graphkv-top", "noedge-top")):
        edges = base.startswith("graphkv")
        m = int(base.split("-top", 1)[1])

        def top_m(p: Prompt) -> GraphPlan:
            k = min(m, len(p.chunks))
            if plan_variant == "official":
                return GraphPlan.rag_top_m(len(p.chunks), k, "official", p.relevance_order())
            return GraphPlan.top_m(len(p.chunks), k, p.relevance_order(), edges=edges, view=view)

        return Method("graphkv", plan=top_m, layout=layout, label=spec)
    if base.startswith("graphkv-chain"):
        # graphkv-chain1-3: Top1 -> Top2-4 -> rest; graphkv-chain1-2-2: Top1 -> Top2-3 -> Top4-5 -> rest
        sizes = [int(x) for x in base[len("graphkv-chain") :].split("-")]
        if variant:
            raise ValueError("chains have no official variant")
        return Method("graphkv", plan=lambda p: GraphPlan.chain(_chain_groups(p.relevance_order(), sizes), len(p.chunks)), label=spec)
    if base == "graphkv-star":
        # citation ego-graph: every chunk but the last is a reference of the last (central) chunk
        return Method("graphkv", plan=lambda p: GraphPlan.star(len(p.chunks) - 1), label=spec)
    raise ValueError(f"unknown method '{spec}'")


def _chain_groups(order: Sequence[int], sizes: Sequence[int]) -> List[List[int]]:
    """``order`` is ascending relevance, so the top chunk is its last entry."""
    ranked = list(order)[::-1]
    n = len(ranked)
    groups, i = [], 0
    for s in sizes:
        groups.append(ranked[i : i + s])
        i += s
    if i < n:
        groups.append(ranked[i:])
    return [g for g in groups if g]
