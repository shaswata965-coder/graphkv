"""Autoregressive decoding with periodic graph-based KV-cache compression.

The cache is compressed after every ``M`` tokens that are fed back into the
model during decoding ("for every M generated tokens", Fig. 1). The prompt is
prefilled into an uncompressed cache; it is first compressed at decode step
``M``, together with everything generated up to then.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch

from .cache import CompressionEvent, GraphKVConfig, KVCache, compress_cache
from .models import DecoderAdapter
from .utils import synchronize


@dataclass
class DecodeResult:
    """Outcome of one :meth:`GraphKVEngine.generate` or :meth:`GraphKVEngine.score` call."""

    prompt_len: int
    tokens: List[int]
    token_nll: List[float]
    prefill_s: float
    decode_s: float
    compress_s: float
    events: List[CompressionEvent] = field(default_factory=list)
    final_entries: int = 0
    final_uncompressed_entries: int = 0
    final_layer_lengths: List[float] = field(default_factory=list)
    final_kv_bytes: int = 0
    final_kv_bytes_allocated: int = 0
    peak_kv_bytes: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s

    @property
    def nll_sum(self) -> float:
        return float(sum(self.token_nll))

    @property
    def perplexity(self) -> float:
        return math.exp(self.nll_sum / len(self.token_nll)) if self.token_nll else float("nan")

    @property
    def max_compression(self) -> float:
        """Largest saving reached right after a compression event (0 if none happened)."""
        return max((e.compression_ratio for e in self.events), default=0.0)

    @property
    def final_compression(self) -> float:
        full = self.final_uncompressed_entries
        return 0.0 if full == 0 else 1.0 - self.final_entries / full


class GraphKVEngine:
    """Runs a model adapter with an optional :class:`GraphKVConfig`.

    ``config=None`` is the standard-cache baseline. Both modes go through the
    same code path, so timings are directly comparable.
    """

    def __init__(self, adapter: DecoderAdapter, config: Optional[GraphKVConfig] = None) -> None:
        self.adapter = adapter
        self.config = config

    @property
    def device(self) -> torch.device:
        return self.adapter.device

    def _maybe_compress(self, cache: KVCache, step: int, events: List[CompressionEvent]) -> float:
        if self.config is None or step % self.config.interval != 0:
            return 0.0
        synchronize(self.device)
        t0 = time.perf_counter()
        event = compress_cache(cache, self.config, step=step)
        synchronize(self.device)
        event.seconds = time.perf_counter() - t0
        events.append(event)
        return event.seconds

    @torch.inference_mode()
    def _run(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        choose: Callable[[torch.Tensor, int], int],
        stop_token_ids: Sequence[int] = (),
    ) -> DecodeResult:
        if len(prompt_ids) == 0:
            raise ValueError("prompt must contain at least one token")
        device = self.device
        cfg = self.config
        cache = self.adapter.new_cache(
            track_weights=cfg is not None and cfg.weighted_merge,
            proportional_attention=cfg is not None and cfg.proportional_attention,
        )
        events: List[CompressionEvent] = []
        tokens: List[int] = []
        nlls: List[float] = []
        compress_s = 0.0

        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
        synchronize(device)
        t0 = time.perf_counter()
        logits = self.adapter(ids, cache, last_only=True)
        synchronize(device)
        t1 = time.perf_counter()
        bytes_per_entry = cache.bytes_per_entry()
        peak_entries = cache.num_entries()

        for step in range(1, max_new_tokens + 1):
            log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
            token = choose(log_probs, step - 1)
            tokens.append(token)
            nlls.append(-float(log_probs[token]))
            if step == max_new_tokens or token in stop_token_ids:
                break
            logits = self.adapter(torch.tensor([[token]], device=device), cache)
            peak_entries = max(peak_entries, cache.num_entries())
            compress_s += self._maybe_compress(cache, step, events)
        synchronize(device)
        t2 = time.perf_counter()

        return DecodeResult(
            prompt_len=len(prompt_ids),
            tokens=tokens,
            token_nll=nlls,
            prefill_s=t1 - t0,
            decode_s=t2 - t1,
            compress_s=compress_s,
            events=events,
            final_entries=cache.num_entries(),
            final_uncompressed_entries=cache.uncompressed_entries(),
            final_layer_lengths=cache.layer_lengths(),
            final_kv_bytes=cache.nbytes(),
            final_kv_bytes_allocated=cache.allocated_nbytes(),
            peak_kv_bytes=peak_entries * bytes_per_entry,
        )

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        generator: Optional[torch.Generator] = None,
        stop_token_ids: Sequence[int] = (),
    ) -> DecodeResult:
        """Generate up to ``max_new_tokens`` tokens (greedy unless ``do_sample``).

        ``token_nll`` holds ``-log p(token)`` of each generated token under the
        model's untempered next-token distribution, i.e. the data for a
        self-perplexity of the generated text.
        """

        def choose(log_probs: torch.Tensor, _: int) -> int:
            if not do_sample:
                return int(log_probs.argmax())
            return _sample(log_probs, temperature, top_k, top_p, generator)

        return self._run(prompt_ids, max_new_tokens, choose, stop_token_ids)

    def score(self, prompt_ids: Sequence[int], target_ids: Sequence[int]) -> DecodeResult:
        """Teacher-forced NLL of ``target_ids`` following ``prompt_ids``.

        The reference tokens are fed one at a time exactly like generated
        tokens would be, so the cache is compressed at the same steps as in
        :meth:`generate`. ``perplexity`` of the result is the perplexity of the
        reference continuation under the (compressed) model.
        """
        targets = list(target_ids)
        if not targets:
            raise ValueError("target_ids must not be empty")
        return self._run(prompt_ids, len(targets), lambda _lp, i: targets[i])


@torch.inference_mode()
def sequence_nll(adapter: DecoderAdapter, ids: Sequence[int]) -> List[float]:
    """Per-token NLL of ``ids[1:]`` under the *uncompressed* model, in one forward pass.

    Used to judge generated text with the full model ("oracle" perplexity).
    """
    x = torch.tensor([list(ids)], dtype=torch.long, device=adapter.device)
    log_probs = torch.log_softmax(adapter(x, adapter.new_cache())[0, :-1].float(), dim=-1)
    return (-log_probs.gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)).tolist()


def _sample(
    log_probs: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Optional[torch.Generator],
) -> int:
    """Temperature / top-k / top-p sampling on CPU, so a seeded generator is reproducible on any device."""
    if temperature <= 0:
        return int(log_probs.argmax())
    logits = log_probs.detach().float().cpu() / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        drop = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[drop] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
