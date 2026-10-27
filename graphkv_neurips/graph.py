"""Graph-KV: structure-aware attention and shared positional encodings (Sec. 3.1).

A *plan* says which representations of the text chunks exist and what each
one attends to:

* ``h^(0)`` of a chunk is its independent encoding (offline parallel encoding).
* In round ``t`` every target chunk ``j`` is re-encoded on top of the KV caches
  of its sources ``N(j)`` (their latest representation), which is the sparse
  attention ``softmax(Q_j K_N(j)^T / sqrt(d)) V_N(j)`` of the paper, plus causal
  attention over its own tokens. This gives ``h^(t)``.
* At query time the question and the answer attend to the prefix and to every
  representation in ``plan.query_view``.

A *layout* assigns RoPE positions. :class:`PaperLayout` is the allocation of
Sec. 3.1: the prefix takes ``[0, P)``; every representation of level ``t``
shares one range that starts where the longest representation of level
``t - 1`` ends (sources ``[P, P + L)``, targets ``[P + L, P + 2L)``, ...);
query tokens come after the last range. :class:`SequentialRAGLayout`
reproduces the official RAG scripts (``pcw.py: gapemp, gapemp_appr``), which
instead lay the source chunks out one after another (see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .model import RopeDecoder, Segment

Instance = Tuple[int, int]  # (chunk index, level)


@dataclass
class GraphPlan:
    """Representations to compute and what they attend to.

    ``rounds[t - 1]`` maps every target chunk of round ``t`` to the source
    instances it attends to; a source instance must have a level below ``t``.
    A target with an empty source list is re-encoded alone at its level's
    positions (used as a no-edge control). ``levels`` optionally declares which
    chunks occupy each position range, so that conditions which hide some
    representations from the query still give the visible ones the same positions.
    """

    num_chunks: int
    rounds: List[Dict[int, List[Instance]]] = field(default_factory=list)
    query_view: Optional[List[Instance]] = None
    levels: Optional[List[List[int]]] = None

    def __post_init__(self) -> None:
        for t, targets in enumerate(self.rounds, start=1):
            for target, sources in targets.items():
                if not 0 <= target < self.num_chunks:
                    raise ValueError(f"target {target} out of range")
                for chunk, level in sources:
                    if not 0 <= chunk < self.num_chunks or not 0 <= level < t:
                        raise ValueError(f"round {t}: invalid source instance {(chunk, level)}")
                    if level > 0 and chunk not in self.rounds[level - 1]:
                        raise ValueError(f"round {t}: source {(chunk, level)} is never computed")
        if self.query_view is None:
            self.query_view = self.default_view()
        for chunk, level in self.query_view:
            if not 0 <= chunk < self.num_chunks or (level > 0 and (level > len(self.rounds) or chunk not in self.rounds[level - 1])):
                raise ValueError(f"query view instance {(chunk, level)} is never computed")

    # -- structure ---------------------------------------------------------

    def instances(self) -> List[Instance]:
        """The representations that have to be computed: the query view and, recursively, their sources."""
        needed = set(self.query_view or [])
        stack = list(needed)
        while stack:
            chunk, level = stack.pop()
            if level == 0:
                continue
            for src in self.rounds[level - 1][chunk]:
                if src not in needed:
                    needed.add(src)
                    stack.append(src)
        return sorted(needed, key=lambda inst: (inst[1], inst[0]))

    def level_members(self) -> List[List[int]]:
        """Chunks that occupy each position range (declared ``levels`` or the computed instances)."""
        if self.levels is not None:
            return [list(level) for level in self.levels]
        members: List[List[int]] = [[] for _ in range(self.num_levels())]
        for chunk, level in self.instances():
            members[level].append(chunk)
        return members

    def final_level(self, chunk: int) -> int:
        level = 0
        for t, targets in enumerate(self.rounds, start=1):
            if chunk in targets:
                level = t
        return level

    def default_view(self) -> List[Instance]:
        """The paper's rule: the query attends to the source chunks and the target chunks.

        That is every instance that serves as a source, plus the final
        representation of every chunk. A chunk that is only a target is seen
        through its updated representation, an isolated chunk through ``h^(0)``.
        """
        view = set()
        for targets in self.rounds:
            for sources in targets.values():
                view.update(sources)
        view.update((chunk, self.final_level(chunk)) for chunk in range(self.num_chunks))
        return sorted(view, key=lambda inst: (inst[1], inst[0]))

    def num_levels(self) -> int:
        return len(self.rounds) + 1

    def num_edges(self) -> int:
        return sum(len(sources) for targets in self.rounds for sources in targets.values())

    # -- constructors ------------------------------------------------------

    @classmethod
    def parallel(cls, num_chunks: int) -> "GraphPlan":
        """No edges: plain parallel encoding with one shared position range."""
        return cls(num_chunks, [])

    @classmethod
    def from_edges(cls, num_chunks: int, edges: Iterable[Tuple[int, int]], query_view: Optional[List[Instance]] = None) -> "GraphPlan":
        """One round over directed ``(source, target)`` edges; targets attend to the sources' ``h^(0)``."""
        targets: Dict[int, List[Instance]] = {}
        for src, dst in edges:
            targets.setdefault(dst, []).append((src, 0))
        rounds = [dict(sorted(targets.items()))] if targets else []
        return cls(num_chunks, rounds, query_view)

    @classmethod
    def star(cls, num_neighbors: int) -> "GraphPlan":
        """Citation ego-graph: chunks ``0..k-1`` are references (sources), chunk ``k`` the central paper (target)."""
        return cls.from_edges(num_neighbors + 1, [(i, num_neighbors) for i in range(num_neighbors)])

    @classmethod
    def top_m(
        cls,
        num_chunks: int,
        m: int,
        order: Optional[Sequence[int]] = None,
        *,
        edges: bool = True,
        view: str = "all",
    ) -> "GraphPlan":
        """Top-``m`` bipartite graph with selectable edges and query view (drop-source ablation).

        The top-``m`` chunks (last ``m`` of ``order``, ascending relevance) are
        sources at level 0, the other chunks targets at level 1 (Fig. 3).

        * ``edges=False`` re-encodes the targets alone at the same positions
          (no information can flow from the sources): the control.
        * ``view="all"`` (paper): the query sees sources and targets;
          ``"targets"``: the sources are dropped after the targets are updated;
          ``"sources"``: only the top-``m`` chunks are visible.
        """
        if not 1 <= m <= num_chunks:
            raise ValueError("m must be between 1 and the number of chunks")
        order = list(range(num_chunks)) if order is None else list(order)
        top = sorted(order[-m:])
        rest = [c for c in range(num_chunks) if c not in top]
        if not rest:
            return cls(num_chunks, [], [(c, 0) for c in top])
        rounds = [{c: ([(s, 0) for s in top] if edges else []) for c in rest}]
        views = {
            "all": [(s, 0) for s in top] + [(c, 1) for c in rest],
            "targets": [(c, 1) for c in rest],
            "sources": [(s, 0) for s in top],
        }
        if view not in views:
            raise ValueError("view must be 'all', 'targets' or 'sources'")
        return cls(num_chunks, rounds, views[view], levels=[top, rest])

    @classmethod
    def rag_top_m(cls, num_chunks: int, m: int, variant: str = "paper", order: Optional[Sequence[int]] = None) -> "GraphPlan":
        """Bipartite RAG graph of Sec. 4.1.

        ``order`` lists the chunks by ascending retrieval score (default: the
        chunk order itself, as in the official data), so the top-``m`` chunks
        are its last ``m`` entries.

        ``variant="paper"``: the top-``m`` chunks are sources and every *remaining*
        chunk is a target that attends to them (Fig. 3); the query sees each chunk
        once. ``variant="official"``: as ``pcw.py: gapemp_appr``, every chunk
        (including the top-``m``) is a target of the top-``m`` sources and the
        query sees both ``h^(0)`` and ``h^(1)`` of every chunk.
        """
        if variant == "paper":
            return cls.top_m(num_chunks, m, order)
        if not 1 <= m <= num_chunks:
            raise ValueError("m must be between 1 and the number of chunks")
        order = list(range(num_chunks)) if order is None else list(order)
        top = sorted(order[-m:])
        if variant == "official":
            rounds = [{c: [(s, 0) for s in top] for c in range(num_chunks)}]
            view = [(c, 0) for c in range(num_chunks)] + [(c, 1) for c in range(num_chunks)]
            return cls(num_chunks, rounds, view)
        raise ValueError("variant must be 'paper' or 'official'")

    @classmethod
    def rag_full(cls, num_chunks: int, self_loops: bool = True, view: str = "all") -> "GraphPlan":
        """Graph-KV-Full: every chunk is both a source and a target (complete bipartite graph).

        The query sees ``h^(0)`` and ``h^(1)`` of every chunk (``view="all"``),
        only the updated ``h^(1)`` (``"targets"``) or only ``h^(0)``
        (``"sources"``). With ``self_loops`` a chunk's target copy also attends
        to its own source copy, as in the official ``gapemp``.
        """
        everything = list(range(num_chunks))
        rounds = [{c: [(s, 0) for s in everything if self_loops or s != c] for c in everything}]
        views = {"all": [(c, 0) for c in everything] + [(c, 1) for c in everything], "targets": [(c, 1) for c in everything], "sources": [(c, 0) for c in everything]}
        if view not in views:
            raise ValueError("view must be 'all', 'targets' or 'sources'")
        return cls(num_chunks, rounds, views[view], levels=[everything, everything])

    @classmethod
    def chain(cls, groups: Sequence[Sequence[int]], num_chunks: int) -> "GraphPlan":
        """Multi-hop Graph-KV (Appendix A.1.2): ``groups[0] -> groups[1] -> ...``, one round per arrow."""
        rounds = []
        for t in range(1, len(groups)):
            rounds.append({c: [(s, t - 1) for s in groups[t - 1]] for c in groups[t]})
        covered = {c for g in groups for c in g}
        if covered != set(range(num_chunks)):
            raise ValueError("the groups must cover every chunk exactly once")
        return cls(num_chunks, rounds)


# ---------------------------------------------------------------------- layouts


class PaperLayout:
    """Shared position ranges per level (Sec. 3.1, Fig. 1 and Fig. 2)."""

    name = "paper"

    def assign(self, plan: GraphPlan, lengths: Sequence[int], prefix_len: int) -> "Placement":
        starts: Dict[int, int] = {0: prefix_len}
        members = plan.level_members()
        for level in range(1, plan.num_levels() + 1):
            prev = [lengths[c] for c in members[level - 1]] if level - 1 < len(members) else []
            starts[level] = starts[level - 1] + (max(prev) if prev else 0)
        compute = {inst: starts[inst[1]] for inst in plan.instances()}
        view_end = max([compute[i] + lengths[i[0]] for i in plan.query_view] + [prefix_len])
        return Placement(compute=compute, view=dict(compute), query_start=view_end)

    def source_starts(self, placement: "Placement", target: Instance, sources: Sequence[Instance], lengths) -> List[int]:
        return [placement.compute[s] for s in sources]


class LocalPaperLayout(PaperLayout):
    """Like :class:`PaperLayout`, but a target starts right after the longest of *its own* sources.

    Identical to the paper layout when all targets share their sources (stars,
    RAG top-m). With several ego-graphs side by side (ArXiv-QA with
    distractors) it reproduces ``pcw_parallel.gapemp_graph_batch``, where each
    central paper follows its own references.
    """

    name = "paper-local"

    def assign(self, plan: GraphPlan, lengths: Sequence[int], prefix_len: int) -> "Placement":
        shared = super().assign(plan, lengths, prefix_len).compute
        needed = set(plan.instances())
        compute: Dict[Instance, int] = {inst: prefix_len for inst in needed if inst[1] == 0}
        for t, targets in enumerate(plan.rounds, start=1):
            for target, sources in targets.items():
                if (target, t) in needed:
                    compute[(target, t)] = max(compute[s] + lengths[s[0]] for s in sources) if sources else shared[(target, t)]
        view_end = max([compute[i] + lengths[i[0]] for i in plan.query_view] + [prefix_len])
        return Placement(compute=compute, view=dict(compute), query_start=view_end)


class SequentialRAGLayout:
    """Positions used by the official RAG scripts (``pcw.py``), kept for parity checks.

    Sources are encoded alone, then a target sees its sources laid out one
    after another from position 0 and itself right after them. At query time
    the prefix, all ``h^(0)`` and all ``h^(1)`` are re-rotated to consecutive
    positions, and the query follows.
    """

    name = "sequential"

    def assign(self, plan: GraphPlan, lengths: Sequence[int], prefix_len: int) -> "Placement":
        if plan.num_levels() > 2:
            raise ValueError("the sequential RAG layout supports a single round")
        compute: Dict[Instance, int] = {}
        for inst in plan.instances():
            if inst[1] == 0:
                compute[inst] = 0
        needed = set(plan.instances())
        for targets in plan.rounds:
            for target, sources in targets.items():
                if (target, 1) in needed:
                    compute[(target, 1)] = sum(lengths[c] for c, _ in sources)
        view: Dict[Instance, int] = {}
        cursor = prefix_len
        for inst in sorted(plan.query_view, key=lambda i: (i[1], i[0])):
            view[inst] = cursor
            cursor += lengths[inst[0]]
        return Placement(compute=compute, view=view, query_start=cursor)

    def source_starts(self, placement: "Placement", target: Instance, sources: Sequence[Instance], lengths) -> List[int]:
        starts, cursor = [], 0
        for chunk, _ in sources:
            starts.append(cursor)
            cursor += lengths[chunk]
        return starts


LAYOUTS = {"paper": PaperLayout, "paper-local": LocalPaperLayout, "sequential": SequentialRAGLayout}


@dataclass
class Placement:
    compute: Dict[Instance, int]  # start position at which each instance is encoded
    view: Dict[Instance, int]  # start position of each visible instance at query time
    query_start: int


# ---------------------------------------------------------------------- encoder


@dataclass
class EncodedContext:
    """Everything the query attends to, plus bookkeeping for the experiments."""

    segment: Segment
    query_start: int
    num_context_tokens: int
    attention_pairs: int  # query-key pairs computed while building the context (complexity, Sec. 3.1)
    max_position: int


def _shift(decoder: RopeDecoder, segment: Segment, start: int) -> Segment:
    return decoder.move(segment, start + torch.arange(len(segment), device=segment.positions.device))


def encode_graph(
    decoder: RopeDecoder,
    prefix_ids: Sequence[int],
    chunk_ids: Sequence[Sequence[int]],
    plan: GraphPlan,
    layout: str = "paper",
    *,
    pad_id: int = 0,
    max_batch_tokens: int = 16384,
    level0_cache: Optional[Dict[int, Segment]] = None,
) -> EncodedContext:
    """Prefill the prefix and every representation in ``plan`` (Graph-KV, Step 1 of Fig. 1).

    ``level0_cache`` maps chunk index -> an independent encoding of that chunk
    (at any positions, e.g. from relevance scoring); such chunks are moved
    instead of being encoded again.
    """
    if len(chunk_ids) != plan.num_chunks:
        raise ValueError("plan and chunk list disagree on the number of chunks")
    lengths = [len(c) for c in chunk_ids]
    lay = LAYOUTS[layout]()
    placement = lay.assign(plan, lengths, len(prefix_ids))
    needed = set(plan.instances())
    reps: Dict[Instance, Segment] = {}
    pairs = 0

    cache = level0_cache or {}
    level0 = [inst for inst in plan.instances() if inst[1] == 0]
    for inst in level0:
        if inst[0] in cache:
            reps[inst] = _shift(decoder, cache[inst[0]], placement.compute[inst])
    todo = [inst for inst in level0 if inst not in reps]
    if todo:
        segs = decoder.encode_batch(
            [chunk_ids[c] for c, _ in todo], [placement.compute[i] for i in todo], pad_id=pad_id, max_batch_tokens=max_batch_tokens
        )
        for inst, seg in zip(todo, segs):
            reps[inst] = seg
    pairs += sum(lengths[c] * (lengths[c] + 1) // 2 for c, _ in level0)

    for t, targets in enumerate(plan.rounds, start=1):
        # targets with the same sources share one context and are batched together
        groups: Dict[Tuple[Instance, ...], List[int]] = {}
        for target, sources in targets.items():
            if (target, t) in needed:
                groups.setdefault(tuple(sources), []).append(target)
        for sources, group in groups.items():
            context = None
            if sources:
                starts = lay.source_starts(placement, (group[0], t), sources, lengths)
                context = Segment.cat([_shift(decoder, reps[s], st) for s, st in zip(sources, starts)])
            segs = decoder.encode_batch(
                [chunk_ids[c] for c in group],
                [placement.compute[(c, t)] for c in group],
                context,
                pad_id=pad_id,
                max_batch_tokens=max_batch_tokens,
            )
            for c, seg in zip(group, segs):
                reps[(c, t)] = seg
                pairs += lengths[c] * (len(context) if context is not None else 0) + lengths[c] * (lengths[c] + 1) // 2

    prefix = None
    if len(prefix_ids):
        _, prefix = decoder.encode(torch.as_tensor([list(prefix_ids)]), torch.arange(len(prefix_ids)).unsqueeze(0))
        pairs += len(prefix_ids) * (len(prefix_ids) + 1) // 2
    visible = [_shift(decoder, reps[inst], placement.view[inst]) for inst in plan.query_view]
    segment = Segment.cat(([prefix] if prefix is not None else []) + visible)
    return EncodedContext(
        segment=segment,
        query_start=placement.query_start,
        num_context_tokens=len(segment),
        attention_pairs=pairs,
        max_position=int(segment.positions.max()) if len(segment) else -1,
    )
