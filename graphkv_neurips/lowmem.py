"""Run a model that does not fit in RAM: keep most layers resident, stream the rest from safetensors.

Used to check the 8B ``Tulu3-Block-FT`` (16 GB of bfloat16 weights) on a
machine with less memory. The embedding matrix is read row by row on demand
and the last ``streamed`` decoder layers are loaded from the memory-mapped
checkpoint just before :class:`~graphkv_neurips.model.RopeDecoder` runs them
and released afterwards. The numbers are identical to a fully loaded model.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Dict, Iterator, List, Optional

import torch
from torch import nn


def _snapshot(model_id: str) -> str:
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])


class _Files:
    def __init__(self, root: str) -> None:
        index = os.path.join(root, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        else:
            from safetensors import safe_open

            with safe_open(os.path.join(root, "model.safetensors"), "pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self.root = root
        self._handles: Dict[str, object] = {}

    def handle(self, name: str):
        from safetensors import safe_open

        fname = self.weight_map[name]
        if fname not in self._handles:
            self._handles[fname] = safe_open(os.path.join(self.root, fname), "pt")
        return self._handles[fname]

    def get(self, name: str) -> torch.Tensor:
        return self.handle(name).get_tensor(name)


class LazyEmbedding(nn.Module):
    """``nn.Embedding`` replacement that reads only the rows it needs from the checkpoint."""

    def __init__(self, files: _Files, name: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.files, self.name, self.dtype = files, name, dtype

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        uniq, inverse = torch.unique(ids, return_inverse=True)
        table = self.files.handle(self.name).get_slice(self.name)
        rows = torch.cat([table[int(u) : int(u) + 1] for u in uniq.tolist()], dim=0).to(self.dtype)
        return rows[inverse]


def _set(model: nn.Module, name: str, tensor: torch.Tensor) -> None:
    *path, leaf = name.split(".")
    mod = model
    for p in path:
        mod = getattr(mod, p)
    mod._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)


class LayerStreamer:
    """Context manager factory for ``RopeDecoder.layer_hook``."""

    def __init__(self, model: nn.Module, files: _Files, layers: List[int], dtype: torch.dtype) -> None:
        self.model, self.files, self.layers, self.dtype = model, files, set(layers), dtype
        self.names = {i: [n for n in files.weight_map if n.startswith(f"model.layers.{i}.")] for i in layers}

    @contextlib.contextmanager
    def __call__(self, i: int) -> Iterator[None]:
        if i not in self.layers:
            yield
            return
        for n in self.names[i]:
            _set(self.model, n, self.files.get(n).to(self.dtype))
        try:
            yield
        finally:
            for n in self.names[i]:
                _set(self.model, n, torch.empty(0, dtype=self.dtype, device="meta"))


def load_streaming(model_id: str, streamed: int = 8, dtype: torch.dtype = torch.bfloat16, lazy_embedding: bool = True):
    """Return ``(tokenizer, model, layer_hook)``; pass ``layer_hook`` to ``RopeDecoder.layer_hook``."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    root = _snapshot(model_id)
    files = _Files(root)
    config = AutoConfig.from_pretrained(root)
    with torch.device("meta"):
        try:
            model = AutoModelForCausalLM.from_config(config, dtype=dtype)
        except TypeError:  # transformers < 4.56
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    n_layers = config.num_hidden_layers
    stream: List[int] = list(range(n_layers - streamed, n_layers)) if streamed else []
    skip_prefixes = tuple(f"model.layers.{i}." for i in stream)
    for name in files.weight_map:
        if name.startswith(skip_prefixes) or (lazy_embedding and name == "model.embed_tokens.weight"):
            continue
        # copy out of the memory map: mapped pages are evictable cache, and evicting them would
        # make every forward pass re-read the whole checkpoint from disk
        _set(model, name, files.get(name).to(dtype).clone())
    if lazy_embedding:
        model.model.embed_tokens = LazyEmbedding(files, "model.embed_tokens.weight", dtype)
    rotary_cls = type(model.model.rotary_emb)
    model.model.rotary_emb = rotary_cls(config=config)
    if getattr(config, "tie_word_embeddings", False):
        raise ValueError("tied embeddings are not supported by the streaming loader")
    model.eval()
    tok = AutoTokenizer.from_pretrained(root)
    return tok, model, LayerStreamer(model, files, stream, dtype)


def loaded_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.device.type != "meta")


def first_real_parameter(model: nn.Module) -> Optional[torch.Tensor]:
    return next((p for p in model.parameters() if p.device.type != "meta"), None)
