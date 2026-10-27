"""Cora / PubMed text-attributed citation graphs for paper topic classification (Sec. 4.3, Table 4).

Data: Hugging Face dataset ``Graph-COM/Text-Attributed-Graphs`` (``<name>/raw_texts.pt``,
``processed_data.pt``, ``categories.csv``). ``processed_data.pt`` is a pickled
``torch_geometric`` ``Data`` object; it is read with an allow-listed unpickler
(tensors, numpy arrays and inert stand-ins for the PyG containers), so neither
``torch_geometric`` nor an unrestricted ``torch.load`` is needed.
"""

from __future__ import annotations

import csv
import pickle
import types
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

HF_REPO = "Graph-COM/Text-Attributed-Graphs"


class _Inert:
    """Stand-in for torch_geometric containers: keeps the pickled state, runs no code."""

    def __init__(self, *args, **kwargs) -> None:
        self.__dict__["_args"] = args

    def __setstate__(self, state) -> None:
        self.__dict__["_state"] = state


_ALLOWED = {
    ("torch._utils", "_rebuild_tensor_v2"),
    ("collections", "OrderedDict"),
    ("_codecs", "encode"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("torch_geometric."):
            return _Inert
        if (module, name) in _ALLOWED:
            if module.startswith("numpy"):
                import numpy

                if name == "_reconstruct":
                    return numpy.core.multiarray._reconstruct if hasattr(numpy, "core") else numpy._core.multiarray._reconstruct
                return getattr(numpy, name)
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


_SAFE_PICKLE = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="graphkv_neurips.tag_data.safe_pickle")


def _store(obj) -> Dict[str, object]:
    """The attribute dict of a pickled PyG ``Data``: ``Data._store`` (a GlobalStorage) ``._mapping``."""
    state = getattr(obj, "_state", None)
    storage = state.get("_store") if isinstance(state, dict) else None
    mapping = getattr(storage, "_state", {}).get("_mapping") if storage is not None else None
    if not isinstance(mapping, dict):
        raise ValueError("unexpected processed_data.pt layout")
    return mapping


@dataclass
class TAGraph:
    name: str
    texts: List[str]
    labels: torch.Tensor
    edge_index: torch.Tensor
    classes: List[str]
    masks: Dict[str, torch.Tensor]

    def neighbors(self, node: int) -> List[int]:
        """Nodes linked to ``node`` (citation edges, either direction, deduplicated, ascending)."""
        src, dst = self.edge_index
        out = set(dst[src == node].tolist()) | set(src[dst == node].tolist())
        out.discard(node)
        return sorted(out)


def load_tag(name: str = "cora", data_dir: Optional[str] = None) -> TAGraph:
    import os

    def path(f: str) -> str:
        if data_dir is not None:
            return os.path.join(data_dir, name, f)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(HF_REPO, f"{name}/{f}", repo_type="dataset")

    texts = torch.load(path("raw_texts.pt"), weights_only=True)
    data = torch.load(path("processed_data.pt"), weights_only=False, pickle_module=_SAFE_PICKLE)
    store = _store(data)
    with open(path("categories.csv"), newline="", encoding="utf-8") as f:
        classes = [row["name"] for row in csv.DictReader(f)]
    masks = {k: v for k, v in store.items() if isinstance(v, torch.Tensor) and v.dtype == torch.bool}
    return TAGraph(name, list(texts), store["y"].view(-1), store["edge_index"], classes, masks)
