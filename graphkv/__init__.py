"""Graph-based dynamic KV-cache compression for transformer inference.

Implementation of "A Graph-Based Methodology for Dynamic KV-Cache Compression
in Transformer Inference" (Bhattacharya et al., ISCAS 2026).
"""

from .cache import CompressionEvent, GraphKVConfig, KVCache, compress_cache, compress_layer
from .clustering import epsilon_adjacency, label_propagation, merge_clusters, pairwise_sq_distances
from .engine import DecodeResult, GraphKVEngine
from .models import SUPPORTED_MODEL_TYPES, make_adapter

__version__ = "0.1.0"

__all__ = [
    "CompressionEvent",
    "DecodeResult",
    "GraphKVConfig",
    "GraphKVEngine",
    "KVCache",
    "SUPPORTED_MODEL_TYPES",
    "compress_cache",
    "compress_layer",
    "epsilon_adjacency",
    "label_propagation",
    "make_adapter",
    "merge_clusters",
    "pairwise_sq_distances",
]
