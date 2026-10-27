"""Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models.

Implementation of Wang et al., NeurIPS 2025 (arXiv:2506.07334) that runs on CPU
or GPU with any transformers release >= 4.45, verified against the official
code (``third_party/GraphKV``) and the released outputs.
"""

from .graph import EncodedContext, GraphPlan, encode_graph
from .methods import METHODS, Generation, Method, Prompt, decode, generate, parse_method, prepare
from .model import APEConfig, KVBuffer, RopeDecoder, Segment

__all__ = [
    "APEConfig",
    "EncodedContext",
    "Generation",
    "GraphPlan",
    "KVBuffer",
    "METHODS",
    "Method",
    "Prompt",
    "RopeDecoder",
    "Segment",
    "decode",
    "encode_graph",
    "generate",
    "parse_method",
    "prepare",
]
