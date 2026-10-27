"""The streaming loader (layers loaded from the checkpoint on demand) changes nothing numerically."""

import torch

from graphkv_neurips.lowmem import load_streaming, loaded_bytes
from graphkv_neurips.model import RopeDecoder
from graphkv_neurips.testing import tiny_model


def test_streamed_layers_and_lazy_embedding_match_the_full_model(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    model = tiny_model("llama", num_hidden_layers=3)
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(tmp_path)
    model.save_pretrained(tmp_path)
    ids = torch.randint(0, 256, (1, 12), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = model(ids).logits
    _, small, hook = load_streaming(str(tmp_path), streamed=2, dtype=torch.float32)
    assert loaded_bytes(small) < loaded_bytes(model)
    dec = RopeDecoder(small)
    dec.layer_hook = hook
    with torch.no_grad():
        ours, _, _ = dec.forward(ids, torch.arange(12).unsqueeze(0), logits="all")
    assert (ours - ref).abs().max() < 1e-5
    assert all(p.device.type == "meta" for n, p in small.named_parameters() if n.startswith("model.layers.2."))
