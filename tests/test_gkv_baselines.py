"""PCW and APE against independent references (their original code is CUDA/flash-attn only).

* PCW (Ratner et al., ``pcw_wrapper.py``): every window is encoded after its own BOS from
  position 0, only one BOS entry is kept, task tokens start at the longest window's length
  and see all windows. Reference: one Hugging Face forward pass with the equivalent block
  mask and position ids.
* APE (Yang et al., ``ape/ape_llama.py``): chunks attend to the shared prefix from the shared
  range ``[P, P + L)``; question/answer tokens combine a temperature-scaled softmax over the
  chunks with a causal softmax over prefix and question through scaled log-sum-exps.
  Reference: a line-by-line transcription of ``llama_attention_prefill_query``.
"""

import math

import torch

from graphkv_neurips.methods import Method, Prompt, prepare
from graphkv_neurips.model import APEConfig, RopeDecoder
from graphkv_neurips.testing import tiny_model

BOS = 1


def _tok(n, seed):
    return torch.randint(3, 256, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _dense_last_logits(model, ids, positions, allow):
    mask = torch.where(allow, 0.0, torch.finfo(torch.float32).min)[None, None]
    with torch.no_grad():
        out = model(torch.as_tensor([ids]), position_ids=torch.as_tensor([positions]), attention_mask=mask)
    return out.logits[0, -1]


def _query_logits(dec, prepared):
    q = prepared.query_ids
    pos = prepared.query_start + torch.arange(len(q)).unsqueeze(0)
    with torch.no_grad():
        logits, _, _ = dec.forward(torch.as_tensor([q]), pos, prepared.segment, ape=prepared.ape)
    return logits[0]


def test_pcw_matches_block_mask_reference():
    model = tiny_model("llama")
    dec = RopeDecoder(model)
    prefix, c0, c1, q = [BOS] + _tok(3, 1), _tok(6, 2), _tok(4, 3), _tok(3, 4)
    ours = _query_logits(dec, prepare(dec, Prompt(prefix, [c0, c1], q), Method("pcw", bos_id=BOS)))
    windows = [prefix, [BOS] + c0, [BOS] + c1]
    longest = max(map(len, windows))
    ids, positions, owner = [], [], []
    for w, win in enumerate(windows):
        ids += win
        positions += list(range(len(win)))
        owner += [w] * len(win)
    n_ctx = len(ids)
    ids += q
    positions += list(range(longest, longest + len(q)))
    n = len(ids)
    allow = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n_ctx):
        for j in range(i + 1):
            allow[i, j] = owner[i] == owner[j]
    starts = {0}
    cursor = 0
    for win in windows:
        starts.add(cursor)
        cursor += len(win)
    for i in range(n_ctx, n):
        for j in range(n_ctx):
            allow[i, j] = j == 0 or j not in starts  # the first window keeps its BOS, the others drop theirs
        allow[i, n_ctx : i + 1] = True
    ref = _dense_last_logits(model, ids, positions, allow)
    assert (ours - ref).abs().max() < 1e-5


def test_ape_context_construction_matches_block_mask_reference():
    """With T = S = 1 APE's merge is an ordinary softmax, so its prefill can be checked against a dense mask."""
    model = tiny_model("llama")
    dec = RopeDecoder(model)
    prefix, c0, c1, q = _tok(4, 5), _tok(6, 6), _tok(3, 7), _tok(3, 8)
    method = Method("ape", ape_temperature=1.0, ape_scale=1.0)
    ours = _query_logits(dec, prepare(dec, Prompt(prefix, [c0, c1], q), method))
    P, L = len(prefix), 6
    ids = prefix + c0 + c1 + q
    positions = list(range(P)) + list(range(P, P + 6)) + list(range(P, P + 3)) + list(range(P + L, P + L + 3))
    n = len(ids)
    allow = torch.zeros(n, n, dtype=torch.bool)
    allow[:P, :P] = torch.ones(P, P, dtype=torch.bool).tril()
    for start, length in ((P, 6), (P + 6, 3)):
        allow[start : start + length, :P] = True
        allow[start : start + length, start : start + length] = torch.ones(length, length, dtype=torch.bool).tril()
    allow[n - 3 :, : n - 3] = True
    allow[n - 3 :, n - 3 :] = torch.ones(3, 3, dtype=torch.bool).tril()
    ref = _dense_last_logits(model, ids, positions, allow)
    assert (ours - ref).abs().max() < 1e-5


def _ape_reference(q, k, v, len_prefix, len_context, temperature, scale, softmax_scale):
    """Transcription of ape_llama.llama_attention_prefill_query (flash_attn_func written out)."""

    def flash(qq, kk, vv, causal, s):
        scores = torch.matmul(qq, kk.transpose(-1, -2)) * s
        if causal:  # FlashAttention aligns the causal mask to the bottom right
            nq, nk = qq.shape[2], kk.shape[2]
            scores = scores.masked_fill(~torch.ones(nq, nk, dtype=torch.bool).tril(diagonal=nk - nq), float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        return torch.matmul(torch.softmax(scores, dim=-1), vv), lse

    key_states_context = k[:, :, len_prefix : len_prefix + len_context]
    key_states_other = torch.cat([k[:, :, :len_prefix], k[:, :, len_prefix + len_context :]], dim=-2)
    value_states_context = v[:, :, len_prefix : len_prefix + len_context]
    value_states_other = torch.cat([v[:, :, :len_prefix], v[:, :, len_prefix + len_context :]], dim=-2)
    out_c, lse_c = flash(q, key_states_context, value_states_context, False, 1 / (math.sqrt(q.shape[-1]) * temperature))
    out_o, lse_o = flash(q, key_states_other, value_states_other, True, softmax_scale)
    lse_c = lse_c.unsqueeze(-1) * (scale * temperature)
    w = torch.softmax(torch.cat([lse_c, lse_o.unsqueeze(-1)], dim=-1), dim=-1).unsqueeze(-2)
    stacked = torch.stack([out_c, out_o], dim=-2)
    return torch.matmul(w, stacked).squeeze(-2)


def test_ape_attention_matches_transcribed_official_formula():
    dec = RopeDecoder(tiny_model("llama", num_key_value_heads=4))
    g = torch.Generator().manual_seed(0)
    P, C, n, d = 3, 7, 4, dec.head_dim
    q = torch.randn(1, dec.num_heads, n, d, generator=g)
    k = torch.randn(1, dec.num_heads, P + C + n, d, generator=g)
    v = torch.randn(1, dec.num_heads, P + C + n, d, generator=g)
    mask = torch.zeros(P + C, dtype=torch.bool)
    mask[P:] = True
    for T, S in ((0.9, 0.9), (0.5, 1.3)):
        ours = dec._ape_attention(q, k, v, P + C, APEConfig(mask, T, S))
        ref = _ape_reference(q, k, v, P, C, T, S, dec.scale)
        assert (ours - ref).abs().max() < 1e-5


def test_ape_puts_bos_before_the_question_like_its_code():
    dec = RopeDecoder(tiny_model("llama"))
    prepared = prepare(dec, Prompt(_tok(3, 1), [_tok(4, 2)], _tok(2, 3)), Method("ape", bos_id=BOS))
    assert prepared.query_ids[0] == BOS and len(prepared.query_ids) == 3
