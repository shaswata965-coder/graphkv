# Verification of `graphkv_neurips` against the Graph-KV paper and the published code

For every component this lists the source of truth, how it was checked, and the result. Where the paper and the
authors' code disagree, both are implemented and the difference is stated. Everything below can be re-run:
the commands are at the end.

Sources of truth:
* **paper**: arXiv:2506.07334v2 (NeurIPS 2025), including Appendix A.
* **official code**: [Graph-COM/GraphKV](https://github.com/Graph-COM/GraphKV) at `ba01bbc` (`third_party/GraphKV`).
* **released outputs**: the generations in the Hugging Face dataset `Graph-COM/GraphKV` (`results/`).
* **baseline code**:
  * [AI21Labs/Parallel-Context-Windows](https://github.com/AI21Labs/Parallel-Context-Windows) (PCW);
  * [Infini-AI-Lab/APE](https://github.com/Infini-AI-Lab/APE);
  * [TemporaryLoRA/Block-Attention](https://github.com/TemporaryLoRA/Block-Attention).

## 1. End to end with the paper's model and data

`python -m graphkv_neurips.verify_released` uses `ldsjmdy/Tulu3-Block-FT` (the paper's 8B-Block backbone, bfloat16,
on CPU) and the official processed 2Wiki data. For each example it builds every method's context with this
implementation and feeds each released generation teacher-forced.

The released outputs are greedy, so under the computation that produced them each released token is the model's
arg-max. Rows are released runs, columns our methods. Each cell gives:
* the token-weighted share of positions where our arg-max equals the released token;
* in brackets, the number of outputs that agree everywhere. Positions where the two candidate tokens are within two
  bfloat16 steps of each other count as agreeing (see below).

Run: 12 stratified 2Wiki examples (seed 0); 60 released outputs; 5,353 answer tokens.
Raw data: [`verify_released_2wiki.json`](verify_released_2wiki.json).

| released run \ our method | sequential | block | graphkv-top1@official | graphkv-top3@official | graphkv-full@official | graphkv-top3 (paper layout) |
|---|---|---|---|---|---|---|
| vanilla (sequential) | **99.8% (12/12)** | 97.0% (2/12) | 97.2% (2/12) | 96.6% (0/12) | 96.7% (3/12) | 97.7% (2/12) |
| block (Block-RAG) | 96.3% (2/12) | **99.6% (10/12)** | 98.0% (4/12) | 97.1% (2/12) | 96.9% (2/12) | 96.9% (2/12) |
| gapemp_appr_1 (Graph-KV Top-1) | 96.3% (1/12) | 97.8% (4/12) | **99.9% (12/12)** | 98.6% (5/12) | 97.2% (3/12) | 96.7% (2/12) |
| gapemp_appr_3 (Graph-KV Top-3) | 96.6% (1/12) | 97.7% (2/12) | 98.8% (5/12) | **99.7% (12/12)** | 98.7% (6/12) | 98.1% (5/12) |
| gapemp (Graph-KV Full) | 95.1% (1/12) | 96.0% (1/12) | 96.6% (2/12) | 98.2% (6/12) | **100.0% (12/12)** | 96.2% (3/12) |

**Diagonal (bold).** Each released run is evaluated under the method that produced it.
* 99.79% of the 5,353 tokens are our arg-max.
* 50 of the 60 outputs agree at every token.
* All 11 disagreements are listed below. Each has a logit margin of 0 to 0.25 between our arg-max and the released
  token, at most two bfloat16 steps at those logit sizes (logits around 16-32 are only resolved to 0.125).
* These are ties that the official run (bf16 FlashAttention on a GPU) broke the other way. Two Block-RAG rows lack
  the step column because they were measured before the logit value was recorded. Their margins are 0.125 and 0,
  which are ties as well.
* No disagreement exceeds bfloat16 resolution.

| example | released run | answer tokens | position | our arg-max | released token | logit margin | bf16 step at that logit |
|---|---|---|---|---|---|---|---|
| 4084 | block | 138 | 90 | `' was'` | `','` | 0.125 |  |
| 5098 | block | 73 | 38 | `' the'` | `' these'` | 0.000 |  |
| 5246 | vanilla | 59 | 28 | `' "'` | `' titled'` | 0.125 | 0.125 |
| 5246 | block | 59 | 55 | `'-'` | `' '` | 0.000 | 0.250 |
| 5246 | gapemp_appr_3 | 55 | 37 | `' King'` | `' he'` | 0.125 | 0.125 |
| 6615 | gapemp_appr_1 | 157 | 96 | `' Jiří'` | `' we'` | 0.250 | 0.125 |
| 6615 | gapemp_appr_3 | 192 | 125 | `' we'` | `' Jiří'` | 0.000 | 0.125 |
| 7772 | vanilla | 82 | 32 | `' bar'` | `' Chamber'` | 0.000 | 0.125 |
| 7772 | block | 159 | 50 | `' Governor'` | `' which'` | 0.125 | 0.125 |
| 7772 | block | 159 | 60 | `'.'` | `','` | 0.250 | 0.250 |
| 10846 | gapemp_appr_3 | 107 | 29 | `','` | `'.\n'` | 0.000 | 0.250 |

**Off-diagonal: the negative control.** The same released texts scored under another method agree on 97.1% of
tokens, and few outputs agree everywhere. The check therefore discriminates between methods.
* Where two methods produced the same answer, their cells coincide. This is expected.
* Teacher forcing makes most next tokens predictable under any reasonable context, so the gap is a few points, not
  tens.

**Last column.** The paper's position layout (Sec. 3.1) is a different computation from the released RAG code:
* the released top-3 outputs agree with it on 98.1% of tokens and 5/12 outputs;
* under `graphkv-top3@official` they agree on 99.7% and 12/12.

The paper's RAG numbers were therefore produced with the released code's sequential source positions, not with the
shared ranges it describes.

How it was run:
* CPU, bfloat16, with `--streamed-layers 16`: 16 of the 32 decoder layers are read from the memory-mapped
  checkpoint when needed, because 16 GB of weights do not fit in this machine's 15 GB of RAM. This is numerically
  identical (`tests/test_gkv_lowmem.py`).
* Released runs not checked here:
  * HotpotQA sequential: not released in full;
  * MorehopQA and MultiHop-RAG: same code paths.
* Notebook cell 6a repeats the check on a GPU with 30 2Wiki and 30 HotpotQA examples.

## 2. Component by component

| component | source of truth | check | result |
|---|---|---|---|
| decoder over an explicit KV context | Hugging Face Llama / Mistral / Qwen2 / Qwen3 | `test_gkv_model.py`: logits vs. HF forward and HF `generate`, incremental decoding, moved keys | max abs logit difference ~2e-7, identical greedy tokens |
| structure-aware attention, Fig. 1 (block mask) | paper Sec. 3.1, Fig. 1 | `test_graph_structured_block_mask`: one HF forward with the graph mask and shared positions | identical KV (≤1e-5) |
| Fig. 2 equivalence (target after one source = reading source then target) | paper Fig. 2 | `test_fig2_target_equals_reading_source_then_target` | identical |
| free from positional bias (source order) | paper Table 1, Sec. 3.1 | every permutation of 3 sources | identical logits; sequential control differs |
| shared position ranges, context window | paper Sec. 3.1, Fig. 7 | positions of every token, span independent of #chunks | as stated: span = P + 2L + q |
| complexity O(\|V\|L² + \|E\|L²) | paper Sec. 3.1 | exact count of query-key pairs | equal |
| ArXiv-QA Graph-KV, one or several ego-graphs | `pcw_parallel.gapemp_graph`, `gapemp_graph_batch` | parity test, built with `run_arxiv.build` | identical greedy output (full length) |
| ArXiv-QA Block-RAG, one or several groups | `pcw_parallel.block`, `block_batch` | parity test | identical |
| ArXiv-QA prompts, distractor sampling, token limits | `arxiv_inference.py`, `pcw.py` | strings found verbatim in the official source; same `random` stream; 512 / 256 / 256 new tokens | identical |
| RAG Graph-KV-Full | `pcw.gapemp` | parity test + section 1 | identical |
| RAG Graph-KV Top-1 / Top-3 | `pcw.gapemp_appr` | parity test + section 1 | identical |
| RAG Block-RAG | `server/block_generate_server.py` (Block-Attention's server, same code) | parity test + section 1 | identical |
| sequential | `server/generate_server.py` (`generate_vanilla`), `pcw.vanilla` | parity test + section 1 | identical |
| RAG prompts | `utils.build_prefix`, `docs2blocks`, `build_suffix` | byte comparison for all 6 tasks | identical |
| PCW | `pcw_wrapper.py` (BOS per window, one BOS kept, task tokens after the longest window) | `test_pcw_matches_block_mask_reference` | identical logits (≤1e-5) |
| APE | `ape/ape_llama.py` (context prefill on the shared prefix; temperature/scale merge), `demo_ape.py` (T = S = 0.9, BOS before the question) | line-by-line transcription of `llama_attention_prefill_query`; dense check of the context prefill | identical (≤1e-5) |
| paper numbers | Tables 2, 3, Fig. 6 | `rescore.py` on the released outputs | see section 3 |
| paper topic classification | Appendix A.2.3 (no code released) | prompt template, ego-graph, seeds 42-44 for order-dependent methods; Cora/PubMed from `Graph-COM/Text-Attributed-Graphs` | implemented; see section 4 for what is unspecified |
| stress test | Appendix A.2.4 (no code released) | Cora first test sample as node text, 500/1000 and 100-800 words, 10 neighbours, `torch.cuda.max_memory_allocated` on GPU | implemented |

## 3. The paper's numbers (from the released outputs)

| table | result |
|---|---|
| Figure 6 (ArXiv-QA) | 23 of 23 released runs reproduce the paper's counts |
| Table 3 | 64 of 67 cells reproduce. Two look like typos (82.29 vs 82.89, 63.08 vs 63.00). One run was not released in full: HotpotQA 8B-Block sequential has 13 of 7405 outputs. |
| Table 2, MorehopQA | The baselines and Top-1/Top-3 are scored with "no `Answer:` → empty" (they reproduce), but Graph-KV-Full only with the released `rag_eval.py` fallback "no `Answer:` → whole output". Scored consistently, Full's advantage over the best sequential baseline is +1.0 or −3.0 points instead of +6.5. |
| Table 2, MultiHop-RAG | Not reproducible: the scorer (`qa_evaluate2`) was not released. With the benchmark's own metric the ordering still favours Graph-KV. |

Details: [`rescored_official_outputs.md`](rescored_official_outputs.md).

## 4. Where the paper, the code and this implementation part ways

1. **RAG positions.** The paper (Sec. 3.1) gives sources one shared range and targets the next. The released RAG
   code (`gapemp`, `gapemp_appr`) instead lays sources out one after another and re-rotates everything to
   consecutive positions. The paper's RAG numbers come from that code.
   * Default: `graphkv-top3` (paper).
   * Released-code variant: `graphkv-top3@official`.
2. **What "L" is.** "A maximum chunk length of L". The only released shared-range code (ArXiv-QA) starts the
   targets after the longest *source*.
   * Default: `layout="paper"` (as the code).
   * Literal reading: `@global` (longest chunk overall).
3. **Top-m targets.** Fig. 3 makes the remaining chunks the targets. The released code makes every chunk a target
   and keeps both copies of every chunk. `graphkv-top<m>` follows the paper, `@official` the code.
4. **Generation length.**
   * The paper says 256 new tokens. The released code allows 1024 (RAG), 512 (ArXiv sequential) and
     256 (ArXiv Graph-KV / Block).
   * Of 1600 released RAG outputs sampled, 6 exceed 256 tokens (all MorehopQA), so the difference is negligible.
   * `--max-new-tokens` sets either.
5. **Dataset name.** "NarrativeQA" in Tables 3 and 6 is the Natural Questions file `nq_eval` of Block-Attention.
6. **Not specified by the paper, chosen and documented in the code:**
   * PCW/APE hyper-parameters: APE's defaults, T = S = 0.9.
   * Classification:
     * test split: the one in `Graph-COM/Text-Attributed-Graphs`. The paper cites TSGFM's split, which may differ;
     * neighbours: undirected;
     * answer parsing: first class name after the last `Answer:`.
   * Multi-hop chain (A.1.2): no code was released; implemented from the text.
7. **Numerics.** On GPU the official code runs FlashAttention-2 in bfloat16 and rounds re-rotated keys to bfloat16.
   This implementation uses PyTorch SDPA and rotates keys at their final positions.
   * Greedy outputs can differ at near-ties; section 1 measures how often.
   * Decoded text can differ in whitespace before punctuation under transformers ≥ 5, which ignores
     `clean_up_tokenization_spaces` for this tokenizer. The scoring removes punctuation, so scores are unaffected.

## 5. Commands

```bash
# component tests (any transformers >= 4.45)
python -m pytest tests/test_gkv_*.py
# parity with the official functions (transformers 4.50, the official pin)
python -m venv .venv-upstream && .venv-upstream/bin/pip install torch transformers==4.50.0 pytest
.venv-upstream/bin/python -m pytest tests/test_gkv_upstream_parity.py
# released outputs -> paper tables
python -m graphkv_neurips.rescore --results-dir <Graph-COM/GraphKV>/results --output docs/neurips/rescored_official_outputs.md
# 8B end-to-end check (GPU; on a 16 GB-RAM CPU machine add --streamed-layers 12)
python -m graphkv_neurips.verify_released --task 2wiki --num-examples 12 --output docs/neurips/verify_released_2wiki.json
```
