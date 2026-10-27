# Graph-KV (NeurIPS 2025): everything done up until now

Paper: *Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models*
(Wang et al., NeurIPS 2025, arXiv:2506.07334v2).
Branch: `graph-kv-neurIPS`. Status as of 2026-10-27.

## At a glance

| # | step | status |
|---|---|---|
| 1 | Find the authors' code and pin it | done |
| 2 | Reimplement the method and every baseline (`graphkv_neurips/`) | done |
| 3 | Tests against Hugging Face, the paper's properties and the official functions | done: 58 Graph-KV tests pass (19 more parity tests under transformers 4.50) |
| 4 | Re-score the authors' released outputs against the paper's tables | done |
| 5 | Audit paper vs official code vs ours, fix deviations | done; differences documented |
| 6 | End-to-end check with the paper's 8B model on CPU | done: 99.79% token agreement |
| 7 | Build the new experiments (top-k windows, chunkers, drop-sources) | done, not yet run at scale |
| 8 | Colab notebook and a 200-unit monthly budget | done |
| 9 | Colab run, cells 1-5 (setup, tests, GPU self-check, debug run) | done on an L4 |
| 10 | Notebook cells that replicate the paper's Tables 3 and 2 | done, waiting to be run |
| 11 | Colab cells 6a-6c, then the experiments E3, E1, E2 | **next** |

---

## Step 1: the authors' code

* The official code exists: [Graph-COM/GraphKV](https://github.com/Graph-COM/GraphKV). It has no license, so it is
  referenced as a git submodule pinned at `ba01bbc` (`third_party/GraphKV`), not copied.
* Its data and released generations are in the Hugging Face dataset `Graph-COM/GraphKV` (`datahub/`, `results/`).
* The paper's three models are on Hugging Face: `ldsjmdy/Tulu3-Block-FT` (8B-Block, the main one),
  `ldsjmdy/Tulu3-SFT` and `ldsjmdy/Tulu3-RAG`, 16 GB each.

## Step 2: the implementation (`graphkv_neurips/`)

The official code gets its attention masks from FlashAttention-2 and needs CUDA, `flash-attn` and
transformers 4.50. This reimplementation runs anywhere (CPU or GPU), with transformers 4.45 to 5.x, on Llama,
Mistral, Qwen2 and Qwen3 models.

| module | what it does |
|---|---|
| `model.py` | decoder over an explicit KV context (reuses the Hugging Face weights), moving cached keys to new positions, batched encoding, APE attention |
| `graph.py` | graph plans: top-m bipartite, Graph-KV-Full, star, multi-hop chain, parallel, no-edge control; position layouts of the paper (`paper`, `@global`) and of the released code (`@official`) |
| `methods.py` | every method as a spec string: `sequential`, `pcw`, `ape`, `block`, `parallel`, `graphkv-top<m>`, `noedge-top<m>`, `graphkv-full`, `graphkv-chain<a>-<b>`, `graphkv-star`, with `@official` / `@global` and the views `:targets` / `:sources` |
| `prompts.py` | the official prompt strings, byte for byte |
| `metrics.py` | the official scorers (subspan EM, MultiHop-RAG, MorehopQA strict and lenient, ArXiv, classification) |
| `data.py`, `tag_data.py` | the official processed data; Cora / PubMed (safe unpickler for the PyG files) |
| `run_rag.py` | the paper's RAG protocol (Tables 2, 3, 5, 6) |
| `run_arxiv.py` | ArXiv-QA (Fig. 6) |
| `run_classify.py` | paper topic classification (Table 4) |
| `stress.py` | the stress test (Sec. 4.4) |
| `run_structure.py`, `chunking.py`, `selection.py`, `attention.py` | the new experiments (step 7) |
| `rescore.py`, `paper_numbers.py` | re-score released outputs against the transcribed paper numbers |
| `verify_released.py`, `compare_official.py` | check our computation against the released generations |
| `replicate.py` | the paper's Tables 3 and 2 on a sample (step 10) |
| `gpucheck.py` | one-minute self-check on the GPU the experiments run on |
| `lowmem.py` | streams layers from disk so the 8B model runs in 15 GB of RAM |

## Step 3: tests

* `tests/test_gkv_*.py`: 58 pass, 1 skipped (the parity module, which needs transformers 4.50).
  The whole repository: 123 pass, 1 skipped.
* What they check:
  * the decoder against Hugging Face: logits within ~2e-7, identical greedy generation;
  * Graph-KV against one Hugging Face forward pass with the graph-structured block mask (Fig. 1);
  * the paper's properties: no positional bias over source order, shared position ranges, exact complexity count;
  * PCW and APE against dense reference implementations of their published code;
  * chunkers, scorers, views, answer NLL, the replication report.
* `tests/test_gkv_upstream_parity.py` (transformers 4.50): 19 tests run the official functions from the submodule
  and ours on the same tiny models. Greedy outputs are identical on every path (RAG Top-m, Full, Block-RAG,
  sequential, ArXiv Graph-KV and Block, single and batched).

## Step 4: the paper's numbers from the released outputs

`rescore.py` scores the authors' released generations with the official metrics
([`docs/neurips/rescored_official_outputs.md`](docs/neurips/rescored_official_outputs.md)).

| table | result |
|---|---|
| Fig. 6 (ArXiv-QA) | 23 of 23 runs give the paper's counts |
| Table 3 (NarrativeQA, 2Wiki, TriviaQA, HotpotQA) | 64 of 67 cells match. Two look like typos (82.29 vs 82.89, 63.08 vs 63.00); HotpotQA 8B-Block sequential was released with only 13 of 7405 outputs |
| Table 2, MorehopQA | reproduces, but with two scorers: strict for the baselines and Top-1/Top-3, lenient for Graph-KV Full. Scored consistently, Full's advantage shrinks from +6.5 to +1.0 or -3.0 points |
| Table 2, MultiHop-RAG | not reproducible: the paper's scorer (`qa_evaluate2`) was never released |

## Step 5: where the paper, the official code and this implementation differ

Full list: [`docs/neurips/VERIFICATION.md`](docs/neurips/VERIFICATION.md), section 4. The main points:

1. **RAG positions.** The paper gives all sources one shared position range and targets the next one. The released
   RAG code lays sources out one after another. The paper's RAG numbers come from the code.
   Ours: `graphkv-top3` (paper) and `graphkv-top3@official` (code).
2. **Top-m targets.** The paper makes the remaining chunks the targets; the code makes every chunk a target and
   keeps both copies. Both are implemented.
3. **What "L" is.** The code starts targets after the longest source; the literal paper text says the longest chunk
   (`@global`).
4. **Generation length.** The paper says 256 new tokens, the released RAG code allows 1024. Only 6 of 1600 sampled
   released outputs exceed 256 tokens.
5. **Unspecified by the paper**, chosen and documented: APE's T = S = 0.9, classification split and answer parsing,
   the multi-hop chain (no code released).

## Step 6: end-to-end check with the paper's 8B model (CPU)

`verify_released.py` with `ldsjmdy/Tulu3-Block-FT` in bfloat16, on 12 2Wiki questions (60 released outputs,
5,353 tokens). Each released answer is fed teacher-forced under each of our methods; because the released runs are
greedy, every released token should be our arg-max under the method that produced it.

* **Same method (the test):** 99.79% of tokens are our arg-max; 50 of 60 outputs agree at every token.
* **All 11 disagreements are bfloat16 ties**: logit margins of 0 to 0.25, within two bfloat16 steps. The official
  GPU run broke these ties the other way.
* **Other methods (the negative control):** 97.1%, and few outputs agree everywhere, so the check tells methods
  apart.
* **Paper vs code positions:** the released Top-3 outputs match `graphkv-top3@official` 12 of 12, the paper layout
  only 5 of 12. The paper's RAG numbers were produced with the code's positions.
* Getting it to run: the 16 GB model did not fit in 15 GB of RAM, so `lowmem.py` streams 16 of the 32 layers
  from disk (numerically identical).

## Step 7: the new experiments you asked for

All run through `run_structure.py`: documents → one context → **chunker** (nodes) → **scorer** (which nodes are
the top-k sources) → **conditions** (edges, and what the question sees).

| experiment | question | how |
|---|---|---|
| E1, top-k windows | does Graph-KV generalize beyond retrieved documents? | `--chunker fixed:128`, scorers `retriever`, `bm25`, `attention` (the LLM's own question attention), controls `random`, `first`, `last`, upper bound `oracle` |
| E2, node structure | does structure matter, or only granularity? | chunkers `documents`, `fixed:N`, `sentences:N`, `paragraphs:N`, `attention:N` (cuts where attention across the boundary is lowest), `attention-sent:N`; compare at matched size, then sweep N |
| E3, drop the sources | does source information actually reach the targets? | `graphkv-top3:targets` (only updated targets kept) minus `noedge-top3:targets` (same tokens and positions, encoded without edges), with paired bootstrap intervals, split by where the gold answer is |

Also built: `--metric nll` (answer likelihood, one forward pass, 5-20x cheaper than generation, for sweeps),
time budgets with resume, cost projection in compute units, `--data-file` for any JSONL dataset.

Directions we flagged as easy to miss ([`docs/neurips/COLAB.md`](docs/neurips/COLAB.md), section 5): the
block-fine-tuned model is essential; paper vs code positions; the duplication confound in Full and top-m; whether
relevance matters at all (`random`); edges learned from attention; multi-hop direction; contexts long enough for
structure to matter; NLL alongside accuracy.

## Step 8: Colab notebook and budget

* Notebook: [`notebooks/graphkv_neurips_colab.ipynb`](notebooks/graphkv_neurips_colab.ipynb).
  Guide: [`docs/neurips/COLAB.md`](docs/neurips/COLAB.md).
* Budget: 200 compute units a month, planned as ~40 a week (5 experiments x ~8 units, plus reserve). Debug on a T4
  or CPU with a 1B model; run the 8B model on an L4; sweep with NLL, confirm with generation.
* Cell 4b (`gpucheck.py`) repeats the key equivalence checks on the actual GPU before any paid run.

## Step 9: Colab run so far (L4)

| cell | result |
|---|---|
| 1. runtime | NVIDIA L4, 23,034 MiB |
| 2. Drive | mounted, results go to `MyDrive/graphkv_runs` |
| 3. code + install | torch 2.11.0+cu130, transformers 5.20.2, CUDA available |
| 4. unit tests | 44 passed in 14 s |
| 4b. GPU self-check | all checks ok (logits identical to Hugging Face; Graph-KV vs block mask 1.3e-7; moving keys 2.2e-7; official Full 8.9e-8) |
| 5. debug run (1B model, 5 questions) | pipeline works end to end, 2.7 s per question, ~0 units. Accuracy numbers are meaningless at this size, as expected. 5.2 chunks of ~112 tokens per question; Graph-KV's position span 369 vs 700 for sequential reading |

## Step 10: replicating the paper's tables (new cells 6b and 6c)

The full tables are ~37,000 questions per method, hundreds of GPU hours. `replicate.py` answers 50 questions per
dataset (stratified by question type) with the paper's model and rows: Sequential, Block-RAG, Graph-KV
Top-1 / Top-3 / Full. Every cell is printed three times:

* **ours:** this code on those questions;
* **released:** the authors' published answers to the same questions, same scorer;
* **paper:** the printed number, over all questions.

*Ours vs released* is the test: expect the same correctness on 95% or more of questions. *Released vs paper* is
only sampling error (about ±14 points at 50 questions). Optional PCW / APE rows use the 8B-SFT and 8B-RAG models.
Checked on CPU with a 0.5B model (pipeline, question matching, report); the GPU cost estimate, ~1.5 h per cell on
an L4, is not yet measured.

## Problems hit along the way, and the fixes

| problem | fix |
|---|---|
| 8B model too large for 15 GB of RAM; two out-of-memory kills | stream 12, then 16 layers from disk; return freed memory to the OS between methods |
| reading weights from the memory map was very slow | keep the resident layers as real copies |
| Qwen3 not available in transformers 4.50 | test configs depend on the installed version |
| recursion when loading the Cora / PubMed pickles | read the stored tensors directly, with an allow-listed unpickler |
| released text decoded with `clean_up_tokenization_spaces` | mismatches that are only a space before punctuation are flagged as such |
| MorehopQA released outputs are in a different order from its data file | released answers are matched by question |
| notebook summary collector crashed on classification summaries | fixed |
| a CPU experiment run was interrupted at your request | everything moved to Colab |

## Next

1. Colab cell 6a (teacher-forced check on GPU, 30 2Wiki + 30 HotpotQA questions). Then I add the GPU results to
   `docs/neurips/VERIFICATION.md`.
2. Cells 6b and 6c (Tables 3 and 2 on a sample).
3. The experiments: E3 (cell 7), E1 (cell 8), E2 (cell 9); optionally Table 4 on Cora.
4. Not in this budget: ArXiv-QA (Fig. 6) needs an 80 GB GPU.

## Commit history (this work)

The branch starts from the repository's earlier, separate ISCAS 2026 KV-compression work (3 commits on 2026-10-08).

| commit | what |
|---|---|
| `0a1e43b` | Graph-KV implementation, verification and Colab experiments |
| `71e7d3e` | audit against the paper and published code; remaining tasks and the 8B check |
| `06421bc` | layouts for star and chain plans; reject meaningless official variants |
| `b0935c9`, `b7971c1`, `cdf83ef` | `verify_released`: mismatching tokens, logit margins, bfloat16 near-ties |
| `6781606`, `3ad67b5` | `verify_released`: memory release, `--report-only` |
| `2b5a233` | end-to-end check with the 8B Tulu3-Block-FT |
| `0ef192e` | GPU self-check; notebook summary collector fix |
| `1bfe693` | `verify_released` prints the disagreement list |
| `796a476` | notebook cells replicating Tables 3 and 2 |
