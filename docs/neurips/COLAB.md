# Running the Graph-KV experiments on Colab with 200 compute units a month

Notebook: [`notebooks/graphkv_neurips_colab.ipynb`](../../notebooks/graphkv_neurips_colab.ipynb)
(in Colab: *File → Open notebook → GitHub*, or upload the file).

## 1. What things cost

Colab does not publish per-GPU rates, and reported figures differ by source and month. Before each session,
open **View resources** next to the RAM/Disk meter. Colab shows the live rate there in units per hour.
Put that number in the notebook's `CU_PER_HOUR`, and every run prints its projected cost after the
first `--warmup` examples.

| runtime | units / hour (reported range) | use it for |
|---|---|---|
| CPU | ~0.1 | unit tests, summaries, re-scoring released outputs |
| T4 (16 GB, fp16 only) | ~1.2-1.8 | debugging with the 1B model |
| L4 (22.5 GB, bf16) | ~1.7-4.8 | every run with the 8B `Tulu3-Block-FT` |
| A100 | ~5-15 | not needed this month |

The plan below assumes the **pessimistic** end (L4 = 4.8/h). If your panel shows ~1.7/h, the same plan costs
about a third as much, and the rest can go to more examples or to the follow-up directions in section 5.

Per-example cost with the 8B model on an L4. These are estimates; check the projection printed by the runner:

| work | seconds per example and condition | 300 examples x 4 conditions |
|---|---|---|
| `--metric nll` (teacher-forced answer likelihood) | ~1 | ~20 min, ~1.6 units |
| `--metric generate` (greedy, up to 256 tokens) | ~4-6 | ~1.7 h, ~8 units |
| attention chunker / attention scorer | +0.5 per example | small |

Fixed overhead per session: ~2 min install + ~3-5 min to download the 8B weights (~0.5 units on an L4).
So one session that runs several experiments costs less than several sessions that run one each.

## 2. Monthly budget

200 units over 4 weeks is 50 a week. Keep 10 a week in reserve, so plan with **40 a week**, which is about
**8 units for each of 5 experiments**:

| per experiment | runtime | units |
|---|---|---|
| 2-3 debug runs (1B model, 5 examples, `nll`) | T4 / CPU | ~0.5-1 |
| stage A sweep, `--metric nll`, 200-300 examples | L4 | ~2-3 |
| stage B confirmation, `--metric generate`, 150 examples, 2-3 best conditions | L4 | ~3-4 |

Suggested month:

| week | experiments | approx. units |
|---|---|---|
| 1 | E0 sanity check against the released outputs (once); E3 drop-sources on documents | 12 + 8 |
| 2 | E1 scorers on fixed windows (NLL); E3 repeated on the best windows; generation confirmation | 3 x 8 |
| 3 | E2 structures at matched size, then a granularity sweep for the best two (NLL); confirmation | 3 x 8 |
| 4 | second dataset (HotpotQA or a long-document set) for the best configurations; reserve | 30 + reserve |

## 3. Step by step

1. **Push access.** If the repository is private, create a GitHub fine-grained token with read access to it,
   and add it in Colab under the key icon (*Secrets*) as `GH_TOKEN`. Enable notebook access.
2. **Runtime.** *Runtime → Change runtime type*: T4 for debugging, L4 for 8B runs.
3. Run cells 1-4: rate, Drive, code + install, unit tests. The unit tests run on tiny random models and
   check the attention masks, positions, chunkers and views (~1 min). Run them again after each `git pull`.
4. **Debug run (cell 5)** on every change: 5 examples, 1B model, ~5 min. Read `summary.md` in the output
   directory on Drive. Only then switch to an L4.
5. **Real runs (cells 6-9).** Each one has `--time-budget-min` and stops cleanly. Rerunning the same cell
   resumes, and finished examples are skipped. Keep the browser tab open; on Pro, background execution is
   not available and an idle tab disconnects.
6. **Collect (cell 10)**, then **disconnect (cell 11)**. A connected runtime bills even when it sits idle.

Debugging rules that keep re-runs cheap:
* A change to flags or code means a new `--output-dir`. Resuming into an old directory mixes settings.
  The debug cell deletes its own output instead.
* Most problems show up in the first example. Use `--num-examples 5 --metric nll --max-new-tokens 32` until
  the pipeline is right.
* Do not pay for a GPU to look at results. Summaries are on Drive and can be read from a CPU runtime.

## 4. The three experiments

All three run through `python -m graphkv_neurips.run_structure` and share one pipeline: documents → one
context text → **chunker** (nodes) → **scorer** (which nodes are the top-k sources) → **conditions**
(edges and what the question sees).

**E1: top-k windows (generalising beyond RAG documents).** `--chunker fixed:128` cuts the context into
windows. `--scorer` picks the top-k windows that become sources:
* `retriever`: the score of the document the window came from.
* `bm25`: lexical match with the question.
* `attention`: the LLM's own question attention over the independently encoded windows. It needs no
  retriever, and the encodings are reused by Graph-KV.
* `random`, `first`, `last`: controls.
* `oracle`: an upper bound.

`graphkv-top<m>` works for any `m`.

**E2: node structure vs granularity.**
* Chunkers: `documents`, `fixed:N`, `sentences[:N]`, `paragraphs[:N]`, `attention:N` and `attention-sent:N`.
* `attention:N` cuts where the following tokens attend least to the preceding text: the cross-boundary
  attention mass over middle layers, with sink tokens excluded. At those points, encoding the two sides
  independently loses the least.
* All chunkers partition exactly the same text, and the sequential baseline is identical across them, which
  serves as a check.
* Compare at matched size first, then sweep `N`. If accuracy or NLL against tokens-per-chunk falls on one
  curve for all structures, the effect is granularity rather than structure.

**E3: drop the sources after the targets are updated.**
* `graphkv-top3:targets` keeps only the updated targets.
* `noedge-top3:targets` keeps the same target tokens at the same positions, encoded without edges.
* Their difference is the information carried from sources into target KV. The summary reports it with a
  paired bootstrap interval, split by where the gold answer occurs.
* In the `source_only` slice, anything above the control can only have come through the edges.

## 5. Directions that are easy to miss

1. **The fine-tuned model is not optional.** Graph-KV's gains are reported on `Tulu3-Block-FT`, which was
   post-trained with block attention. On untuned models every parallel method loses accuracy (Appendix A.1.1).
   Your structure findings may not transfer to a model that was never trained to read independent blocks.
   Two cheap checks:
   * run the best configuration on `ldsjmdy/Tulu3-SFT` (same base model, no block training);
   * consider block fine-tuning a 1B model with the Block-Attention recipe. That would cut every later
     experiment's cost ~8x.
2. **Paper positions vs released positions.** The released RAG code places sources one after another instead
   of in the shared range of Sec. 3.1, and its top-m variant keeps two copies of every chunk.
   The paper's RAG tables were produced with that code. E0 measures the difference (`graphkv-top3` vs
   `graphkv-top3@official`).
3. **Duplication confound.**
   * Graph-KV-Full and the released top-m let the query see every chunk twice. Part of any gain can be
     re-reading.
   * `graphkv-full:targets` (each chunk once, after seeing all others) against `parallel` separates the two.
4. **Relevance vs any extra attention.** The `random` scorer answers whether the top-k needs to be relevant
   at all.
5. **Edges from attention, not just nodes.** The attention statistics here pick boundaries and sources.
   The natural next step connects chunk pairs with high cross-attention as edges: a learned sparse graph
   instead of the fixed bipartite one.
6. **Multi-hop rounds and direction.** `graphkv-chain1-3` (Top1 → Top2-4 → rest, Appendix A.1.2) is already
   available. The reverse direction and two-way rounds are untested.
7. **Contexts long enough for structure to matter.** The RAG prompts are ~1-2k tokens. Use
   `--data-file` with long single documents (e.g. Qasper, QuALITY, LongBench single-doc QA converted to
   `{question, answers, context}` JSONL) to test windows where sequential reading actually struggles.
8. **Metrics.** Substring accuracy rewards long answers. Report NLL alongside it, keep the example set fixed
   across conditions (the runner does), and read the bootstrap intervals: with 150 examples an accuracy
   difference under ~5 points is noise.
