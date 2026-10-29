# Colab results (L4, `ldsjmdy/Tulu3-Block-FT`, bfloat16)

Analysis of every run in the Colab account's `MyDrive/graphkv_runs` folder, read on 2026-10-29.
All runs used the paper's 8B-Block model on an NVIDIA L4 (torch 2.11, transformers 5.20). Every interval below is
a paired 95% bootstrap interval over the same questions. NLL is the answer's negative log-likelihood in nats per
token; lower is better.

## What the folder holds

| run | notebook cell | what | questions | status |
|---|---|---|---|---|
| `e0_verify_2wiki.json` | 6a | released 2Wiki outputs, teacher-forced | 30 (150 outputs) | complete; the out-of-memory fix worked |
| `e0_verify_hqa.json` | 6a | the same, HotpotQA | 30 (120 outputs) | complete (already in `VERIFICATION.md`) |
| `table3_block` | 6b | paper Table 3 on a sample | 39-40 of 50 per dataset | stopped by the 100-minute budget |
| `table2_block` | 6c | paper Table 2 on a sample | 50 per dataset | complete |
| `e3_docs_nll` | 7, stage A | E3, answer NLL | 300 | complete |
| `e3_docs_gen` | 7, stage B | E3, generation | 126 of 150 | stopped by the 35-minute budget |
| `e1_fixed128_{retriever,bm25,attention,random}` | 8 | E1, answer NLL | 300 each | complete |
| `e2_{documents,fixed128,sentences128,paragraphs128,attention128}` | 9 | E2, answer NLL | 200 each | complete |
| `e2_attention-sent128` | 9 | E2, answer NLL | 138 of 200 | stopped by the 15-minute budget; no summary |
| `debug/smoke` | 5 | debug run, 1B model | 5 | pipeline check only |
| E2 granularity sweep, Cora | 9, optional | | | not run |

Measured compute (prefill + decode, without model loading): Table 3 99.5 min, Table 2 95.3 min, E1 31 min, E2 33 min,
E3 48 min. That is about 5.1 GPU hours, roughly 25 compute units at 4.8 per hour.

## Key findings

1. **The reimplementation reproduces the authors' code.** Teacher-forced, 99.2-99.8% of released tokens are our
   arg-max under their own method. All 81 disagreements across 270 outputs are bfloat16 near-ties. Generating freely,
   our answers are as correct as the released ones on 98.6% of Table 3 questions and 95.2% of Table 2 questions.
2. **The paper's RAG numbers come from the released code's positions.** On 2Wiki, as on HotpotQA, the released
   Top-3 outputs agree with `graphkv-top3@official` on all 30 questions, and with the paper's layout on only 8.
3. **Graph-KV Full scored best on 5 of the 6 sampled datasets**, but with 39-50 questions per dataset an accuracy is
   only known to within ±14-16 points, so the order is not significant. In the paper's own Table 3 rows for this model,
   sequential reading beats every Graph-KV variant on all four datasets (average 73.6 vs 72.3 for Full). Graph-KV's
   gain there is over Block-RAG (+2 to +3 points).
4. **Graph-KV is slower per question in this implementation.** Relative to sequential, Top-1 takes 1.23x, Top-3
   1.29x and Full 1.46x the time (up to 1.73x), at contexts of about 1,000 tokens.
5. **In every NLL experiment, Graph-KV's edges made the answer less likely.** Graph-KV Top-3 is worse than plain
   parallel encoding in all 9 runs (+0.13 to +0.31). It is also worse than the same chunks at the same positions
   without edges in all 6 comparisons (+0.30 to +0.40). Every interval excludes zero.
6. **The NLL metric is a weak proxy here; read findings 5, 7 and 8 as provisional.** It scores the bare gold answer
   as the first tokens of the reply. The model never answers that way: it opens with "Based on the..." or
   "I apologize, but...". The gold answer therefore costs 8-9 nats per token, against 0.05 for the model's own
   wording. In E3, the only place both were measured, NLL and generation disagree in direction (details below).
7. **Choosing better sources does not matter while every chunk stays visible (E1).** With all chunks kept, every
   scorer is within 0.03 nats of random. Scorers only matter once the sources are dropped, and then they matter
   just as much without edges, so the gain does not come from the graph.
8. **BM25 is the best way to pick 128-token windows.** It puts the answer among the top 3 for 66.7% of questions,
   ahead of the retriever (60.3%), the model's own attention (57.7%) and random (44.3%).
9. **The updated targets carry almost none of the sources' facts (E3, generation).** With the sources dropped,
   compositional questions whose answer appears only in the sources score 2 of 37 with edges and 0 of 37 without.
   Overall, edges add +1.6 points [-5.6, +8.7].

## E0: does this code reproduce the released generations? (teacher-forced)

2Wiki, 30 stratified questions, 150 released outputs. Each cell shows the share of released tokens that are our
arg-max, then the outputs that agree at every token once bfloat16 near-ties are allowed, then strict agreement.
Bold cells are each released run against its own method.

| released run \ our method | sequential | block | top1@official | top3@official | full@official | top3 (paper layout) |
|---|---|---|---|---|---|---|
| vanilla (Sequential) | **99.7% (30/30; 24)** | 96.4% (11; 6) | 96.2% (7; 6) | 96.0% (7; 5) | 96.1% (9; 4) | 96.7% (9; 4) |
| block (Block-RAG) | 96.2% (7; 6) | **99.6% (30/30; 22)** | 97.4% (9; 8) | 97.2% (9; 5) | 96.8% (7; 6) | 96.4% (5; 3) |
| gapemp_appr_1 (Top-1) | 97.0% (5; 5) | 98.1% (13; 8) | **99.8% (30/30; 26)** | 98.9% (21; 14) | 97.9% (15; 9) | 97.2% (8; 3) |
| gapemp_appr_3 (Top-3) | 96.3% (4; 4) | 97.7% (8; 7) | 98.5% (18; 14) | **99.7% (30/30; 23)** | 98.0% (14; 8) | 96.8% (8; 3) |
| gapemp (Full) | 95.7% (3; 3) | 96.6% (8; 6) | 96.8% (13; 9) | 97.9% (14; 8) | **99.2% (30/30; 19)** | 96.1% (6; 2) |

| | 2Wiki (GPU, new) | HotpotQA (GPU) | 2Wiki (CPU, earlier) |
|---|---|---|---|
| released outputs checked | 150 | 120 | 60 |
| own-method token agreement | 99.2-99.8% | 99.6-99.7% | 99.79% |
| outputs agreeing at every token, strictly | 114 (76%) | 97 (81%) | 50 (83%) |
| disagreements | 53 | 28 | 11 |
| largest logit margin | 0.5 (2 bfloat16 steps) | 0.25 (2 steps) | 0.25 |
| outputs not explained by ties | 0 | 0 | 0 |
| closest other method, below the diagonal | 0.9 points (Top-1 row) | 1.8 points | |

* The released Top-1 and Top-3 2Wiki answers are often the same text, so the negative control separates those two
  less sharply here (0.9 points) than on HotpotQA.
* The teacher-forced NLL of the released answers under their own method is 0.048 nats per token (2Wiki) and 0.062
  (HotpotQA). This is the scale to compare the E1-E3 NLLs (8-9 nats) against.

## Table 3 on a sample (cell 6b): NarrativeQA, 2Wiki, TriviaQA, HotpotQA

Accuracy on the "all" column. Each cell gives ours / the released answers to the same questions / the paper's number
over all questions. 40 NarrativeQA and 39 questions of each other dataset were answered before the 100-minute budget
ran out.

| method | NarrativeQA | 2Wiki | TriviaQA | HotpotQA | average (ours) | average (paper) | same correctness as released |
|---|---|---|---|---|---|---|---|
| Sequential | 80.0 / 77.5 / 63.21 | 79.5 / 79.5 / 77.29 | 69.2 / 69.2 / 76.64 | 79.5 / - / 77.43 | 77.0 | **73.64** | 117/118 |
| Block-RAG | 77.5 / 77.5 / 59.39 | 76.9 / 71.8 / 71.35 | 64.1 / 66.7 / 74.44 | 79.5 / 76.9 / 71.31 | 74.5 | 69.12 | 153/157 |
| Graph-KV Top-1 | 75.0 / 75.0 / 62.04 | 74.4 / 74.4 / 73.60 | 66.7 / 66.7 / 75.17 | **84.6** / 84.6 / 73.39 | 75.2 | 71.05 | 155/157 |
| Graph-KV Top-3 | 77.5 / - / 62.29 | 71.8 / 71.8 / 75.15 | 66.7 / 66.7 / 75.66 | 79.5 / 79.5 / 75.19 | 73.9 | 72.07 | 117/117 |
| Graph-KV Full | **82.5** / 82.5 / 62.88 | **82.1** / 76.9 / 74.38 | **71.8** / 74.4 / 75.85 | 82.1 / 82.1 / 76.17 | **79.6** | 72.32 | 154/157 |

* **Replication: 696 of 706 answers (98.6%) are as correct as the released ones**; 514 (72.8%) are the same text.
  The differing texts are bfloat16 ties breaking the other way, as in E0.
* The NarrativeQA sample is much easier than the full set: 75-82 here against 59-63 in the paper. The released
  answers score the same as ours on these questions, so this is sampling, not an error.
* At 39 questions, ±16 points (95%) swamps every gap between methods.
* "-" means the authors released no outputs: HotpotQA Sequential (13 of 7,405) and NarrativeQA Top-3.

2Wiki by question type (ours / paper):

| method | comparison (n=8) | inference (n=6) | bridge_comparison (n=9) | compositional (n=16) |
|---|---|---|---|---|
| Sequential | 75.0 / 84.24 | 83.3 / 63.08 | 77.8 / 90.07 | 81.2 / 70.76 |
| Block-RAG | 75.0 / 83.25 | 83.3 / 50.41 | 77.8 / 89.85 | 75.0 / 60.90 |
| Graph-KV Top-1 | 75.0 / 82.76 | 66.7 / 52.42 | 77.8 / 90.00 | 75.0 / 65.94 |
| Graph-KV Top-3 | 87.5 / 83.25 | 50.0 / 56.74 | 77.8 / 90.47 | 68.8 / 67.83 |
| Graph-KV Full | 87.5 / 82.69 | 66.7 / 54.16 | 77.8 / 90.43 | 87.5 / 67.11 |

## Table 2 on a sample (cell 6c): MultiHop-RAG, MorehopQA

50 questions per dataset. Each cell gives ours / released / paper.

| method | MultiHop-RAG | MorehopQA | same correctness as released |
|---|---|---|---|
| Sequential | 32.0 / 30.0 / 43.60 | 36.0 / 34.0 / 37.38 | 94/100 |
| Block-RAG | 40.0 / 38.0 / 43.32 | 40.0 / 44.0 / 37.92 | 93/100 |
| Graph-KV Top-1 | 42.0 / 42.0 / 44.81 | 44.0 / 48.0 / 37.96 | 96/100 |
| Graph-KV Top-3 | 40.0 / 44.0 / 45.79 | 46.0 / 44.0 / 38.11 | 95/100 |
| Graph-KV Full | **48.0** / 48.0 / 46.41 | **52.0** / 48.0 / 44.90 | 98/100 |

| MultiHop-RAG type (ours) | inference (16) | comparison (17) | temporal (12) | null (5) |
|---|---|---|---|---|
| Sequential | 62.5 | 5.9 | 16.7 | 60.0 |
| Block-RAG | 87.5 | 11.8 | 25.0 | 20.0 |
| Graph-KV Top-1 / Top-3 / Full | 75.0 / 68.8 / 81.2 | 0.0 / 0.0 / 0.0 | 33.3 / 33.3 / 50.0 | 100 / 100 / 100 |

| MorehopQA hops (ours) | 1 hop (20) | 2 hops (19) | 3 hops (7) | 5 hops (4) |
|---|---|---|---|---|
| Sequential | 55.0 | 26.3 | 28.6 | 0.0 |
| Block-RAG | 70.0 | 15.8 | 28.6 | 25.0 |
| Graph-KV Top-1 / Top-3 / Full | 60.0 / 60.0 / 60.0 | 21.1 / 31.6 / 42.1 | 57.1 / 57.1 / 57.1 | 50.0 / 25.0 / 50.0 |

* **Replication: 476 of 500 (95.2%) are as correct as the released answers**; 272 (54.4%) are the same text.
  Answers here are longer (116-152 tokens against 73-124 for Table 3), so a tie flips the text more often.
* Graph-KV Full leads on both datasets, but with two caveats:
  * MorehopQA scores Full leniently and every other row strictly, because that is how the printed numbers were
    computed. Scored consistently, Full's lead in the paper shrinks from +6.5 to +1.0 or -3.0 points
    (`VERIFICATION.md`).
  * MultiHop-RAG uses the benchmark's own scorer, because the paper's was never released, so the paper column is not
    comparable. Under it, comparison queries score 0-12% for every method.
* Graph-KV's lead comes from MultiHop-RAG's null queries (5/5 against 1-3/5) and from MorehopQA's questions of
  2-5 hops. On 1-hop MorehopQA questions, Block-RAG is best (70.0).

## Cost per question (Tables 3 and 2)

Mean seconds per question on the L4, and the ratio to sequential.

| dataset | Sequential | Block-RAG | Top-1 | Top-3 | Full |
|---|---|---|---|---|---|
| NarrativeQA | 8.5 | 8.1 (0.95x) | 9.3 (1.09x) | 9.4 (1.11x) | 10.4 (1.22x) |
| 2Wiki | 5.7 | 6.4 (1.12x) | 7.2 (1.26x) | 7.1 (1.25x) | 8.0 (1.40x) |
| TriviaQA | 6.1 | 6.9 (1.13x) | 7.3 (1.20x) | 8.1 (1.33x) | 9.0 (1.48x) |
| HotpotQA | 5.3 | 5.9 (1.11x) | 6.8 (1.28x) | 7.4 (1.40x) | 8.7 (1.64x) |
| MorehopQA | 9.8 | 10.8 (1.10x) | 11.5 (1.17x) | 12.2 (1.24x) | 12.9 (1.32x) |
| MultiHop-RAG | 8.5 | 9.9 (1.16x) | 11.8 (1.39x) | 12.1 (1.42x) | 14.7 (1.73x) |
| **mean ratio** | 1.00 | 1.10 | 1.23 | 1.29 | 1.46 |

Times include generation, and answer lengths differ by method. At these context lengths, the extra prefill of the
targets outweighs any saving from the shorter attention span.

## E3: does source information reach the targets? (2Wiki, document chunks, retriever)

### Answer NLL, 300 questions

| condition | the question sees | context KV | NLL |
|---|---|---|---|
| `graphkv-top3` | top-3 sources and the updated targets | 985 | 8.980 |
| `noedge-top3` | the same chunks at the same positions, encoded without edges | 985 | 8.679 |
| `graphkv-top3:targets` | updated targets only (sources dropped) | 601 | 8.923 |
| `noedge-top3:targets` | targets only, encoded without edges | 601 | 8.526 |
| `graphkv-top3:sources` | sources only | 422 | 9.140 |

| contrast | difference | 95% CI |
|---|---|---|
| edges, everything visible (`graphkv-top3` - `noedge-top3`) | +0.300 | [+0.237, +0.373] |
| edges, targets only (`graphkv-top3:targets` - `noedge-top3:targets`) | +0.397 | [+0.307, +0.489] |
| drop the sources (`graphkv-top3:targets` - `graphkv-top3`) | -0.057 | [-0.269, +0.176] |
| drop the targets (`graphkv-top3:sources` - `graphkv-top3`) | +0.160 | [+0.024, +0.301] |

### Generation, 126 questions

| | Graph-KV targets | no-edge targets | difference |
|---|---|---|---|
| accuracy | 42.9 | 41.3 | +1.6 [-5.6, +8.7] |
| questions only one condition gets right | 10 | 8 | |

Questions whose gold answer appears only in the dropped sources (79):

| question type | n | Graph-KV targets | no-edge targets |
|---|---|---|---|
| comparison (yes/no, which one) | 15 | 100 | 87 |
| bridge_comparison | 19 | 84 | 84 |
| compositional (an entity, e.g. a birthplace) | 37 | 5 | 0 |
| inference | 8 | 0 | 0 |
| all | 79 | 41.8 | 36.7 |

* **With the sources dropped, the targets almost never supply a fact that only the sources had**: 2 of 37
  compositional questions.
* The two comparison types score high either way: their answers come from the targets plus the model's prior. They
  dominate the accuracy of the "source only" bucket.
* **NLL and generation disagree in direction.** NLL says edges hurt (+0.40, significant). Generation says they help
  slightly, and on source-only questions the edges win 5 to 1. See the caveat below.

## E1: top-k windows instead of retrieved documents (2Wiki, fixed 128-token windows, 300 questions)

7.9 windows of 119 tokens per question; the top 3 are the sources.

| scorer | answer among top-3 | `parallel` | `graphkv-top3` | `graphkv-top3:targets` | `noedge-top3:targets` | edges (targets only) |
|---|---|---|---|---|---|---|
| retriever | 60.3% | 8.600 | 8.809 | 8.821 | 8.438 | +0.382 [+0.292, +0.479] |
| BM25 | **66.7%** | 8.600 | 8.812 | 8.724 | **8.355** | +0.369 [+0.260, +0.480] |
| attention (the LLM's own) | 57.7% | 8.600 | 8.767 | 8.995 | 8.652 | +0.343 [+0.240, +0.454] |
| random | 44.3% | 8.600 | 8.782 | 9.018 | 8.676 | +0.341 [+0.228, +0.451] |

Scorer minus random, on the same questions:

| view | retriever | BM25 | attention |
|---|---|---|---|
| `graphkv-top3` (all visible) | +0.027 [-0.033, +0.087] | +0.029 [-0.029, +0.086] | -0.016 [-0.067, +0.033] |
| `graphkv-top3:targets` | -0.197 [-0.369, -0.022] | -0.294 [-0.452, -0.132] | -0.023 [-0.176, +0.137] |
| `noedge-top3:targets` | -0.238 [-0.402, -0.073] | -0.321 [-0.460, -0.177] | -0.024 [-0.165, +0.117] |

* `graphkv-top3` is worse than `parallel` for every scorer: +0.21, +0.21, +0.17, +0.18, each interval above zero.
* The scorer does not matter while all windows are visible. Once the sources are dropped, BM25 and the retriever help
  by the same amount with or without edges.
* The model's own attention picks windows no better than random under NLL, even though it finds the answer 13
  points more often.
* Check: `parallel` does not depend on the scorer, and it is identical in all four runs.

## E2: what should a node be? (2Wiki, BM25, 200 questions)

| chunker | chunks | tokens per chunk | answer among top-3 | sources' share of context | `parallel` | `graphkv-top3` | `graphkv-top3:targets` |
|---|---|---|---|---|---|---|---|
| documents | 10.0 | 94 | 65.5% | 39% | 7.901 | 8.216 | 8.079 |
| fixed:128 | 7.8 | 119 | 70.0% | 37% | **7.867** | **8.045** | 7.991 |
| sentences:128 | 5.1 | **183** | 80.0% | 51% | 8.015 | 8.192 | 7.996 |
| paragraphs:128 | 4.5 | **207** | 83.5% | 54% | 8.077 | 8.203 | 8.069 |
| attention:128 | 7.4 | 128 | 69.0% | 43% | 7.924 | 8.133 | **7.867** |

Chunker minus documents, on the same questions:

| view | fixed:128 | sentences:128 | paragraphs:128 | attention:128 |
|---|---|---|---|---|
| `parallel` | -0.035 [-0.137, +0.074] | +0.114 [+0.046, +0.186] | +0.175 [+0.108, +0.247] | +0.022 [-0.075, +0.121] |
| `graphkv-top3` | **-0.171 [-0.274, -0.064]** | -0.024 [-0.103, +0.048] | -0.012 [-0.089, +0.061] | -0.083 [-0.173, +0.006] |
| `graphkv-top3:targets` | -0.088 [-0.264, +0.078] | -0.083 [-0.293, +0.111] | -0.010 [-0.223, +0.194] | **-0.212 [-0.427, -0.016]** |

* **The sizes are not matched.** `sentences:128` and `paragraphs:128` produced chunks of 183 and 207 tokens. Their
  top 3 therefore hold over half of the context, which alone explains their higher answer-in-top-3 rates. Only
  fixed, attention and documents are comparable in size.
* Graph-KV is worse than `parallel` under every chunker (+0.13 to +0.31, each interval above zero). Targets only is
  level with `parallel` (every interval spans zero), although it holds only 46-63% of the context.
* Fixed 128-token windows beat documents for Graph-KV (-0.17). Attention-placed boundaries give the best
  targets-only score (-0.21), the only targets-only gain whose interval excludes zero.
* `attention-sent:128` stopped at 138 questions. On those questions it is within 0.14 of documents in every condition.
  The attention chunkers take about 30% longer to prepare a condition (0.67 s against 0.51 s for fixed windows).

## Caveat: what the answer NLL measures

The E1-E3 NLL is the teacher-forced loss of the bare gold answer (for example `Bratislava`), placed right after the
question as the first tokens of the reply.

* **Wrong format.** The model replies with a sentence. Of 126 generations, 85 open with "Based on the" and 30 with
  "I apologize, but". The gold answer therefore costs about 8-9 nats per token, against 0.05 for the model's own
  wording in E0. A change of 0.2-0.4 nats is a small shift in the probability of a reply the model would not give.
* **Inconsistent with visibility.** It does separate an answer present in the context (about 7) from one absent
  everywhere (about 20). But on 88 questions where BM25 drops the window with the answer string and random keeps it,
  the targets-only NLL is unchanged: -0.10 [-0.43, +0.24]. In the opposite case (24 questions), it falls by 1.7-2.1.
* **Inconsistent with generation.** E3 measured both: NLL puts the edges' effect at +0.40 (worse, significant), and
  generation at +1.6 points (better, not significant).

So the NLL results (edges hurt, scorers do not matter while everything is visible, fixed and attention chunks are
best) are worth confirming with generation, or with an NLL that scores the answer inside the model's usual reply.
For example, score it after a forced prefix such as `Based on the provided search documents, the answer is`.

## Next

1. Fix or replace the NLL proxy, then repeat the E1 and E2 contrasts that matter. A 150-question generation run costs
   about 35 minutes per condition pair.
2. Resume cell 6b to finish 50 questions per dataset. Resume `e3_docs_gen` (24 left) and `e2_attention-sent128`
   (62 left). All three continue where they stopped.
3. Add a generation run of `graphkv-top3` and `noedge-top3` with everything visible to E3, so the cost of dropping
   the sources is measured in accuracy.
4. Still to run: the E2 granularity sweep (attention:32, 64, 256) and Table 4 on Cora.
