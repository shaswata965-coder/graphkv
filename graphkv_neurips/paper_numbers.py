"""Numbers reported in the paper (arXiv:2506.07334v2), transcribed for comparison.

Accuracy in %. Keys of each row are the column names of the table.
"""

# Table 3 -- NarrativeQA, 2Wiki, TriviaQA, HotpotQA. The 2Wiki header of the paper reads
# "Compare, Infer, Bridge, Compose" with 3040 / 1549 / 2751 / 5236 questions, i.e. the
# 2WikiMultihopQA types comparison / inference / bridge_comparison / compositional.
TABLE3 = {
    ("8B-SFT", "Sequential"): {"nqa": 60.60, "2wiki/comparison": 86.84, "2wiki/inference": 24.66, "2wiki/bridge_comparison": 74.62, "2wiki/compositional": 67.20, "2wiki/all": 68.33, "tqa": 76.19, "hqa/comparison": 74.37, "hqa/bridge": 73.18, "hqa/all": 73.42},
    ("8B-SFT", "PCW"): {"nqa": 39.22, "2wiki/comparison": 70.13, "2wiki/inference": 14.07, "2wiki/bridge_comparison": 67.17, "2wiki/compositional": 32.58, "2wiki/all": 46.94, "tqa": 60.13, "hqa/comparison": 54.87, "hqa/bridge": 33.28, "hqa/all": 37.62},
    ("8B-SFT", "APE"): {"nqa": 49.05, "2wiki/comparison": 74.44, "2wiki/inference": 18.26, "2wiki/bridge_comparison": 61.17, "2wiki/compositional": 37.41, "2wiki/all": 49.20, "tqa": 66.28, "hqa/comparison": 60.52, "hqa/bridge": 46.24, "hqa/all": 49.11},
    ("8B-RAG", "Sequential"): {"nqa": 62.38, "2wiki/comparison": 82.29, "2wiki/inference": 60.94, "2wiki/bridge_comparison": 90.54, "2wiki/compositional": 68.22, "2wiki/all": 75.75, "tqa": 76.38, "hqa/comparison": 73.43, "hqa/bridge": 78.59, "hqa/all": 77.55},
    ("8B-RAG", "PCW"): {"nqa": 46.59, "2wiki/comparison": 66.21, "2wiki/inference": 18.71, "2wiki/bridge_comparison": 72.88, "2wiki/compositional": 19.49, "2wiki/all": 42.37, "tqa": 64.18, "hqa/comparison": 58.84, "hqa/bridge": 33.40, "hqa/all": 33.51},
    ("8B-RAG", "APE"): {"nqa": 49.14, "2wiki/comparison": 74.40, "2wiki/inference": 21.88, "2wiki/bridge_comparison": 84.55, "2wiki/compositional": 29.96, "2wiki/all": 51.65, "tqa": 67.31, "hqa/comparison": 63.34, "hqa/bridge": 42.41, "hqa/all": 46.61},
    ("8B-Block", "Sequential"): {"nqa": 63.21, "2wiki/comparison": 84.24, "2wiki/inference": 63.08, "2wiki/bridge_comparison": 90.07, "2wiki/compositional": 70.76, "2wiki/all": 77.29, "tqa": 76.64, "hqa/comparison": 71.88, "hqa/bridge": 78.82, "hqa/all": 77.43},
    ("8B-Block", "Block-RAG"): {"nqa": 59.39, "2wiki/comparison": 83.25, "2wiki/inference": 50.41, "2wiki/bridge_comparison": 89.85, "2wiki/compositional": 60.90, "2wiki/all": 71.35, "tqa": 74.44, "hqa/comparison": 70.20, "hqa/bridge": 71.59, "hqa/all": 71.31},
    ("8B-Block", "Graph-KV Top-1"): {"nqa": 62.04, "2wiki/comparison": 82.76, "2wiki/inference": 52.42, "2wiki/bridge_comparison": 90.00, "2wiki/compositional": 65.94, "2wiki/all": 73.60, "tqa": 75.17, "hqa/comparison": 70.41, "hqa/bridge": 74.14, "hqa/all": 73.39},
    ("8B-Block", "Graph-KV Top-3"): {"nqa": 62.29, "2wiki/comparison": 83.25, "2wiki/inference": 56.74, "2wiki/bridge_comparison": 90.47, "2wiki/compositional": 67.83, "2wiki/all": 75.15, "tqa": 75.66, "hqa/comparison": 71.01, "hqa/bridge": 76.24, "hqa/all": 75.19},
    ("8B-Block", "Graph-KV Full"): {"nqa": 62.88, "2wiki/comparison": 82.69, "2wiki/inference": 54.16, "2wiki/bridge_comparison": 90.43, "2wiki/compositional": 67.11, "2wiki/all": 74.38, "tqa": 75.85, "hqa/comparison": 70.41, "hqa/bridge": 77.62, "hqa/all": 76.17},
}

# Table 2 -- MultiHop-RAG and MorehopQA (LongBench-v2 columns omitted: no released outputs)
TABLE2 = {
    ("8B-SFT", "Sequential"): {"multihopqa/inference_query": 42.54, "multihopqa/comparison_query": 33.49, "multihopqa/temporal_query": 33.37, "multihopqa/null_query": 99.34, "multihopqa/all": 39.18, "morehopqa/hop1": 57.65, "morehopqa/hop2": 24.27, "morehopqa/hop3": 29.87, "morehopqa/hop4": 15.38, "morehopqa/hop5": 26.37, "morehopqa/all": 38.37},
    ("8B-RAG", "Sequential"): {"multihopqa/inference_query": 73.12, "multihopqa/comparison_query": 34.66, "multihopqa/temporal_query": 39.00, "multihopqa/null_query": 43.43, "multihopqa/all": 44.27, "morehopqa/hop1": 54.72, "morehopqa/hop2": 22.59, "morehopqa/hop3": 22.07, "morehopqa/hop4": 53.85, "morehopqa/hop5": 25.27, "morehopqa/all": 35.86},
    ("8B-Block", "Sequential"): {"multihopqa/inference_query": 63.55, "multihopqa/comparison_query": 34.54, "multihopqa/temporal_query": 38.03, "multihopqa/null_query": 53.09, "multihopqa/all": 43.60, "morehopqa/hop1": 59.23, "morehopqa/hop2": 20.43, "morehopqa/hop3": 25.32, "morehopqa/hop4": 23.07, "morehopqa/hop5": 30.76, "morehopqa/all": 37.38},
    ("8B-Block", "Block-RAG"): {"multihopqa/inference_query": 72.73, "multihopqa/comparison_query": 34.16, "multihopqa/temporal_query": 38.03, "multihopqa/null_query": 40.73, "multihopqa/all": 43.32, "morehopqa/hop1": 56.75, "morehopqa/hop2": 25.72, "morehopqa/hop3": 24.67, "morehopqa/hop4": 15.38, "morehopqa/hop5": 27.47, "morehopqa/all": 37.92},
    ("8B-Block", "Graph-KV Top-1"): {"multihopqa/inference_query": 73.25, "multihopqa/comparison_query": 34.19, "multihopqa/temporal_query": 38.18, "multihopqa/null_query": 53.85, "multihopqa/all": 44.81, "morehopqa/hop1": 54.72, "morehopqa/hop2": 25.72, "morehopqa/hop3": 24.67, "morehopqa/hop4": 23.07, "morehopqa/hop5": 31.86, "morehopqa/all": 37.96},
    ("8B-Block", "Graph-KV Top-3"): {"multihopqa/inference_query": 73.12, "multihopqa/comparison_query": 34.43, "multihopqa/temporal_query": 38.69, "multihopqa/null_query": 63.64, "multihopqa/all": 45.79, "morehopqa/hop1": 56.53, "morehopqa/hop2": 25.72, "morehopqa/hop3": 24.67, "morehopqa/hop4": 23.07, "morehopqa/hop5": 29.67, "morehopqa/all": 38.11},
    ("8B-Block", "Graph-KV Full"): {"multihopqa/inference_query": 69.04, "multihopqa/comparison_query": 34.63, "multihopqa/temporal_query": 38.48, "multihopqa/null_query": 88.79, "multihopqa/all": 46.41, "morehopqa/hop1": 64.86, "morehopqa/hop2": 30.52, "morehopqa/hop3": 25.97, "morehopqa/hop4": 30.76, "morehopqa/hop5": 47.25, "morehopqa/all": 44.90},
}

# Figure 6 -- number of correct ArXiv-QA answers out of 60 (0 / 1 / 2 distractors)
FIGURE6 = {
    0: {"8B-SFT+SEQ": 27, "8B-SFT+PCW": 3, "8B-SFT+APE": 5, "8B-RAG+SEQ": 25, "8B-RAG+PCW": 3, "8B-RAG+APE": 4, "8B-Block+SEQ": 30, "8B-Block+Block-RAG": 7, "8B-Block+GraphKV": 27},
    1: {"8B-SFT+SEQ+First": 9, "8B-SFT+SEQ+Last": 12, "8B-RAG+SEQ+First": 7, "8B-RAG+SEQ+Last": 18, "8B-Block+SEQ+First": 7, "8B-Block+SEQ+Last": 14, "8B-Block+Block-RAG+First": 3, "8B-Block+Block-RAG+Last": 5, "8B-Block+GraphKV": 17},
    2: {"8B-SFT+SEQ+First": 0, "8B-SFT+SEQ+Last": 6, "8B-RAG+SEQ+First": 1, "8B-RAG+SEQ+Last": 4, "8B-Block+SEQ+First": 0, "8B-Block+SEQ+Last": 9, "8B-Block+Block-RAG+First": 0, "8B-Block+Block-RAG+Last": 0, "8B-Block+GraphKV": 9},
}

# Tables 5/6 -- untuned Llama-3.1-8B-SFT backbone (Appendix A.1.1)
TABLE5_6 = {
    "PCW": {"morehopqa/all": 11.35, "2wiki/all": 46.94, "multihopqa/all": 42.01, "hqa/all": 37.62, "nqa": 39.22, "tqa": 60.13},
    "APE": {"morehopqa/all": 14.13, "2wiki/all": 49.20, "multihopqa/all": 40.34, "hqa/all": 49.11, "nqa": 49.05, "tqa": 66.28},
    "Block-RAG": {"morehopqa/all": 13.32, "2wiki/all": 48.99, "multihopqa/all": 38.41, "hqa/all": 41.49, "nqa": 46.53, "tqa": 65.55},
    "Graph-KV Top-3": {"morehopqa/all": 15.47, "2wiki/all": 54.30, "multihopqa/all": 42.35, "hqa/all": 53.90, "nqa": 51.08, "tqa": 69.47},
}

# released output files (Graph-COM/GraphKV results/) -> (backbone, method) rows of the tables
RELEASED_RAG_RUNS = {
    ("ldsjmdyTulu3-SFT", "vanilla"): ("8B-SFT", "Sequential"),
    ("ldsjmdyTulu3-RAG", "vanilla"): ("8B-RAG", "Sequential"),
    ("ldsjmdyTulu3-Block-FT", "vanilla"): ("8B-Block", "Sequential"),
    ("ldsjmdyTulu3-Block-FT", "block"): ("8B-Block", "Block-RAG"),
    ("ldsjmdyTulu3-Block-FT", "gapemp_appr_1"): ("8B-Block", "Graph-KV Top-1"),
    ("ldsjmdyTulu3-Block-FT", "gapemp_appr_3"): ("8B-Block", "Graph-KV Top-3"),
    ("ldsjmdyTulu3-Block-FT", "gapemp_appr_5"): ("8B-Block", "Graph-KV Top-5 (not in paper)"),
    ("ldsjmdyTulu3-Block-FT", "gapemp"): ("8B-Block", "Graph-KV Full"),
}

RELEASED_ARXIV_RUNS = {
    0: {
        "8B-SFT+SEQ": "ldsjmdyTulu3-SFT/vanilla_1_last_42.jsonl",
        "8B-RAG+SEQ": "ldsjmdyTulu3-RAG/vanilla_1_last_42.jsonl",
        "8B-Block+SEQ": "ldsjmdyTulu3-Block-FT/vanilla_1_last_42.jsonl",
        "8B-Block+Block-RAG": "ldsjmdyTulu3-Block-FT/block_1_last_42.jsonl",
        "8B-Block+GraphKV": "ldsjmdyTulu3-Block-FT/gapemp_graph_1_last_42.jsonl",
    },
    1: {
        "8B-SFT+SEQ+First": "ldsjmdyTulu3-SFT/vanilla_2_first_42.jsonl",
        "8B-SFT+SEQ+Last": "ldsjmdyTulu3-SFT/vanilla_2_last_42.jsonl",
        "8B-RAG+SEQ+First": "ldsjmdyTulu3-RAG/vanilla_2_first_42.jsonl",
        "8B-RAG+SEQ+Last": "ldsjmdyTulu3-RAG/vanilla_2_last_42.jsonl",
        "8B-Block+SEQ+First": "ldsjmdyTulu3-Block-FT/vanilla_2_first_42.jsonl",
        "8B-Block+SEQ+Last": "ldsjmdyTulu3-Block-FT/vanilla_2_last_42.jsonl",
        "8B-Block+Block-RAG+First": "ldsjmdyTulu3-Block-FT/block_2_first_42.jsonl",
        "8B-Block+Block-RAG+Last": "ldsjmdyTulu3-Block-FT/block_2_last_42.jsonl",
        "8B-Block+GraphKV": "ldsjmdyTulu3-Block-FT/gapemp_graph_2_first_42.jsonl",
    },
    2: {
        "8B-SFT+SEQ+First": "ldsjmdyTulu3-SFT/vanilla_3_first_42.jsonl",
        "8B-SFT+SEQ+Last": "ldsjmdyTulu3-SFT/vanilla_3_last_42.jsonl",
        "8B-RAG+SEQ+First": "ldsjmdyTulu3-RAG/vanilla_3_first_42.jsonl",
        "8B-RAG+SEQ+Last": "ldsjmdyTulu3-RAG/vanilla_3_last_42.jsonl",
        "8B-Block+SEQ+First": "ldsjmdyTulu3-Block-FT/vanilla_3_first_42.jsonl",
        "8B-Block+SEQ+Last": "ldsjmdyTulu3-Block-FT/vanilla_3_last_42.jsonl",
        "8B-Block+Block-RAG+First": "ldsjmdyTulu3-Block-FT/block_3_first_42.jsonl",
        "8B-Block+Block-RAG+Last": "ldsjmdyTulu3-Block-FT/block_3_last_42.jsonl",
        "8B-Block+GraphKV": "ldsjmdyTulu3-Block-FT/gapemp_graph_3_first_42.jsonl",
    },
}
