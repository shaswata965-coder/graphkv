## RAG (Tables 2 and 3): released outputs re-scored

| task | scorer | backbone | method | column | n | re-scored | paper | diff |
|---|---|---|---|---|---|---|---|---|
| nqa | nqa | 8B-SFT | Sequential | nqa | 3610 | 60.60 | 60.60 | +0.00 |
| nqa | nqa | 8B-RAG | Sequential | nqa | 3610 | 62.38 | 62.38 | +0.00 |
| nqa | nqa | 8B-Block | Sequential | nqa | 3610 | 63.21 | 63.21 | +0.00 |
| nqa | nqa | 8B-Block | Block-RAG | nqa | 3610 | 59.39 | 59.39 | +0.00 |
| nqa | nqa | 8B-Block | Graph-KV Top-1 | nqa | 3610 | 62.04 | 62.04 | +0.00 |
| nqa | nqa | 8B-Block | Graph-KV Top-5 (not in paper) | nqa | 3610 | 62.18 |  |  |
| nqa | nqa | 8B-Block | Graph-KV Full | nqa | 3610 | 62.88 | 62.88 | +0.00 |
| 2wiki | 2wiki | 8B-SFT | Sequential | 2wiki/compositional | 12576 | 67.20 | 67.20 | +0.00 |
| 2wiki | 2wiki | 8B-SFT | Sequential | 2wiki/comparison | 12576 | 86.84 | 86.84 | +0.00 |
| 2wiki | 2wiki | 8B-SFT | Sequential | 2wiki/inference | 12576 | 24.66 | 24.66 | +0.00 |
| 2wiki | 2wiki | 8B-SFT | Sequential | 2wiki/bridge_comparison | 12576 | 74.62 | 74.62 | +0.00 |
| 2wiki | 2wiki | 8B-SFT | Sequential | 2wiki/all | 12576 | 68.33 | 68.33 | +0.00 |
| 2wiki | 2wiki | 8B-RAG | Sequential | 2wiki/compositional | 12576 | 68.22 | 68.22 | +0.00 |
| 2wiki | 2wiki | 8B-RAG | Sequential | 2wiki/comparison | 12576 | 82.89 | 82.29 | +0.60 |
| 2wiki | 2wiki | 8B-RAG | Sequential | 2wiki/inference | 12576 | 60.94 | 60.94 | +0.00 |
| 2wiki | 2wiki | 8B-RAG | Sequential | 2wiki/bridge_comparison | 12576 | 90.54 | 90.54 | +0.00 |
| 2wiki | 2wiki | 8B-RAG | Sequential | 2wiki/all | 12576 | 75.75 | 75.75 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Sequential | 2wiki/compositional | 12576 | 70.76 | 70.76 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Sequential | 2wiki/comparison | 12576 | 84.24 | 84.24 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Sequential | 2wiki/inference | 12576 | 63.00 | 63.08 | -0.08 |
| 2wiki | 2wiki | 8B-Block | Sequential | 2wiki/bridge_comparison | 12576 | 90.07 | 90.07 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Sequential | 2wiki/all | 12576 | 77.29 | 77.29 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Block-RAG | 2wiki/compositional | 12576 | 60.90 | 60.90 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Block-RAG | 2wiki/comparison | 12576 | 83.25 | 83.25 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Block-RAG | 2wiki/inference | 12576 | 50.41 | 50.41 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Block-RAG | 2wiki/bridge_comparison | 12576 | 89.85 | 89.85 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Block-RAG | 2wiki/all | 12576 | 71.35 | 71.35 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-1 | 2wiki/compositional | 12576 | 65.94 | 65.94 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-1 | 2wiki/comparison | 12576 | 82.76 | 82.76 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-1 | 2wiki/inference | 12576 | 52.42 | 52.42 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-1 | 2wiki/bridge_comparison | 12576 | 90.00 | 90.00 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-1 | 2wiki/all | 12576 | 73.60 | 73.60 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-3 | 2wiki/compositional | 12576 | 67.83 | 67.83 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-3 | 2wiki/comparison | 12576 | 83.25 | 83.25 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-3 | 2wiki/inference | 12576 | 56.74 | 56.74 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-3 | 2wiki/bridge_comparison | 12576 | 90.47 | 90.47 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-3 | 2wiki/all | 12576 | 75.15 | 75.15 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-5 (not in paper) | 2wiki/compositional | 12576 | 67.51 |  |  |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-5 (not in paper) | 2wiki/comparison | 12576 | 82.96 |  |  |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-5 (not in paper) | 2wiki/inference | 12576 | 56.48 |  |  |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-5 (not in paper) | 2wiki/bridge_comparison | 12576 | 90.47 |  |  |
| 2wiki | 2wiki | 8B-Block | Graph-KV Top-5 (not in paper) | 2wiki/all | 12576 | 74.91 |  |  |
| 2wiki | 2wiki | 8B-Block | Graph-KV Full | 2wiki/compositional | 12576 | 67.11 | 67.11 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Full | 2wiki/comparison | 12576 | 82.69 | 82.69 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Full | 2wiki/inference | 12576 | 54.16 | 54.16 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Full | 2wiki/bridge_comparison | 12576 | 90.43 | 90.43 | +0.00 |
| 2wiki | 2wiki | 8B-Block | Graph-KV Full | 2wiki/all | 12576 | 74.38 | 74.38 | +0.00 |
| tqa | tqa | 8B-SFT | Sequential | tqa | 11313 | 76.19 | 76.19 | +0.00 |
| tqa | tqa | 8B-RAG | Sequential | tqa | 11313 | 76.38 | 76.38 | +0.00 |
| tqa | tqa | 8B-Block | Sequential | tqa | 11313 | 76.64 | 76.64 | +0.00 |
| tqa | tqa | 8B-Block | Block-RAG | tqa | 11313 | 74.44 | 74.44 | +0.00 |
| tqa | tqa | 8B-Block | Graph-KV Top-1 | tqa | 11313 | 75.17 | 75.17 | +0.00 |
| tqa | tqa | 8B-Block | Graph-KV Top-3 | tqa | 11313 | 75.66 | 75.66 | +0.00 |
| tqa | tqa | 8B-Block | Graph-KV Top-5 (not in paper) | tqa | 11313 | 76.16 |  |  |
| tqa | tqa | 8B-Block | Graph-KV Full | tqa | 11313 | 75.85 | 75.85 | +0.00 |
| hqa | hqa | 8B-SFT | Sequential | hqa/comparison | 7405 | 74.37 | 74.37 | +0.00 |
| hqa | hqa | 8B-SFT | Sequential | hqa/bridge | 7405 | 73.18 | 73.18 | +0.00 |
| hqa | hqa | 8B-SFT | Sequential | hqa/all | 7405 | 73.42 | 73.42 | +0.00 |
| hqa | hqa | 8B-RAG | Sequential | hqa/comparison | 7405 | 73.43 | 73.43 | +0.00 |
| hqa | hqa | 8B-RAG | Sequential | hqa/bridge | 7405 | 78.59 | 78.59 | +0.00 |
| hqa | hqa | 8B-RAG | Sequential | hqa/all | 7405 | 77.55 | 77.55 | +0.00 |
| hqa | hqa | 8B-Block | Sequential | hqa/all | 13 | 84.61 | 77.43 | +7.18 |
| hqa | hqa | 8B-Block | Block-RAG | hqa/comparison | 7405 | 70.20 | 70.20 | +0.00 |
| hqa | hqa | 8B-Block | Block-RAG | hqa/bridge | 7405 | 71.59 | 71.59 | +0.00 |
| hqa | hqa | 8B-Block | Block-RAG | hqa/all | 7405 | 71.31 | 71.31 | +0.00 |
| hqa | hqa | 8B-Block | Graph-KV Top-1 | hqa/comparison | 7405 | 70.41 | 70.41 | +0.00 |
| hqa | hqa | 8B-Block | Graph-KV Top-1 | hqa/bridge | 7405 | 74.14 | 74.14 | +0.00 |
| hqa | hqa | 8B-Block | Graph-KV Top-1 | hqa/all | 7405 | 73.39 | 73.39 | +0.00 |
| hqa | hqa | 8B-Block | Graph-KV Top-3 | hqa/comparison | 7405 | 71.01 | 71.01 | +0.00 |
| hqa | hqa | 8B-Block | Graph-KV Top-3 | hqa/bridge | 7405 | 76.24 | 76.24 | +0.00 |
| hqa | hqa | 8B-Block | Graph-KV Top-3 | hqa/all | 7405 | 75.19 | 75.19 | +0.00 |
| hqa | hqa | 8B-Block | Graph-KV Top-5 (not in paper) | hqa/comparison | 7405 | 70.88 |  |  |
| hqa | hqa | 8B-Block | Graph-KV Top-5 (not in paper) | hqa/bridge | 7405 | 77.05 |  |  |
| hqa | hqa | 8B-Block | Graph-KV Top-5 (not in paper) | hqa/all | 7405 | 75.81 |  |  |
| hqa | hqa | 8B-Block | Graph-KV Full | hqa/comparison | 7405 | 70.41 | 70.41 | +0.00 |
| hqa | hqa | 8B-Block | Graph-KV Full | hqa/bridge | 7405 | 77.62 | 77.62 | +0.00 |
| hqa | hqa | 8B-Block | Graph-KV Full | hqa/all | 7405 | 76.17 | 76.17 | +0.00 |
| multihopqa | multihopqa | 8B-SFT | Sequential | multihopqa/inference_query | 2556 | 32.47 | 42.54 | -10.07 |
| multihopqa | multihopqa | 8B-SFT | Sequential | multihopqa/comparison_query | 2556 | 0.70 | 33.49 | -32.79 |
| multihopqa | multihopqa | 8B-SFT | Sequential | multihopqa/null_query | 2556 | 99.66 | 99.34 | +0.32 |
| multihopqa | multihopqa | 8B-SFT | Sequential | multihopqa/temporal_query | 2556 | 0.17 | 33.37 | -33.20 |
| multihopqa | multihopqa | 8B-SFT | Sequential | multihopqa/all | 2556 | 22.37 | 39.18 | -16.81 |
| multihopqa | multihopqa | 8B-RAG | Sequential | multihopqa/inference_query | 2556 | 81.61 | 73.12 | +8.49 |
| multihopqa | multihopqa | 8B-RAG | Sequential | multihopqa/comparison_query | 2556 | 5.72 | 34.66 | -28.94 |
| multihopqa | multihopqa | 8B-RAG | Sequential | multihopqa/null_query | 2556 | 34.88 | 43.43 | -8.55 |
| multihopqa | multihopqa | 8B-RAG | Sequential | multihopqa/temporal_query | 2556 | 21.78 | 39.00 | -17.22 |
| multihopqa | multihopqa | 8B-RAG | Sequential | multihopqa/all | 2556 | 37.05 | 44.27 | -7.22 |
| multihopqa | multihopqa | 8B-Block | Sequential | multihopqa/inference_query | 2556 | 71.32 | 63.55 | +7.77 |
| multihopqa | multihopqa | 8B-Block | Sequential | multihopqa/comparison_query | 2556 | 5.25 | 34.54 | -29.29 |
| multihopqa | multihopqa | 8B-Block | Sequential | multihopqa/null_query | 2556 | 55.81 | 53.09 | +2.72 |
| multihopqa | multihopqa | 8B-Block | Sequential | multihopqa/temporal_query | 2556 | 18.52 | 38.03 | -19.51 |
| multihopqa | multihopqa | 8B-Block | Sequential | multihopqa/all | 2556 | 35.32 | 43.60 | -8.28 |
| multihopqa | multihopqa | 8B-Block | Block-RAG | multihopqa/inference_query | 2556 | 81.25 | 72.73 | +8.52 |
| multihopqa | multihopqa | 8B-Block | Block-RAG | multihopqa/comparison_query | 2556 | 3.62 | 34.16 | -30.54 |
| multihopqa | multihopqa | 8B-Block | Block-RAG | multihopqa/null_query | 2556 | 27.24 | 40.73 | -13.49 |
| multihopqa | multihopqa | 8B-Block | Block-RAG | multihopqa/temporal_query | 2556 | 18.52 | 38.03 | -19.51 |
| multihopqa | multihopqa | 8B-Block | Block-RAG | multihopqa/all | 2556 | 34.58 | 43.32 | -8.74 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-1 | multihopqa/inference_query | 2556 | 81.74 | 73.25 | +8.49 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-1 | multihopqa/comparison_query | 2556 | 3.73 | 34.19 | -30.46 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-1 | multihopqa/null_query | 2556 | 57.14 | 53.85 | +3.29 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-1 | multihopqa/temporal_query | 2556 | 19.03 | 38.18 | -19.15 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-1 | multihopqa/all | 2556 | 38.41 | 44.81 | -6.40 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-3 | multihopqa/inference_query | 2556 | 81.61 | 73.12 | +8.49 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-3 | multihopqa/comparison_query | 2556 | 4.78 | 34.43 | -29.65 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-3 | multihopqa/null_query | 2556 | 71.42 | 63.64 | +7.78 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-3 | multihopqa/temporal_query | 2556 | 20.75 | 38.69 | -17.94 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-3 | multihopqa/all | 2556 | 40.80 | 45.79 | -4.99 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-5 (not in paper) | multihopqa/inference_query | 2556 | 81.61 |  |  |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-5 (not in paper) | multihopqa/comparison_query | 2556 | 5.14 |  |  |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-5 (not in paper) | multihopqa/null_query | 2556 | 81.72 |  |  |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-5 (not in paper) | multihopqa/temporal_query | 2556 | 18.86 |  |  |
| multihopqa | multihopqa | 8B-Block | Graph-KV Top-5 (not in paper) | multihopqa/all | 2556 | 41.70 |  |  |
| multihopqa | multihopqa | 8B-Block | Graph-KV Full | multihopqa/inference_query | 2556 | 77.57 | 69.04 | +8.53 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Full | multihopqa/comparison_query | 2556 | 5.60 | 34.63 | -29.03 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Full | multihopqa/null_query | 2556 | 93.68 | 88.79 | +4.89 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Full | multihopqa/temporal_query | 2556 | 20.06 | 38.48 | -18.42 |
| multihopqa | multihopqa | 8B-Block | Graph-KV Full | multihopqa/all | 2556 | 42.25 | 46.41 | -4.16 |
| morehopqa | morehopqa | 8B-SFT | Sequential | morehopqa/hop5 | 1118 | 40.65 | 26.37 | +14.28 |
| morehopqa | morehopqa | 8B-SFT | Sequential | morehopqa/hop4 | 1118 | 46.15 | 15.38 | +30.77 |
| morehopqa | morehopqa | 8B-SFT | Sequential | morehopqa/hop3 | 1118 | 35.06 | 29.87 | +5.19 |
| morehopqa | morehopqa | 8B-SFT | Sequential | morehopqa/hop2 | 1118 | 31.00 | 24.27 | +6.73 |
| morehopqa | morehopqa | 8B-SFT | Sequential | morehopqa/hop1 | 1118 | 69.59 | 57.65 | +11.94 |
| morehopqa | morehopqa | 8B-SFT | Sequential | morehopqa/all | 1118 | 47.85 | 38.37 | +9.48 |
| morehopqa | morehopqa-strict | 8B-SFT | Sequential | morehopqa/hop5 | 1118 | 26.37 | 26.37 | +0.00 |
| morehopqa | morehopqa-strict | 8B-SFT | Sequential | morehopqa/hop4 | 1118 | 15.38 | 15.38 | +0.00 |
| morehopqa | morehopqa-strict | 8B-SFT | Sequential | morehopqa/hop3 | 1118 | 29.87 | 29.87 | +0.00 |
| morehopqa | morehopqa-strict | 8B-SFT | Sequential | morehopqa/hop2 | 1118 | 24.27 | 24.27 | +0.00 |
| morehopqa | morehopqa-strict | 8B-SFT | Sequential | morehopqa/hop1 | 1118 | 57.65 | 57.65 | +0.00 |
| morehopqa | morehopqa-strict | 8B-SFT | Sequential | morehopqa/all | 1118 | 38.37 | 38.37 | +0.00 |
| morehopqa | morehopqa | 8B-RAG | Sequential | morehopqa/hop5 | 1118 | 25.27 | 25.27 | +0.00 |
| morehopqa | morehopqa | 8B-RAG | Sequential | morehopqa/hop4 | 1118 | 61.53 | 53.85 | +7.68 |
| morehopqa | morehopqa | 8B-RAG | Sequential | morehopqa/hop3 | 1118 | 22.07 | 22.07 | +0.00 |
| morehopqa | morehopqa | 8B-RAG | Sequential | morehopqa/hop2 | 1118 | 24.03 | 22.59 | +1.44 |
| morehopqa | morehopqa | 8B-RAG | Sequential | morehopqa/hop1 | 1118 | 54.95 | 54.72 | +0.23 |
| morehopqa | morehopqa | 8B-RAG | Sequential | morehopqa/all | 1118 | 36.58 | 35.86 | +0.72 |
| morehopqa | morehopqa-strict | 8B-RAG | Sequential | morehopqa/hop5 | 1118 | 25.27 | 25.27 | +0.00 |
| morehopqa | morehopqa-strict | 8B-RAG | Sequential | morehopqa/hop4 | 1118 | 53.84 | 53.85 | -0.01 |
| morehopqa | morehopqa-strict | 8B-RAG | Sequential | morehopqa/hop3 | 1118 | 22.07 | 22.07 | +0.00 |
| morehopqa | morehopqa-strict | 8B-RAG | Sequential | morehopqa/hop2 | 1118 | 22.59 | 22.59 | +0.00 |
| morehopqa | morehopqa-strict | 8B-RAG | Sequential | morehopqa/hop1 | 1118 | 54.72 | 54.72 | +0.00 |
| morehopqa | morehopqa-strict | 8B-RAG | Sequential | morehopqa/all | 1118 | 35.86 | 35.86 | +0.00 |
| morehopqa | morehopqa | 8B-Block | Sequential | morehopqa/hop5 | 1118 | 41.75 | 30.76 | +10.99 |
| morehopqa | morehopqa | 8B-Block | Sequential | morehopqa/hop4 | 1118 | 46.15 | 23.07 | +23.08 |
| morehopqa | morehopqa | 8B-Block | Sequential | morehopqa/hop3 | 1118 | 27.27 | 25.32 | +1.95 |
| morehopqa | morehopqa | 8B-Block | Sequential | morehopqa/hop2 | 1118 | 28.36 | 20.43 | +7.93 |
| morehopqa | morehopqa | 8B-Block | Sequential | morehopqa/hop1 | 1118 | 68.24 | 59.23 | +9.01 |
| morehopqa | morehopqa | 8B-Block | Sequential | morehopqa/all | 1118 | 45.34 | 37.38 | +7.96 |
| morehopqa | morehopqa-strict | 8B-Block | Sequential | morehopqa/hop5 | 1118 | 30.76 | 30.76 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Sequential | morehopqa/hop4 | 1118 | 23.07 | 23.07 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Sequential | morehopqa/hop3 | 1118 | 25.32 | 25.32 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Sequential | morehopqa/hop2 | 1118 | 20.43 | 20.43 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Sequential | morehopqa/hop1 | 1118 | 59.23 | 59.23 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Sequential | morehopqa/all | 1118 | 37.38 | 37.38 | +0.00 |
| morehopqa | morehopqa | 8B-Block | Block-RAG | morehopqa/hop5 | 1118 | 35.16 | 27.47 | +7.69 |
| morehopqa | morehopqa | 8B-Block | Block-RAG | morehopqa/hop4 | 1118 | 46.15 | 15.38 | +30.77 |
| morehopqa | morehopqa | 8B-Block | Block-RAG | morehopqa/hop3 | 1118 | 27.27 | 24.67 | +2.60 |
| morehopqa | morehopqa | 8B-Block | Block-RAG | morehopqa/hop2 | 1118 | 31.00 | 25.72 | +5.28 |
| morehopqa | morehopqa | 8B-Block | Block-RAG | morehopqa/hop1 | 1118 | 63.96 | 56.75 | +7.21 |
| morehopqa | morehopqa | 8B-Block | Block-RAG | morehopqa/all | 1118 | 44.09 | 37.92 | +6.17 |
| morehopqa | morehopqa-strict | 8B-Block | Block-RAG | morehopqa/hop5 | 1118 | 27.47 | 27.47 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Block-RAG | morehopqa/hop4 | 1118 | 15.38 | 15.38 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Block-RAG | morehopqa/hop3 | 1118 | 24.67 | 24.67 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Block-RAG | morehopqa/hop2 | 1118 | 25.72 | 25.72 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Block-RAG | morehopqa/hop1 | 1118 | 56.75 | 56.75 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Block-RAG | morehopqa/all | 1118 | 37.92 | 37.92 | +0.00 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-1 | morehopqa/hop5 | 1118 | 38.46 | 31.86 | +6.60 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-1 | morehopqa/hop4 | 1118 | 38.46 | 23.07 | +15.39 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-1 | morehopqa/hop3 | 1118 | 26.62 | 24.67 | +1.95 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-1 | morehopqa/hop2 | 1118 | 31.25 | 25.72 | +5.53 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-1 | morehopqa/hop1 | 1118 | 64.41 | 54.72 | +9.69 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-1 | morehopqa/all | 1118 | 44.45 | 37.96 | +6.49 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-1 | morehopqa/hop5 | 1118 | 31.86 | 31.86 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-1 | morehopqa/hop4 | 1118 | 23.07 | 23.07 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-1 | morehopqa/hop3 | 1118 | 24.67 | 24.67 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-1 | morehopqa/hop2 | 1118 | 25.72 | 25.72 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-1 | morehopqa/hop1 | 1118 | 54.72 | 54.72 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-1 | morehopqa/all | 1118 | 37.56 | 37.96 | -0.40 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-3 | morehopqa/hop5 | 1118 | 40.65 | 29.67 | +10.98 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-3 | morehopqa/hop4 | 1118 | 38.46 | 23.07 | +15.39 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-3 | morehopqa/hop3 | 1118 | 27.27 | 24.67 | +2.60 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-3 | morehopqa/hop2 | 1118 | 28.84 | 25.72 | +3.12 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-3 | morehopqa/hop1 | 1118 | 65.76 | 56.53 | +9.23 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-3 | morehopqa/all | 1118 | 44.36 | 38.11 | +6.25 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-3 | morehopqa/hop5 | 1118 | 29.67 | 29.67 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-3 | morehopqa/hop4 | 1118 | 23.07 | 23.07 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-3 | morehopqa/hop3 | 1118 | 24.67 | 24.67 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-3 | morehopqa/hop2 | 1118 | 22.83 | 25.72 | -2.89 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-3 | morehopqa/hop1 | 1118 | 56.53 | 56.53 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-3 | morehopqa/all | 1118 | 37.03 | 38.11 | -1.08 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/hop5 | 1118 | 49.45 |  |  |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/hop4 | 1118 | 46.15 |  |  |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/hop3 | 1118 | 25.97 |  |  |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/hop2 | 1118 | 30.52 |  |  |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/hop1 | 1118 | 66.89 |  |  |
| morehopqa | morehopqa | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/all | 1118 | 46.06 |  |  |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/hop5 | 1118 | 40.65 |  |  |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/hop4 | 1118 | 30.76 |  |  |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/hop3 | 1118 | 24.67 |  |  |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/hop2 | 1118 | 25.96 |  |  |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/hop1 | 1118 | 59.68 |  |  |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Top-5 (not in paper) | morehopqa/all | 1118 | 40.42 |  |  |
| morehopqa | morehopqa | 8B-Block | Graph-KV Full | morehopqa/hop5 | 1118 | 47.25 | 47.25 | +0.00 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Full | morehopqa/hop4 | 1118 | 30.76 | 30.76 | +0.00 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Full | morehopqa/hop3 | 1118 | 25.97 | 25.97 | +0.00 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Full | morehopqa/hop2 | 1118 | 30.52 | 30.52 | +0.00 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Full | morehopqa/hop1 | 1118 | 64.86 | 64.86 | +0.00 |
| morehopqa | morehopqa | 8B-Block | Graph-KV Full | morehopqa/all | 1118 | 44.90 | 44.90 | +0.00 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Full | morehopqa/hop5 | 1118 | 39.56 | 47.25 | -7.69 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Full | morehopqa/hop4 | 1118 | 23.07 | 30.76 | -7.69 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Full | morehopqa/hop3 | 1118 | 23.37 | 25.97 | -2.60 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Full | morehopqa/hop2 | 1118 | 26.68 | 30.52 | -3.84 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Full | morehopqa/hop1 | 1118 | 57.20 | 64.86 | -7.66 |
| morehopqa | morehopqa-strict | 8B-Block | Graph-KV Full | morehopqa/all | 1118 | 39.35 | 44.90 | -5.55 |

* scorer `nqa`: 6 of 6 cells with a paper value agree within 0.01.
* scorer `2wiki`: 33 of 35 cells with a paper value agree within 0.01.
* scorer `tqa`: 7 of 7 cells with a paper value agree within 0.01.
* scorer `hqa`: 18 of 19 cells with a paper value agree within 0.01.
* scorer `multihopqa`: 0 of 35 cells with a paper value agree within 0.01.
* scorer `morehopqa`: 8 of 42 cells with a paper value agree within 0.01.
* scorer `morehopqa-strict`: 33 of 42 cells with a paper value agree within 0.01.

## ArXiv-QA (Figure 6): correct answers out of 60

| distractors | run | file | re-scored | paper |
|---|---|---|---|---|
| 0 | 8B-SFT+SEQ | ldsjmdyTulu3-SFT/vanilla_1_last_42.jsonl | 27 | 27 |
| 0 | 8B-RAG+SEQ | ldsjmdyTulu3-RAG/vanilla_1_last_42.jsonl | 25 | 25 |
| 0 | 8B-Block+SEQ | ldsjmdyTulu3-Block-FT/vanilla_1_last_42.jsonl | 30 | 30 |
| 0 | 8B-Block+Block-RAG | ldsjmdyTulu3-Block-FT/block_1_last_42.jsonl | 7 | 7 |
| 0 | 8B-Block+GraphKV | ldsjmdyTulu3-Block-FT/gapemp_graph_1_last_42.jsonl | 27 | 27 |
| 1 | 8B-SFT+SEQ+First | ldsjmdyTulu3-SFT/vanilla_2_first_42.jsonl | 9 | 9 |
| 1 | 8B-SFT+SEQ+Last | ldsjmdyTulu3-SFT/vanilla_2_last_42.jsonl | 12 | 12 |
| 1 | 8B-RAG+SEQ+First | ldsjmdyTulu3-RAG/vanilla_2_first_42.jsonl | 7 | 7 |
| 1 | 8B-RAG+SEQ+Last | ldsjmdyTulu3-RAG/vanilla_2_last_42.jsonl | 18 | 18 |
| 1 | 8B-Block+SEQ+First | ldsjmdyTulu3-Block-FT/vanilla_2_first_42.jsonl | 7 | 7 |
| 1 | 8B-Block+SEQ+Last | ldsjmdyTulu3-Block-FT/vanilla_2_last_42.jsonl | 14 | 14 |
| 1 | 8B-Block+Block-RAG+First | ldsjmdyTulu3-Block-FT/block_2_first_42.jsonl | 3 | 3 |
| 1 | 8B-Block+Block-RAG+Last | ldsjmdyTulu3-Block-FT/block_2_last_42.jsonl | 5 | 5 |
| 1 | 8B-Block+GraphKV | ldsjmdyTulu3-Block-FT/gapemp_graph_2_first_42.jsonl | 17 | 17 |
| 2 | 8B-SFT+SEQ+First | ldsjmdyTulu3-SFT/vanilla_3_first_42.jsonl | 0 | 0 |
| 2 | 8B-SFT+SEQ+Last | ldsjmdyTulu3-SFT/vanilla_3_last_42.jsonl | 6 | 6 |
| 2 | 8B-RAG+SEQ+First | ldsjmdyTulu3-RAG/vanilla_3_first_42.jsonl | 1 | 1 |
| 2 | 8B-RAG+SEQ+Last | ldsjmdyTulu3-RAG/vanilla_3_last_42.jsonl | 4 | 4 |
| 2 | 8B-Block+SEQ+First | ldsjmdyTulu3-Block-FT/vanilla_3_first_42.jsonl | 0 | 0 |
| 2 | 8B-Block+SEQ+Last | ldsjmdyTulu3-Block-FT/vanilla_3_last_42.jsonl | 9 | 9 |
| 2 | 8B-Block+Block-RAG+First | ldsjmdyTulu3-Block-FT/block_3_first_42.jsonl | 0 | 0 |
| 2 | 8B-Block+Block-RAG+Last | ldsjmdyTulu3-Block-FT/block_3_last_42.jsonl | 0 | 0 |
| 2 | 8B-Block+GraphKV | ldsjmdyTulu3-Block-FT/gapemp_graph_3_first_42.jsonl | 9 | 9 |
