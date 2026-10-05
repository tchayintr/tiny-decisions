# Results

Accuracy in %. Every number is read from the JSON files next to this one.

## Zero-shot: after decision training, no examples of the held-out tasks

| model | params | Banking77 | TREC | SMS spam | Emotion | AG News | average |
|---|---:|---:|---:|---:|---:|---:|---:|
| random guess | – | 1.3 | 16.7 | 50.0 | 16.7 | 25.0 | **21.9** |
| from scratch, xxs | 0.6M | 3.4 | 26.8 | 49.4 | 28.8 | 58.1 | **33.3** |
| from scratch, xs | 1.9M | 9.6 | 19.0 | 53.0 | 38.3 | 54.1 | **34.8** |
| from scratch, s | 6.8M | 20.0 | 38.2 | 48.1 | 42.0 | 61.5 | **41.9** |
| from scratch, m | 17M | 18.7 | 20.8 | 62.1 | 45.3 | 64.1 | **42.2** |
| from scratch, l | 29M | 30.4 | 35.0 | 50.5 | 43.5 | 66.7 | **45.2** |
| Ettin start, ettin-17m | 17M | 43.3 | 22.6 | 49.6 | 50.6 | 62.0 | **45.6** |
| Ettin start, ettin-32m | 32M | 51.7 | 40.0 | 49.7 | 53.9 | 71.0 | **53.3** |
| Ettin start, ettin-68m | 68M | 57.0 | 31.0 | 49.6 | 55.6 | 73.7 | **53.4** |
| Ettin start, ettin-150m | 150M | 61.4 | 44.4 | 49.6 | 53.9 | 75.3 | **56.9** |
| Ettin start, ettin-400m | 396M | 64.7 | 51.8 | 49.9 | 57.8 | 79.2 | **60.7** |
| Qwen3.5-0.8B, zero-shot | 0.8B | 24.5 | 62.0 | 60.0 | 57.4 | 70.7 | **54.9** |
| Qwen3.5-9B, zero-shot | 9B | 69.6 | 86.4 | 93.4 | 56.6 | 86.6 | **78.5** |

## SMS spam: fine-tuning on k examples from the train split, accuracy on the 747-item test split (mean of 3 draws)

| model | k=0 | k=8 | k=32 | k=128 |
|---|---:|---:|---:|---:|
| from scratch, xxs | 49.4 | 73.2 | 88.1 | 93.3 |
| from scratch, xs | 53.0 | 85.6 | 92.2 | 94.3 |
| from scratch, s | 48.7 | 80.6 | 91.8 | 94.5 |
| from scratch, m | 61.3 | 89.5 | 94.1 | 94.7 |
| from scratch, l | 50.2 | 89.0 | 91.6 | 94.7 |
| Ettin start, ettin-17m | 49.7 | 78.7 | 93.4 | 94.6 |
| Ettin start, ettin-32m | 49.7 | 88.7 | 93.5 | 95.2 |
| Ettin start, ettin-68m | 49.8 | 81.4 | 91.6 | 95.6 |
| Ettin start, ettin-150m | 49.7 | 86.8 | 94.9 | 94.9 |
| Ettin start, ettin-400m | 50.3 | 92.4 | 93.0 | 96.2 |
| TF-IDF + logistic regression, same examples | – | 61.7 | 86.5 | 95.0 |
| Qwen3.5-0.8B, same examples in its prompt | 58.5 | 61.8 | 79.3 | – |
| Laya, English checkpoint | 92.5 | – | – | – |
| Bekko System One v0 17M (task in its training data) | 67.2 | – | – | – |

## TREC: fine-tuning on k examples from the train split, accuracy on the 250-item test split (mean of 3 draws)

| model | k=0 | k=8 | k=32 | k=128 |
|---|---:|---:|---:|---:|
| from scratch, xxs | 25.2 | 27.1 | 39.6 | 68.5 |
| from scratch, xs | 19.2 | 33.6 | 51.3 | 77.3 |
| from scratch, s | 37.2 | 40.8 | 52.7 | 80.7 |
| from scratch, m | 18.8 | 38.1 | 50.7 | 77.9 |
| from scratch, l | 32.4 | 37.2 | 58.0 | 76.7 |
| Ettin start, ettin-17m | 21.2 | 46.8 | 74.4 | 86.4 |
| Ettin start, ettin-32m | 37.2 | 46.9 | 76.7 | 88.7 |
| Ettin start, ettin-68m | 31.6 | 61.3 | 81.2 | 92.1 |
| Ettin start, ettin-150m | 45.6 | 61.6 | 82.8 | 93.5 |
| Ettin start, ettin-400m | 50.4 | 72.0 | 88.0 | 93.1 |
| TF-IDF + logistic regression, same examples | – | 37.6 | 59.1 | 73.6 |
| Qwen3.5-0.8B, same examples in its prompt | 60.8 | 65.9 | 75.2 | – |
| Laya, English checkpoint | 81.6 | – | – | – |
| Bekko System One v0 17M (task not in its training data) | 46.4 | – | – | – |

## Size: weights rounded to 8 bits

| model | params | MB at 8 bits | zero-shot average | after rounding to 8 bits |
|---|---:|---:|---:|---:|
| from scratch, xxs | 0.6M | 0.6 | 33.3 | 33.3 |
| from scratch, xs | 1.9M | 1.9 | 34.8 | 34.9 |
| from scratch, s | 6.8M | 6.8 | 41.9 | 42.0 |
| from scratch, m | 17M | 17.4 | 42.2 | 42.3 |
| from scratch, l | 29M | 29.4 | 45.2 | 45.2 |
| Ettin start, ettin-17m | 17M | 16.9 | 45.6 | 45.6 |
| Ettin start, ettin-32m | 32M | 32.0 | 53.3 | 52.9 |
| Ettin start, ettin-68m | 68M | 68.4 | 53.4 | 53.4 |
| Ettin start, ettin-150m | 150M | 149.6 | 56.9 | 56.9 |
| Ettin start, ettin-400m | 396M | 395.8 | 60.7 | 60.4 |

## Speed: time per decision on a CPU and a GPU

The same Banking77 decisions with 5 answers offered, batch 1, tokenization included, the answer read back; median ms per decision. On the CPU every model runs in fp32. On the GPU, ours and Laya run in eager PyTorch (Laya through its own code, in bf16) and Qwen3.5-0.8B in vLLM (bf16, CUDA graphs, structured output), as in the accuracy runs.

| model | Intel(R) Xeon(R) Platinum 8468, 1 thread | NVIDIA H100 80GB HBM3 |
|---|---:|---:|
| Decisions Model, ettin-17m | 6.03 | 3.34 |
| Laya, English checkpoint | 394.7 | 15.8 |
| Qwen3.5-0.8B, zero-shot | 1,650.4 | 31.5 |
| Bekko System One v0 17M, its own code (timed in a separate run) | 14.72 | – |

Jev (TypeSafe AI) is a closed API and is not timed here: third parties report 236–276 ms p50 per call, network included. Bekko's GPU time is not measured here (the shared GPU was busy); its model card reports 4.6 ms eager on an RTX 5090.

## Overlap check: ettin-17m after fine-tuning (mean of 3 draws)

| task | k | on its k examples | test split | test items with no near-copy in the train split | TF-IDF on its k examples | TF-IDF, test split |
|---|---:|---:|---:|---:|---:|---:|
| SMS spam | 8 | 100.0 | 78.7 (n=747) | 77.2 (n=667) | 100.0 | 61.7 |
| SMS spam | 32 | 100.0 | 93.4 (n=747) | 92.9 (n=667) | 100.0 | 86.5 |
| SMS spam | 128 | 100.0 | 94.6 (n=747) | 94.1 (n=667) | 100.0 | 95.0 |
| TREC | 8 | 100.0 | 46.8 (n=250) | 46.8 (n=250) | 100.0 | 37.6 |
| TREC | 32 | 100.0 | 74.4 (n=250) | 74.4 (n=250) | 100.0 | 59.1 |
| TREC | 128 | 100.0 | 86.3 (n=250) | 86.3 (n=250) | 100.0 | 73.6 |
