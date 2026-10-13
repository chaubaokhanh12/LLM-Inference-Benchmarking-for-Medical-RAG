Backend `ollama` / model `medrag-qwen2.5-3b-gpu` (tag `ollama-qwen2.5-3b`), judge `gpt-4.1-mini-2025-04-14`

| Metric | Group | Value | 95% CI | n |
| --- | --- | --- | --- | --- |
| faithfulness | answerable/all | 0.978 | [0.958, 0.995] | 100 |
| relevance (1-5) | answerable/all | 4.940 | [4.880, 4.990] | 100 |
| correctness (1-5) | answerable/all | 4.690 | [4.550, 4.810] | 100 |
| language match | answerable/all | 1.000 | [1.000, 1.000] | 100 |
| has valid citation | answerable/all | 1.000 | [1.000, 1.000] | 100 |
| cites missing passage | answerable/all | 0.010 | [0.000, 0.030] | 100 |
| abstained (bad) | answerable/all | 0.000 | [0.000, 0.000] | 100 |
| gold chunk in context | answerable/all | 1.000 | [1.000, 1.000] | 100 |
| faithfulness | answerable/en | 0.987 | [0.967, 1.000] | 50 |
| relevance (1-5) | answerable/en | 4.980 | [4.940, 5.000] | 50 |
| correctness (1-5) | answerable/en | 4.840 | [4.740, 4.920] | 50 |
| language match | answerable/en | 1.000 | [1.000, 1.000] | 50 |
| has valid citation | answerable/en | 1.000 | [1.000, 1.000] | 50 |
| cites missing passage | answerable/en | 0.000 | [0.000, 0.000] | 50 |
| abstained (bad) | answerable/en | 0.000 | [0.000, 0.000] | 50 |
| gold chunk in context | answerable/en | 1.000 | [1.000, 1.000] | 50 |
| faithfulness | answerable/vi | 0.970 | [0.930, 1.000] | 50 |
| relevance (1-5) | answerable/vi | 4.900 | [4.780, 4.980] | 50 |
| correctness (1-5) | answerable/vi | 4.540 | [4.280, 4.760] | 50 |
| language match | answerable/vi | 1.000 | [1.000, 1.000] | 50 |
| has valid citation | answerable/vi | 1.000 | [1.000, 1.000] | 50 |
| cites missing passage | answerable/vi | 0.020 | [0.000, 0.060] | 50 |
| abstained (bad) | answerable/vi | 0.000 | [0.000, 0.000] | 50 |
| gold chunk in context | answerable/vi | 1.000 | [1.000, 1.000] | 50 |
| faithfulness | crosslingual | 0.924 | [0.828, 0.995] | 33 |
| relevance (1-5) | crosslingual | 3.925 | [3.424, 4.400] | 40 |
| correctness (1-5) | crosslingual | 3.375 | [2.850, 3.900] | 40 |
| language match | crosslingual | 0.975 | [0.925, 1.000] | 40 |
| has valid citation | crosslingual | 0.970 | [0.909, 1.000] | 33 |
| cites missing passage | crosslingual | 0.025 | [0.000, 0.075] | 40 |
| abstained (bad) | crosslingual | 0.175 | [0.075, 0.300] | 40 |
| gold chunk in context | crosslingual | 0.300 | [0.174, 0.450] | 40 |
| abstained (good) | unanswerable | 0.900 | [0.800, 0.980] | 50 |
| language match | unanswerable | 0.960 | [0.900, 1.000] | 50 |

Latency (median over requests that called the LLM)

| retrieve | rerank | LLM TTFT | LLM total | end-to-end | LLM share |
| --- | --- | --- | --- | --- | --- |
| 79 ms | 835 ms | 2934 ms | 3959 ms | 4920 ms | 81% |
