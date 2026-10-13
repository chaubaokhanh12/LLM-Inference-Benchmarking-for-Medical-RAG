Backend `ollama` / model `medrag-qwen2.5-3b-gpu` (tag `ollama-qwen2.5-3b`), judge `gpt-4.1-mini-2025-04-14`

| Metric | Group | Value | 95% CI | n |
| --- | --- | --- | --- | --- |
| faithfulness | answerable/all | 0.975 | [0.954, 0.993] | 100 |
| relevance (1-5) | answerable/all | 4.980 | [4.950, 5.000] | 100 |
| correctness (1-5) | answerable/all | 4.780 | [4.670, 4.880] | 100 |
| language match | answerable/all | 1.000 | [1.000, 1.000] | 100 |
| has valid citation | answerable/all | 0.330 | [0.240, 0.420] | 100 |
| cites missing passage | answerable/all | 0.000 | [0.000, 0.000] | 100 |
| abstained (bad) | answerable/all | 0.000 | [0.000, 0.000] | 100 |
| gold chunk in context | answerable/all | 1.000 | [1.000, 1.000] | 100 |
| faithfulness | answerable/en | 0.989 | [0.974, 1.000] | 50 |
| relevance (1-5) | answerable/en | 5.000 | [5.000, 5.000] | 50 |
| correctness (1-5) | answerable/en | 4.940 | [4.860, 5.000] | 50 |
| language match | answerable/en | 1.000 | [1.000, 1.000] | 50 |
| has valid citation | answerable/en | 0.360 | [0.220, 0.500] | 50 |
| cites missing passage | answerable/en | 0.000 | [0.000, 0.000] | 50 |
| abstained (bad) | answerable/en | 0.000 | [0.000, 0.000] | 50 |
| gold chunk in context | answerable/en | 1.000 | [1.000, 1.000] | 50 |
| faithfulness | answerable/vi | 0.962 | [0.922, 0.990] | 50 |
| relevance (1-5) | answerable/vi | 4.960 | [4.900, 5.000] | 50 |
| correctness (1-5) | answerable/vi | 4.620 | [4.400, 4.800] | 50 |
| language match | answerable/vi | 1.000 | [1.000, 1.000] | 50 |
| has valid citation | answerable/vi | 0.300 | [0.180, 0.420] | 50 |
| cites missing passage | answerable/vi | 0.000 | [0.000, 0.000] | 50 |
| abstained (bad) | answerable/vi | 0.000 | [0.000, 0.000] | 50 |
| gold chunk in context | answerable/vi | 1.000 | [1.000, 1.000] | 50 |
| faithfulness | crosslingual | 0.935 | [0.857, 0.990] | 32 |
| relevance (1-5) | crosslingual | 3.925 | [3.400, 4.400] | 40 |
| correctness (1-5) | crosslingual | 3.425 | [2.925, 3.900] | 40 |
| language match | crosslingual | 0.950 | [0.875, 1.000] | 40 |
| has valid citation | crosslingual | 0.531 | [0.374, 0.719] | 32 |
| cites missing passage | crosslingual | 0.000 | [0.000, 0.000] | 40 |
| abstained (bad) | crosslingual | 0.200 | [0.075, 0.325] | 40 |
| gold chunk in context | crosslingual | 0.300 | [0.174, 0.450] | 40 |
| abstained (good) | unanswerable | 0.900 | [0.800, 0.980] | 50 |
| language match | unanswerable | 0.960 | [0.900, 1.000] | 50 |

Latency (median over requests that called the LLM)

| retrieve | rerank | LLM TTFT | LLM total | end-to-end | LLM share |
| --- | --- | --- | --- | --- | --- |
| 33 ms | 836 ms | 2942 ms | 4271 ms | 5120 ms | 83% |
