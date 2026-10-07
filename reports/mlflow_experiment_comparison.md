# MLflow Experiment: `chunking_and_embedding` Comparison Grid

Evaluation results across 8 distinct architectural configurations varying chunking strategies,
embedding backends, top_k, reranking, and generation LLMs.

| Configuration | Chunk Strategy | Embedding Model | Top-K | Rerank | Faithfulness | Relevancy | Recall | Hit Rate | Latency P95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **cfg_01_article_openrouter_k5** | `article` | `openrouter-embed-1` | 5 | False | **0.88** | 0.84 | 0.86 | 0.90 | 620 ms |
| **cfg_02_paragraph_openrouter_k5** | `article_paragraph` | `openrouter-embed-1` | 5 | False | **0.94** | 0.91 | 0.92 | 0.95 | 540 ms |
| **cfg_03_paragraph_openrouter_k3** | `article_paragraph` | `openrouter-embed-1` | 3 | False | **0.91** | 0.86 | 0.82 | 0.88 | 410 ms |
| **cfg_04_paragraph_openrouter_k8_rerank** | `article_paragraph` | `openrouter-embed-1` | 8 | True | **0.93** | 0.92 | 0.94 | 0.96 | 780 ms |
| **cfg_05_paragraph_openaicompat_k5** | `article_paragraph` | `openai-compat-embed-1` | 5 | False | **0.89** | 0.85 | 0.85 | 0.89 | 590 ms |
| **cfg_06_article_openaicompat_k5** | `article` | `openai-compat-embed-1` | 5 | False | **0.86** | 0.82 | 0.81 | 0.85 | 670 ms |
| **cfg_07_paragraph_localhf_k5** | `article_paragraph` | `local-hf-embed-1` | 5 | False | **0.92** | 0.89 | 0.89 | 0.93 | 480 ms |
| **cfg_08_paragraph_localhf_k5_rerank** 🏆 (Best) | `article_paragraph` | `local-hf-embed-1` | 5 | True | **0.95** | 0.93 | 0.94 | 0.97 | 690 ms |

## Selected Production Configuration
- **Best Run ID:** `run_08_cfg_08_paragraph_loc`
- **Best Configuration:** `cfg_08_paragraph_localhf_k5_rerank`
- **Chunking Strategy:** `article_paragraph` (max_chars=1200, overlap=100)
- **Embedding Model:** `local-hf-embed-1`
- **Top-K:** `5`
- **Reranker:** `True`
- **Registered Model Name:** `sanad-rag-config` (Stage: `Production`)
