# Sanad (سند) — Operational Runbook

This guide covers operational workflows for deploying, evaluating, monitoring, and extending the Sanad Legal Q&A platform.

---

## 1. Quickstart & Local Setup

### 1.1 Prerequisites
- **Python**: 3.11+
- **Node.js**: 20+ (with npm 10+)
- **Docker & Docker Compose**: v2.20+
- **DVC**: 3.x+

### 1.2 Environment Configuration
Clone the repository and copy the environment template:
```bash
cp .env.example .env
```
Edit `.env` and add your LLM API credentials:
```ini
OPENROUTER_API_KEY=sk-or-v1-...
ADMIN_SECRET_KEY=supersecretadminkey
```

---

## 2. Running the System

### Option A: Complete Docker Compose Stack (Recommended)
Launch all 7 services (Backend, Frontend, Qdrant, MLflow, Langfuse, Prometheus, Grafana):
```bash
docker compose -f infra/docker-compose.yml up -d --build
```

Verify service health using the automated sanity check script:
```bash
./infra/check_services.sh
```

**Port Mappings:**
| Service | URL | Credentials |
| :--- | :--- | :--- |
| **Frontend UI** | `http://localhost:3000` | - |
| **Backend API** | `http://localhost:8000/docs` | - |
| **Qdrant Vector DB** | `http://localhost:6333/dashboard` | - |
| **MLflow Server** | `http://localhost:5000` | - |
| **Grafana Dashboards** | `http://localhost:3002` | `admin` / `admin` |
| **Langfuse Tracing** | `http://localhost:3001` | - |
| **Prometheus Metrics**| `http://localhost:9090` | - |

---

### Option B: Local Development
Install dependencies:
```bash
make setup
```

Run the backend:
```bash
uvicorn sanad.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Run the frontend:
```bash
cd frontend && npm install && npm run dev
```

---

## 3. Corpus Management & Ingestion

### 3.1 Extract and Validate Articles
Extract 1,149 articles from source PDF:
```bash
python -m sanad.cli extract \
  --pdf-path data/raw/egyptian_civil_code_1948.pdf \
  --output-dir data/processed \
  --report-path reports/corpus_spotcheck.md
```

Validate corpus schema and integrity:
```bash
python -m sanad.cli validate --articles-path data/processed/articles.json
```

### 3.2 Index into Qdrant Vector DB
Index articles into a dedicated partitioned collection:
```bash
python -m sanad.cli ingest --embedding-model openrouter-embed-1
```
*(Use `--force` to rebuild an existing collection).*

### 3.3 Reproduce Entire Pipeline via DVC
Execute the end-to-end reproducible DVC workflow:
```bash
dvc repro
```

---

## 4. Evaluation & Quality Gates

### 4.1 Run RAGAS Quality Gate
Execute the automated 20-sample CI evaluation gate:
```bash
python -m sanad.cli eval --subset ci --gate --threshold 0.75
```
*Expected Output:*
- Exit code `0` if Faithfulness >= 0.75.
- Reports written to `reports/ragas_results.json` and `reports/ragas_results.md`.

### 4.2 Run MLflow Grid Experimentation
Evaluate 8 architectural configurations across chunking strategies and embedding backends:
```bash
python backend/src/sanad/evaluation/mlflow_experiment.py
```
*Outputs:*
- Registers top configuration as `sanad-rag-config` and promotes to `Production`.
- Saves comparison table to `reports/mlflow_experiment_comparison.md`.
- Saves interactive HTML report to `reports/mlflow_experiment_comparison.html`.

### 4.3 Detect Embedding Drift
Evaluate query distribution divergence against the statutory centroid:
```bash
python -m sanad.cli drift --threshold 0.45
```
*Generates:* `reports/drift_report.json` and `reports/drift_report.md`.

---

## 5. How to Add a New Model (Config-Only)

Adding a new chat or embedding model requires **zero code changes**—modify `config/models.yaml` only.

### Example: Adding a New Chat Model
```yaml
chat_models:
  - id: custom-deepseek-v3
    label: "DeepSeek V3 (Custom)"
    provider: openrouter
    model: "deepseek/deepseek-chat"
    params: { temperature: 0.1, max_tokens: 1500 }
    pricing: { input_per_1m: 0.14, output_per_1m: 0.28 }
    supports_streaming: true
    enabled: true
```

### Example: Adding a New Embedding Model
```yaml
embedding_models:
  - id: custom-bge-large-ar
    label: "BAAI BGE Large Arabic"
    provider: local_hf
    model: "BAAI/bge-large-zh-v1.5"
    dimension: 1024
    batch_size: 32
    enabled: true
```
Upon restart, the backend automatically registers the model, exposes it via `/models`, and partitions Qdrant collections accordingly.

---

## 6. Canary Deployment & Traffic Management

Production traffic routing between stable and canary releases is governed by `infra/nginx/canary.conf`.

### 6.1 Rollout Stages
1. **Stage 0 (Internal Testing)**: 0% canary traffic. Developers send `X-Canary: always` HTTP header to route directly to canary candidate.
2. **Stage 1 (Initial Release)**: 5% canary / 95% stable split. Monitor Prometheus P95 latency and error rates for 1 hour.
3. **Stage 2 (Expansion)**: 25% canary / 75% stable split.
4. **Stage 3 (Pre-Cutover)**: 50% canary / 50% stable split.
5. **Stage 4 (Full Cutover)**: 100% traffic promoted to new release.

### 6.2 Rollback Triggers
Immediately abort and revert traffic to stable if:
- HTTP 5xx error rate exceeds 0.5% over a 5-minute rolling window.
- P95 latency exceeds 2,000 ms.
- RAGAS faithfulness gate drops below 0.75.
