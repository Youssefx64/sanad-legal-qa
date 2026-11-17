# AUDIT: Authenticity and Integrity Verification of Sanad Repository

Date of Audit: 2026-11-17
Auditor: Antigravity Assistant & Engineering Pair
Repository: `sanad-legal-qa` (`https://github.com/Youssefx64/sanad-legal-qa`)

---

## 1. Audit Findings by Item

### a. `backend/src/sanad/evaluation/mlflow_experiment.py`: Hardcoded Metrics Literals

**Finding: YES, metrics are hardcoded literals.**

#### Evidence & Proof:
* **File:** `backend/src/sanad/evaluation/mlflow_experiment.py` (lines 30–155)
* **Code definition:**
  ```python
  EXPERIMENT_CONFIGS: list[dict[str, Any]] = [
      {
          "name": "cfg_01_article_openrouter_k5",
          "chunk_strategy": "article",
          "max_chars": 2500,
          "overlap": 0,
          "embedding_model": "openrouter-embed-1",
          "top_k": 5,
          "rerank": False,
          "chat_model": "openrouter-chat-1",
          "faithfulness": 0.88,
          "answer_relevancy": 0.84,
          "context_precision": 0.81,
          "context_recall": 0.86,
          "hit_rate": 0.90,
          "mrr": 0.85,
          "latency_p95": 620.0,
          "tokens_per_query": 240.0,
      },
      ...
  ```
* **Logging loop (lines 209–227):**
  The script directly logs `metrics = {"faithfulness": cfg["faithfulness"], ...}` and calls `mlflow.log_metric(k, v)` on the dictionary literals without executing any queries, chunking, or evaluation.
* **Commit that introduced this:**
  * **Commit Hash:** `8fc41a5f7e6ddbf59064b3d12564a848bd4eabce`
  * **Commit Date:** Wed Oct 7 13:58:20 2026 +0300
  * **Commit Message:** `feat(mlops): implement RAGAS quality gate, MLflow experiment grid, Locust load tests, and CI/CD (Phase 7)`
* **Command verification:**
  ```bash
  git log --diff-filter=A --oneline -- backend/src/sanad/evaluation/mlflow_experiment.py
  # Output: 8fc41a5 feat(mlops): implement RAGAS quality gate, MLflow experiment grid, Locust load tests, and CI/CD (Phase 7)
  ```

---

### b. `reports/ragas_results.json`: Sample Count, 401 Fallbacks, and Judge Fallback Values

**Finding: 20 samples total; 20 out of 20 (100%) were produced by HTTP 401 Unauthorized fallbacks.**

#### Evidence & Proof:
* **Analysis command:**
  ```bash
  python3 -c '
  import json
  with open("reports/ragas_results.json") as f:
      data = json.load(f)
  results = data.get("results", [])
  print("sample_count:", data.get("sample_count"))
  print("len results:", len(results))
  reasons = [r.get("details", {}).get("reason", "") for r in results]
  count_401 = sum(1 for r in reasons if "401" in r or "Unauthorized" in r)
  print("count_401:", count_401)
  '
  ```
* **Command Output:**
  ```text
  sample_count: 20
  len results: 20
  count_401: 20
  ```
* **Fallback reason in each record:**
  ```text
  "Fallback due to: Client error '401 Unauthorized' for url 'https://openrouter.ai/api/v1/chat/completions'\nFor more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401"
  ```
* **Commit that introduced this file:**
  * **Commit Hash:** `8fc41a5f7e6ddbf59064b3d12564a848bd4eabce` (Wed Oct 7 13:58:20 2026 +0300)
* **Judge Fallback Value in `backend/src/sanad/evaluation/judge.py` at commit `8fc41a5`:**
  ```python
  except Exception as e:
      logger.warning(f"LLM judge completion failed ({e}); falling back to heuristic.")
      return 0.90, 0.85, f"Fallback due to: {e}"
  ```
  On any exception/401, `faithfulness` was hardcoded to `0.90` and `answer_relevancy` was hardcoded to `0.85`.

---

### c. `reports/locust_report.html`: Creation Method, Commit, and Runtime State

**Finding: The report is hand-written HTML, NOT an export from Locust.**

#### Evidence & Proof:
* **File Structure:**
  `reports/locust_report.html` is a 155-line hand-crafted static HTML document using a dark palette (`#0f172a`), inline CSS, custom `.metric-card` flex grids, and hardcoded text:
  `Simulation: 50 concurrent users • Ramp-up: 5 users/sec • Duration: 5m`.
  Native Locust reports generate HTML with embedded ECharts visualizations, automated timing distribution tables, and dynamic JavaScript bundles.
* **Commit History:**
  * Created in commit `8fc41a5f7e6ddbf59064b3d12564a848bd4eabce` (Wed Oct 7 13:58:20 2026 +0300).
  * Minor styling/trailing newline touch-up in commit `0d3e15c6e0d2f0eb5e0dd0e0fd9f609f0bde0068` (Wed Oct 7 18:45:04 2026 +0300).
* **LLM Live vs. MOCK_MODE during the run:**
  * At commit `8fc41a5`, `MOCK_MODE` did not exist anywhere in the codebase (it was implemented later in commit `b675302`).
  * In `.env.example` at commit `8fc41a5`, `OPENROUTER_API_KEY=` was empty.
  * Because the file was authored manually rather than produced by running the `locust` CLI process, neither live LLM nor mock load was actually executed.

---

### d. Data Version Control (DVC) and Git Tracking

**Finding: DVC is NOT initialized. All datasets are tracked directly in Git.**

#### Evidence & Proof:
* **Directory checks:**
  * `.dvc/` directory: **DOES NOT EXIST** (`ls .dvc` -> No such file or directory)
  * `dvc.lock`: **DOES NOT EXIST** (`ls dvc.lock` -> No such file or directory)
  * DVC remote: **NONE** (`dvc remote list` -> No remotes configured)
* **Git tracking of `data/`:**
  * Command: `git ls-files data/`
  * Output:
    ```text
    data/eval/eval_questions.jsonl
    data/processed/articles.json
    data/processed/corpus_metadata.json
    data/processed/issuance_law.json
    data/raw/egyptian_civil_code_1948.pdf
    ```
  * All raw, processed, and evaluation data files are committed directly to Git.
* **Status of `dvc.yaml`:**
  A pipeline definition `dvc.yaml` was written in commit `8fc41a5`, but `dvc init` was never run, and no remote was configured.

---

### e. Container Registry, Pull Requests, and Git Tags

**Finding: NO registry push in CI, NO GitHub PRs, and NO git tags exist.**

#### Evidence & Proof:
* **GitHub Actions Workflow (`.github/workflows/ci.yml`):**
  * The `docker-build` job only runs local docker build commands:
    ```yaml
    - name: Build Backend Docker Image
      run: docker build -t sanad-backend:latest -f backend/Dockerfile .
    - name: Build Frontend Docker Image
      run: docker build -t sanad-frontend:latest -f frontend/Dockerfile ./frontend
    ```
  * There are **NO** `docker/login-action`, **NO** `docker/build-push-action`, and **NO** steps pushing images to GHCR or Docker Hub.
* **Git Tags:**
  * Command: `git tag`
  * Output: `<empty>` (Zero tags exist in the repository).
* **Pull Requests & Merge Commits:**
  * Command: `git log --merges --oneline`
  * Output: `<empty>` (Zero merge commits).
  * All commits in the repo history were made directly on `main`.
  * `gh` CLI status: `gh` command is not installed on the system environment.

---

### f. README Claims vs. Actual Code / Behavior

| # | README Claim | Actual Code Reality | Status |
|---|---|---|---|
| 1 | *"The baseline evaluation report from the real RAGAS run is preserved in `reports/ragas_results.json` and `.md`"* | `reports/ragas_results.json` was 100% produced by HTTP 401 Unauthorized errors with hardcoded fallback values of 0.90/0.85 in `judge.py`. No live model successfully evaluated it. | **FALSE** |
| 2 | `/health` endpoint response documentation showed: `{"status": "ok", "version": "0.1.0", "mock_mode": true, "qdrant_connected": true, "articles_count": 1149}` | Actual endpoint output in `backend/src/sanad/api/routes/health.py`: `{"status": "ok", "qdrant": true, "models_loaded": 9, "version": "0.1.0", "mock_mode": true}`. The fields `qdrant_connected` and `articles_count` do not exist. | **FALSE** |
| 3 | *"Locust Load Testing: 50 concurrent users simulated with P95 latency of 418 ms and 0% errors (14,820 reqs, 49.4 RPS)"* | Hand-written static HTML file committed directly to `reports/locust_report.html` in commit `8fc41a5`, not generated by a real Locust run. | **FALSE** |
| 4 | *"MLflow Tracking Grid: 8 configurations evaluated across chunking, embeddings, and rerankers. Best configuration registered as sanad-rag-config in Production"* | `backend/src/sanad/evaluation/mlflow_experiment.py` has a hardcoded dictionary of 8 configurations with literal metrics (`faithfulness: 0.88`, `0.94`, etc.) that are logged without running evaluations. | **FALSE** |

---

## 2. Current BLOCKED Items

Under **Hard Rule 4**, any item requiring external services, credentials, or missing system tools that cannot be automated without user authorization is stopped and documented here:

1. **GitHub CLI (`gh`) Missing:**
   * `gh` is not installed on this Linux system (`gh: command not found`).
   * Creating branches and opening PRs via `gh pr create` requires `gh` and an authenticated `GITHUB_TOKEN`.
2. **Git Push Credentials:**
   * Running `git push origin <branch>` prompts interactively for username/password. No SSH key or Personal Access Token (PAT) is stored in the git credential helper.
3. **OpenRouter API Key Status:**
   * **VERIFIED LIVE:** A call to `https://openrouter.ai/api/v1/chat/completions` using the key in `.env` returned HTTP 200 (`nvidia/nemotron-3-ultra-550b-a55b:free`).
   * A call to `https://openrouter.ai/api/v1/embeddings` using the key in `.env` returned HTTP 200 (`nvidia/nemotron-3-embed-1b:free`, 2048 dimensions).
   * **Task 2 is NOT blocked by key.**
4. **Locust Package in Local Environment:**
   * `locust` command is available inside virtual environments/containers, but native load test requires running against the active server.

---

## 3. Preservation of 401 Fallback Artifact

The original 401 fallback evaluation file has been documented. In Task 2, it will be moved to `reports/ragas_results_FALLBACK_401.json` so historical transparency is preserved.

