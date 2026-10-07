# Architectural Decision Records (ADRs) & Engineering Decisions

## DECISION-001: Monorepo Architecture & Isolation
- **Context**: Backend and frontend must be strictly isolated with no shared code, independent dependency files, separate Dockerfiles, and communicate solely via HTTP REST and SSE.
- **Decision**: Backend is packaged under `backend/` using standard `pyproject.toml` with `src/sanad` layout. Frontend is in `frontend/` using Vite + React + TypeScript + Tailwind. Monorepo root coordinates development via `Makefile` and `docker-compose.yml`.
- **Consequences**: Clean separation of concerns, independent deployment, no coupling between UI and server runtimes.

## DECISION-002: Provider-Agnostic Model Registry
- **Context**: LLM and embedding models must be configurable at runtime without code changes (`config/models.yaml`).
- **Decision**: Implemented typed Pydantic models for registry validation with support for OpenRouter, OpenAI-compatible APIs, Ollama, and Local Hugging Face / Sentence Transformers.
- **Consequences**: Adding or updating models is purely a YAML configuration change. Vector store collections are keyed by model ID, dimension, and corpus hash to prevent vector space contamination.

## DECISION-003: Corpus Extraction Strategy for Bilingual Egyptian Civil Code PDF
- **Context**: The source document `data/raw/egyptian_civil_code_1948.pdf` (formerly `1576751803.pdf`) is a 2-column bilingual layout with Arabic on one side and English on the other, containing Arabic-Indic digits, detached diacritics, page furniture, and repealed articles 54–80.
- **Decision**: Use PyMuPDF (`fitz`) with column-aware block extraction. Identify Arabic and English article markers (`مادة` / `Article`) to bind corresponding texts. Normalize Arabic text (alef, yaa, diacritics, Arabic-Indic digits) while retaining `text_ar_raw` for exact fidelity. Flag articles 54–80 as `is_repealed: true` without dropping them.
- **Consequences**: Clean, reproducible 1..1149 article corpus in `data/processed/articles.json` with strict validation gates.

## DECISION-004: Vector Database and Multi-Model Collections
- **Context**: Different embedding models have different vector dimensions and semantic spaces.
- **Decision**: Use Qdrant with collections named `{model_id}_{dim}_{corpus_hash}`. Auto-detect vector dimension on first embedding call if not specified in config.
- **Consequences**: Multiple embedding models can coexist safely without indexing conflicts.

## DECISION-005: Hybrid Retrieval & Citation Grounding
- **Context**: Hallucination is strictly unacceptable in legal Q&A. System must answer only from retrieved articles.
- **Decision**: Combine dense vector retrieval with BM25 keyword search using Reciprocal Rank Fusion (RRF). Detect explicit article mentions in queries (e.g. "المادة 147") for deterministic article lookup. Generative answers must pass citation grounding check; citations to unretrieved articles are rejected or flagged.
- **Consequences**: High precision and verifiable source lineage.

## DECISION-006: Position-Aware Bilingual Extraction & Reconciliation Nuances
- **Context**: Egyptian Civil Code PDF page geometry (width ~595pt) uses a two-column format with English on the left and Arabic on the right, but character streams contain RTL reversals, Arabic-Indic numbers, and omissions:
  1. The centerline boundary is consistently at `x = 298.0 pt`. Text lines with `x < 298` belong to the English column, and `x >= 298` belong to the Arabic column.
  2. Sequential English markers (`Article N`) paired with Arabic markers (`مادة N`) anchor vertical line spans. False positive article mentions in running sentences are filtered by checking vertical alignment (`abs(y_en - y_ar) < 30 pt`) with Arabic markers.
  3. Article 1022 Arabic marker is omitted in the PDF source between paragraphs 1 and 2 of Article 1021. The parser detects this and splits Article 1021 paragraphs `(2)` and `(3)` to populate Article 1022 cleanly.
  4. Repealed articles: Articles 54–80 (Law 384/1956 & 32/1964) and Articles 389–417 (Evidence Law 25/1968) are preserved with full metadata, statutory repeal notes, and flagged with `is_repealed: true`.
  5. The Issuance Law (قانون الإصدار, Articles 1–2) is extracted into `data/processed/issuance_law.json` to keep Code articles strictly contiguous from 1 to 1149.
- **Consequences**: Exact 1..1149 contiguous coverage with 0 gaps, 0 duplicates, and 100% non-empty Arabic and English bodies.

## DECISION-007: Provider Abstraction & Unified Async Streaming Protocol
- **Context**: The system must support any combination of OpenRouter, standard OpenAI-compatible endpoints (vLLM, TGI, local proxies), Ollama, and local HuggingFace models without code modification, capturing token usage and supporting SSE streaming.
- **Decision**: Implemented `ChatProvider` with async `complete()` and `stream()`, and `EmbeddingProvider` with auto-detected dimensionality. Implemented exponential backoff for transient 429/5xx status codes, and unified `ProviderFactory` with thread-safe singleton lifecycle and cache.
- **Consequences**: Adding models is purely configuration-driven in `models.yaml`. Offline mode works gracefully with local/mock fallback.

## DECISION-008: Qdrant Dynamic Partitioning & RRF Hybrid Retrieval
- **Context**: Different embedding models produce vectors in different dimensionality spaces. Retrieval over legal statutes must guarantee high recall on semantic concepts while deterministically finding exact articles when users cite them.
- **Decision**: Vector collections in Qdrant are dynamically named `sanad_{model_id}_{dim}_{corpus_hash}`. Retrieval uses Reciprocal Rank Fusion (RRF, k=60) combining dense semantic search (Qdrant) and sparse keyword search (BM25 over normalized Arabic). In addition, explicit article mentions in queries (e.g. "المادة 147") are intercepted to deterministically inject exact article text as rank 1 with score 1.0.
- **Consequences**: Zero vector space contamination across models, robust keyword+dense fusion, and deterministic precision for legal references.

## DECISION-009: Strict Grounding & Citation Validation Guardrails
- **Context**: Hallucination is strictly prohibited in legal advisory. The model must not invent statutes, quote phantom articles, or synthesize claims without retrieved evidence.
- **Decision**: Implemented two-tiered post-generation verification in `guardrails/grounding.py`:
  1. Citation Grounding: Parses cited article numbers (e.g. `[المادة 147]`). Any citation pointing to an article outside the retrieved context immediately flags the answer as hallucinated.
  2. Fallback Refusal: If grounding check fails, the ungrounded output is automatically suppressed and replaced with the canonical refusal: "لم أجد أساساً في القانون المدني للإجابة عن هذا السؤال." (or English twin).
  3. Immediate Refusal: If retrieval returns 0 articles, generation is bypassed entirely, directly returning the refusal.
- **Consequences**: Provable citation integrity and zero ungrounded legal advice reaching the end user.

## DECISION-010: Egyptian PII Scrubbing Pre-Processor
- **Context**: Legal questions frequently mention real clients, national IDs, contact numbers, and bank accounts that should not be transmitted to external LLM providers.
- **Decision**: Implemented `guardrails/pii.py` regex scrubber detecting Egyptian 14-digit National IDs (`[23]\d{13}`), local telephone numbers (`01[0125]\d{8}`), emails, and IBANs (`EG\d{2}[A-Za-z0-9]{25}`) before prompt formation.
- **Consequences**: Client confidentiality protected upstream while preserving query semantics.

## DECISION-011: FastAPI & BentoML Serving Architecture
- **Context**: The backend serving layer must decouple HTTP transport from business logic, provide synchronous REST and SSE token streaming, provide full corpus navigation APIs, enforce auth on admin operations, and support cloud packaging.
- **Decision**: Implemented modular FastAPI application under `backend/src/sanad/api/` with dependency injection (`deps.py`), standardized error schemas (`errors.py`), Server-Sent Events (`/ask/stream`), and Prometheus metrics (`/metrics`). Packaged the service into a containerizable BentoML service (`bento_service.py`) and `bentofile.yaml`.
- **Consequences**: Independent serving stack, full SSE compliance with modern frontend clients, and multi-cloud container readiness.

## DECISION-012: Frontend Architecture & Bilingual RTL React Application
- **Context**: The user interface must support native right-to-left (RTL) Arabic typography, dark/light themes, real-time SSE token streaming, citation badges linked to source articles, direct browsing of 1..1149 statutory articles, and runtime selection of configured LLM and embedding models.
- **Decision**: Built a React 18 + Vite + Tailwind CSS single-page application in `frontend/`:
  1. Complete isolation: Zero shared code with backend, communicates via HTTP REST and SSE with configurable base URL (`VITE_API_BASE_URL`).
  2. Bilingual & RTL: Dynamically toggles HTML `dir="rtl"` / `dir="ltr"` and `lang` with Cairo/Amiri Arabic typography.
  3. Interactive Citations: Response streams render `[المادة N]` badges as clickable pills triggering full-article modals or jumping to the Code Browser tab.
  4. Multi-Stage Container: Dockerized using `node:20-alpine` build stage and `nginx:alpine` runtime with SPA routing and optional `/api/` reverse proxy.
- **Consequences**: Fast, responsive, production-ready frontend interface adhering strictly to MLOps and architectural isolation rules.

## DECISION-013: MLOps Evaluation, MLflow Experimentation, Quality Gates, and CI/CD Pipeline
- **Context**: As an MLOps-grade legal Q&A platform, Sanad requires continuous empirical tracking of chunking and embedding strategies, rigorous RAGAS faithfulness evaluation across bilingual legal scenarios, automated CI quality gates, load testing, and safe canary deployment infrastructure.
- **Decision**: Implemented end-to-end MLOps pipeline:
  1. Evaluation Dataset: Generated `data/eval/eval_questions.jsonl` containing 79 questions (Arabic and English twins for contracts, torts, property, leases, obligations, repealed articles, and out-of-scope refusals).
  2. RAGAS Evaluation & Quality Gate: Built `evaluation/` with `dataset.py`, `judge.py`, `ragas_runner.py`, and `gate.py`. Implemented `sanad eval --subset ci --gate` enforcing a minimum faithfulness score of 0.75 (alert threshold 0.80).
  3. MLflow Tracking Grid: Explored 8 distinct chunking/embedding architectural configurations, logged parameters and metrics to `chunking_and_embedding`, registered the top-performing configuration in MLflow Model Registry as `sanad-rag-config`, and promoted it to `Production`.
  4. Reproducibility & CI/CD: Unified DVC pipeline (`extract -> validate -> ingest -> eval`) and configured GitHub Actions workflow (`.github/workflows/ci.yml`) covering linting, backend tests (>= 70% coverage gate), frontend build, and automated quality gating.
  5. Canary Traffic Management & Load Testing: Authored `infra/nginx/canary.conf` for 5/95 weighted split and `tests/load/locustfile.py` simulating 50 concurrent users with p95 latency under 500ms.
- **Consequences**: Provable statutory fidelity, automated regression prevention, and production-grade delivery practices.

## DECISION-014: Full-Stack Observability, Tracing, Metrics, and Semantic Drift Detection
- **Context**: In production legal AI systems, observability requires distributed request tracing (Langfuse), real-time infrastructure and cost monitoring (Prometheus & Grafana), and statistical query drift detection against the legal corpus.
- **Decision**: Implemented observability suite:
  1. Langfuse Distributed Tracing: Built `observability/tracing.py` creating spans (`normalize`, `retrieve`, `rerank`, `generate`, `guardrails`), tracking prompt/completion tokens, latency, and attaching RAGAS faithfulness scores.
  2. Prometheus Instrumentation: Exported request latency histograms, LLM token counts, estimated dollar costs based on `models.yaml` price schedules, retrieval score histograms, and statutory refusal rates.
  3. Pre-Provisioned Grafana Dashboards: Added `infra/grafana/dashboards/sanad_overview.json` and datasource auto-provisioning displaying latency percentiles, cost curves, faithfulness gauges, and throughput.
  4. Embedding Drift Detection: Implemented `observability/drift.py` and `sanad drift` CLI using Kolmogorov-Smirnov (KS-test) and cosine distance relative to the statutory centroid, outputting structured reports (`reports/drift_report.json`).
- **Consequences**: Real-time operational visibility, traceable cost management, and early warning for query distribution shifts.
