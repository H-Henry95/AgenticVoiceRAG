# Design Document — Agentic Voice RAG

> This is a working design doc, pre-filled with the structure that matters and
> `TODO`s where your specifics go. The JD explicitly lists documentation of
> "model design, experiments, data provenance and solution rationale" as a
> responsibility — most candidates skip this, so a real design doc is cheap
> differentiation. Fill the TODOs as you build.

---

## 1. Problem statement

**Domain:** `TODO — the domain you chose (e.g. product support, personal finance).`

**User & job-to-be-done:** `TODO — who asks, what they need, in what channel.`

**Why AI / why now:** `TODO — what makes this hard for search or rules alone.`

**Success criteria:**
- Answer quality: grounded, cited, no hallucinated facts.
- Latency: `TODO` p95 target for text; `TODO` for voice round-trip.
- Cost: `TODO` target cost per resolved query.

---

## 2. Architecture & rationale

The system is a LangGraph agent fronted by a FastAPI orchestrator, with retrieval,
an MCP tool server, and an optional voice loop as separable components. See the
diagram in `README.md`.

**Key principle — interfaces over integrations.** Every external dependency sits
behind an abstract interface (`VectorStore`, `LLMClient`, `SpeechToText`,
`TextToSpeech`). Consequences:
- Local↔hosted is a config switch, not a rewrite.
- Components are independently testable and mockable.
- Vendor lock-in is contained to one file per dependency.

**Why agentic (LangGraph) rather than a linear chain.** A conditional router lets
the system choose retrieval vs. tool-use vs. direct response at runtime, and a
verification node validates answers against context before returning them. This
is the difference between "used an LLM" and "built an agent."

---

## 3. Data provenance

Provenance is attached at ingestion and never dropped, so every answer is
traceable to its source.

| Field | Captured where | Purpose |
|-------|----------------|---------|
| `source` | ingestion | cite the originating document |
| `chunk_index` | ingestion | locate the passage within the doc |
| `ingested_at` | ingestion | audit freshness; detect stale content |
| `content_hash` | ingestion | dedup; detect changed source material |

**Source inventory:** `TODO — list document sets, owners, licenses, update cadence.`

**PII / sensitivity:** `TODO — what sensitive data exists and how it's handled
(redaction at ingest, access controls, EU residency, retention).`

---

## 4. Retrieval design

- **Chunking:** character-based with overlap (`CHUNK_SIZE`/`CHUNK_OVERLAP`).
  `TODO — note the config you settled on and why, backed by eval numbers.`
- **Embeddings:** BGE-small (local). Query-side instruction prefix applied.
- **Index:** FAISS inner-product on normalized vectors (= cosine). `TODO — total
  chunks, index size, build time.`
- **Reranking:** `TODO — none yet / cross-encoder if added.`

---

## 5. Evaluation methodology

Offline eval runs a golden set through the full agent and logs to MLflow, so
retrieval/model/chunking configs are compared as tracked experiments.

**Metrics:**
- `keyword_recall` — coarse grounding check (stand-in; see below).
- `verified_rate` — fraction of answers that pass the verification node.
- `TODO — upgrade to RAGAS: faithfulness, answer_relevancy, context_precision.`

**Golden set:** `src/eval/golden_set.json` — `TODO — grow to N≥30 questions
grounded in YOUR documents, including "unanswerable" cases to test refusal.`

**Experiment log:**

| Run | Change | keyword_recall | verified_rate | Notes |
|-----|--------|----------------|---------------|-------|
| `TODO` | baseline | | | |

---

## 6. Monitoring in production

Answers the "drift, bias, fairness, reliability" responsibility.

- **Embedding drift:** cosine distance of recent-query mean vs. a reference mean;
  alert above a calibrated threshold (~0.15). Rising drift ⇒ users asking about
  things the index covers poorly. Upgrade path: Evidently dashboards.
- **Output safety / bias gate:** heuristic gate pre-return (`monitoring/drift.py`);
  upgrade to a small toxicity/bias classifier. `TODO — define the bias tests
  relevant to your domain (e.g. consistent answers across demographic phrasings).`
- **Reliability:** `TODO — timeouts, retries, fallback model, circuit breaker on
  the LLM endpoint.`
- **Tracing:** Langfuse captures every node's input/output for debugging and
  post-hoc review.

---

## 7. Cost & deployment

**Local-first, hosted-for-demo — a deliberate tradeoff.**

- *Development:* everything local (Ollama, sentence-transformers, FAISS,
  faster-whisper, Kokoro). Zero marginal cost, no rate limits, works offline.
- *Shareable demo:* a public deploy of a local LLM needs compute that free
  hosting won't provide, so the demo points the LLM at a free hosted tier
  (e.g. Groq) via env vars. Embeddings and vector store stay local/cheap.

| Component | Local (dev) | Hosted (demo) |
|-----------|-------------|---------------|
| LLM | Ollama (free) | Groq free tier |
| Embeddings | BGE-small (free) | same (local) |
| Vector store | FAISS (free) | FAISS / Pinecone free tier |
| STT / TTS | faster-whisper / Kokoro | Deepgram / ElevenLabs (paid) |

**Cost per query (hosted path):** `TODO — measure tokens in/out × price; note the
verification node roughly doubles LLM calls, and whether that's worth it.`

**Deployment target:** `TODO — Render/Fly/Railway/cloud; single container vs.
split microservices.`

---

## 8. Risks & limitations

- `TODO — small local models hallucinate more; how the verify node mitigates it.`
- `TODO — retrieval gaps on out-of-domain questions; refusal behavior.`
- `TODO — voice latency budget and interruption handling.`
- `TODO — eval set coverage limits.`

---

## 9. Future work

- `TODO — reranking, hybrid (BM25 + dense) retrieval, LoRA fine-tune, multi-turn
  memory, per-tenant indexes, etc.`
