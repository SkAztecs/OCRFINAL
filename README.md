# Medical Document Intelligence Pipeline

OCR + LLM extraction service that turns clinical PDFs (discharge summaries, radiology reports, pathology reports, and other document types) into structured JSON.

## Architecture

```
Client (browser/curl)
      │  HTTPS (443) / HTTP (80)
      ▼
  Nginx  ──────────────────────────  reverse proxy, TLS termination,
      │                              rate limiting, hides EC2 IP
      │  proxies to 127.0.0.1:8000
      ▼
  FastAPI (api.py, uvicorn)  ───────  request validation, job queueing,
      │                              auth (X-Api-Key), serializes GPU
      │                              access via asyncio.Lock
      │  multiprocessing (spawn context)
      ▼
  Warm Model Worker (model_worker.py) ─ persistent child process,
      │                                imports medical_agent ONCE,
      │                                keeps models loaded across requests
      ▼
  medical_agent.py  ─────────────────  OCR phase → extraction phase,
                                       talks to Ollama over HTTP
      │
      ▼
  Ollama (localhost:11434)
      ├── LightOnOCR-2:1b   (vision/OCR model)
      └── qwen3:8b          (extraction model)
```

**Why this shape:** the original design spawned a new process per request with the default `fork` start method, which corrupts CUDA state and caused silent 500 errors. The fix moves all GPU work into one persistent worker process started with `spawn` (clean interpreter, no inherited CUDA state), so models load from disk exactly once and stay resident for the life of the server — no reload overhead per request.

## Files

| File | Role |
|---|---|
| `api.py` | FastAPI app. Routes: `GET /health`, `POST /extract`, `GET /` (upload UI), `/static`. Spawns and manages the warm worker via a lifespan hook. Validates uploads (PDF only, size cap, doc-type/compression/OCR-engine enums), writes each job to an isolated `jobs/<job_id>/` directory, and serializes GPU access with an `asyncio.Lock` so only one job runs at a time. |
| `model_worker.py` | Persistent child process spawned once at startup. Imports `medical_agent` (triggering the one-time model load), then loops: pull a job off the queue → run OCR phase → flush GPU cache (`torch.cuda.empty_cache()` + `gc.collect()`) → run extraction phase → push the result back. |
| `medical_agent.py` | Core pipeline logic. PDF rendering, OCR routing (PaddleOCR for digital PDFs, LightOnOCR via Ollama for scanned/poor-quality PDFs), per-document-type system prompts (`PROMPT_MAP`, 14 document types), LLM extraction calls to Ollama (`_call_gemini`), JSON schema validation, post-processing. Key functions: `run_ocr_phase`, `run_pipeline_extraction`, `transcribe_all_pages`, `texts_to_json`. |
| `start.sh` | Launch script. Requires `MEDQUERY_API_KEY` env var. Runs `uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1` (production) or with `--reload` (dev, pass `--reload` as arg). |
| `nginx.conf` | Public-facing reverse proxy. Binds 80/443, proxies to `127.0.0.1:8000`, strips `Server`/`X-Powered-By` headers, rate-limits at 5 req/s per IP (burst 10), caps upload size at 100MB, extended timeouts for slow OCR jobs. EC2's IP is never directly reachable — Nginx (optionally behind Cloudflare) is the only public entry point. |

## Request flow

1. Client uploads a PDF to `POST /extract` with `X-Api-Key` header and form fields (`document_type`, `compression_mode`, `ocr_engine`, `enhance_contrast`).
2. `api.py` validates the request, saves the PDF to `jobs/<job_id>/input.pdf`, and dispatches a job dict to the warm worker via a multiprocessing queue.
3. `model_worker.py` runs the job in-process (no new process spawned per request):
   - **OCR phase** (`run_ocr_phase`): renders PDF pages to images, routes to PaddleOCR or LightOnOCR depending on document quality/engine override, stitches page text together.
   - GPU cache flush between phases.
   - **Extraction phase** (`run_pipeline_extraction`): classifies/splits the document if needed, runs the appropriate per-document-type prompt through the LLM, validates and post-processes the JSON output.
4. Result is written to `jobs/<job_id>/result.json` and returned to the client through the same queue → FastAPI → HTTP response path.

## Known operational notes

- **GPU residency (Ollama):** `medical_agent.py` calls Ollama with explicit `keep_alive` settings per model — the vision model is set to release VRAM quickly after use, while the extraction model can be kept warm across chunks within a job. Explicit unload calls guard the handoff between OCR and extraction phases to avoid both models being resident in VRAM simultaneously, which causes GPU contention and significant slowdown on single-GPU instances.
- **Single-GPU serialization:** `api.py` holds one `asyncio.Lock` for the whole `/extract` endpoint, so only one job's GPU work runs at a time — appropriate for a single-GPU deployment (e.g. one NVIDIA T4), not yet built for multi-GPU/multi-model-instance parallelism.
- **Job timeout:** hard ceiling of 900 seconds per job (`JOB_TIMEOUT_SECS` in `api.py`); requests exceeding this return `504`.
- **Auth:** simple shared-secret header (`X-Api-Key` / `MEDQUERY_API_KEY`). No per-user accounts or rate limiting beyond Nginx's IP-based limiter.

## Deployment

```bash
# one-time
sudo apt install nginx -y
sudo cp nginx.conf /etc/nginx/nginx.conf
sudo nginx -t && sudo systemctl reload nginx

# start the API + warm worker
MEDQUERY_API_KEY=<secret> bash start.sh
```

Nginx handles ports 80/443; uvicorn binds to `127.0.0.1:8000` only and is never directly exposed.
