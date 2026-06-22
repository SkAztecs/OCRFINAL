"""
api.py — FastAPI orchestrator for the medical document pipeline.

What changed vs the original and why
─────────────────────────────────────
PROBLEM 1 — Internal Server Error 500
  Root cause: the original code called multiprocessing.Process() with the
  default "fork" start method.  On Linux, forking after CUDA has been
  initialised (which happens the moment any torch/ML code runs in the
  parent process) corrupts the GPU driver state in child processes.
  The child exits with a non-zero code, the error detail is silently
  swallowed, and FastAPI returns a bare 500.

  Fix: all GPU work is moved into a single persistent worker process that
  is started with the "spawn" context (clean Python interpreter, no
  inherited CUDA state) via model_worker.warm_worker.  api.py itself
  never touches CUDA and never calls multiprocessing.Process() again.

PROBLEM 2 — Models reloaded on every request
  Root cause: the original design spawned two new processes per request,
  so model weights were loaded from disk for every call.

  Fix: model_worker imports medical_agent ONCE at startup.  All subsequent
  requests reuse the already-loaded models — no weight-loading overhead.
  VRAM is reclaimed between OCR and extraction phases with
  torch.cuda.empty_cache() instead of process respawning.

PROBLEM 3 — AWS IP exposure
  Fix: uvicorn is bound to 127.0.0.1 only.  Nginx (nginx.conf) handles
  all public traffic, strips server headers, rate-limits, and terminates
  TLS so the EC2 IP is never directly reachable by clients.

Run with:
    MEDQUERY_API_KEY=<secret> uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
    (use start.sh for convenience)

Nginx handles port 80/443 and proxies to 127.0.0.1:8000.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import multiprocessing
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# medical_agent is imported here ONLY to read the static PROMPT_MAP dict.
# If importing medical_agent triggers CUDA initialisation (e.g. it loads
# model weights at module level), remove this import and hard-code
# VALID_DOC_TYPES below instead — see the comment in that section.
import medical_agent as ma

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

API_KEY = os.environ.get("MEDQUERY_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "MEDQUERY_API_KEY is not set. Refusing to start with no API key "
        "configured — set it via: MEDQUERY_API_KEY=<secret> bash start.sh"
    )
BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# If importing medical_agent above causes CUDA errors in the worker, replace
# `list(ma.PROMPT_MAP.keys())` with your actual doc-type strings:
#   e.g.  ["prescription", "lab_report", "discharge_summary"]
VALID_DOC_TYPES   = ["auto"] + list(ma.PROMPT_MAP.keys()) + ["other"]
VALID_COMP_MODES  = ["tiny", "small", "base", "large", "high"]
VALID_OCR_ENGINES = ["auto", "paddleocr", "lighton"]

JOB_TIMEOUT_SECS  = 900   # 15-minute hard ceiling per job
MAX_UPLOAD_BYTES  = 100 * 1024 * 1024   # 100MB — matches nginx client_max_body_size

# ─────────────────────────────────────────────────────────────────────────────
# WARM-WORKER STATE  (module-level so lifespan + endpoint share it)
# ─────────────────────────────────────────────────────────────────────────────
_job_q:   multiprocessing.Queue | None = None
_res_q:   multiprocessing.Queue | None = None
_worker:  multiprocessing.Process | None = None
_gpu_lock: asyncio.Lock | None = None   # created inside the running event loop


def _start_worker() -> None:
    """Spawn the persistent warm-model worker (called once at startup)."""
    global _job_q, _res_q, _worker

    # MUST use "spawn" — "fork" after CUDA init causes GPU driver corruption.
    ctx    = multiprocessing.get_context("spawn")
    _job_q = ctx.Queue()
    _res_q = ctx.Queue()

    # Import warm_worker inside this function (not at module level) so
    # medical_agent is not imported a second time in the parent process.
    from model_worker import warm_worker

    _worker = ctx.Process(
        target=warm_worker,
        args=(_job_q, _res_q),
        name="ModelWorker",
        daemon=True,   # auto-killed when the FastAPI process exits
    )
    _worker.start()
    log.info(f"Warm worker spawned — PID {_worker.pid}")


def _stop_worker() -> None:
    """Gracefully stop the worker (called on shutdown)."""
    if _job_q is not None:
        _job_q.put(None)  # poison pill
    if _worker is not None and _worker.is_alive():
        _worker.join(timeout=10)
        if _worker.is_alive():
            _worker.kill()
            log.warning("Model worker force-killed (did not exit cleanly).")
    log.info("Model worker stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN  (replaces deprecated @app.on_event)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gpu_lock
    _gpu_lock = asyncio.Lock()  # must be created inside the running event loop
    _start_worker()
    yield                       # server is live
    _stop_worker()


app = FastAPI(title="MedQuery Pipeline API", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────
def _require_key(x_api_key: Optional[str]) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ─────────────────────────────────────────────────────────────────────────────
# JOB DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
def _blocking_get(timeout: float) -> dict | None:
    """Queue.get with timeout — runs in a thread pool executor."""
    try:
        return _res_q.get(timeout=timeout)
    except Exception:
        return None  # timeout or queue error


async def _dispatch(job: dict) -> dict:
    """
    Push a job to the warm worker and await the result dict.
    The asyncio.Lock in the caller ensures only one job is in flight,
    so _res_q.get() will always receive the correct response.
    """
    loop = asyncio.get_running_loop()

    # Enqueue (fast; run in executor so we don't block the event loop)
    await loop.run_in_executor(None, _job_q.put, job)
    log.info(f"[{job['job_id']}] Dispatched to warm worker")

    # Wait for result with a hard timeout
    response = await loop.run_in_executor(
        None, _blocking_get, float(JOB_TIMEOUT_SECS)
    )

    if response is None:
        log.error(f"[{job['job_id']}] Timed out after {JOB_TIMEOUT_SECS}s")
        raise HTTPException(504, "Pipeline job timed out — the document may be too large.")

    return response


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Liveness + readiness check.
    Returns 200 when the warm worker is alive and accepting jobs.
    Returns 503 when the worker has crashed (needs a server restart).
    """
    alive = _worker is not None and _worker.is_alive()
    body  = {
        "status":       "ok" if alive else "degraded — worker not running",
        "worker_pid":   _worker.pid if _worker else None,
        "worker_alive": alive,
    }
    status_code = 200 if alive else 503
    return JSONResponse(content=body, status_code=status_code)


@app.post("/extract")
async def extract(
    file:             UploadFile    = File(...),
    document_type:    str           = Form("auto"),
    compression_mode: str           = Form("large"),
    ocr_engine:       str           = Form("auto"),
    enhance_contrast: bool          = Form(False),
    x_api_key:        Optional[str] = Header(None),
):
    _require_key(x_api_key)

    # ── Input validation ─────────────────────────────────────────────────────
    if document_type not in VALID_DOC_TYPES:
        raise HTTPException(400, f"Invalid document_type. Valid: {VALID_DOC_TYPES}")
    if compression_mode not in VALID_COMP_MODES:
        raise HTTPException(400, f"Invalid compression_mode. Valid: {VALID_COMP_MODES}")
    if ocr_engine not in VALID_OCR_ENGINES:
        raise HTTPException(400, f"Invalid ocr_engine. Valid: {VALID_OCR_ENGINES}")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    # ── Size guard (defense in depth — nginx should also enforce this) ──────
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large ({file_size / 1024 / 1024:.1f}MB). "
            f"Max allowed: {MAX_UPLOAD_BYTES / 1024 / 1024:.0f}MB.",
        )

    # Guard: reject immediately if the worker has crashed
    if not (_worker and _worker.is_alive()):
        raise HTTPException(
            503,
            "Model worker is not running. Restart the server and check logs.",
        )

    # ── Per-job isolated workspace ───────────────────────────────────────────
    job_id        = uuid.uuid4().hex[:12]
    job_dir       = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path      = str(job_dir / "input.pdf")
    workspace_dir = str(job_dir / "workspace")
    stitched_path = str(job_dir / "stitched.txt")
    result_path   = str(job_dir / "result.json")
    os.makedirs(workspace_dir, exist_ok=True)

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    log.info(
        f"[{job_id}] New job — file={file.filename!r} "
        f"doc_type={document_type} engine={ocr_engine} "
        f"compression={compression_mode} contrast={enhance_contrast}"
    )

    job = dict(
        job_id=job_id,
        pdf_path=pdf_path,
        workspace_dir=workspace_dir,
        stitched_path=stitched_path,
        result_path=result_path,
        document_type=document_type,
        compression_mode=compression_mode,
        ocr_engine=ocr_engine,
        enhance_contrast=enhance_contrast,
    )

    # ── Serialise GPU access — single T4, one job at a time ─────────────────
    async with _gpu_lock:
        response = await _dispatch(job)

    if not response.get("ok"):
        error_detail = response.get("error", "Unknown pipeline error.")
        log.error(f"[{job_id}] Pipeline failed:\n{error_detail}")
        # Full traceback is logged above for your own debugging via SSH/logs.
        # Only a generic message goes back to the client — a full traceback
        # can leak file paths and internals to anyone with the API key.
        raise HTTPException(
            500,
            detail=f"Pipeline processing failed (job_id={job_id}). Check server logs for details.",
        )

    log.info(f"[{job_id}] Returning result to client.")
    return JSONResponse(content=response["result"])


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = BASE_DIR / "static" / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# ── Static assets ────────────────────────────────────────────────────────────
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")