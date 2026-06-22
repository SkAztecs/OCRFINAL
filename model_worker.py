"""
model_worker.py — Persistent warm-model worker.

Started ONCE by api.py via a multiprocessing SPAWN context at server startup.
Because it uses spawn (not fork), CUDA initialises cleanly inside this child
process with no state leak from the FastAPI main process.

medical_agent is imported HERE (inside the child), so all GPU models are
loaded exactly once and kept alive for the lifetime of the server.
Subsequent requests reuse the already-loaded models — zero reload overhead.

VRAM between pipeline phases is reclaimed with an explicit cache flush
(torch.cuda.empty_cache + gc.collect) instead of killing/respawning a
subprocess, which is faster and avoids the fork-CUDA crash entirely.

Communication:
  job_q  ← dicts pushed by api.py  (one dict per request)
  res_q  → dicts pulled by api.py  {"job_id", "ok", "result" | "error"}

Shutdown: api.py sends None (poison pill) on job_q.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import traceback
from multiprocessing import Queue
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: do NOT import medical_agent at module level.
# It is imported inside warm_worker() so it only executes in the child
# process — the spawn context gives us a clean interpreter with no prior
# CUDA state, which is the only safe way to use CUDA + multiprocessing.
# ─────────────────────────────────────────────────────────────────────────────


def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] model_worker — %(message)s",
    )
    return logging.getLogger("model_worker")


def _flush_gpu(log: logging.Logger) -> None:
    """Release unused CUDA memory between pipeline phases."""
    try:
        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log.info("GPU cache flushed.")
    except Exception:
        pass
    gc.collect()


def _run_phases(ma: Any, job: dict, log: logging.Logger) -> dict:
    """
    Execute OCR then extraction in-process (no subprocess spawning).
    Models stay hot between calls; VRAM is reclaimed with an explicit flush.
    """
    job_id = job["job_id"]

    # Set the global config flags that medical_agent reads.
    # This is safe here because we are inside the single-threaded worker
    # process — no concurrent mutation is possible.
    ma.OCR_ENGINE       = job["ocr_engine"]
    ma.COMPRESSION_MODE = job["compression_mode"]

    # ── Phase 1: OCR ─────────────────────────────────────────────────────────
    log.info(f"[{job_id}] OCR phase starting")
    ma.run_ocr_phase(
        job["enhance_contrast"],
        True,                       # verbose=True
        job["compression_mode"],
        job["pdf_path"],
        job["workspace_dir"],
        job["stitched_path"],
    )
    log.info(f"[{job_id}] OCR phase complete — flushing GPU cache")
    _flush_gpu(log)

    # ── Phase 2: Extraction ───────────────────────────────────────────────────
    log.info(f"[{job_id}] Extraction phase starting")
    ma.run_pipeline_extraction(
        document_type=job["document_type"],
        stitched_path=job["stitched_path"],
        result_path=job["result_path"],
    )
    log.info(f"[{job_id}] Extraction phase complete")

    result_path = job["result_path"]
    if not os.path.exists(result_path):
        raise FileNotFoundError(
            f"Extraction finished but wrote no result file at: {result_path}"
        )

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def warm_worker(job_q: Queue, res_q: Queue) -> None:
    """
    Entry point for the spawned child process.

    Imports medical_agent (triggering one-time GPU model load), then
    loops forever: dequeue job → run pipeline → enqueue result.
    """
    log = _setup_logging()

    log.info("Worker process started — importing medical_agent (GPU models loading) …")
    import medical_agent as ma  # ← all model weights load here, ONCE

    log.info("medical_agent ready — worker is warm and accepting jobs.")

    while True:
        job = job_q.get()  # blocks until a job (or shutdown signal) arrives

        if job is None:    # poison pill → clean shutdown
            log.info("Shutdown signal received — exiting.")
            break

        job_id = job.get("job_id", "??")
        log.info(f"[{job_id}] Job received: doc_type={job.get('document_type')}")

        try:
            result = _run_phases(ma, job, log)
            res_q.put({"job_id": job_id, "ok": True, "result": result})
            log.info(f"[{job_id}] Job succeeded.")

        except Exception:
            err = traceback.format_exc()
            log.error(f"[{job_id}] Job FAILED:\n{err}")
            res_q.put({"job_id": job_id, "ok": False, "error": err})
