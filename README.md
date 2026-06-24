# Complete Mac Setup Guide — MedQuery Medical Document Extraction Pipeline

This guide walks you from a fresh Mac to a running pipeline: Ollama (local LLM), PaddleOCR, FastAPI async API, and optional web UI.

---

## What You Are Building

```
PDF upload → FastAPI (api_server.py) → Worker process (Apple MPS)
                ↓                              ↓
           job_id (202)              OCR (PaddleOCR or LightOnOCR)
                ↓                              ↓
         poll /jobs/{id}              LLM extraction (qwen3:8b via Ollama)
                ↓                              ↓
           JSON result                 jobs/{id}/result.json
```

On Mac (Apple Silicon), the pipeline uses **one worker** with **MPS** (Metal). PaddleOCR runs on **CPU**. Ollama handles all LLM inference.

---

## Part 0 — Before You Start

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Mac | Apple Silicon (M1/M2/M3/M4) or Intel | M2 Pro / M3 with 16 GB+ unified memory |
| RAM | 16 GB | 24–32 GB (qwen3:8b + OCR models are memory-heavy) |
| Disk | 20 GB free | 40 GB+ (models + job workspaces) |

### What you need installed globally

- macOS 12.3+ (for Apple Silicon MPS)
- Internet for model downloads
- A terminal (Terminal.app or iTerm2)

---

## Part 1 — Install System Tools

### 1.1 Xcode Command Line Tools

```bash
xcode-select --install
```

If prompted, click **Install** and wait for it to finish.

Verify:

```bash
xcode-select -p
# Expected: /Library/Developer/CommandLineTools
```

### 1.2 Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After install, follow the on-screen instructions to add Homebrew to your PATH (especially on Apple Silicon).

Verify:

```bash
brew --version
```

### 1.3 Python 3.11

```bash
brew install python@3.11
```

Verify:

```bash
python3.11 --version
# Expected: Python 3.11.x
```

### 1.4 Ollama (local LLM server)

```bash
brew install ollama
```

Or download the macOS app from https://ollama.com/download

Verify:

```bash
ollama --version
```

---

## Part 2 — Get the Project Files

Create a project folder and copy your pipeline files into it.

```bash
mkdir -p ~/medquery-pipeline
cd ~/medquery-pipeline
```

You need these files in that folder:

| File | Purpose |
|------|---------|
| `medical_agent.py` | Core OCR + extraction pipeline |
| `api_server.py` | Async FastAPI API (202 + polling) |
| `model_worker.py` | GPU/MPS worker process |
| `gpu_worker_pool.py` | Worker pool manager |
| `start.sh` | Convenience startup script |
| `index.html` | Web UI (optional) |

Optional (legacy synchronous API):

| File | Purpose |
|------|---------|
| `api.py` | Blocking single-request API |

Create the folders the app expects:

```bash
mkdir -p static jobs ocr_output
```

Copy the web UI into `static/`:

```bash
cp index.html static/index.html
```

Your layout should look like:

```
~/medquery-pipeline/
├── medical_agent.py
├── api_server.py
├── model_worker.py
├── gpu_worker_pool.py
├── start.sh
├── static/
│   └── index.html
├── jobs/              ← created automatically per job
└── ocr_output/        ← logs + local CLI output
```

---

## Part 3 — Python Virtual Environment

Always use a venv so dependencies stay isolated.

```bash
cd ~/medquery-pipeline

python3.11 -m venv .venv
source .venv/bin/activate
```

Your prompt should show `(.venv)`.

Upgrade pip:

```bash
pip install --upgrade pip setuptools wheel
```

---

## Part 4 — Install Python Dependencies

Install in this order (Mac-specific choices included).

### 4.1 PyTorch (Apple Silicon MPS support)

```bash
pip install torch torchvision
```

Verify MPS:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Expected on Apple Silicon: MPS: True
```

On Intel Macs without MPS, you'll see `MPS: False` — the pipeline still runs on CPU.

### 4.2 Core ML / NLP stack

```bash
pip install transformers python-dotenv tqdm pydantic python-dateutil json-repair
```

### 4.3 Image / PDF processing

```bash
pip install Pillow numpy opencv-python-headless pypdfium2
```

### 4.4 PaddleOCR (CPU on Mac)

On Mac, use the **CPU** PaddlePaddle build (not `paddlepaddle-gpu`):

```bash
pip install paddlepaddle paddleocr
```

If `paddleocr` fails to import, the pipeline can still run using the LightOnOCR + Ollama path — but PaddleOCR is much faster for digital PDFs.

### 4.5 FastAPI server

```bash
pip install fastapi uvicorn python-multipart
```

### 4.6 Quick sanity check

```bash
python -c "
import torch, cv2, pypdfium2, fastapi, uvicorn
from dotenv import load_dotenv
print('Core imports OK')
print('MPS available:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
try:
    from paddleocr import PaddleOCR
    print('PaddleOCR OK')
except ImportError as e:
    print('PaddleOCR missing (optional):', e)
"
```

---

## Part 5 — Pull Ollama Models

The pipeline uses two Ollama models (defined in `medical_agent.py`):

| Model | Used for |
|-------|----------|
| `qwen3:8b` | Structured JSON extraction |
| `maternion/LightOnOCR-2:1b` | OCR on scanned/low-quality PDFs |

### 5.1 Start Ollama

**Option A — macOS app:** Open Ollama from Applications.

**Option B — terminal:**

```bash
ollama serve
```

Leave this running in a dedicated terminal tab, or use the Ollama menubar app.

Verify:

```bash
curl http://localhost:11434/api/tags
```

You should get JSON (possibly an empty model list at first).

### 5.2 Download models

In a **new** terminal:

```bash
ollama pull qwen3:8b
ollama pull maternion/LightOnOCR-2:1b
```

This can take several minutes and several GB of disk space.

Verify:

```bash
ollama list
```

You should see both models listed.

### 5.3 Test Ollama chat

```bash
ollama run qwen3:8b "Reply with exactly: OK"
```

---

## Part 6 — Environment Configuration

### 6.1 Required: API key

Generate a secret key:

```bash
openssl rand -hex 32
```

Save it — you'll use it in every API request.

Add to your shell profile (`~/.zshrc` on modern Macs):

```bash
echo 'export MEDQUERY_API_KEY="paste-your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 6.2 Optional: `.env` file

Create `~/medquery-pipeline/.env`:

```bash
cat > ~/medquery-pipeline/.env << 'EOF'
# Required by api_server.py (can also be exported in shell)
MEDQUERY_API_KEY=paste-your-key-here

# Mac: force 1 worker (default on MPS anyway)
MEDQUERY_GPU_WORKERS=1

# Ollama endpoint (default is fine for single Mac)
OLLAMA_BASE_URL=http://localhost:11434
EOF
```

`medical_agent.py` calls `load_dotenv()`, so values in `.env` are picked up automatically.

### 6.3 Mac-specific environment notes

| Variable | Mac value | Why |
|----------|-----------|-----|
| `MEDQUERY_GPU_WORKERS` | `1` | Apple MPS supports one device; auto-detected |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Default local Ollama |
| `CUDA_VISIBLE_DEVICES` | **Do not set** | Not used on Mac |

The worker automatically sets `PADDLE_USE_GPU=False` on MPS and uses `torch.mps.empty_cache()` between pipeline phases.

---

## Part 7 — Make `start.sh` Executable

```bash
cd ~/medquery-pipeline
chmod +x start.sh
```

---

## Part 8 — Start the Pipeline

You need **two things running**:

1. **Ollama** (if not using the menubar app)
2. **The API server**

### Terminal 1 — Ollama (skip if using Ollama app)

```bash
ollama serve
```

### Terminal 2 — API server

```bash
cd ~/medquery-pipeline
source .venv/bin/activate
export MEDQUERY_API_KEY="your-key-here"

bash start.sh
```

Or run uvicorn directly:

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8000 --workers 1
```

Expected log lines:

```
GPU worker pool started — 1 worker(s)
Spawned worker 0 ...
Worker runtime configured — device=mps ollama=http://localhost:11434
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "ok",
  "worker_pool": {
    "num_workers": 1,
    "alive_workers": 1
  }
}
```

If `alive_workers` is `0`, check Terminal 2 for worker crash logs.

---

## Part 9 — Run Your First Extraction

### 9.1 Submit a job (async — returns immediately)

Replace paths and API key:

```bash
curl -X POST "http://127.0.0.1:8000/extract" \
  -H "X-Api-Key: $MEDQUERY_API_KEY" \
  -F "file=@/path/to/your/report.pdf" \
  -F "document_type=auto" \
  -F "compression_mode=large" \
  -F "ocr_engine=auto" \
  -F "enhance_contrast=false"
```

Response (**202 Accepted**):

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "queued",
  "status_url": "/jobs/a1b2c3d4e5f6",
  "poll_interval_seconds": 3
}
```

Save the `job_id`.

### 9.2 Poll until complete

```bash
JOB_ID="a1b2c3d4e5f6"

curl -s "http://127.0.0.1:8000/jobs/$JOB_ID" \
  -H "X-Api-Key: $MEDQUERY_API_KEY" | python3 -m json.tool
```

Status progression:

```
queued → processing → completed
                   └→ failed (check "error" field)
```

When `status` is `"completed"`, the response includes a `"result"` object with extracted JSON.

### 9.3 Poll in a loop (convenience script)

```bash
JOB_ID="paste-job-id-here"

while true; do
  RESP=$(curl -s "http://127.0.0.1:8000/jobs/$JOB_ID" -H "X-Api-Key: $MEDQUERY_API_KEY")
  STATUS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
  echo "Status: $STATUS"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    echo "$RESP" | python3 -m json.tool
    break
  fi
  sleep 3
done
```

### 9.4 Where files land on disk

Each job gets an isolated folder:

```
jobs/a1b2c3d4e5f6/
├── meta.json        ← status, timestamps, errors
├── input.pdf        ← uploaded file
├── workspace/       ← rendered page images, OCR cache
├── stitched.txt     ← combined OCR text
└── result.json      ← final extracted JSON
```

Logs also go to:

```
ocr_output/pipeline.log
```

---

## Part 10 — Web UI (Optional)

Open in browser:

```
http://127.0.0.1:8000/
```

Enter your API key, upload a PDF, click **Extract**.

**Important:** The bundled `index.html` was written for the **old synchronous** API (expects JSON directly from `POST /extract`). The new `api_server.py` returns **202 + job_id** and requires polling.

For the web UI to work with the async API, the JavaScript needs a polling loop on `GET /jobs/{job_id}`. Until that's updated, use **curl** (Part 9) or the **CLI** (Part 11).

**Alternative for synchronous web UI:** run legacy `api.py` instead:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

That blocks until extraction finishes and returns JSON directly — works with the current `index.html`, but only one job at a time.

---

## Part 11 — CLI Mode (No API)

Run the pipeline interactively from the terminal:

```bash
cd ~/medquery-pipeline
source .venv/bin/activate

# Place a PDF in the project folder or edit PDF_PATH in medical_agent.py
python medical_agent.py
```

You'll be prompted for:

- Document type (`auto` recommended)
- Compression mode (`large` recommended)
- OCR engine (`auto` recommended)
- Enhance contrast (y/n)
- Force OCR re-run (y/n)

Output:

```
ocr_output/{pdf_name}_stitched.txt
ocr_output/{pdf_name}.json
```

---

## Part 12 — Document Types & OCR Engines

### Document types (`document_type`)

Use `auto` unless you know the exact type — it runs the structure classifier to detect document type(s).

Fixed types (faster, no auto-detection):

- `discharge_summary`
- `histopathology_report`
- `radiotherapy_report`
- `pet_ct_scan`
- `ct_scan`
- `mri_scan`
- `chemotherapy_admission`
- `outpatient_note`
- `mammogram`
- `cytology_report`
- `ultrasound_scan`
- `referral_letter`
- `registration_receipt`
- `dna_test`
- `other`

### OCR engines (`ocr_engine`)

| Value | When to use | Mac speed |
|-------|-------------|-----------|
| `auto` | Default — inspects PDF quality | Best choice |
| `paddleocr` | Clean/digital PDFs with text layer | Fast (CPU) |
| `lighton` | Scanned, photographed, poor-quality PDFs | Slow (~30–60s/page via Ollama) |

### Compression modes (`compression_mode`)

Affects OCR image resolution and memory use:

`tiny` → `small` → `base` → `large` (default) → `high`

On Mac with limited RAM, try `base` or `small` for large scanned documents.

---

## Part 13 — Run on Every Boot (Optional)

### Ollama as a background service

If you installed via the macOS app, Ollama starts automatically. Otherwise use `brew services`:

```bash
brew services start ollama
```

### API server with a simple launch script

Create `~/medquery-pipeline/run-api.sh`:

```bash
#!/usr/bin/env bash
cd ~/medquery-pipeline
source .venv/bin/activate
export MEDQUERY_API_KEY="your-key-here"
exec uvicorn api_server:app --host 127.0.0.1 --port 8000 --workers 1
```

```bash
chmod +x ~/medquery-pipeline/run-api.sh
```

Run manually or add to Login Items / `launchd` if you want it always on.

---

## Part 14 — Troubleshooting

### `MEDQUERY_API_KEY is not set`

```bash
export MEDQUERY_API_KEY="your-key"
# or add to ~/.zshrc and source it
```

### `503 — No GPU workers are running`

- Worker crashed on startup — read the uvicorn terminal output.
- Common cause: missing Python dependency in the worker process.
- Fix: re-run the import sanity check from Part 4.6 inside the venv.

### `401 — Invalid or missing API key`

Header must be exactly:

```
X-Api-Key: your-key-here
```

(curl is case-insensitive; some clients require exact casing.)

### Ollama connection errors / timeouts

```bash
# Is Ollama running?
curl http://localhost:11434/api/tags

# Are models pulled?
ollama list
```

Restart Ollama:

```bash
brew services restart ollama
# or quit and reopen the Ollama app
```

### `MPS: False` on Apple Silicon

- Update macOS to 12.3+
- Reinstall PyTorch: `pip install --upgrade torch`
- Intel Macs don't have MPS — pipeline falls back to CPU (slower but functional)

### PaddleOCR import / install failures

The pipeline still works without PaddleOCR — it falls back to LightOnOCR via Ollama for all pages. Force that path:

```bash
-F "ocr_engine=lighton"
```

To retry Paddle install:

```bash
pip uninstall paddlepaddle paddleocr -y
pip install paddlepaddle==2.6.2 paddleocr
```

(Use the latest compatible versions for your Python/macOS combo if this fails.)

### Job stuck in `processing`

- Check `ocr_output/pipeline.log` and the uvicorn terminal.
- Large scanned PDFs can take **many minutes** (LightOnOCR is per-page).
- Ollama may be loading models — first request after idle is slower.

### Out of memory / Mac becomes sluggish

- Close other apps.
- Use `compression_mode=small` or `base`.
- Use `document_type=discharge_summary` (or known type) instead of `auto`.
- Use `ocr_engine=paddleocr` for digital PDFs.
- Ensure only **one** Ollama instance is running.
- Keep `MEDQUERY_GPU_WORKERS=1` on Mac.

### `Only PDF files are accepted`

Upload must be `.pdf` and sent as multipart form field `file`.

### Port 8000 already in use

```bash
lsof -i :8000
kill <PID>
```

Or run on another port:

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8001 --workers 1
```

---

## Part 15 — Full Quick-Start Checklist

Copy this checklist and tick each step:

```
[ ] Xcode CLI tools installed
[ ] Homebrew installed
[ ] Python 3.11 installed
[ ] Ollama installed
[ ] Project folder created at ~/medquery-pipeline
[ ] All .py files copied
[ ] static/index.html in place
[ ] python3.11 -m venv .venv && source .venv/bin/activate
[ ] pip packages installed (torch, fastapi, paddleocr, etc.)
[ ] ollama pull qwen3:8b
[ ] ollama pull maternion/LightOnOCR-2:1b
[ ] MEDQUERY_API_KEY exported
[ ] ollama serve (or Ollama app running)
[ ] bash start.sh
[ ] curl /health → alive_workers: 1
[ ] curl POST /extract with a test PDF → 202 + job_id
[ ] curl GET /jobs/{id} → status: completed + result JSON
```

---

## Part 16 — One-Command Reference

```bash
# Activate environment (run this in every new terminal)
cd ~/medquery-pipeline && source .venv/bin/activate

# Start API
export MEDQUERY_API_KEY="your-key" && bash start.sh

# Health
curl http://127.0.0.1:8000/health

# Submit
curl -X POST http://127.0.0.1:8000/extract \
  -H "X-Api-Key: $MEDQUERY_API_KEY" \
  -F "file=@report.pdf" \
  -F "document_type=auto" \
  -F "ocr_engine=auto"

# Poll
curl -H "X-Api-Key: $MEDQUERY_API_KEY" http://127.0.0.1:8000/jobs/JOB_ID
```

---

## Appendix — Key Files Reference

| File | Description |
|------|-------------|
| `medical_agent.py` | OCR, LLM extraction, pipeline orchestration |
| `api_server.py` | Async FastAPI (POST /extract → 202, GET /jobs/{id}) |
| `model_worker.py` | Worker process entry point (MPS/CUDA pinned) |
| `gpu_worker_pool.py` | Spawns and supervises worker pool |
| `start.sh` | Starts uvicorn with api_server |
| `api.py` | Legacy synchronous API (optional) |
| `jobs/` | Per-job workspaces and results |
| `ocr_output/pipeline.log` | Pipeline log file |
| `.env` | Optional environment variables |

---

*Generated for MedQuery Pipeline — Mac (Apple Silicon / Intel)*
