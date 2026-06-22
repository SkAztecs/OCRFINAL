#!/usr/bin/env bash
# start.sh — Launch the MedQuery stack
#
# Usage:
#   bash start.sh            # production
#   bash start.sh --reload   # hot-reload for development
#
# Required env var:
#   MEDQUERY_API_KEY   — shared secret for X-Api-Key header
#
# The warm worker (model_worker.py) is started automatically by FastAPI
# via the lifespan hook — you do NOT need to run it separately.

set -euo pipefail

: "${MEDQUERY_API_KEY:?ERROR: MEDQUERY_API_KEY env var must be set}"

HOST="127.0.0.1"   # localhost only — Nginx handles public traffic
PORT=8000
WORKERS=1          # single-GPU: always 1 uvicorn worker (the warm worker
                   # serialises GPU access internally via asyncio.Lock)
LOG_LEVEL="info"

echo "──────────────────────────────────────────────"
echo " MedQuery API  →  http://${HOST}:${PORT}"
echo " Public traffic via Nginx on port 80/443"
echo "──────────────────────────────────────────────"

if [[ "${1-}" == "--reload" ]]; then
    echo " Mode: DEVELOPMENT (hot reload)"
    uvicorn api:app \
        --host "$HOST" --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        --reload
else
    echo " Mode: PRODUCTION"
    uvicorn api:app \
        --host "$HOST" --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL"
fi
