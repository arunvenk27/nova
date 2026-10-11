#!/usr/bin/env bash
# NOVA dashboard - start on macOS / Linux / WSL.   Usage: ./start.sh [port]
set -e
cd "$(dirname "$0")"
PORT="${1:-8000}"
PY=python3; command -v python3 >/dev/null 2>&1 || PY=python
if ! "$PY" -c "import fastapi, uvicorn, numpy, pandas, scipy, sklearn, matplotlib" 2>/dev/null; then
  echo "Installing required packages (first run only)..."
  "$PY" -m pip install -r requirements.txt
fi
echo "NOVA running at http://localhost:$PORT  (Ctrl+C to stop)"
( sleep 3; (xdg-open "http://localhost:$PORT" || open "http://localhost:$PORT") >/dev/null 2>&1 || true ) &
exec "$PY" -m uvicorn backend.app:app --host 127.0.0.1 --port "$PORT"
