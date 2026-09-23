#!/usr/bin/env bash
# Start the FastAPI server + an ngrok tunnel for a remote Flutter developer to hit during testing.
# Usage: API_KEY="some-long-random-secret" ./scripts/run_dev_server.sh
set -e
cd "$(dirname "$0")/.."

: "${API_KEY:?Set API_KEY first, e.g. API_KEY=your-secret-here ./scripts/run_dev_server.sh}"

echo "Starting FastAPI on :8000 ..."
API_KEY="$API_KEY" .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!
trap 'kill $UVICORN_PID 2>/dev/null' EXIT
sleep 2

echo "Starting ngrok tunnel — public URL will appear below. Ctrl+C to stop both."
ngrok http 8000
