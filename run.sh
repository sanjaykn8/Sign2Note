#!/usr/bin/env bash
# run.sh -- start all three services in the background (macOS/Linux).
# See SETUP.md for the equivalent Windows PowerShell commands (run in
# separate terminals there instead of backgrounded, since Windows
# terminals don't share Ctrl+C-to-kill-all semantics the same way).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Starting ML service (port 8000)..."
cd "$ROOT/ml_service"
python3 -m uvicorn api:app --host 127.0.0.1 --port 8000 &
ML_PID=$!

echo "Starting backend gateway (port 3001)..."
cd "$ROOT/backend"
node server.js &
BE_PID=$!

echo "Starting frontend (port 8080)..."
cd "$ROOT/frontend"
npm run dev &
FE_PID=$!

echo ""
echo "All services started."
echo "  Frontend -> http://localhost:8080  (/ for upload, /webcam for live session)"
echo "  Backend  -> http://localhost:3001"
echo "  ML API   -> http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all."

trap "kill $ML_PID $BE_PID $FE_PID 2>/dev/null" EXIT
wait $ML_PID $BE_PID $FE_PID
