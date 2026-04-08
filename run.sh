#!/usr/bin/env bash
# run.sh – start all three services in separate terminals (macOS/Linux)
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Starting ML service (port 8000)..."
cd "$ROOT/ml_service"
uvicorn api:app --host 0.0.0.0 --port 8000 &
ML_PID=$!

echo "Starting backend (port 3001)..."
cd "$ROOT/backend"
node server.js &
BE_PID=$!

echo "Starting frontend (port 3000)..."
cd "$ROOT/frontend/react-app"
npm start &
FE_PID=$!

echo ""
echo "All services started."
echo "  Frontend → http://localhost:3000"
echo "  Backend  → http://localhost:3001"
echo "  ML API   → http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all."

wait $ML_PID $BE_PID $FE_PID
