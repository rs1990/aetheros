#!/bin/bash
set -e

echo "Starting Supply Chain Intelligence Platform..."

cd "$(dirname "$0")"

# Backend
python -m uvicorn backend.main:app --port 8001 --reload &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID (http://localhost:8001)"
echo "API docs: http://localhost:8001/docs"

# Frontend
cd frontend && npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID (http://localhost:5173)"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
