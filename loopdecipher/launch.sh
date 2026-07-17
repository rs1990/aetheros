#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d node_modules ]; then
  echo "Installing dependencies..."
  npm install
fi

npm run dev &
DEV_PID=$!
trap 'kill "$DEV_PID" 2>/dev/null' EXIT

echo "Starting Intervue..."
until curl -s -o /dev/null http://localhost:3000; do
  sleep 0.5
done

echo "Ready at http://localhost:3000"
open http://localhost:3000 2>/dev/null || true

wait "$DEV_PID"
