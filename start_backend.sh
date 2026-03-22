#!/usr/bin/env bash
# Start the AutoML Platform backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR"

echo "Installing backend dependencies..."
pip install -r backend/requirements.txt -q

echo "Starting backend on http://0.0.0.0:8000 ..."
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
