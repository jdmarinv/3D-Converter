#!/bin/bash
# 2D to 3D Studio - Offline Converter macOS Launcher
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "          2D TO 3D STUDIO - OFFLINE CONVERTER"
echo "=========================================================="

if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

python3 gui.py
