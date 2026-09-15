#!/bin/bash
# 2D to 3D Studio - Offline Converter Linux Launcher
set -e

# Resolve script root directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "          2D TO 3D STUDIO - OFFLINE CONVERTER"
echo "=========================================================="

# 1. Activate Python virtual environment if found
if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (venv)..."
    source venv/bin/activate
fi

# 2. Check for Python executable
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3 was not found in PATH."
    echo "Install it via your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip ffmpeg"
    echo "  Fedora:        sudo dnf install python3 python3-pip ffmpeg"
    echo "  Arch:          sudo pacman -S python python-pip ffmpeg"
    exit 1
fi

echo "[INFO] Starting 2D to 3D Studio Web GUI..."
$PYTHON_CMD gui.py
