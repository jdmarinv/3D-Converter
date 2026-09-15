#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "          2D TO 3D STUDIO - OFFLINE CONVERTER"
echo "=========================================================="
echo "Iniciando servidor local y abriendo interfaz gráfica..."

source .venv/bin/activate
python gui.py
