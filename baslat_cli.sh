#!/bin/bash
# Tahmin Aracını Başlatma Scripti
# Bu script Desktop/clean_project içindeki LOCAL venv'i kullanır.

PROJECT_DIR="/home/didim_mehmet/Desktop/clean_project"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

echo "🦅 Tahmin Aracı Başlatılıyor (Local Venv)..."
cd "$PROJECT_DIR" || exit
"$VENV_PYTHON" tahmin_et.py
