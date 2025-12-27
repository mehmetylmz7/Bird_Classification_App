#!/bin/bash
# Web Uygulamasını Başlatma Scripti
# Bu script Desktop/clean_project içindeki LOCAL venv'i kullanır.

PROJECT_DIR="/home/didim_mehmet/Desktop/clean_project"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

echo "🦅 Yeni Konumdan Başlatılıyor (Local Venv): $PROJECT_DIR"
cd "$PROJECT_DIR" || exit
"$VENV_PYTHON" -m streamlit run "$PROJECT_DIR/web_app.py"
