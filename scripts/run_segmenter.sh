#!/bin/bash
# Запуск сервиса сегментации

cd "$(dirname "$0")/.." || exit

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -r services/segmenter/requirements.txt

echo "Starting segmenter service on http://localhost:8090"
cd services/segmenter
python app.py