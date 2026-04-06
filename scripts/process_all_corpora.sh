#!/bin/bash
# scripts/process_all_corpora.sh

set -e  # остановка при ошибке

SEGMENTER_URL="http://localhost:8090"
OUTPUT_DIR="/mnt/archive/book2bert/data/processed/sentences_full"
WORKERS=10

echo "=== Processing militera ==="
go run cmd/process-corpus/main.go \
    --corpus /mnt/archive/corpus/militera_2023_11359_txt \
    --output "$OUTPUT_DIR" \
    --segmenter "$SEGMENTER_URL" \
    --workers $WORKERS

echo "=== Processing flibusta 2023 ==="
go run cmd/process-corpus/main.go \
    --corpus /mnt/archive/corpus/flibusta_2023_143861_txt \
    --output "$OUTPUT_DIR" \
    --segmenter "$SEGMENTER_URL" \
    --workers $WORKERS

echo "=== Processing flibusta 2025 ==="
go run cmd/process-corpus/main.go \
    --corpus /mnt/archive/corpus/flibusta_2025_txt \
    --output "$OUTPUT_DIR" \
    --segmenter "$SEGMENTER_URL" \
    --workers $WORKERS

echo "=== Processing geography ==="
go run cmd/process-corpus/main.go \
    --corpus /mnt/archive/corpus/geomatrix_geo_library_txt \
    --output "$OUTPUT_DIR" \
    --segmenter "$SEGMENTER_URL" \
    --workers $WORKERS

echo "=== All done ==="