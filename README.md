# Запуск

### Запуск сервиса сегментации
```bash
cd services/segmenter
python app.py  # порт 8090

## 1. Собрать sentences_militera с фильтрацией
go run cmd/process-corpus/main.go \
    --corpus /mnt/archive/corpus/militera_2023_11359_txt \
    --output data/processed/sentences_militera_filtered \
    --segmenter http://localhost:8090 \
    --workers 10

## 2. Обучить токенизатор
python scripts/train_tokenizer.py \
    --sentences-dir data/processed/sentences_militera_filtered \
    --output-dir data/processed/tokenizer_militera_filtered \
    --vocab-size 50000 \
    --min-frequency 2

## 3. Собрать датасет
python scripts/build_dataset.py \
    --sentences-dir data/processed/sentences_militera_filtered \
    --tokenizer-path data/processed/tokenizer_militera_filtered \
    --output-dir data/processed/dataset_militera_filtered \
    --max-length 512 \
    --val-split 0.05

## 4. Запустить обучение
python training/train_tiny_streaming.py \
    --config training/config/tiny_bert_streaming.yaml \
    --dataset_path data/processed/dataset_militera_filtered \
    --output_dir data/models/tiny_bert_militera_filtered