# Приложение А: Все команды для воспроизведения эксперимента

## 1. Запуск сервиса сегментации
```bash
cd services/segmenter
gunicorn -w 8 -b 0.0.0.0:8090 app:app
```

## 2. Обработка militera корпуса
```bash
go run cmd/process-corpus/main.go \
    --corpus /mnt/archive/corpus/militera_2023_11359_txt \
    --output data/processed/sentences_militera_v3 \
    --segmenter http://localhost:8090 \
    --workers 10
```

## 3. Сбор корпуса для токенизатора
```bash
go run cmd/corpus-builder/main.go \
    --sentences data/processed/sentences_militera_v3 \
    --output data/processed/corpus_militera_v3.txt \
    --workers 16 \
    --filter-cjk true
```

## 4. Обучение токенизатора
```bash
python scripts/train_tokenizer_from_corpus.py \
    --corpus data/processed/corpus_militera_v3.txt \
    --output-dir data/processed/tokenizer_militera_v3 \
    --vocab-size 50000
```

## 5. Сборка датасета
```bash
python scripts/build_dataset.py \
    --sentences-dir data/processed/sentences_militera_v3 \
    --tokenizer-path data/processed/tokenizer_militera_v3 \
    --output-dir data/processed/dataset_militera_v3 \
    --max-length 512 \
    --val-split 0.05 \
    --batch-size 2000
```

## 6. Обучение модели
```bash
python training/train_tiny_streaming.py \
    --config training/config/tiny_bert_streaming.yaml \
    --dataset_path data/processed/dataset_militera_v3 \
    --output_dir data/models/tiny_bert_militera_v3
```

## 7. Анализ результатов
```bash
# MLM предсказания
python scripts/evaluate_mlm_v3.py --model_dir data/models/tiny_bert_militera_v3

# Визуализация attention
python scripts/visualize_attention_v3.py \
    --model_dir data/models/tiny_bert_militera_v3 \
    --text "Генерал Шкуро командовал дивизией"

# Анализ [CLS] эмбеддингов
python scripts/analyze_cls_v3.py --model_dir data/models/tiny_bert_militera_v3
```
