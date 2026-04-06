## 5. План нового эксперимента

### Фаза 1: Подготовка корпуса (20-30 часов)

```bash
# Обработка всех txt.gz файлов (180k)
go run cmd/process-corpus/main.go \
    --corpus /mnt/archive/corpus/all_genres \
    --output /mnt/archive/book2bert/data/processed/sentences_full \
    --segmenter http://localhost:8090 \
    --workers 10
```

### Фаза 2: Тестирование словаря (2-3 часа)

```bash
# Собрать выборку (10%)
go run cmd/corpus-builder/main.go \
    --sentences data/processed/sentences_full \
    --output data/processed/corpus_sample.txt \
    --sample 0.1 \
    --workers 16

# Тестировать разные vocab_size
python scripts/train_tokenizer_from_corpus.py --corpus corpus_sample.txt --vocab-size 30000
python scripts/train_tokenizer_from_corpus.py --corpus corpus_sample.txt --vocab-size 50000
python scripts/train_tokenizer_from_corpus.py --corpus corpus_sample.txt --vocab-size 70000
python scripts/train_tokenizer_from_corpus.py --corpus corpus_sample.txt --vocab-size 100000
python scripts/train_tokenizer_from_corpus.py --corpus corpus_sample.txt --vocab-size 120000
```

### Фаза 3: Сборка датасета (10-20 часов)

```bash
# Собрать корпус для токенизатора (полный)
go run cmd/corpus-builder/main.go \
    --sentences data/processed/sentences_full \
    --output data/processed/corpus_full.txt \
    --workers 16

# Обучить финальный токенизатор с оптимальным vocab_size
python scripts/train_tokenizer_from_corpus.py \
    --corpus data/processed/corpus_full.txt \
    --output-dir data/processed/tokenizer_full_optimal \
    --vocab-size 80000  # например

# Собрать датасет
python scripts/build_dataset.py \
    --sentences-dir data/processed/sentences_full \
    --tokenizer-path data/processed/tokenizer_full_optimal \
    --output-dir data/processed/dataset_full \
    --max-length 512 \
    --val-split 0.05 \
    --batch-size 2000
```

### Фаза 4: Обучение BERT-small (100-150 часов)

```bash
python training/train_small_streaming.py \
    --config experiments/exp_02_full_corpus_small/config.yaml \
    --dataset_path data/processed/dataset_full \
    --output_dir data/models/full_small
```
