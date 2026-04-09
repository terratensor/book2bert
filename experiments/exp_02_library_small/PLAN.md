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



# Итоговые команды

## Следующие шаги

### 1. Сбор корпуса для токенизатора

```bash
go run cmd/corpus-builder/main.go \
    --sentences /mnt/archive/book2bert/data/processed/sentences_full \
    --output /mnt/archive/book2bert/data/processed/corpus_full.txt \
    --workers 32 \
    --filter-cjk true
```

**Ожидаемое время:** 2-3 часа (408 GB → один файл)

### 2. Тестирование оптимального размера словаря

```bash
# Взять выборку (10%)
head -n 100000000 /mnt/archive/book2bert/data/processed/corpus_full.txt > corpus_sample.txt

# Протестировать разные vocab_size
for size in 30000 50000 70000 100000 120000; do
    python scripts/train_tokenizer_from_corpus.py \
        --corpus corpus_sample.txt \
        --output-dir tokenizer_test_$size \
        --vocab-size $size
done
```

### 3. Сборка датасета

```bash
python scripts/build_dataset.py \
    --sentences-dir /mnt/archive/book2bert/data/processed/sentences_full \
    --tokenizer-path data/processed/tokenizer_full_optimal \
    --output-dir /mnt/archive/book2bert/data/processed/dataset_full \
    --max-length 512 \
    --val-split 0.05 \
    --batch-size 2000
```

---
