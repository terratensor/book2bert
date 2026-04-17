# 1. Запуск анализа

```bash
# Создаём директорию для результатов
mkdir -p data/analysis

# Запускаем анализ
go run cmd/analyze-corpus/main.go \
    --dir data/processed/sentences_full \
    --workers 32 \
    --output data/analysis
```

## Что мы получим
Файл	Содержание
stats_summary.json	Общая статистика, перцентили, языковой состав
stats_length_histogram.csv	Распределение длин предложений
stats_per_file.csv	Статистика по каждому файлу

### 3.1 Сначала сохраним длинные предложения

```bash
# Сохраним все предложения длиннее 1000 символов
python scripts/save_long_sentences.py \
    --sentences-dir data/processed/sentences_full \
    --output data/analysis/long_sentences.jsonl \
    --min-length 1000
```

### 3.2 Построим графики

```bash
python scripts/analyze_corpus_stats.py \
    --csv data/analysis/stats_length_histogram.csv \
    --json data/analysis/stats_summary.json \
    --output data/analysis
```

### 4.2 Аномально длинные предложения
После сохранения long_sentences.jsonl мы сможем посмотреть, что это за монстры:

```bash
# Посмотрим первые 5 длинных предложений
head -5 data/analysis/long_sentences.jsonl | jq -r '.text' | head -20

# Или посчитаем, сколько их по жанрам
cat data/analysis/long_sentences.jsonl | jq -r '.genre' | sort | uniq -c | sort -rn
```


## 3. Запуск фильтрации

```bash
# Создаём выходную директорию
mkdir -p data/processed/sentences_filtered

# Запускаем фильтрацию
go run cmd/filter-corpus/main.go \
    --input data/processed/sentences_full \
    --output data/processed/sentences_filtered \
    --workers 32 \
    --num-outputs 10000
```
### Результаты
```bash
2026/04/17 01:38:20 [PROGRESS] 176406/176407 files (100.0%), kept: 742718952/1001364952 sentences (74.2%)

============================================================
FILTERING COMPLETE
============================================================
Time:           1h16m40s
Input files:    176407
Total sentences: 1001367746
Kept sentences:  742720187 (74.17%)
Filtered out:    258647559 (25.83%)
------------------------------------------------------------
Filter reasons:
  uppercase_ratio: 2581422 (0.26%)
  punct_ratio    : 7895 (0.00%)
  min_length     : 181479482 (18.12%)
  language       : 68500100 (6.84%)
  garbage        : 263059 (0.03%)
  list_marker    : 4375832 (0.44%)
  digit_ratio    : 1432929 (0.14%)
  max_length     : 6840 (0.00%)
============================================================
```

### Запуск анализа отфильтрованного корпуса
```bash
go run cmd/analyze-filtered/main.go \
    --input data/processed/sentences_filtered \
    --workers 32
```    

```bash
============================================================
FILTERED CORPUS ANALYSIS
============================================================
Files:           10000
Total sentences: 742720181
Total chars:     76388273002
Avg length:      102.85
Min length:      20
Max length:      2000
Time:            3m50s
------------------------------------------------------------
Percentiles:
  75%: 135
  95%: 246
  99%: 367
  1%: 21
  10%: 30
  90%: 198
  5%: 25
  25%: 48
  50%: 83
============================================================
```

### Запуск дедупликации
```bash
# Создаём выходную директорию
mkdir -p data/processed/sentences_dedup

# Запускаем дедупликацию
go run cmd/deduplicate-corpus/main.go \
    --input data/processed/sentences_filtered \
    --output data/processed/sentences_dedup \
    --workers 32 \
    --num-outputs 10000
```    
```bash
2026/04/17 10:33:25 [PROGRESS] 9984/10000 files (99.8%), total: 742494192, kept: 544464999, dup: 198029187 (26.7%)

============================================================
DEDUPLICATION COMPLETE
============================================================
Time:            1h18m44s
Input files:     10000
Total sentences: 742720181
Kept sentences:  544588775 (73.32%)
Duplicates:      198131406 (26.68%)
============================================================
```
## Итоги дедупликации

| Показатель | Значение |
|------------|----------|
| **Время** | 1 час 18 минут |
| **Всего предложений** | 742,720,181 |
| **Уникальных** | 544,588,775 (**73.32%**) |
| **Дубликатов** | 198,131,406 (**26.68%**) |

---

## Потребление ресурсов

| Ресурс | Пиковое значение |
|--------|------------------|
| **RAM** | ~122 GB |
| **Swap** | ~20 GB |
| **Итого память** | ~142 GB |

**Вывод:** Дедупликация 742 млн предложений требует ~140 GB памяти. Ваш сервер справился, но был на пределе.

---

## Сравнение этапов

| Этап | Предложений | Размер | Доля от исходного |
|------|-------------|--------|-------------------|
| Исходный корпус | 1,001,367,746 | 408 GB | 100% |
| После фильтрации | 742,720,181 | 92 GB | 74.2% |
| **После дедупликации** | **544,588,775** | **~65-70 GB** | **54.4%** |

**Общее сокращение:** 1 млрд → 545 млн (**45.6% удалено**)

---

## Что удалили

| Этап | Удалено | Доля от исходного |
|------|---------|-------------------|
| Фильтрация (мусор) | 258,647,559 | 25.8% |
| Дедупликация (повторы) | 198,131,406 | 19.8% |
| **Итого** | **456,778,971** | **45.6%** |

---

## Что теперь

| Параметр | Значение |
|----------|----------|
| **Уникальных предложений** | **544,588,775** |
| **Ожидаемый размер датасета** | **~65-70 GB** (сжатый) |
| **Количество выходных файлов** | 10,000 |
| **Формат** | JSONL.gz |

---

## Следующие шаги

| Шаг | Статус | Что делаем |
|-----|--------|-----------|
| 1. Анализ корпуса | ✅ Завершён | — |
| 2. Фильтрация | ✅ Завершена | — |
| 3. **Дедупликация** | ✅ **Завершена** | — |
| 4. Семантическая дедупликация | ⏸ Опционально | Можно пропустить |
| 5. Балансировка | ⏸ Опционально | Можно пропустить |
| 6. Сборка датасета для BERT | ⏳ **Следующий шаг** | build_dataset.py |

---

## Рекомендация

**Семантическая дедупликация не нужна.** Мы уже удалили 26.8% точных дубликатов. Семантические дубликаты (разные формулировки одного смысла) в военных текстах встречаются редко.

**Балансировка не нужна.** Корпус уже сбалансирован по жанрам (история, биографии, философия и т.д.).

---

## Переходим к сборке датасета для BERT

```bash
python scripts/build_dataset.py \
    --sentences-dir data/processed/sentences_dedup \
    --tokenizer-path data/processed/tokenizer_full \
    --output-dir data/processed/dataset_ideal \
    --max-length 512 \
    --val-split 0.05 \
    --batch-size 2000 \
    --sample-ratio 0.15
```

**Ожидаемый размер датасета:** ~65-70 GB → ~10-12 GB после сборки (примеры по 512 токенов)

## Запуск сборки корпуса для токенизатора
```bash
go run cmd/corpus-builder/main.go \
    --sentences data/processed/sentences_dedup \
    --output data/processed/corpus_clean.txt \
    --workers 32 \
    --filter-cjk true
```    
```bash
2026/04/17 12:31:50 === Corpus Builder (Go) ===
2026/04/17 12:31:50 Sentences dir: data/processed/sentences_dedup
2026/04/17 12:31:50 Output file: data/processed/corpus_clean.txt
2026/04/17 12:31:50 Workers: 32
2026/04/17 12:31:50 Filter CJK: true
2026/04/17 12:31:50 Found 10000 JSONL files
2026/04/17 12:38:21 Done! Total sentences: 544588775
2026/04/17 12:38:21 Corpus saved to: data/processed/corpus_clean.txt
```