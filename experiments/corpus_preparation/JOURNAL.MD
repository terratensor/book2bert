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