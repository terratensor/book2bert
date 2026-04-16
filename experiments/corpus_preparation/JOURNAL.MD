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