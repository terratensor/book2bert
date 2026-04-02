# Эксперимент book2bert

## Цель
Создание BERT-модели с нуля для понимания архитектуры трансформеров и получения полезной модели для работы с корпусом нехудожественной литературы (40 ГБ, 180k книг).

## Фаза 0: Подготовка окружения

### 0.1 Установка зависимостей
```bash
pip install -r requirements.txt
```

### 0.2 Запуск сервиса сегментации
```bash
cd services/segmenter
python app.py  # порт 8090
```

## Фаза 1: Обработка 50 книг (чистый эксперимент)

### 1.1 Исходные данные
- 50 книг философской тематики (27.6 МБ)
- Путь: `data/sample/texts/`
- Кодировка: Windows-1251 (требуется конвертация)

### 1.2 Сегментация предложений
```bash
export CUDA_VISIBLE_DEVICES=0
go run cmd/process-books/main.go \
    --books data/sample/texts \
    --output data/processed/sentences \
    --segmenter http://localhost:8090
```
**Результат:** 152,444 предложения

### 1.3 Обучение токенизатора (пробный запуск — НЕВЕРНЫЙ)
```bash
python scripts/train_tokenizer.py --vocab_size 30000
```
**Проблема:** В словаре нет целых слов (учение, граница, материя) — только субворды.

### 1.4 Корректное обучение токенизатора
```bash
python scripts/train_tokenizer.py --vocab_size 50000
```
**Ожидание:** Словарь содержит целые слова.

### 1.5 Проверка словаря
```bash
head -100 data/processed/tokenizer/vocab.txt | grep -E "^(учение|граница|материя|философия)$"
```
**Ожидание:** Найдены все слова.

### 1.6 Создание датасета
```bash
python scripts/build_dataset.py \
    --sentences-dir data/processed/sentences \
    --tokenizer-path data/processed/tokenizer \
    --output-dir data/processed/dataset \
    --max-length 512 \
    --val-split 0.05
```
**Результат:** train/val примеры для BERT.

### 1.7 Обучение моделей

#### BERT-tiny (22M)
```bash
python training/train_tiny.py \
    --config training/config/tiny_bert.yaml \
    --dataset_path data/processed/dataset \
    --output_dir data/models
```

#### BERT-small (45M)
```bash
python training/train_small.py \
    --config training/config/small_bert.yaml \
    --dataset_path data/processed/dataset \
    --output_dir data/models
```

#### BERT-base (108M)
```bash
python training/train_base.py \
    --config training/config/base_bert.yaml \
    --dataset_path data/processed/dataset \
    --output_dir data/models
```

### 1.8 Оценка качества
```bash
# MLM предсказания
python scripts/evaluate_mlm.py

# Визуализация attention
python scripts/visualize_attention.py

# Анализ [CLS] эмбеддингов
python scripts/analyze_cls.py
```

## Фаза 2: Масштабирование на 4 ГБ (военно-исторический корпус)

### 2.1 Подготовка данных
- 11,200 файлов военно-исторической тематики
- Сегментация предложений
- Обучение токенизатора (vocab_size=50000)

### 2.2 Обучение моделей
- BERT-base на 4 ГБ данных

## Фаза 3: Полный корпус (40 ГБ)
- Аренда мощностей при необходимости

## Требования к документации
- Каждый шаг фиксируется в NOTES.md
- Результаты сохраняются в data/logs/
- Модели сохраняются с меткой времени