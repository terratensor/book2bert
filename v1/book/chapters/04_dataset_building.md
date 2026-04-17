# Глава 4: Сборка датасета для BERT

> "Датасет — это то, что модель видит во время обучения. От его качества зависит всё."

## 4.1 Что такое датасет для BERT

BERT обучается на двух задачах:
- **MLM (Masked Language Model)** — предсказание маскированных слов
- **NSP (Next Sentence Prediction)** — определение, следует ли предложение B за A

В нашем эксперименте мы используем только MLM (NSP отключен, так как наша задача — semantic search и эмбеддинги).

**Формат одного примера:**

```json
{
  "input_ids": [2, 30651, 757, 5010, 153, 38344, 18, 3],
  "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1],
  "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0],
  "book_id": "0a2a41ed-6746-4f42-a240-a269c8e137c1",
  "genre": "Unknown"
}
```

**Что означают поля:**

| Поле | Описание |
|------|----------|
| `input_ids` | ID токенов (включая `[CLS]` в начале и `[SEP]` в конце) |
| `attention_mask` | 1 для реальных токенов, 0 для padding |
| `token_type_ids` | 0 для первого предложения, 1 для второго (не используется) |
| `book_id` | Метаданные (для анализа) |

## 4.2 Структура датасета

```
data/processed/dataset_militera_v3/
├── train/
│   ├── part_0000.jsonl
│   ├── part_0001.jsonl
│   └── ... (100 файлов)
└── val/
    ├── part_0000.jsonl
    └── ... (100 файлов)
```

**Почему JSONL, а не Parquet/Arrow?**
- Построчное чтение (streaming) — не нужно загружать всё в память
- Легко отлаживать (можно посмотреть примеры)
- Простая запись из разных воркеров

## 4.3 Проблема: двойной `[CLS]` и двойной `[SEP]`

**Что случилось:** В первой версии `build_dataset.py` мы вручную добавляли `[CLS]` и `[SEP]`, но `tokenizer.encode()` уже делал это автоматически.

```python
# Неправильно (было):
input_ids = [cls_id] + tokens + [sep_id]  # tokens уже содержат [CLS] и [SEP]

# Результат:
[2, 2, 363, 18, ..., 3, 3]  # два [CLS] в начале, два [SEP] в конце
```

**Исправление:** Убрать ручное добавление.

```python
# Правильно (стало):
input_ids = tokens  # tokenizer.encode() уже добавил [CLS] и [SEP]
```

## 4.4 Проблема: потеря `[SEP]` между предложениями

**Что случилось:** При склейке предложений через `" ".join(sentences)` мы теряли `[SEP]` между ними.

```python
# Было:
text = " ".join(sentences)  # предложение1 предложение2

# Результат: [CLS] предложение1 предложение2 [SEP]  # нет границы!
```

**Исправление:** Вставлять `[SEP]` между предложениями.

```python
# Стало:
text = " [SEP] ".join(sentences)  # предложение1 [SEP] предложение2
```

**Результат:** `[CLS] предложение1 [SEP] предложение2 [SEP]` — границы видны.

## 4.5 Проблема: сборка датасета занимала 6-8 часов

**Что случилось:** Каждый вызов `tokenizer.encode()` в группировке приводил к накладным расходам на FFI (Python → C++).

```python
# Медленно (было):
for sentence in sentences:
    tokenizer.encode(sentence)  # вызов на каждое предложение

for ts in tokenized_sentences:
    test_text = " [SEP] ".join(test_group)
    tokenizer.encode(test_text)  # еще один вызов на каждую проверку!
```

**Исправление:** 
1. Батчевая токенизация (`encode_batch`)
2. Кэширование ID предложений
3. Группировка без вызова токенизатора

```python
# Быстро (стало):
# 1. Токенизируем все предложения книги батчами
encoded_batch = tokenizer.encode_batch(texts)  # один вызов на 1000 предложений

# 2. Сохраняем ID в кэш
tokenized.append({"ids": encoded.ids, "length": len(encoded.ids), "text": text})

# 3. Группируем, используя заранее посчитанные длины
for ts in tokenized:
    needed = ts["length"] + 1  # +1 для [SEP]
    if current_tokens + needed > max_length:
        # сохраняем группу
    else:
        current_group.append(ts)
```

**Результат:** Время сборки сократилось с **6-8 часов до 17 минут**.

## 4.6 Точная группировка по токенам

**Алгоритм группировки:**

```python
def group_sentences_exact(tokenized_sentences, max_length=512):
    groups = []
    current_group = []
    # [CLS] в начале + финальный [SEP] в конце = 2 токена
    current_tokens = 2
    
    for ts in tokenized_sentences:
        # +1 для [SEP] после предложения
        needed = ts["length"] + 1
        
        if current_tokens + needed > max_length and current_group:
            groups.append(current_group)
            current_group = [ts]
            current_tokens = 2 + ts["length"] + 1
        else:
            current_group.append(ts)
            current_tokens += needed
    
    if current_group:
        groups.append(current_group)
    
    return groups
```

**Почему это правильно:**
- Учитывает `[CLS]` в начале (1 токен)
- Учитывает `[SEP]` после каждого предложения (1 токен)
- Учитывает финальный `[SEP]` (1 токен)
- Гарантирует, что группа не превысит `max_length`

## 4.7 Кодирование группы

```python
def encode_group(tokenizer, group, max_length=512):
    # Собираем ID всех предложений, вставляя [SEP] между ними
    sep_id = tokenizer.token_to_id("[SEP]")
    all_ids = []
    
    for i, ts in enumerate(group):
        if i > 0:
            all_ids.append(sep_id)
        all_ids.extend(ts["ids"])
    
    all_ids.append(sep_id)  # финальный [SEP]
    
    # [CLS] в начало
    cls_id = tokenizer.token_to_id("[CLS]")
    input_ids = [cls_id] + all_ids
    
    # Паддинг
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    
    attention_mask = [1] * len(input_ids)
    
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids.extend([tokenizer.token_to_id("[PAD]")] * padding_length)
        attention_mask.extend([0] * padding_length)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": [0] * max_length
    }
```

## 4.8 Запуск сборки датасета

```bash
python scripts/build_dataset.py \
    --sentences-dir data/processed/sentences_militera_v3 \
    --tokenizer-path data/processed/tokenizer_militera_v3 \
    --output-dir data/processed/dataset_militera_v3 \
    --max-length 512 \
    --val-split 0.05 \
    --batch-size 2000
```

**Результат:**
```
Processing books: 11351it [16:59, 11.13it/s]

=== Done ===
Train examples: 3,931,780
Val examples: 223,044
```

## 4.9 Итоговая статистика датасета

| Показатель | Значение |
|------------|----------|
| Train примеры | 3,931,780 |
| Val примеры | 223,044 |
| **Всего** | **4,154,824** |
| Размер на диске | ~25 GB |
| Время сборки | **17 минут** |
| Скорость | 11.13 книг/сек |

## 4.10 Промежуточные итоги

| Проблема | Решение | Эффект |
|----------|---------|--------|
| Двойной `[CLS]`/`[SEP]` | Убрать ручное добавление | Корректный формат |
| Потеря `[SEP]` между предложениями | Вставка `[SEP]` при склейке | Модель видит границы |
| Медленная токенизация | Батчевая + кэширование | **6-8 часов → 17 минут** |
| Неточная группировка | Алгоритм с учетом `[SEP]` | Без обрезания (кроме 0.0008%) |

**Ключевой вывод:** Оптимизация сборки датасета (батчевая токенизация + кэширование) дала ускорение в **20-30 раз** без потери качества. Теперь мы можем пересобрать датасет за 17 минут вместо 6-8 часов, что критически важно для итеративной разработки.

---
[Далее: Глава 5 - Архитектура BERT...]
