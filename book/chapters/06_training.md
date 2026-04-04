# Глава 6: Обучение — от теории к практике

> "Обучение модели — это искусство баланса между скоростью, памятью и качеством."

## 6.1 Проблема: датасет не влезает в память

Наш датасет — **4.15 млн примеров, 25 ГБ**. Загрузить его целиком в оперативную память невозможно.

**Решение:** Streaming — читаем данные построчно, не загружая всё.

```python
class StreamingJSONLDataset(IterableDataset):
    def __init__(self, data_dir, split):
        self.files = list(Path(data_dir).glob(f"{split}/*.jsonl"))
    
    def __iter__(self):
        for filepath in self.files:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
```

**Как это работает:**
- `IterableDataset` — специальный класс PyTorch для потоковых данных
- `num_workers=0` (нельзя использовать multiprocessing со streaming)
- Данные читаются по одному примеру за раз

## 6.2 Mixed Precision (fp16) — память в 2 раза меньше

**Проблема:** Модель с 30M параметров в fp32 занимает ~120 МБ (веса) + градиенты + оптимизатор ≈ 500 МБ. Но с активациями и батчами память растет.

**Решение:** Использовать fp16 (16-bit floating point).

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(input_ids, attention_mask, token_type_ids, labels)
    loss = outputs['loss']

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Что изменилось:**

| Тип | Размер | Когда использовать |
|-----|--------|-------------------|
| fp32 | 4 байта | Точные вычисления (оптимизатор) |
| fp16 | 2 байта | Быстрые вычисления (forward/backward) |

**Влияние на качество:** Для BERT-base разница <0.5% по метрикам. На нашей модели разница незаметна.

## 6.3 Masked Language Modeling (MLM)

**Что это:** Маскируем 15% токенов, модель учится их предсказывать.

```python
def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    labels = input_ids.clone()
    
    # 1. Выбираем 15% токенов для маскирования
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # ignore index
    
    # 2. Из выбранных 15%:
    #    - 80% заменяем на [MASK]
    #    - 10% заменяем на случайный токен
    #    - 10% оставляем без изменений
    
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id
    
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.vocab_size, labels.shape)
    input_ids[indices_random] = random_words[indices_random]
    
    return input_ids, labels
```

**Пример:**

```
Исходный текст:  [CLS] Философия есть учение о всеобщем . [SEP]
Маскированный:   [CLS] Философия есть [MASK] о всеобщем . [SEP]
Labels:          -100  -100       -100 5010   -100 -100    -100 -100
                              ↑
                    нужно предсказать "учение" (ID 5010)
```

## 6.4 OneCycleLR — правильный scheduler

**Проблема:** Learning rate — самый важный гиперпараметр. Если он слишком большой — модель расходится. Если слишком маленький — учится слишком медленно.

**Решение:** OneCycleLR — начинаем с маленького LR, разогреваемся до максимума, затем снижаем.

```python
steps_per_epoch = train_examples // batch_size
total_steps = steps_per_epoch * num_epochs

scheduler = OneCycleLR(
    optimizer,
    max_lr=5e-5,
    total_steps=total_steps,
    pct_start=0.1  # 10% времени на разогрев
)
```

**График LR:**
```
LR
  ^
  |         /‾‾‾‾‾‾‾\
  |        /        \
  |       /          \
  |      /            \
  |_____/              \______
  +--------------------------> steps
       разогрев    снижение
       (10%)        (90%)
```

**Важно:** `total_steps` должно быть точным! Иначе scheduler упадет.

```python
# Рассчитываем автоматически при старте
train_examples = count_examples(dataset_path, "train")  # 3,931,780
steps_per_epoch = train_examples // batch_size  # 245,736
total_steps = steps_per_epoch * num_epochs  # 1,228,680
```

## 6.5 Стратегия чекпоинтов

**Проблема:** Обучение длится 50-60 часов. При сбое мы теряем весь прогресс.

**Решение:** Сохраняем чекпоинты с разной частотой.

| Тип чекпоинта | Частота | Размер | Назначение |
|--------------|---------|--------|------------|
| `last_checkpoint.pt` | Каждые 1000 шагов (перезапись) | 360 MB | Быстрое восстановление после сбоя |
| `checkpoint_step_N.pt` | Каждые 10000 шагов | 360 MB | Возобновление с любой точки |
| `best_model.pt` | Когда val_loss улучшается | 360 MB | Лучшая модель |
| `epoch_N.pt` | В конце каждой эпохи | 360 MB | Long-term хранение |

**Структура чекпоинта:**

```python
checkpoint = {
    'epoch': epoch,
    'global_step': global_step,
    'step_in_epoch': step_in_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_val_loss': best_val_loss,
    'config': config
}
```

## 6.6 Возобновление обучения

```python
def resume_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint['global_step']
    step_in_epoch = checkpoint['step_in_epoch']
    
    return start_epoch, global_step, step_in_epoch
```

**Важно:** При возобновлении нужно пропустить уже обработанные шаги:

```python
# Пропускаем step_in_epoch шагов
data_iter = iter(dataloader)
for _ in range(step_in_epoch):
    next(data_iter)

# Продолжаем обучение
for step, batch in enumerate(data_iter, start=step_in_epoch):
    # ...
```

## 6.7 Логирование

Мы используем три вида логирования:

### 1. CSV (пошаговые метрики)

```csv
global_step,epoch,step_in_epoch,loss,learning_rate,grad_norm
1000,1,1000,8.234,4.2e-05,1.23
2000,1,2000,7.891,4.5e-05,1.45
```

### 2. CSV (эпохальные метрики)

```csv
epoch,train_loss,val_loss,val_perplexity,learning_rate,best_val_loss,time_seconds
1,8.94,7.88,2634,4.85e-05,7.88,36264
2,7.94,7.81,2456,4.85e-05,7.81,44435
```

### 3. TensorBoard (визуализация в реальном времени)

```bash
tensorboard --logdir=data/models/tiny_bert_militera_v3/tensorboard
```

**Что видим в TensorBoard:**
- `train/step_loss` — пошаговая кривая обучения
- `train/learning_rate` — график LR
- `epoch/train_loss` — средний loss за эпоху
- `epoch/val_loss` — валидационный loss
- `epoch/val_perplexity` — perplexity (exp(val_loss))

## 6.8 Градиентное накопление (Gradient Accumulation)

**Проблема:** Большой batch size требует много памяти. Мы хотим эффективный batch size = 16, но памяти хватает только на 8.

**Решение:** Делаем 2 маленьких шага, накапливаем градиенты, обновляем веса раз в 2 шага.

```python
gradient_accumulation_steps = 2
effective_batch_size = batch_size * gradient_accumulation_steps  # 8 * 2 = 16

for step, batch in enumerate(dataloader):
    loss = model(batch) / gradient_accumulation_steps
    loss.backward()
    
    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 6.9 Полный цикл обучения

```python
def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, tokenizer):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader):
        # 1. Переносим на GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # 2. Маскируем токены
        masked_input_ids, labels = mask_tokens(input_ids, tokenizer)
        
        # 3. Forward (fp16)
        with autocast():
            outputs = model(masked_input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
        
        # 4. Backward (fp16)
        scaler.scale(loss).backward()
        
        # 5. Обновляем веса
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

## 6.10 Скорость обучения

**Наша конфигурация:**
- GPU: NVIDIA GeForce RTX 5060 Ti (16 GB)
- batch_size: 8 (эффективный 16)
- gradient_accumulation_steps: 2
- fp16: включен

**Реальная скорость:**
```
Epoch 1/Training: 245736it [11:02:30, 6.18it/s, loss=8.2341, lr=4.2e-05]
```

**Расчет:**
- 245,736 шагов × 0.16 секунды = 39,317 секунд ≈ **10.9 часов** на эпоху

## 6.11 Мониторинг прогресса

**По логам:**
```bash
tail -f data/models/tiny_bert_militera_v3/training.log
```

**По TensorBoard:**
```bash
tensorboard --logdir=data/models/tiny_bert_militera_v3/tensorboard
```

**По CSV:**
```bash
tail -20 data/models/tiny_bert_militera_v3/csv/step_metrics.csv
```

## 6.12 Промежуточные итоги

| Компонент | Что дает | Цена |
|-----------|----------|------|
| Streaming | Возможность обучать на 25 ГБ данных | Медленнее на 10-20% |
| Mixed Precision (fp16) | Память в 2 раза меньше, скорость на 30% выше | Минимальная потеря качества |
| Gradient Accumulation | Эффективный batch size 16 при физическом 8 | На 20% медленнее |
| OneCycleLR | Стабильное обучение | Нужно точно рассчитать total_steps |
| Чекпоинты | Возможность возобновления после сбоев | Дополнительное место на диске |

**Ключевой вывод:** Комбинация streaming + fp16 + gradient accumulation позволяет обучать модель на данных, которые не влезают в память, с приемлемой скоростью (10 часов на эпоху).

---
[Далее: Глава 7 - Результаты...]
