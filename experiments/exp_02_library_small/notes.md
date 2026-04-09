# Эксперимент 2: Полный корпус + BERT-small

## Цель
Обучить BERT-small (45M) на полном корпусе (180k книг, 145 ГБ текста)

## Ожидаемые результаты
- Val loss: <2.5
- Perplexity: <12
- Точность на частях слов: >30%

## План
1. Обработка полного корпуса
2. Тестирование vocab_size (30k-120k)
3. Обучение токенизатора
4. Сборка датасета
5. Обучение BERT-small

## Статус
⏳ Планирование

## 2026-04-06: Проблема исправления разрывов слов

### Типы разрывов
1. Разрыв пробелом внутри слова: "внима тельно" → "внимательно"
2. Перенос с дефисом: "внима- \n тельно" → "внимательно"
3. Отсутствие пробелов: "подводяитог" → "подводя итог"

### Решение
- Тип 2: исправляем (fixHyphenatedWords)
- Тип 1: не исправляем (нужен словарь)
- Тип 3: не возникает при правильной очистке

### Почему не исправляем тип 1
- Нужен словарь русского языка (100k+ слов)
- Без словаря будут ложные срабатывания
- Модель BERT может выучить эти паттерны сама


## 2026-04-07: Обработка полного корпуса завершена

### Статистика
| Корпус | Файлы | Предложения |
|--------|-------|-------------|
| Flibusta 2023 | 143,861 | 776,606,995 |
| Flibusta 2025 | 21,388 | 189,639,159 |
| Militera | 11,361 | 107,900,756 |
| География | 191 | 7,247,353 |
| **Итого** | **176,801** | **1,081,394,263** |

- Размер на диске: 408 GB
- Ошибок: 0
- Пропущено: 394 (0.22%)

### Следующие шаги
1. Сбор корпуса для токенизатора (2-3 часа)
2. Тестирование оптимального vocab_size
3. Сборка датасета (3-5 часов)
4. Обучение BERT-small (100-120 часов)

## 2026-04-07: Пересмотр выбора vocab_size

### Проблема
- BERT не понимает слова из субвордов (тест показал 0%)
- Нужны целые слова в словаре

### Решение
- Увеличить vocab_size до 120,000
- Увеличить модель до BERT-small (45M)
- Баланс: память vs понимание редких слов

### Сравнение
| vocab_size | Целых слов | Память (fp32) |
|------------|------------|---------------|
| 100,000 | 31% | 154 MB |
| 120,000 | 35% | 185 MB |
| 150,000 | 38% | 230 MB |

### Выбор: 120,000

EOF

git add NOTES.md
git commit -m "decision: vocab_size = 120,000 (пересмотр)

- Проблема: BERT не понимает слова из субвордов
- Решение: больше целых слов в словаре
- Выбран компромисс: 120,000 (35% целых слов)
- Модель: BERT-small (45M) для большей ёмкости"


```
(venv) audetv@home:/mnt/work/audetv/go/src/github.com/terratensor/book2bert$ python scripts/train_tokenizer_from_corpus.py \                                                                      --corpus /mnt/archive/book2bert/data/processed/corpus_full.txt \
    --output-dir data/processed/tokenizer_full \
    --vocab-size 120000
Training tokenizer on /mnt/archive/book2bert/data/processed/corpus_full.txt...
[00:32:51] Pre-processing files (150272 M ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                100%[00:00:51] Tokenize words                 ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 53922870 / 53922870
[00:00:30] Count pairs                    ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 53922870 / 53922870
[00:04:25] Compute merges                 ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 119377   /   119377
Tokenizer saved to data/processed/tokenizer_full
Vocabulary size: 120000

````

```
Processing books: 176234it [4:20:29,  6.84it/s]WARNING: Group exceeded 512 tokens (638), truncating!
WARNING: Group exceeded 512 tokens (513), truncating!
WARNING: Group exceeded 512 tokens (528), truncating!
WARNING: Group exceeded 512 tokens (756), truncating!
WARNING: Group exceeded 512 tokens (570), truncating!
Processing books: 176240it [4:20:29, 12.41it/s]WARNING: Group exceeded 512 tokens (705), truncating!
Processing books: 176263it [4:20:31, 10.69it/s]WARNING: Group exceeded 512 tokens (517), truncating!
Processing books: 176407it [4:20:43, 11.28it/s]

=== Done ===
Train examples: 40817321
Val examples: 2150997
(venv) audetv@home:/mnt/
```
```
audetv@home:/mnt/work/audetv/go/src/github.com/terratensor/book2bert$ du -h /mnt/archive/book2bert/data/processed/dataset_full/
226G	/mnt/archive/book2bert/data/processed/dataset_full/train
12G	/mnt/archive/book2bert/data/processed/dataset_full/val
238G	/mnt/archive/book2bert/data/processed/dataset_full/
```