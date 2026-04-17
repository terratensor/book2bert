# book2bert

[![Go Version](https://img.shields.io/badge/Go-1.25+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![Python Version](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python)](https://python.org)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**BERT модель для военно-исторических текстов, обученная с нуля**

Этот проект содержит BERT-tiny модель (29.9M параметров), обученную на военно-историческом корпусе militera (11,361 книга, 93M предложений, 14 ГБ текста).

## Возможности

- ✅ **MLM (Masked Language Model)** — предсказание маскированных слов
- ✅ **Semantic search** — поиск похожих текстов через [CLS] эмбеддинги
- ✅ **Извлечение эмбеддингов** — для downstream задач (классификация, кластеризация)
- ✅ **Понимание военной терминологии** (дивизия, корпус, армия, командование)
- ✅ **Понимание философских концептов** (информация, мера, триединство)

## Быстрый старт

### Установка

```bash
git clone https://github.com/terratensor/book2bert.git
cd book2bert

# Установка Python зависимостей
pip install -r requirements.txt

# (Опционально) Установка Go зависимостей
go mod download
```

### Загрузка обученной модели

Модель доступна в папке `data/models/tiny_bert_militera_v3/` после обучения.

```bash
ls data/models/tiny_bert_militera_v3/
# best_model.pt  config.yaml  checkpoints/  csv/  tensorboard/  training.log
```

## Использование модели

### 1. Предсказание маскированных слов (MLM)

```python
from pathlib import Path
import torch
import sys

sys.path.insert(0, "training")
from model import BERT, BERTForMLM
from tokenizers import BertWordPieceTokenizer

# Загрузка модели
model_path = "data/models/tiny_bert_militera_v3/best_model.pt"
tokenizer_path = "data/processed/tokenizer_militera_v3"

tokenizer = BertWordPieceTokenizer(
    str(Path(tokenizer_path) / "vocab.txt"),
    lowercase=False
)

bert = BERT(
    vocab_size=50000,
    hidden_size=384,
   num_layers=6,
    num_heads=12,
    intermediate_size=1536,
    max_position=512,
    dropout=0.1
)
model = BERTForMLM(bert, vocab_size=50000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Предсказание
text = "Генерал [MASK] командовал дивизией в трудных условиях."
encoded = tokenizer.encode(text)
input_ids = torch.tensor([encoded.ids]).to(device)

mask_id = tokenizer.token_to_id("[MASK]")
mask_pos = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0][0]

with torch.no_grad():
    outputs = model(input_ids=input_ids)
    logits = outputs['logits']
    top_k = torch.topk(logits[0, mask_pos, :], 10)

print("Top-10 predictions:")
for idx in top_k.indices:
    print(f"  - {tokenizer.id_to_token(idx.item())}")
```

**Ожидаемый вывод:**
```
Top-10 predictions:
  - корпуса
  - армии
  - дивизии
  - артиллерии
  - командующего
  - и
  - его
  - ##у
  - ##ом
```

### 2. Извлечение эмбеддингов (для semantic search)

```python
def get_cls_embedding(text):
    """Возвращает [CLS] эмбеддинг для текста"""
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    attention_mask = torch.tensor([encoded.attention_mask]).to(device)
    
    with torch.no_grad():
        hidden_states, _ = bert(input_ids, attention_mask)
    
    return hidden_states[0, 0, :].cpu().numpy()

# Пример
emb1 = get_cls_embedding("Генерал Шкуро командовал дивизией.")
emb2 = get_cls_embedding("Танковая дивизия прорвала оборону.")
emb3 = get_cls_embedding("Философия есть учение о всеобщем.")

# Косинусное сходство
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity([emb1], [emb2]))  # ~0.85 (близки по смыслу)
print(cosine_similarity([emb1], [emb3]))  # ~0.45 (далеки)
```

### 3. Semantic search (поиск похожих текстов)

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self, model, bert, tokenizer, device):
        self.model = model
        self.bert = bert
        self.tokenizer = tokenizer
        self.device = device
        self.corpus = []
        self.embeddings = []
    
    def add_texts(self, texts):
        """Добавляет тексты в индекс"""
        for text in texts:
            emb = get_cls_embedding(text)
            self.corpus.append(text)
            self.embeddings.append(emb)
        self.embeddings = np.array(self.embeddings)
    
    def search(self, query, top_k=5):
        """Ищет похожие тексты"""
        query_emb = get_cls_embedding(query)
        similarities = cosine_similarity([query_emb], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.corpus[idx],
                'score': float(similarities[idx])
            })
        return results

# Использование
searcher = SemanticSearch(model, bert, tokenizer, device)
searcher.add_texts([
    "Генерал Шкуро командовал дивизией.",
    "Танки прорвали оборону противника.",
    "Философия есть учение о всеобщем.",
    "Артиллерия вела огонь по позициям врага."
])

results = searcher.search("Командир дивизии Шкуро", top_k=3)
for r in results:
    print(f"{r['score']:.4f}: {r['text']}")
```

## Запуск через командную строку

### MLM предсказания

```bash
python scripts/evaluate_mlm_v3.py --model_dir data/models/tiny_bert_militera_v3
```

### Визуализация attention

```bash
python scripts/visualize_attention_v3.py \
    --model_dir data/models/tiny_bert_militera_v3 \
    --text "Генерал Шкуро командовал дивизией."
```

### Анализ [CLS] эмбеддингов

```bash
python scripts/analyze_cls_v3.py --model_dir data/models/tiny_bert_militera_v3
```

## Результаты модели

### Метрики (после 1 эпохи)

| Показатель | Значение |
|------------|----------|
| Train loss | 5.67 |
| Val loss | 3.41 |
| Perplexity | 30.21 |
| Время эпохи | 11.05 ч |

### Примеры предсказаний

| Вход | Топ-5 предсказаний |
|------|-------------------|
| "Философия есть [MASK] о всеобщем" | упоминание, представление, понятие, литература, книга |
| "Мера — это [MASK]" | объект, направление, положение, время, место |
| "Триединство: [MASK], информация и мера" | **информация**, разведка, телефон, связь, почта |
| "Генерал [MASK] командовал дивизией" | корпуса, армии, дивизии, артиллерии, командующего |

## Запуск сервиса для инференса

### HTTP сервер

```python
# inference_server.py
from flask import Flask, request, jsonify
import torch
from model import BERT, BERTForMLM
from tokenizers import BertWordPieceTokenizer

app = Flask(__name__)

# Загрузка модели при старте
# ... (код загрузки)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs['logits']
    
    return jsonify({'logits': logits.cpu().tolist()})

@app.route('/embed', methods=['POST'])
def embed():
    data = request.get_json()
    text = data.get('text', '')
    
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    attention_mask = torch.tensor([encoded.attention_mask]).to(device)
    
    with torch.no_grad():
        hidden_states, _ = bert(input_ids, attention_mask)
    
    return jsonify({'embedding': hidden_states[0, 0, :].cpu().tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083)
```

Запуск:
```bash
python inference_server.py
curl -X POST http://localhost:8083/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Генерал Шкуро командовал дивизией."}'
```

## Структура проекта

```
book2bert/
├── cmd/                    # Go-утилиты
│   ├── process-corpus/    # Обработка корпуса
│   └── corpus-builder/    # Сбор корпуса для токенизатора
├── pkg/                    # Go-библиотеки
│   ├── adapters/          # HTTP-клиенты, репозитории
│   ├── app/               # Use cases
│   ├── core/              # Доменные модели, порты
│   └── textutils/         # Утилиты для текста
├── scripts/               # Python-скрипты
│   ├── build_dataset.py
│   ├── train_tokenizer_from_corpus.py
│   ├── evaluate_mlm_v3.py
│   ├── visualize_attention_v3.py
│   └── analyze_cls_v3.py
├── services/              # HTTP-сервисы
│   └── segmenter/         # Сегментация предложений (razdel)
├── training/              # Обучение модели
│   ├── model.py           # Архитектура BERT
│   ├── train_tiny_streaming.py
│   └── config/
└── data/                  # Данные (создаются при запуске)
```

## Требования

- **Python**: 3.13+
- **Go**: 1.25+
- **CUDA**: 13.0+ (для GPU)
- **RAM**: 32+ GB (для обработки корпуса)
- **VRAM**: 8+ GB (для обучения)

## Документация

Подробная книга о создании модели доступна в [book/](book/).

- [Введение](book/README.md)
- [Глава 1: Зачем создавать BERT с нуля](book/chapters/01_intro.md)
- [Глава 2: Подготовка данных](book/chapters/02_data_preparation.md)
- [Глава 3: Токенизация](book/chapters/03_tokenization.md)
- [Глава 4: Сборка датасета](book/chapters/04_dataset_building.md)
- [Глава 5: Архитектура BERT](book/chapters/05_architecture.md)
- [Глава 6: Обучение](book/chapters/06_training.md)
- [Глава 7: Результаты](book/chapters/07_results.md)
- [Глава 8: Выводы](book/chapters/08_conclusions.md)

## Лицензия

MIT

## Ссылки

- [Репозиторий](https://github.com/terratensor/book2bert)
- [Книга](book/)
- [Словарь терминов](book/chapters/appendix_glossary.md)
