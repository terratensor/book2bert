# Глава 5: Архитектура BERT — реализация на PyTorch

> "Чтобы понять модель, нужно построить её своими руками."

## 5.1 От текста к числам: полный путь

Прежде чем погрузиться в архитектуру, давайте проследим путь от сырого текста до входа в модель.

```
Текст: "Философия есть учение"
         ↓ токенизация (WordPiece)
Токены: ["[CLS]", "Философия", "есть", "учение", "[SEP]"]
         ↓ словарь (vocab.txt)
ID:     [2,        7659,       757,    5010,     3]
         ↓ Embedding Layer
Векторы: [v_cls, v_философия, v_есть, v_учение, v_sep]  # каждый размером hidden_size (384)
         ↓ + Positional Encoding + Token Type Embedding
         ↓ Multi-Head Self-Attention (x N слоев)
         ↓ Выходные векторы (такой же формы)
```

**Что происходит на каждом этапе — разберем в этой главе.**

## 5.2 Общая архитектура BERT

BERT (Bidirectional Encoder Representations from Transformers) состоит из:

1. **Embedding Layer** — превращает ID токенов в векторы
2. **Positional Encoding** — добавляет информацию о позиции
3. **Token Type Embedding** — различает предложения A и B (для NSP)
4. **Stack TransformerBlock** — N слоев (у нас 6)
5. **MLM Head** — предсказывает маскированные токены

```python
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size)
        self.token_type_embedding = nn.Embedding(2, hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(...) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(hidden_size)
```

## 5.3 Embedding Layer (таблица векторов)

**Что это:** Таблица, где каждой строке соответствует вектор для конкретного токена.

```python
self.token_embedding = nn.Embedding(vocab_size, hidden_size)
# vocab_size = 50,000 (размер словаря)
# hidden_size = 384 (размер эмбеддинга)
```

**Аналогия:** Представьте огромную книгу, где на каждой странице (токен) записан вектор чисел, описывающий смысл этого токена. Модель учит эти векторы во время обучения.

**Вход:** ID токена (например, 7659 для "философия")
**Выход:** Вектор размера 384 (например, [0.12, -0.34, 0.56, ...])

## 5.4 Positional Encoding (позиционное кодирование)

**Проблема:** Self-attention не различает порядок токенов. Для модели "кошка ест рыбу" и "рыба ест кошку" — одинаково.

**Решение:** Добавить информацию о позиции.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Синусы для четных позиций, косинусы для нечетных
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                             -(math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, hidden_size]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

**Почему синусы и косинусы?**
- Разные частоты для разных измерений
- Позволяет модели легко определить относительную позицию
- Не требует обучения (но можно и обучаемые)

## 5.5 Multi-Head Self-Attention (сердце трансформера)

### 5.5.1 Что такое внимание?

**Self-attention** позволяет каждому токену "посмотреть" на другие токены и решить, какие из них важны для понимания текущего.

**Аналогия:** Читая слово "учение", модель может посмотреть на предыдущие слова "Философия есть", чтобы понять контекст.

### 5.5.2 Query, Key, Value (Q, K, V)

Каждый токен создает три вектора:

| Вектор | Что делает | Аналогия |
|--------|-----------|----------|
| **Query** | "Что я ищу?" | Вопрос от текущего слова |
| **Key** | "Что я предлагаю?" | Ответ от других слов |
| **Value** | "Какая у меня информация?" | Содержание, которое передается |

```python
# Проекции Q, K, V
self.query = nn.Linear(hidden_size, hidden_size)
self.key = nn.Linear(hidden_size, hidden_size)
self.value = nn.Linear(hidden_size, hidden_size)
```

### 5.5.3 Формула внимания

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

**Разбор по шагам:**

1. **Q × K^T** — матрица "внимания" (score). Размер: [seq_len, seq_len]
   - Элемент [i, j] показывает, насколько токен i должен обратить внимание на токен j

2. **/ √d** — масштабирование (чтобы градиенты не затухали)

3. **softmax** — превращает числа в вероятности (сумма по строке = 1)

4. **× V** — взвешенная сумма векторов значений

### 5.5.4 Multi-Head: внимание с разных точек зрения

**Почему несколько голов?** Разные головы могут фокусироваться на разных типах связей:

- Одна голова может следить за синтаксисом (подлежащее-сказуемое)
- Другая — за семантикой (связанные по смыслу слова)
- Третья — за дальними связями

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Проецируем Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 2. Транспонируем для параллельного вычисления
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 3. Считаем scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 4. Маскируем padding (если есть)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax и dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 6. Применяем attention к V
        context = torch.matmul(attn_weights, V)
        
        # 7. Объединяем головы
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # 8. Выходная проекция
        return self.out_proj(context), attn_weights
```

### 5.5.5 Визуализация attention (пример из нашей модели)

После обучения мы визуализировали attention weights. Ранние слои фокусируются на синтаксисе, поздние — на семантике.

```python
# Layer 0, Head 0: внимание к соседним токенам (синтаксис)
# Layer 5, Head 3: [CLS] собирает информацию со всего предложения
```

## 5.6 Feed-Forward Network (FFN)

**Что это:** Два линейных слоя с активацией GELU между ними.

```python
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()  # Gaussian Error Linear Unit
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

**Почему intermediate_size = 4 × hidden_size?** Это стандартное соотношение в BERT:

| Модель | hidden_size | intermediate_size | Соотношение |
|--------|-------------|-------------------|-------------|
| BERT-tiny | 384 | 1536 | 1:4 |
| BERT-base | 768 | 3072 | 1:4 |

FFN добавляет **нелинейность** и **выразительную способность**. Примерно 2/3 всех параметров модели находятся в FFN.

## 5.7 Transformer Block (полный блок)

```python
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention с residual connection
        attn_output, attn_weights = self.attention(x, mask)
        x = self.ln1(x + self.dropout(attn_output))  # residual + layer norm
        
        # Feed-forward с residual connection
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))
        
        return x, attn_weights
```

**Ключевые элементы:**

| Компонент | Зачем |
|-----------|-------|
| **Residual connection** | `x + f(x)` — позволяет градиентам проходить сквозь много слоев |
| **Layer Normalization** | Стабилизирует обучение (нормализует по признакам) |
| **Dropout** | Регуляризация (случайно обнуляет нейроны) |

## 5.8 BERTForMLM (голова для предсказания)

```python
class BERTForMLM(nn.Module):
    def __init__(self, bert, vocab_size):
        super().__init__()
        self.bert = bert
        self.mlm_head = nn.Linear(bert.hidden_size, vocab_size)
        
        # Связываем веса с эмбеддингами (опционально)
        self.mlm_head.weight = self.bert.token_embedding.weight
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        hidden_states, attention_weights = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.mlm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {'loss': loss, 'logits': logits, 'attention_weights': attention_weights}
```

**Что делает MLM head:**
- Принимает выходы BERT (размер [batch, seq_len, hidden_size])
- Проецирует на размер словаря (vocab_size=50000)
- Для каждой позиции предсказывает, какой токен должен быть

## 5.9 Параметры нашей модели

```yaml
model:
  vocab_size: 50000
  hidden_size: 384
  num_layers: 6
  num_heads: 12
  intermediate_size: 1536
  max_position: 512
  dropout: 0.1
```

**Подсчет параметров:**

| Компонент | Формула | Результат |
|-----------|---------|-----------|
| Token Embedding | 50000 × 384 | 19,200,000 |
| Positional Encoding | 512 × 384 | 196,608 |
| Token Type Embedding | 2 × 384 | 768 |
| Attention (6 слоев) | 6 × (4 × 384 × 384) | ~3,538,944 |
| FFN (6 слоев) | 6 × (2 × 384 × 1536) | ~7,077,888 |
| Layer Norm (6 слоев) | 6 × (2 × 384) | 4,608 |
| MLM Head | 384 × 50000 | 19,200,000 |
| **Итого** | | **~29,898,320** |

## 5.10 Почему именно эти числа?

| Параметр | Почему 384? | Почему 6? | Почему 12? |
|----------|-------------|-----------|------------|
| hidden_size | Компромисс между качеством и памятью (768 для base слишком много для 16 ГБ) | — | — |
| num_layers | Больше слоев = дольше обучение, 6 достаточно для нашей задачи | — | — |
| num_heads | hidden_size / head_dim = 384 / 32 = 12 | — | — |

## 5.11 Сравнение с оригинальным BERT

| Параметр | BERT-base | Наша модель (tiny) |
|----------|-----------|-------------------|
| vocab_size | 30,522 | 50,000 |
| hidden_size | 768 | 384 |
| num_layers | 12 | 6 |
| num_heads | 12 | 12 |
| Параметры | 110M | 29.9M |
| Память (fp16) | ~8-10 GB | ~4-5 GB |

**Почему мы выбрали tiny:**
- Умещается в 16 ГБ GPU
- Быстрее обучается (10 часов на эпоху вместо 30-40)
- Для militera корпуса (4 ГБ) достаточно
- После отладки пайплайна можно масштабировать

## 5.12 Что дает каждая компонента (итоговая таблица)

| Компонента | Вход | Выход | Что делает |
|------------|------|-------|-----------|
| Token Embedding | ID токена (0-49999) | Вектор (384) | Превращает число в смысл |
| Positional Encoding | Позиция (0-511) | Вектор (384) | Добавляет информацию о порядке |
| Multi-Head Attention | Векторы (384) | Векторы (384) | Взаимодействие между токенами |
| Feed-Forward | Векторы (384) | Векторы (384) | Нелинейное преобразование |
| Layer Norm | Векторы (384) | Векторы (384) | Стабилизация |
| MLM Head | Векторы (384) | Логиты (50000) | Предсказание следующего токена |

---
[Далее: Глава 6 - Обучение...]