"""
BERT реализация с нуля на PyTorch.
Каждый компонент написан явно для полного понимания.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Механизм multi-head self-attention — основа трансформера.
    Позволяет каждому токену "взаимодействовать" с другими токенами.
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size должен делиться на num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Проекции для Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Выходная проекция
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: [batch, seq_len, hidden_size]
        mask: [batch, seq_len] или [batch, 1, 1, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Проецируем Q, K, V
        Q = self.query(x)  # [batch, seq_len, hidden_size]
        K = self.key(x)
        V = self.value(x)
        
        # 2. Разделяем на heads: [batch, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Attention scores: [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 4. Применяем маску (если есть)
        if mask is not None:
            # Расширяем маску для совместимости с attention heads
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax и dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 6. Применяем attention к V
        context = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]
        
        # 7. Объединяем heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # 8. Выходная проекция
        output = self.out_proj(context)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Применяется одинаково к каждой позиции.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Один блок трансформера:
    - Multi-head self-attention с residual и layer norm
    - Feed-forward с residual и layer norm
    """
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Self-attention с residual
        attn_output, attn_weights = self.attention(x, mask)
        x = self.ln1(x + self.dropout(attn_output))
        
        # Feed-forward с residual
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))
        
        return x, attn_weights


class PositionalEncoding(nn.Module):
    """
    Синусоидальное позиционное кодирование.
    Позволяет модели учитывать порядок токенов.
    """
    def __init__(self, hidden_size: int, max_len: int = 512):
        super().__init__()
        
        # Создаем матрицу позиционных эмбеддингов
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, hidden_size]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        return x + self.pe[:, :x.size(1), :]


class BERT(nn.Module):
    """
    Полная модель BERT.
    """
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 12,
        intermediate_size: int = 1536,
        max_position: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Эмбеддинги
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size, max_position)
        self.token_type_embedding = nn.Embedding(2, hidden_size)  # для NSP (пока не используем)
        
        self.emb_dropout = nn.Dropout(dropout)
        
        # Stack трансформерных блоков
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(hidden_size)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None
    ):
        """
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len] (1 для реальных токенов, 0 для padding)
        token_type_ids: [batch, seq_len] (0 для первого предложения, 1 для второго)
        """
        batch_size, seq_len = input_ids.shape
        
        # Эмбеддинги
        token_emb = self.token_embedding(input_ids)
        
        # Позиционные эмбеддинги
        pos_emb = self.position_encoding(token_emb)
        
        # Token type эмбеддинги (если не переданы, используем нули)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        type_emb = self.token_type_embedding(token_type_ids)
        
        # Суммируем и применяем dropout
        x = self.emb_dropout(token_emb + pos_emb + type_emb)
        
        # Attention mask для transformer
        # Преобразуем [batch, seq_len] -> [batch, 1, 1, seq_len]
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Проход через все слои
        attention_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, attention_mask)
            attention_weights_list.append(attn_weights)
        
        # Финальная нормализация
        x = self.ln_final(x)
        
        return x, attention_weights_list


class BERTForMLM(nn.Module):
    """
    BERT с головой для Masked Language Modeling.
    Предсказывает маскированные токены.
    """
    def __init__(self, bert: BERT, vocab_size: int):
        super().__init__()
        self.bert = bert
        self.mlm_head = nn.Linear(bert.hidden_size, vocab_size)
        
        # Связываем веса с эмбеддингами (опционально)
        self.mlm_head.weight = self.bert.token_embedding.weight
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        """
        input_ids: [batch, seq_len] (с маскированными токенами)
        labels: [batch, seq_len] (исходные токены, -100 для тех, которые не нужно предсказывать)
        """
        # Проход через BERT
        hidden_states, attention_weights = self.bert(
            input_ids, attention_mask, token_type_ids
        )
        
        # Предсказания на всех позициях
        logits = self.mlm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'attention_weights': attention_weights,
            'hidden_states': hidden_states
        }


def count_parameters(model):
    """Подсчет количества параметров в модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Быстрый тест
    bert = BERT(
        vocab_size=30000,
        hidden_size=384,
        num_layers=6,
        num_heads=12
    )
    model = BERTForMLM(bert, vocab_size=30000)
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Тестовый forward
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    output = model(input_ids, attention_mask)
    print(f"Output logits shape: {output['logits'].shape}")