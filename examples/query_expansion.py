#!/usr/bin/env python3
"""
Query Expansion для военно-исторического поиска
Использует MLM голову вашей модели для генерации семантических вариаций запроса
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import List, Tuple, Optional
import random
from sklearn.metrics.pairwise import cosine_similarity

# Добавляем путь к training модулю
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT, BERTForMLM
from tokenizers import BertWordPieceTokenizer

class QueryExpander:
    """
    Расширение поисковых запросов через MLM предсказания
    
    Принцип работы:
    1. Маскируем ключевые слова в запросе
    2. Предсказываем альтернативы через MLM
    3. Генерируем семантические вариации
    4. Усредняем эмбеддинги для поиска
    """
    
    def __init__(
        self, 
        model_path: str, 
        tokenizer_path: str,
        device: str = "cuda",
        top_k_predictions: int = 5,
        mask_probability: float = 0.3
    ):
        """
        Args:
            model_path: путь к best_model.pt
            tokenizer_path: путь к токенизатору
            device: cuda/cpu
            top_k_predictions: сколько вариантов для каждого маска
            mask_probability: вероятность маскирования токена
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.top_k_predictions = top_k_predictions
        self.mask_probability = mask_probability
        
        # Загрузка токенизатора
        vocab_path = Path(tokenizer_path) / "vocab.txt"
        self.tokenizer = BertWordPieceTokenizer(str(vocab_path), lowercase=False)
        
        # Загрузка модели
        self._load_model(model_path)
        
        # Специальные токены
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.cls_token_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        
    def _load_model(self, model_path: str):
        """Загрузка BERT модели с MLM головой"""
        # Конфигурация из вашего README
        self.bert = BERT(
            vocab_size=50000,
            hidden_size=384,
            num_layers=6,
            num_heads=12,
            intermediate_size=1536,
            max_position=512,
            dropout=0.1
        )
        
        self.model = BERTForMLM(self.bert, vocab_size=50000)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def mask_tokens(self, tokens: List[int], mask_ratio: float = 0.15) -> Tuple[List[int], List[int]]:
        """
        Маскирует случайные токены (кроме специальных)
        
        Returns:
            masked_tokens: токены с [MASK]
            masked_positions: позиции где был маск
        """
        masked = tokens.copy()
        masked_positions = []
        
        for i, token_id in enumerate(tokens):
            # Не маскируем специальные токены
            if token_id in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                continue
                
            if random.random() < mask_ratio:
                masked[i] = self.mask_token_id
                masked_positions.append(i)
                
        return masked, masked_positions
    
    def predict_masked(self, input_ids: torch.Tensor, positions: List[int]) -> List[List[Tuple[str, float]]]:
        """
        Предсказывает токены для замаскированных позиций
        
        Returns:
            List[List[(token, score)]] для каждой позиции
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs['logits']  # [batch, seq_len, vocab_size]
            
        predictions = []
        for pos in positions:
            pos_logits = logits[0, pos, :]  # [vocab_size]
            
            # Top-k предсказаний
            top_k_logits, top_k_indices = torch.topk(pos_logits, self.top_k_predictions)
            top_k_probs = torch.softmax(top_k_logits, dim=0)
            
            # Конвертируем в токены
            pos_predictions = []
            for idx, prob in zip(top_k_indices, top_k_probs):
                token = self.tokenizer.id_to_token(idx.item())
                pos_predictions.append((token, prob.item()))
            
            predictions.append(pos_predictions)
            
        return predictions
    
    def expand_query(self, query: str, num_variations: int = 5) -> List[str]:
        """
        Генерирует вариации запроса через MLM
        
        Args:
            query: исходный запрос
            num_variations: количество вариаций
            
        Returns:
            список расширенных запросов (включая оригинал)
        """
        # Токенизация
        encoded = self.tokenizer.encode(query)
        tokens = encoded.ids
        
        # Генерируем разные маскировки
        variations = [query]  # оригинал всегда включен
        
        for _ in range(num_variations):
            # Случайная маскировка
            masked_tokens, masked_positions = self.mask_tokens(tokens, self.mask_probability)
            
            if not masked_positions:
                continue
                
            # Предсказание
            input_ids = torch.tensor([masked_tokens]).to(self.device)
            predictions = self.predict_masked(input_ids, masked_positions)
            
            # Создаем вариацию, заменяя маски на топ-1 предсказание
            expanded_tokens = masked_tokens.copy()
            for pos, preds in zip(masked_positions, predictions):
                if preds:
                    best_token = preds[0][0]
                    expanded_tokens[pos] = self.tokenizer.token_to_id(best_token)
            
            # Декодируем
            expanded_query = self.tokenizer.decode(expanded_tokens)
            variations.append(expanded_query)
            
        return variations
    
    def expand_query_smart(self, query: str, num_variations: int = 5) -> List[str]:
        """
        Умное расширение: маскируем только значимые слова (существительные, глаголы)
        
        Использует простые эвристики для определения значимых токенов
        """
        # TODO: можно улучшить с помощью POS-теггера
        # Пока пропускаем специальные токены и короткие слова
        encoded = self.tokenizer.encode(query)
        tokens = encoded.ids
        token_strings = [self.tokenizer.id_to_token(t) for t in tokens]
        
        variations = [query]
        
        for _ in range(num_variations):
            masked = tokens.copy()
            masked_positions = []
            
            for i, (token_id, token_str) in enumerate(zip(tokens, token_strings)):
                # Пропускаем специальные токены и короткие слова (<3 символов)
                if token_id in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                    continue
                if len(token_str) < 3 or token_str.startswith('##'):
                    continue
                if random.random() < self.mask_probability:
                    masked[i] = self.mask_token_id
                    masked_positions.append(i)
            
            if not masked_positions:
                continue
                
            input_ids = torch.tensor([masked]).to(self.device)
            predictions = self.predict_masked(input_ids, masked_positions)
            
            expanded_tokens = masked.copy()
            for pos, preds in zip(masked_positions, predictions):
                if preds:
                    # Берем топ-1, но можно и случайный
                    best_token = preds[0][0]
                    expanded_tokens[pos] = self.tokenizer.token_to_id(best_token)
            
            expanded_query = self.tokenizer.decode(expanded_tokens)
            variations.append(expanded_query)
            
        return variations
    
    def get_expanded_embedding(self, query: str, strategy: str = "mean") -> np.ndarray:
        """
        Получает расширенный эмбеддинг для запроса
        
        Args:
            query: исходный запрос
            strategy: 
                - "mean": среднее по всем вариациям
                - "max": поэлементный максимум
                - "concat": конкатенация (увеличивает размерность)
        
        Returns:
            эмбеддинг размерности 384 (или 384*num_variations для concat)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        variations = self.expand_query_smart(query)
        
        # Получаем [CLS] эмбеддинги для всех вариаций
        embeddings = []
        for var in variations:
            encoded = self.tokenizer.encode(var)
            input_ids = torch.tensor([encoded.ids]).to(self.device)
            attention_mask = torch.tensor([encoded.attention_mask]).to(self.device)
            
            with torch.no_grad():
                hidden_states, _ = self.bert(input_ids, attention_mask)
                cls_embedding = hidden_states[0, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
        
        embeddings = np.array(embeddings)  # [num_variations, 384]
        
        if strategy == "mean":
            return np.mean(embeddings, axis=0)
        elif strategy == "max":
            return np.max(embeddings, axis=0)
        elif strategy == "concat":
            return embeddings.flatten()  # [num_variations * 384]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def search_with_expansion(
        self, 
        query: str, 
        corpus_embeddings: np.ndarray,
        corpus_texts: List[str],
        top_k: int = 10,
        expansion_weight: float = 0.5
    ) -> List[dict]:
        """
        Поиск с расширением запроса
        
        Args:
            query: поисковый запрос
            corpus_embeddings: эмбеддинги корпуса [N, 384]
            corpus_texts: тексты корпуса
            top_k: количество результатов
            expansion_weight: вес расширенного эмбеддинга (0-1)
        """
        # Оригинальный эмбеддинг
        original_emb = self.get_embedding(query)
        
        # Расширенный эмбеддинг
        expanded_emb = self.get_expanded_embedding(query, strategy="mean")
        
        # Комбинированный
        combined_emb = (1 - expansion_weight) * original_emb + expansion_weight * expanded_emb
        
        # Нормализация
        combined_emb = combined_emb / np.linalg.norm(combined_emb)
        corpus_embeddings_norm = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        
        # Поиск
        similarities = np.dot(corpus_embeddings_norm, combined_emb)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': corpus_texts[idx],
                'score': float(similarities[idx]),
                'original_score': float(np.dot(corpus_embeddings_norm[idx], original_emb / np.linalg.norm(original_emb)))
            })
        
        return results
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Базовое получение [CLS] эмбеддинга"""
        encoded = self.tokenizer.encode(text)
        input_ids = torch.tensor([encoded.ids]).to(self.device)
        attention_mask = torch.tensor([encoded.attention_mask]).to(self.device)
        
        with torch.no_grad():
            hidden_states, _ = self.bert(input_ids, attention_mask)
        
        return hidden_states[0, 0, :].cpu().numpy()


def demo():
    """Демонстрация работы Query Expansion"""
    
    # Пути к модели (из вашего README)
    model_path = "data/models/tiny_bert_militera_v3/best_model.pt"
    tokenizer_path = "data/processed/tokenizer_militera_v3"
    
    # Инициализация
    expander = QueryExpander(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        top_k_predictions=5,
        mask_probability=0.3
    )
    
    # Тестовые запросы
    test_queries = [
        "бои на танках Т-34",
        "противостояние Тигров и ИС-2",
        "генерал командовал дивизией",
        "философия информации и меры"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Исходный запрос: {query}")
        print(f"{'='*60}")
        
        # Генерация вариаций
        variations = expander.expand_query_smart(query, num_variations=5)
        
        print(f"\nВариации:")
        for i, var in enumerate(variations[1:], 1):
            print(f"  {i}. {var}")
        
        # Покажем топ-предсказания для ключевых слов
        print(f"\nАнализ ключевых слов:")
        encoded = expander.tokenizer.encode(query)
        tokens = [expander.tokenizer.id_to_token(t) for t in encoded.ids]
        
        # Проходим по каждому токену
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            if len(token) < 3 or token.startswith('##'):
                continue
                
            # Маскируем этот токен
            masked_tokens = encoded.ids.copy()
            masked_tokens[i] = expander.mask_token_id
            input_ids = torch.tensor([masked_tokens]).to(expander.device)
            predictions = expander.predict_masked(input_ids, [i])
            
            if predictions:
                print(f"\n  '{token}' → альтернативы:")
                for pred_token, prob in predictions[0][:3]:
                    print(f"    - {pred_token} ({prob:.3f})")


if __name__ == "__main__":
    demo()