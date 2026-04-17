#!/usr/bin/env python3
"""
Semantic Search с Query Expansion для военно-исторического корпуса
Интеграция с вашим существующим кодом
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import List, Dict, Optional
import pickle
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT
from tokenizers import BertWordPieceTokenizer
from v1.examples.query_expansion import QueryExpander


class EnhancedSemanticSearch:
    """
    Улучшенный семантический поиск с поддержкой:
    1. Query Expansion через MLM
    2. Гибридного поиска (keyword + semantic)
    3. Кэширования эмбеддингов
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "cuda",
        use_expansion: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Загрузка токенизатора
        vocab_path = Path(tokenizer_path) / "vocab.txt"
        self.tokenizer = BertWordPieceTokenizer(str(vocab_path), lowercase=False)
        
        # Загрузка BERT (без MLM головы для эмбеддингов)
        self._load_bert(model_path)
        
        # Query Expander
        self.expander = None
        if use_expansion:
            self.expander = QueryExpander(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                device=device
            )
        
        self.corpus = []
        self.embeddings = None
        self.metadata = []
        
    def _load_bert(self, model_path: str):
        """Загрузка BERT модели для извлечения эмбеддингов"""
        from model import BERT  # локальный импорт
        
        self.bert = BERT(
            vocab_size=50000,
            hidden_size=384,
            num_layers=6,
            num_heads=12,
            intermediate_size=1536,
            max_position=512,
            dropout=0.1
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        # Извлекаем только BERT часть (без MLM головы)
        bert_state = {k.replace('bert.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                     if k.startswith('bert.')}
        self.bert.load_state_dict(bert_state)
        self.bert.to(self.device)
        self.bert.eval()
    
    def get_embedding(self, text: str, pooling: str = "cls") -> np.ndarray:
        """
        Получение эмбеддинга текста
        
        Args:
            text: входной текст
            pooling: "cls" или "mean"
        """
        encoded = self.tokenizer.encode(text)
        input_ids = torch.tensor([encoded.ids]).to(self.device)
        attention_mask = torch.tensor([encoded.attention_mask]).to(self.device)
        
        with torch.no_grad():
            hidden_states, _ = self.bert(input_ids, attention_mask)
        
        if pooling == "cls":
            return hidden_states[0, 0, :].cpu().numpy()
        elif pooling == "mean":
            # Mean pooling с учетом маски
            mask = attention_mask[0].cpu().numpy()
            embeddings = hidden_states[0].cpu().numpy()
            mean_emb = np.sum(embeddings * mask[:, np.newaxis], axis=0) / np.sum(mask)
            return mean_emb
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
    
    def add_texts(
        self, 
        texts: List[str], 
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32
    ):
        """
        Добавление текстов в индекс
        
        Args:
            texts: список текстов
            metadata: список метаданных (книга, страница, и т.д.)
            batch_size: размер батча для GPU
        """
        self.corpus.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])
        
        # Вычисляем эмбеддинги батчами
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embs = []
            for text in batch:
                emb = self.get_embedding(text, pooling="mean")
                batch_embs.append(emb)
            all_embeddings.extend(batch_embs)
            print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)}")
        
        # Обновляем общий массив
        new_embeddings = np.array(all_embeddings)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        use_expansion: bool = True,
        expansion_strategy: str = "mean",
        expansion_weight: float = 0.5
    ) -> List[Dict]:
        """
        Поиск похожих текстов
        
        Args:
            query: поисковый запрос
            top_k: количество результатов
            use_expansion: использовать ли Query Expansion
            expansion_strategy: "mean", "max", "concat"
            expansion_weight: вес расширенного эмбеддинга (0-1)
        """
        if self.embeddings is None:
            raise ValueError("Корпус пуст. Сначала вызовите add_texts()")
        
        # Получаем эмбеддинг запроса
        if use_expansion and self.expander:
            query_emb = self.expander.get_expanded_embedding(
                query, 
                strategy=expansion_strategy
            )
            # Комбинируем с оригинальным
            original_emb = self.get_embedding(query, pooling="mean")
            query_emb = (1 - expansion_weight) * original_emb + expansion_weight * query_emb
        else:
            query_emb = self.get_embedding(query, pooling="mean")
        
        # Нормализация
        query_emb = query_emb / np.linalg.norm(query_emb)
        corpus_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Косинусное расстояние
        similarities = np.dot(corpus_norm, query_emb)
        
        # Топ-K
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.corpus[idx],
                'score': float(similarities[idx]),
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        keyword_results: List[Dict],  # из Manticore
        top_k: int = 10,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ) -> List[Dict]:
        """
        Гибридный поиск: объединение keyword (Manticore) и semantic (BERT) результатов
        
        Args:
            query: поисковый запрос
            keyword_results: результаты из Manticore с полем 'score' (BM25)
            top_k: количество результатов
            semantic_weight: вес семантической близости
            keyword_weight: вес keyword близости
        """
        # Получаем семантические эмбеддинги для всех кандидатов
        candidate_texts = [r['text'] for r in keyword_results]
        candidate_embs = []
        
        for text in candidate_texts:
            emb = self.get_embedding(text, pooling="mean")
            candidate_embs.append(emb)
        
        candidate_embs = np.array(candidate_embs)
        candidate_embs = candidate_embs / np.linalg.norm(candidate_embs, axis=1, keepdims=True)
        
        # Семантический эмбеддинг запроса (с расширением)
        if self.expander:
            query_emb = self.expander.get_expanded_embedding(query, strategy="mean")
            original_emb = self.get_embedding(query, pooling="mean")
            query_emb = 0.7 * original_emb + 0.3 * query_emb
        else:
            query_emb = self.get_embedding(query, pooling="mean")
        
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Семантические оценки
        semantic_scores = np.dot(candidate_embs, query_emb)
        
        # Нормализация keyword оценок (BM25)
        max_keyword_score = max([r['score'] for r in keyword_results]) if keyword_results else 1
        keyword_scores = np.array([r['score'] / max_keyword_score for r in keyword_results])
        
        # Комбинированный скор
        combined_scores = semantic_weight * semantic_scores + keyword_weight * keyword_scores
        
        # Сортировка
        sorted_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        results = []
        for idx in sorted_indices:
            results.append({
                'text': keyword_results[idx]['text'],
                'combined_score': float(combined_scores[idx]),
                'semantic_score': float(semantic_scores[idx]),
                'keyword_score': float(keyword_scores[idx]),
                'metadata': keyword_results[idx].get('metadata', {})
            })
        
        return results
    
    def save_index(self, path: str):
        """Сохранение индекса на диск"""
        index_data = {
            'corpus': self.corpus,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Загрузка индекса с диска"""
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        self.corpus = index_data['corpus']
        self.embeddings = index_data['embeddings']
        self.metadata = index_data['metadata']
        print(f"Index loaded from {path}")


def demo_search():
    """Демонстрация поиска с расширением"""
    
    # Инициализация
    searcher = EnhancedSemanticSearch(
        model_path="data/models/tiny_bert_militera_v3/best_model.pt",
        tokenizer_path="data/processed/tokenizer_militera_v3",
        use_expansion=True
    )
    
    # Тестовый корпус (в реальности загружаете из ваших книг)
    test_corpus = [
        "Танки Т-34 прорвали оборону противника под Курском",
        "Тигры и Пантеры вели тяжелые бои с советскими танками",
        "Генерал Шкуро командовал казачьей дивизией",
        "ИС-2 был мощным советским тяжелым танком",
        "Философия информации рассматривает меру как ключевой концепт",
        "Артиллерийская подготовка предшествовала танковой атаке",
        "Командир дивизии отдал приказ на наступление"
    ]
    
    searcher.add_texts(test_corpus)
    
    # Тестовый запрос
    query = "бои с немецкими танками"
    
    print(f"\nПоиск по запросу: '{query}'")
    print("=" * 60)
    
    # Без расширения
    print("\nБез Query Expansion:")
    results = searcher.search(query, use_expansion=False, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.4f}] {r['text']}")
    
    # С расширением
    print("\nС Query Expansion:")
    results_exp = searcher.search(query, use_expansion=True, top_k=3)
    for i, r in enumerate(results_exp, 1):
        print(f"{i}. [{r['score']:.4f}] {r['text']}")


if __name__ == "__main__":
    demo_search()