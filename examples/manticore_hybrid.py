#!/usr/bin/env python3
"""
Гибридный поиск с Query Expansion для вашего Manticore API
ИСПРАВЛЕНА ОШИБКА: обрезка текста до 512 токенов
"""

import sys
from pathlib import Path
import requests
import json
import numpy as np
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT
from tokenizers import BertWordPieceTokenizer
from query_expansion import QueryExpander

import torch


class ManticoreHybridSearch:
    """
    Гибридный поиск для вашей библиотеки library2026
    """
    
    # Максимальная длина последовательности для BERT
    MAX_SEQ_LEN = 512
    
    def __init__(
        self,
        bert_model_path: str,
        tokenizer_path: str,
        manticore_url: str = "http://localhost:9308/search",
        api_key: str = "",
        table_name: str = "library2026",
        device: str = "cuda"
    ):
        self.manticore_url = manticore_url
        self.api_key = api_key
        self.table_name = table_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        # Загрузка BERT модели
        self._load_bert(bert_model_path, tokenizer_path)
        
        # Query Expander
        self.expander = QueryExpander(
            model_path=bert_model_path,
            tokenizer_path=tokenizer_path,
            device=device
        )
        
    def _load_bert(self, model_path: str, tokenizer_path: str):
        """Загрузка BERT для эмбеддингов"""
        from model import BERT
        
        # Токенизатор
        vocab_path = Path(tokenizer_path) / "vocab.txt"
        self.tokenizer = BertWordPieceTokenizer(str(vocab_path), lowercase=False)
        
        # Модель
        self.bert = BERT(
            vocab_size=50000,
            hidden_size=384,
            num_layers=6,
            num_heads=12,
            intermediate_size=1536,
            max_position=self.MAX_SEQ_LEN,
            dropout=0.1
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        # Загружаем только BERT часть
        bert_state = {k.replace('bert.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                     if k.startswith('bert.')}
        self.bert.load_state_dict(bert_state)
        self.bert.to(self.device)
        self.bert.eval()
        
        print("BERT model loaded successfully")
    
    def truncate_to_max_tokens(self, text: str, max_tokens: int = None) -> str:
        """
        Обрезает текст до максимального количества токенов
        
        Args:
            text: исходный текст
            max_tokens: максимальное количество токенов (по умолчанию MAX_SEQ_LEN - 2 для [CLS] и [SEP])
        """
        if max_tokens is None:
            max_tokens = self.MAX_SEQ_LEN - 2  # оставляем место для [CLS] и [SEP]
        
        # Токенизируем
        encoded = self.tokenizer.encode(text)
        tokens = encoded.tokens
        
        if len(tokens) <= max_tokens:
            return text
        
        # Обрезаем до max_tokens токенов
        truncated_tokens = tokens[:max_tokens]
        
        # Декодируем обратно в текст
        truncated_text = self.tokenizer.decode(self.tokenizer.encode(truncated_tokens).ids)
        
        return truncated_text
    
    def get_embedding(self, text: str, pooling: str = "mean") -> np.ndarray:
        """
        Получение эмбеддинга текста с автоматической обрезкой до 512 токенов
        """
        # Обрезаем текст если нужно
        text = self.truncate_to_max_tokens(text)
        
        encoded = self.tokenizer.encode(text)
        input_ids = torch.tensor([encoded.ids]).to(self.device)
        attention_mask = torch.tensor([encoded.attention_mask]).to(self.device)
        
        # Проверка длины (для отладки)
        seq_len = input_ids.shape[1]
        if seq_len > self.MAX_SEQ_LEN:
            print(f"WARNING: Sequence length {seq_len} exceeds {self.MAX_SEQ_LEN}, truncating...")
            input_ids = input_ids[:, :self.MAX_SEQ_LEN]
            attention_mask = attention_mask[:, :self.MAX_SEQ_LEN]
        
        with torch.no_grad():
            hidden_states, _ = self.bert(input_ids, attention_mask)
        
        if pooling == "cls":
            return hidden_states[0, 0, :].cpu().numpy()
        else:  # mean pooling
            mask = attention_mask[0].cpu().numpy()
            embeddings = hidden_states[0].cpu().numpy()
            # Усредняем только по реальным токенам (не padding)
            mean_emb = np.sum(embeddings * mask[:, np.newaxis], axis=0) / (np.sum(mask) + 1e-8)
            return mean_emb
    
    def search_manticore(
        self, 
        query: str, 
        limit: int = 100,
        offset: int = 0
    ) -> Dict:
        """
        Поиск в Manticore через ваш API
        """
        headers = {
            'Content-Type': 'application/json'
        }
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        
        payload = {
            "table": self.table_name,
            "query": {
                "query_string": query
            },
            "limit": limit,
            "offset": offset
        }
        
        try:
            response = requests.post(
                self.manticore_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Manticore error: {response.status_code}")
                print(f"Response: {response.text}")
                return {"hits": {"hits": [], "total": 0}}
                
        except Exception as e:
            print(f"Request error: {e}")
            return {"hits": {"hits": [], "total": 0}}
    
    def expand_query_for_manticore(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Расширение запроса для Manticore с сохранением оригинальных терминов
        """
        try:
            variations = self.expander.expand_query_smart(query, num_variations)
        except Exception as e:
            print(f"Query expansion error: {e}")
            return []
        
        # Фильтруем слишком длинные или пустые вариации
        filtered = []
        for var in variations:
            if var and len(var) > 0 and var != query:
                if abs(len(var) - len(query)) / max(len(query), 1) < 0.8:
                    filtered.append(var)
        
        return filtered[:num_variations]
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        use_expansion: bool = True,
        manticore_limit: int = 200
    ) -> List[Dict]:
        """
        Гибридный поиск с Query Expansion
        """
        
        print(f"\n{'='*70}")
        print(f"ГИБРИДНЫЙ ПОИСК")
        print(f"Запрос: {query}")
        print(f"{'='*70}")
        
        # 1. Собираем кандидатов из Manticore
        all_candidates = {}
        
        # Основной запрос
        print(f"\n1. Поиск в Manticore (основной запрос)...")
        main_results = self.search_manticore(query, limit=manticore_limit)
        total_main = main_results.get('hits', {}).get('total', 0)
        print(f"   Найдено: {total_main} документов")
        
        for hit in main_results.get('hits', {}).get('hits', []):
            doc_id = hit.get('_id')
            all_candidates[doc_id] = hit
        
        # Расширенные запросы
        if use_expansion:
            print(f"\n2. Генерация расширенных запросов...")
            variations = self.expand_query_for_manticore(query, num_variations=3)
            print(f"   Вариации: {variations}")
            
            for var_query in variations:
                print(f"\n   Поиск: '{var_query}'...")
                var_results = self.search_manticore(var_query, limit=manticore_limit // 2)
                var_total = var_results.get('hits', {}).get('total', 0)
                print(f"      Найдено: {var_total} документов")
                
                for hit in var_results.get('hits', {}).get('hits', []):
                    doc_id = hit.get('_id')
                    if doc_id not in all_candidates:
                        all_candidates[doc_id] = hit
                    else:
                        current_score = all_candidates[doc_id].get('_score', 0)
                        new_score = hit.get('_score', 0)
                        if new_score > current_score:
                            all_candidates[doc_id]['_score'] = new_score
        
        print(f"\n3. Уникальных кандидатов: {len(all_candidates)}")
        
        if not all_candidates:
            print("Нет результатов!")
            return []
        
        # 2. Преобразуем в список
        candidate_list = list(all_candidates.values())
        
        # 3. Извлекаем тексты и эмбеддинги для кандидатов
        print(f"\n4. Вычисление семантических эмбеддингов...")
        candidate_texts = []
        candidate_scores = []
        
        for hit in candidate_list:
            source = hit.get('_source', {})
            content = source.get('content', '')
            if content:
                candidate_texts.append(content)
                candidate_scores.append(hit.get('_score', 0))
        
        if not candidate_texts:
            return []
        
        # Получаем эмбеддинги батчами
        batch_size = 8  # Уменьшил с 16 для экономии памяти
        candidate_embs = []
        
        for i in range(0, len(candidate_texts), batch_size):
            batch = candidate_texts[i:i+batch_size]
            batch_embs = []
            for j, text in enumerate(batch):
                try:
                    emb = self.get_embedding(text, pooling="mean")
                    batch_embs.append(emb)
                except Exception as e:
                    print(f"   Error processing text {i+j}: {e}")
                    # Если ошибка, добавляем нулевой вектор
                    batch_embs.append(np.zeros(384))
            candidate_embs.extend(batch_embs)
            print(f"   Обработано {min(i+batch_size, len(candidate_texts))}/{len(candidate_texts)}")
        
        candidate_embs = np.array(candidate_embs)
        # Нормализация с защитой от нулевых векторов
        norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        candidate_embs = candidate_embs / norms
        
        # 4. Эмбеддинг запроса
        print(f"\n5. Вычисление эмбеддинга запроса...")
        
        try:
            if use_expansion:
                expanded_emb = self.expander.get_expanded_embedding(query, strategy="mean")
                original_emb = self.get_embedding(query, pooling="mean")
                query_emb = (1 - semantic_weight) * original_emb + semantic_weight * expanded_emb
            else:
                query_emb = self.get_embedding(query, pooling="mean")
            
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        except Exception as e:
            print(f"Error computing query embedding: {e}")
            # Fallback: используем оригинальный эмбеддинг без расширения
            query_emb = self.get_embedding(query, pooling="mean")
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        # 5. Семантические оценки
        semantic_scores = np.dot(candidate_embs, query_emb)
        
        # 6. Нормализация keyword оценок
        max_keyword = max(candidate_scores) if candidate_scores else 1
        keyword_scores_norm = np.array([s / max_keyword for s in candidate_scores])
        
        # 7. Комбинированный скор
        combined_scores = semantic_weight * semantic_scores + keyword_weight * keyword_scores_norm
        
        # 8. Сортировка
        sorted_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        # 9. Формируем результат
        results = []
        for idx in sorted_indices:
            hit = candidate_list[idx]
            source = hit.get('_source', {})
            
            content = source.get('content', '')
            content_preview = content[:500] + '...' if len(content) > 500 else content
            
            results.append({
                'id': hit.get('_id'),
                'score': float(combined_scores[idx]),
                'semantic_score': float(semantic_scores[idx]),
                'keyword_score': float(keyword_scores_norm[idx]),
                'keyword_raw_score': float(candidate_scores[idx]),
                'title': source.get('title', ''),
                'author': source.get('author', ''),
                'source': source.get('source', ''),
                'genre': source.get('genre', ''),
                'content': content_preview,
                'chunk': source.get('chunk', 0),
            })
        
        return results


def demo_hybrid_search():
    """Демонстрация гибридного поиска"""
    
    # Инициализация
    searcher = ManticoreHybridSearch(
        bert_model_path="data/models/tiny_bert_militera_v3/best_model.pt",
        tokenizer_path="data/processed/tokenizer_militera_v3",
        manticore_url="http://localhost:9308/search",
        api_key="",
        table_name="library2026"
    )
    
    # Тестовые запросы
    test_queries = [
        "Кто командовал ленинградским фронтом в 1941-19412 годах?",
        "противостояние Тигров и ИС-2",
        "генерал командовал дивизией"
    ]
    
    for query in test_queries:
        try:
            results = searcher.hybrid_search(
                query=query,
                top_k=5,
                semantic_weight=0.6,
                keyword_weight=0.4,
                use_expansion=True,
                manticore_limit=50  # Уменьшил для теста
            )
            
            print(f"\n{'='*70}")
            print(f"РЕЗУЛЬТАТЫ ДЛЯ: '{query}'")
            print(f"{'='*70}")
            
            if not results:
                print("Нет результатов")
                continue
            
            for i, r in enumerate(results, 1):
                print(f"\n{i}. [COMBINED={r['score']:.4f} | "
                      f"SEM={r['semantic_score']:.4f} | "
                      f"KW={r['keyword_score']:.4f}]")
                print(f"   📖 {r['title']}")
                print(f"   ✍️ {r['author']}")
                print(f"   📂 {r['source']}")
                print(f"   🏷️ Жанр: {r['genre']}")
                print(f"   📄 Chunk: {r['chunk']}")
                print(f"   📝 {r['content'][:200]}...")
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo_hybrid_search()