#!/usr/bin/env python3
"""
Интерактивный гибридный поиск с Query Expansion для Manticore API
Пользователь может вводить запросы в консоли и получать результаты поиска
"""

import sys
from pathlib import Path
import requests
import json
import numpy as np
from typing import List, Dict, Optional
import readline  # для удобного ввода в консоли

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
        
        print(f"🔧 Используется устройство: {self.device}")
        
        # Загрузка BERT модели
        self._load_bert(bert_model_path, tokenizer_path)
        
        # Query Expander
        self.expander = QueryExpander(
            model_path=bert_model_path,
            tokenizer_path=tokenizer_path,
            device=device
        )
        
        # Настройки поиска (можно менять в интерактивном режиме)
        self.settings = {
            'top_k': 10,
            'semantic_weight': 0.6,
            'keyword_weight': 0.4,
            'use_expansion': False,
            'manticore_limit': 100
        }
        
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
        
        print("✅ BERT модель загружена успешно")
    
    def truncate_to_max_tokens(self, text: str, max_tokens: int = None) -> str:
        """Обрезает текст до максимального количества токенов"""
        if max_tokens is None:
            max_tokens = self.MAX_SEQ_LEN - 2
        
        encoded = self.tokenizer.encode(text)
        tokens = encoded.tokens
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(self.tokenizer.encode(truncated_tokens).ids)
        return truncated_text

    def clean_text_for_tokenizer(self, text) -> str:
        """Очистка текста: приводим к строке, убираем проблемные символы"""
        # 1. Приводим к строке (даже если пришел int, list, None)
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        # 2. Если строка слишком короткая или не содержит текста
        if len(text) < 2:
            return ""
        
        # 3. Если строка похожа на ID или служебные данные
        if text.strip().startswith('[') and text.strip().endswith(']'):
            # Проверяем, не состоит ли она только из цифр внутри скобок
            inner = text.strip()[1:-1]
            if inner.isdigit() or inner == '':
                return ""
        
        # 4. Оставляем только символы в нужных диапазонах
        cleaned = []
        for ch in text:
            code = ord(ch)
            if code == 9 or code == 10 or code == 13:  # tab, LF, CR
                cleaned.append(ch)
            elif 32 <= code <= 126:  # ASCII printable
                cleaned.append(ch)
            elif 0x0400 <= code <= 0x04FF:  # Cyrillic
                cleaned.append(ch)
            # остальные символы пропускаем
        
        result = ''.join(cleaned).strip()
        
        # 5. Если после очистки текст стал слишком коротким
        if len(result) < 2:
            return ""
        
        return result


    def get_embedding(self, text, pooling: str = "mean") -> np.ndarray:
        """Получение эмбеддинга текста"""
        try:
            # Очистка и приведение к строке
            text = self.clean_text_for_tokenizer(text)
            
            if not text or len(text.strip()) < 2:
                return np.zeros(384)
            
            # Обрезаем до максимальной длины
            text = self.truncate_to_max_tokens(text)
            
            # Токенизация
            encoded = self.tokenizer.encode(text)
            
            input_ids = torch.tensor([encoded.ids]).to(self.device)
            attention_mask = torch.tensor([encoded.attention_mask]).to(self.device)
            
            # Обрезаем, если превышает MAX_SEQ_LEN
            if input_ids.shape[1] > self.MAX_SEQ_LEN:
                input_ids = input_ids[:, :self.MAX_SEQ_LEN]
                attention_mask = attention_mask[:, :self.MAX_SEQ_LEN]
            
            with torch.no_grad():
                hidden_states, _ = self.bert(input_ids, attention_mask)
            
            if pooling == "cls":
                return hidden_states[0, 0, :].cpu().numpy()
            else:
                mask = attention_mask[0].cpu().numpy()
                embeddings = hidden_states[0].cpu().numpy()
                sum_mask = np.sum(mask)
                if sum_mask == 0:
                    return np.zeros(384)
                mean_emb = np.sum(embeddings * mask[:, np.newaxis], axis=0) / sum_mask
                return mean_emb
                
        except Exception as e:
            print(f"   [get_embedding ERROR] {e}")
            print(f"   text type: {type(text)}, value: {str(text)[:100] if text else 'empty'}")
            return np.zeros(384)
    
    def search_manticore(
        self, 
        query: str, 
        limit: int = 100,
        offset: int = 0
    ) -> Dict:
        """Поиск в Manticore через ваш API"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        
        payload = {
            "table": self.table_name,
            "query": {"query_string": query},
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
                return {"hits": {"hits": [], "total": 0}}
                
        except Exception as e:
            return {"hits": {"hits": [], "total": 0}}
    
    def expand_query_for_manticore(self, query: str, num_variations: int = 3) -> List[str]:
        """Расширение запроса для Manticore"""
        try:
            variations = self.expander.expand_query_smart(query, num_variations)
        except Exception as e:
            return []
        
        filtered = []
        for var in variations:
            if var and len(var) > 0 and var != query:
                if abs(len(var) - len(query)) / max(len(query), 1) < 0.8:
                    filtered.append(var)
        
        return filtered[:num_variations]
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        semantic_weight: float = None,
        keyword_weight: float = None,
        use_expansion: bool = None,
        manticore_limit: int = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Гибридный поиск с Query Expansion
        """
        # Используем текущие настройки, если параметры не переданы
        top_k = top_k or self.settings['top_k']
        semantic_weight = semantic_weight or self.settings['semantic_weight']
        keyword_weight = keyword_weight or self.settings['keyword_weight']
        use_expansion = use_expansion if use_expansion is not None else self.settings['use_expansion']
        manticore_limit = manticore_limit or self.settings['manticore_limit']
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"ГИБРИДНЫЙ ПОИСК")
            print(f"Запрос: {query}")
            print(f"{'='*70}")
        
        # 1. Собираем кандидатов из Manticore
        all_candidates = {}
        
        if verbose:
            print(f"\n1. Поиск в Manticore (основной запрос)...")
        main_results = self.search_manticore(query, limit=manticore_limit)
        total_main = main_results.get('hits', {}).get('total', 0)
        if verbose:
            print(f"   Найдено: {total_main} документов")
        
        for hit in main_results.get('hits', {}).get('hits', []):
            doc_id = hit.get('_id')
            all_candidates[doc_id] = hit
        
        # Расширенные запросы
        if use_expansion:
            if verbose:
                print(f"\n2. Генерация расширенных запросов...")
            variations = self.expand_query_for_manticore(query, num_variations=3)
            if verbose:
                print(f"   Вариации: {variations}")
            
            for var_query in variations:
                if verbose:
                    print(f"\n   Поиск: '{var_query}'...")
                var_results = self.search_manticore(var_query, limit=manticore_limit // 2)
                var_total = var_results.get('hits', {}).get('total', 0)
                if verbose:
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
        
        if verbose:
            print(f"\n3. Уникальных кандидатов: {len(all_candidates)}")
        
        if not all_candidates:
            if verbose:
                print("Нет результатов!")
            return []
        
        candidate_list = list(all_candidates.values())
        
        # 2. Вычисляем эмбеддинги
        if verbose:
            print(f"\n4. Вычисление семантических эмбеддингов...")
        
        candidate_texts = []
        candidate_scores = []

        for idx, hit in enumerate(candidate_list):
            source = hit.get('_source', {})
            content = source.get('content', '')
            
            # ДИАГНОСТИКА
            if not isinstance(content, str):
                print(f"   [DIAG] Документ {idx}: content имеет тип {type(content)}, а не str")
                print(f"   [DIAG] Содержимое: {repr(content)[:200]}")
                content = str(content) if content is not None else ""
            
            if content:
                candidate_texts.append(content)
                candidate_scores.append(hit.get('_score', 0))
                
                if not candidate_texts:
                    return []
        
        # Получаем эмбеддинги батчами
        batch_size = 8
        candidate_embs = []
        
        for i in range(0, len(candidate_texts), batch_size):
            batch = candidate_texts[i:i+batch_size]
            batch_embs = []
            for text in batch:
                emb = self.get_embedding(text, pooling="mean")
                batch_embs.append(emb)
            candidate_embs.extend(batch_embs)
            if verbose:
                print(f"   Обработано {min(i+batch_size, len(candidate_texts))}/{len(candidate_texts)}")
        
        candidate_embs = np.array(candidate_embs)
        norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        candidate_embs = candidate_embs / norms
        
        # 3. Эмбеддинг запроса
        if verbose:
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
            if verbose:
                print(f"Error computing query embedding: {e}")
            query_emb = self.get_embedding(query, pooling="mean")
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        # 4. Оценки
        semantic_scores = np.dot(candidate_embs, query_emb)
        
        max_keyword = max(candidate_scores) if candidate_scores else 1
        keyword_scores_norm = np.array([s / max_keyword for s in candidate_scores])
        
        combined_scores = semantic_weight * semantic_scores + keyword_weight * keyword_scores_norm
        
        # 5. Сортировка
        sorted_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        # 6. Формируем результат
        results = []
        for idx in sorted_indices:
            hit = candidate_list[idx]
            source = hit.get('_source', {})
            
            content = source.get('content', '')
            
            results.append({
                'id': hit.get('_id'),
                'score': float(combined_scores[idx]),
                'semantic_score': float(semantic_scores[idx]),
                'keyword_score': float(keyword_scores_norm[idx]),
                'keyword_raw_score': float(candidate_scores[idx]),
                'title': source.get('title', 'Нет названия'),
                'author': source.get('author', 'Неизвестен'),
                'source': source.get('source', 'Неизвестно'),
                'genre': source.get('genre', 'Не указан'),
                'content': content,
                'chunk': source.get('chunk', 0),
            })
        
        return results


def print_results(results: List[Dict], query: str):
    """Красивый вывод результатов поиска с ПОЛНЫМ текстом"""
    if not results:
        print("\n❌ Результатов не найдено")
        return
    
    print(f"\n{'='*70}")
    print(f"РЕЗУЛЬТАТЫ ДЛЯ: '{query}'")
    print(f"{'='*70}")
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [COMBINED={r['score']:.4f} | "
              f"SEM={r['semantic_score']:.4f} | "
              f"KW={r['keyword_score']:.4f}]")
        print(f"   📖 {r['title']}")
        print(f"   ✍️ {r['author']}")
        print(f"   📂 {r['source']}")
        print(f"   🏷️ Жанр: {r['genre']}")
        print(f"   📄 Chunk: {r['chunk']}")
        print(f"\n   📝 Текст фрагмента:")
        print(f"   {r['content']}")
        print(f"\n   ---")


def main():
    """Основная функция интерактивного поиска"""
    
    # Параметры для подключения
    bert_model_path = "data/models/tiny_bert_militera_v3/best_model.pt"
    tokenizer_path = "data/processed/tokenizer_militera_v3"
    manticore_url = "http://localhost:9308/search"
    api_key = ""
    table_name = "library2026"
    
    # Проверка существования файлов модели
    if not Path(bert_model_path).exists():
        print(f"❌ Модель не найдена: {bert_model_path}")
        sys.exit(1)
    
    if not Path(tokenizer_path).exists():
        print(f"❌ Токенизатор не найден: {tokenizer_path}")
        sys.exit(1)
    
    # Инициализация поиска
    print("🚀 Загрузка модели и инициализация поиска...")
    
    try:
        searcher = ManticoreHybridSearch(
            bert_model_path=bert_model_path,
            tokenizer_path=tokenizer_path,
            manticore_url=manticore_url,
            api_key=api_key,
            table_name=table_name
        )
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Интерактивный гибридный поиск")
    print("Команды: /quit - выход, /help - справка, /settings - настройки")
    print("="*70)
    
    while True:
        try:
            query = input("\n🔍 Поиск> ").strip()
            
            if not query:
                continue
            
            # Обработка команд
            if query.lower() in ['/quit', '/exit', '/q']:
                print("👋 До свидания!")
                break
            
            elif query.lower() == '/help':
                print("""
Доступные команды:
  /quit, /exit, /q - выход
  /help            - эта справка
  /settings        - показать текущие настройки
  /expansion on/off - включить/выключить расширение запросов
  /set top_k <число> - изменить количество результатов
  /set semantic_weight <0-1> - изменить вес семантики
  /set keyword_weight <0-1> - изменить вес ключевых слов
""")
                continue
            
            elif query.lower() == '/settings':
                s = searcher.settings
                print(f"""
Текущие настройки:
  top_k: {s['top_k']}
  semantic_weight: {s['semantic_weight']}
  keyword_weight: {s['keyword_weight']}
  use_expansion: {'вкл' if s['use_expansion'] else 'выкл'}
  manticore_limit: {s['manticore_limit']}
""")
                continue
            
            elif query.lower().startswith('/expansion'):
                parts = query.split()
                if len(parts) == 2:
                    if parts[1].lower() == 'on':
                        searcher.settings['use_expansion'] = True
                        print("✅ Расширение запросов ВКЛЮЧЕНО")
                    elif parts[1].lower() == 'off':
                        searcher.settings['use_expansion'] = False
                        print("✅ Расширение запросов ВЫКЛЮЧЕНО")
                    else:
                        print("Используйте: /expansion on  или  /expansion off")
                else:
                    print(f"Текущее состояние: {'включено' if searcher.settings['use_expansion'] else 'выключено'}")
                continue
            
            elif query.lower().startswith('/set'):
                parts = query.split()
                if len(parts) == 3:
                    param = parts[1].lower()
                    value = parts[2]
                    if param == 'top_k':
                        try:
                            searcher.settings['top_k'] = int(value)
                            print(f"✅ top_k установлен в {value}")
                        except:
                            print("❌ Введите число")
                    elif param == 'semantic_weight':
                        try:
                            v = float(value)
                            if 0 <= v <= 1:
                                searcher.settings['semantic_weight'] = v
                                print(f"✅ semantic_weight установлен в {v}")
                            else:
                                print("❌ Значение должно быть от 0 до 1")
                        except:
                            print("❌ Введите число")
                    elif param == 'keyword_weight':
                        try:
                            v = float(value)
                            if 0 <= v <= 1:
                                searcher.settings['keyword_weight'] = v
                                print(f"✅ keyword_weight установлен в {v}")
                            else:
                                print("❌ Значение должно быть от 0 до 1")
                        except:
                            print("❌ Введите число")
                    else:
                        print(f"Неизвестный параметр: {param}")
                else:
                    print("Используйте: /set <параметр> <значение>")
                continue
            
            # Выполняем поиск
            results = searcher.hybrid_search(
                query=query,
                verbose=True  # Показываем весь процесс
            )
            
            print_results(results, query)
            
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()