#!/usr/bin/env python3
"""
Простая версия гибридного поиска для вашего API
"""

import requests
import json
import numpy as np
from typing import List, Dict

# Конфигурация
MANTICORE_URL = "http://localhost:9308/search"
API_KEY = ""
TABLE_NAME = "library2026"


def search_manticore(query: str, limit: int = 50) -> List[Dict]:
    """Простой поиск в Manticore"""
    headers = {
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
    }
    
    payload = {
        "table": TABLE_NAME,
        "query": {"query_string": query},
        "limit": limit
    }
    
    response = requests.post(MANTICORE_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        return data.get('hits', {}).get('hits', [])
    else:
        print(f"Error: {response.status_code}")
        return []


def simple_hybrid_search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Простой гибридный поиск:
    1. Ищем в Manticore
    2. Сортируем по _score (BM25)
    3. Возвращаем топ-k
    """
    results = search_manticore(query, limit=top_k)
    
    formatted = []
    for hit in results:
        source = hit.get('_source', {})
        formatted.append({
            'score': hit.get('_score', 0),
            'title': source.get('title', ''),
            'author': source.get('author', ''),
            'content': source.get('content', '')[:300],
            'source': source.get('source', '')
        })
    
    return formatted


if __name__ == "__main__":
    query = "бои на танках Т-34"
    results = simple_hybrid_search(query, top_k=5)
    
    print(f"\nРезультаты для: {query}")
    print("=" * 60)
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Score: {r['score']}")
        print(f"   {r['title']} - {r['author']}")
        print(f"   {r['content'][:150]}...")