#!/usr/bin/env python3
"""
Query Expansion (Расширение запроса)
"""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT
from tokenizers import BertWordPieceTokenizer

def expand_query(query, num_augmentations=5):
    """
    Генерируем похожие запросы через ваш Tiny BERT (MLM голову)
    """
    expanded_queries = [query]
    
    # Маскируем ключевые слова и предсказываем
    # "бои на [MASK] Т-34" → "бои на танках Т-34", "бои на броне Т-34"
    
    for _ in range(num_augmentations):
        # MLM предсказание
        masked = mask_random_tokens(query, mask_prob=0.3)
        predicted = mlm_predict(masked)  # используем вашу MLM голову
        expanded_queries.append(predicted)
    
    # Усредняем эмбеддинги расширенных запросов
    query_emb = mean_pool(embedder.encode(expanded_queries))
    return query_emb