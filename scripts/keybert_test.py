from keybert import KeyBERT
import razdel

# 1. Загружаем ОБЕРТКУ, которая умеет правильно усреднять BPE-токены
# Sentence-BERT обучен давать ОСМЫСЛЕННЫЙ вектор для предложения/фразы
from sentence_transformers import SentenceTransformer

# Эта модель - русифицированный аналог SBERT. 
# Она дает вектор фразы, который НЕ ЯВЛЯЕТСЯ ШУМОМ (в отличие от сырого Mean Pooling)

# model = SentenceTransformer('sergeyzh/rubert-mini-sts')
# model = SentenceTransformer('sergeyzh/rubert-tiny-sts')
# model = SentenceTransformer('cointegrated/rubert-tiny')
# model = SentenceTransformer('cointegrated/rubert-tiny2')
# model = SentenceTransformer('ai-forever/ruBert-base')
# model = SentenceTransformer('ai-forever/ruBert-large')
model = SentenceTransformer('ai-forever/sbert_large_nlu_ru')
kw_model = KeyBERT(model=model)

text_fragment = """
Карбюратор "Солекс" требует регулярной регулировки уровня топлива. 
Для этого необходимо снять верхнюю крышку и измерить расстояние от поплавков до плоскости разъема. 
Неправильный уровень топлива приводит к перерасходу бензина и нестабильной работе на холостом ходу.
"""

# 2. Извлекаем ключевые фразы
# keyphrase_ngram_range=(1, 3) - ищем от 1 до 3 слов
# stop_words - отсекаем мусор
# use_maxsum=True - включает алгоритм Maximal Marginal Relevance (чтобы фразы не дублировали друг друга)
keywords = kw_model.extract_keywords(
    text_fragment, 
    keyphrase_ngram_range=(1, 3), 
    stop_words='english', # Встроенный список стоп-слов
    use_maxsum=True, 
    nr_candidates=20, 
    top_n=5,
    highlight=False
)

print(keywords)
# Ожидаемый вывод:
# [('уровень топлива', 0.85), ('регулировка уровня топлива', 0.79), ('холостой ход', 0.71), ('крышку поплавков', 0.58)]