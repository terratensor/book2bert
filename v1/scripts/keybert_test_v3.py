from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from v1.scripts.our_sentence_transformer import OurSentenceTransformer

import spacy  # Нам нужен морфологический анализатор
import os
import torch
# Устанавливаем GPU до импорта torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Проверка какая GPU используется
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Загружаем нашу модель
our_model = OurSentenceTransformer(
    model_path="data/models/tiny_bert_militera_v3/best_model.pt",
    tokenizer_path="data/processed/tokenizer_militera",
    device="cuda"
)

# Оборачиваем для совместимости с KeyBERT
class KeyBERTCompatible:
    def __init__(self, model):
        self.model = model
    
    def encode(self, sentences, **kwargs):
        return self.model.encode(sentences, **kwargs)

# Создаём KeyBERT
kw_model = KeyBERT(model=KeyBERTCompatible(our_model))

# Загружаем spaCy для фильтрации мусора (только существительные и прилагательные)
# python -m spacy download ru_core_news_sm
nlp = spacy.load("ru_core_news_sm")

text_fragment = """
Годовалые дети регулярно совершают одну и ту же ошибку. Если несколько раз подряд положить игрушку в один из двух контейнеров (например, в правый), а потом на глазах у малыша спрятать игрушку в левый контейнер, ребенок всё равно продолжает искать желанный предмет в правом контейнере. Это объясняли просто неразвитостью интеллекта. Однако эксперименты показали, что дети совершают эту ошибку только в том случае, когда игрушку прячет взрослый человек, поддерживающий с ребенком визуальный контакт. Если же взрослый не смотрит при этом на ребенка, или вообще предметы перемещаются как бы сами (то есть экспериментатор управляет ими из-за ширмы, так что ребенок его не видит), то дети гораздо реже совершают эту ошибку. По-видимому, дети воспринимают действия экспериментатора как сеанс обучения. Человек несколько раз прячет игрушку в правый контейнер, а ребенок при этом думает, что ему тем самым объясняют: смотри, игрушка всегда находится под правым контейнером. И ребенок быстро усваивает этот урок. В дальнейшем, когда игрушку кладут под другой контейнер, ребенок просто не верит своим глазам и продолжает искать там, где взрослые его научили искать. Если же игрушка как бы сама помещается много раз подряд под правый контейнер, ребенок не делает из этого таких далеко идущих выводов (см.: Детские ошибки помогают понять эволюцию разума). Дизайн эксперимента, в котором была показана склонность детей делать слишком далеко идущие выводы из сигналов, подаваемых взрослыми. József Тopál. György Gergely, Ádám Miklysi, Ágnes Erdõhegyi, Gergely Csibra. Infants' Perseverative Search Errors Are Induced by Pragmatic Misinterpretation // Science. 2003. V. 321. P. 1331–1334 Сознание малыша настроено на то, чтобы извлекать общую информацию об устройстве мира не столько из наблюдений за этим миром, сколько из общения со взрослыми. Дети постоянно ждут от взрослых, что те поделятся с ними своей мудростью. Когда взрослый передает ребенку какую-то информацию ребенок пытается найти в ней некий общий смысл, объяснение правил, порядков и законов окружающего мира. Дети склонны обобщать информацию, но не любую, а прежде всего ту, которая получена от взрослого человека при прямом контакте с ним.
"""

# 1. Генерируем кандидатов (много)
# ИСПРАВЛЕНИЕ: use_mmr=True вместо use_maxsum
# diversity=0.5 - стандартное значение (можно играться 0.3-0.7)
keywords = kw_model.extract_keywords(
    text_fragment, 
    keyphrase_ngram_range=(1, 3), 
    stop_words=None, 
    use_maxsum=False,
    use_mmr=False,          # <-- ВОТ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
    diversity=0.5,
    nr_candidates=50, 
    top_n=5
)


# Пост-фильтрация (оставляем только именные группы)
filtered_keywords = []
for phrase, score in keywords:
    doc = nlp(phrase)
    if len(doc) > 0 and doc[-1].pos_ != "NOUN":
        continue
    has_verb = any(token.pos_ == "VERB" for token in doc)
    if has_verb:
        continue
    filtered_keywords.append((phrase, score))

print(keywords)
print("\n\n")
print(filtered_keywords[:5])

# Ожидаемый вывод (близкий к идеалу):
# [('уровень топлива', 0.57), ('холостом ходу', 0.54), ('верхнюю крышку', 0.53), ('плоскости разъема', 0.51)]