# scripts/keybert_test_v3_debug.py
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from v1.scripts.our_sentence_transformer import OurSentenceTransformer

import spacy
import os
import torch
import numpy as np
import hashlib

# Устанавливаем GPU до импорта torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("=" * 80)
print("DEBUG VERSION - MODEL COMPARISON")
print("=" * 80)

# Проверка GPU
print(f"\n1. CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Using GPU: {torch.cuda.get_device_name(0)}")

# 2. Контрольная фраза для сравнения эмбеддингов
test_sentence = "суверенитет россии"

print(f"\n2. Test sentence for embedding comparison: '{test_sentence}'")

# 3. Загружаем нашу модель
print("\n3. Loading OurSentenceTransformer...")
our_model = OurSentenceTransformer(
    model_path="data/models/tiny_bert_full_15pct/best_model.pt",
    tokenizer_path="data/processed/tokenizer_mera_50000",
    device="cuda"
)

# 4. Получаем эмбеддинг от нашей модели
print("\n4. Getting embedding from OurSentenceTransformer...")
our_embedding = our_model.encode([test_sentence])[0]
our_hash = hashlib.md5(our_embedding.tobytes()).hexdigest()
print(f"   Shape: {our_embedding.shape}")
print(f"   First 10 values: {our_embedding[:10]}")
print(f"   MD5 hash (first 16 chars): {our_hash[:16]}")
print(f"   Norm: {np.linalg.norm(our_embedding):.6f}")

# 5. Загружаем стандартную модель для сравнения
print("\n5. Loading standard paraphrase-multilingual-MiniLM-L12-v2...")
try:
    standard_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
    print("   Standard model loaded successfully")
    
    # 6. Получаем эмбеддинг от стандартной модели
    print("\n6. Getting embedding from standard model...")
    standard_embedding = standard_model.encode([test_sentence])[0]
    standard_hash = hashlib.md5(standard_embedding.tobytes()).hexdigest()
    print(f"   Shape: {standard_embedding.shape}")
    print(f"   First 10 values: {standard_embedding[:10]}")
    print(f"   MD5 hash (first 16 chars): {standard_hash[:16]}")
    print(f"   Norm: {np.linalg.norm(standard_embedding):.6f}")
    
    # 7. Сравниваем эмбеддинги
    print("\n7. Comparing embeddings:")
    cosine_sim = np.dot(our_embedding, standard_embedding) / (np.linalg.norm(our_embedding) * np.linalg.norm(standard_embedding))
    print(f"   Cosine similarity between models: {cosine_sim:.6f}")
    
    if cosine_sim > 0.99:
        print("   ⚠️  WARNING: Embeddings are almost identical! Models might be the same!")
    elif cosine_sim > 0.8:
        print("   ⚠️  NOTE: High similarity, but not identical")
    else:
        print("   ✓ Embeddings are different - models are distinct")
        
except Exception as e:
    print(f"   Error loading standard model: {e}")
    print("   Continuing without comparison...")
    standard_model = None

# 8. Создаём обёртку для KeyBERT
class KeyBERTCompatible:
    def __init__(self, model, model_name="custom"):
        self.model = model
        self.model_name = model_name
    
    def encode(self, sentences, **kwargs):
        print(f"   [DEBUG] KeyBERT.encode() called with {len(sentences) if isinstance(sentences, list) else 1} sentences")
        if isinstance(sentences, list) and len(sentences) > 0:
            print(f"   [DEBUG] First sentence: '{sentences[0][:50]}...'")
        result = self.model.encode(sentences, **kwargs)
        print(f"   [DEBUG] Returned embeddings shape: {result.shape}")
        return result

# 9. Создаём KeyBERT с нашей моделью
print("\n8. Creating KeyBERT with custom model...")
kw_model_custom = KeyBERT(model=KeyBERTCompatible(our_model, "custom"))

# 10. Загружаем spaCy
print("\n9. Loading spaCy...")
nlp = spacy.load("ru_core_news_sm")
print("   spaCy loaded")

# 11. Текст для анализа (очищенная версия без тегов)
print("\n10. Preparing text...")
text_fragment_clean = """
        Гораздо проще и правильней вынести этот код в функцию (это может быть и несколько функций).

И тогда будет производиться вызов этой функции - в этом файле или каком-то другом.

В этом разделе рассматривается ситуация, когда функция находится в том же файле.
"""

text_fragment_original = """
Хиппи с их длинными волосами, экзотической восточной вышитой одеждой, нарисованными на лице цветами и босыми ногами в корне изменили моду и стиль красоты в 1968–1969 годах, а последовавшее за ними увлечение стилем «ретро» вернуло женщинам моду эпохи их бабушек — элементы тридцатых и отчасти сороковых годов: брюки клеш, батники, безрукавки из джерси, юбки «макси» и платформы. Диско с блестящими тканями, пластиком, золотым лаком и музыкой АВВА и Bony М сделали фигуру манящей и обтянутой, а секс — нормой жизни.

Панки подарили миру черный цвет в моде 1980-х годов. Творцы моды Клод Монтана и Тьерри Мюглер вернули женщинам в самом начале этого периода подкладные плечи, а от японцев наши красавицы переняли манеру одеваться многослойно. Богатая мода восьмидесятых годов никак не соответствует стилю нынешнего времени. Крушение Берлинской стены и Кувейтский кризис ввели мир моды в период некоторого запустения. Из Америки пришел стиль «гранж»: трикотажные обвислые бесформенные одежды экологически чистых цветов. Наступило время неприбранных волос и невзрачного макияжа. Гламур вернулся в моду лишь в середине 90-х годов, в период возвращения стиля «диско». Минимализм, аскетические формы простых приталенных и длинных однотонных платьев лидировали в сезоне 1996 — начала 1997 годов, но пришедшие в мир парижского вкуса и элегантности англичане Александр Маккуин и Джон Гальяно повернули стиль конца XX века вспять — уверенно глядя в исторически далекую от нас Викторианскую эпоху, они пытаются заставить всех носить длинные платья, турнюры, корсеты, кринолины, «шпильки», шляпки, вуалетки и вышитые аксессуары в сочетании с натуральным мехом. Стиль этого времени, полностью вытеснивший аскетический минимализм прошлого, следует именовать «историческим максимализмом». Тем не менее подобная ретроспективная мода, питающаяся традициями и соками прошлого, никак не может ввести женщин в стиль XXI века: века Интернета, скоростей, единой Европы и новых технологий.
"""

print(f"   Original text length: {len(text_fragment_original)} chars")
print(f"   Clean text length: {len(text_fragment_clean)} chars")



# 14. Если есть стандартная модель, тестируем её на чистом тексте
if standard_model:
    print("\n12. Testing standard model on CLEAN text...")
    kw_model_standard = KeyBERT(model=standard_model)
    keywords_standard = kw_model_standard.extract_keywords(
        text_fragment_original, 
        keyphrase_ngram_range=(1, 3), 
        stop_words=None, 
        use_maxsum=False,
        use_mmr=False,
        diversity=0.5,
        nr_candidates=50, 
        top_n=15
    )
    
    print("\n   Results from standard model (clean text):")
    for i, (phrase, score) in enumerate(keywords_standard[:10], 1):
        print(f"   {i}. '{phrase}' - {score:.4f}")


# 12. Тестируем на чистом тексте
print("\n11. Extracting keywords from CLEAN text with custom model...")
keywords_clean = kw_model_custom.extract_keywords(
    text_fragment_original, 
    keyphrase_ngram_range=(1, 3), 
    stop_words=None, 
    use_maxsum=False,
    use_mmr=False,
    diversity=0.5,
    nr_candidates=50, 
    top_n=15
)

print("\n   Results from CLEAN text:")
for i, (phrase, score) in enumerate(keywords_clean[:10], 1):
    print(f"   {i}. '{phrase}' - {score:.4f}")

# 13. Пост-фильтрация для чистого текста
filtered_clean = []
for phrase, score in keywords_clean:
    doc = nlp(phrase)
    if len(doc) > 0 and doc[-1].pos_ != "NOUN":
        continue
    has_verb = any(token.pos_ == "VERB" for token in doc)
    if has_verb:
        continue
    filtered_clean.append((phrase, score))

print("\n   Filtered results (nouns only, no verbs):")
for i, (phrase, score) in enumerate(filtered_clean[:5], 1):
    print(f"   {i}. '{phrase}' - {score:.4f}")
    
    # Сравниваем результаты двух моделей
    print("\n13. Comparing results between models:")
    custom_phrases = set([p for p, _ in keywords_clean[:10]])
    standard_phrases = set([p for p, _ in keywords_standard[:10]])
    overlap = custom_phrases & standard_phrases
    print(f"   Overlap in top-10 phrases: {len(overlap)}/10")
    print(f"   Common phrases: {overlap if overlap else 'None'}")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)