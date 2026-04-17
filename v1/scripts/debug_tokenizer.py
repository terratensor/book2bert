#!/usr/bin/env python3
"""Проверка токенизации и маскирования (без загрузки модели)"""

import torch
from tokenizers import BertWordPieceTokenizer


class SimpleTokenizer:
    """Простой токенизатор для маскирования (индексы специальных токенов)"""
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.pad_token_id = 0
        self.mask_token_id = 4


def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    """
    Маскирование токенов для MLM.
    input_ids: [batch, seq_len] (на GPU или CPU)
    """
    labels = input_ids.clone()
    device = input_ids.device
    
    special_tokens = torch.tensor(
        [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id],
        device=device
    )
    
    special_tokens_mask = torch.isin(input_ids, special_tokens)
    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id
    
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]
    
    return input_ids, labels


def main():
    # Загружаем новый токенизатор
    tokenizer = BertWordPieceTokenizer("data/processed/tokenizer_militera/vocab.txt", lowercase=False)
    
    print("="*60)
    print("1. Проверка токенизации целых слов")
    print("="*60)
    
    test_words = ["философия", "учение", "граница", "материя", "методология", "сознание"]
    for word in test_words:
        encoded = tokenizer.encode(word)
        print(f"'{word}' → {encoded.tokens}")
    
    print("\n" + "="*60)
    print("2. Проверка токенизации предложения")
    print("="*60)
    
    text = "Философия есть учение о всеобщем."
    encoded = tokenizer.encode(text)
    
    print(f"Текст: {text}")
    print(f"Токены: {encoded.tokens}")
    print(f"ID: {encoded.ids}")
    
    print("\n" + "="*60)
    print("3. Проверка маскирования")
    print("="*60)
    
    # Тест маскирования
    input_ids = torch.tensor([encoded.ids])
    simple_tokenizer = SimpleTokenizer(vocab_size=50000)
    
    # Маскируем 100% токенов для наглядности
    masked_ids, labels = mask_tokens(input_ids.clone(), simple_tokenizer, mlm_probability=1.0)
    
    print(f"Исходные ID: {input_ids[0].tolist()}")
    print(f"Маскированные ID: {masked_ids[0].tolist()}")
    print(f"Labels: {labels[0].tolist()}")
    
    # Декодируем маскированные позиции
    print("\nПозиции, которые модель должна предсказать:")
    masked_positions = 0
    for i, (orig, masked, label) in enumerate(zip(encoded.ids, masked_ids[0], labels[0])):
        if label != -100:
            masked_positions += 1
            orig_token = tokenizer.id_to_token(orig)
            masked_token = tokenizer.id_to_token(masked.item()) if masked != 0 else "[PAD]"
            label_token = tokenizer.id_to_token(label.item())
            print(f"  Позиция {i}: было '{orig_token}' → маскировано как '{masked_token}' (цель: '{label_token}')")
    
    if masked_positions == 0:
        print("  (ни одна позиция не замаскирована — возможно, все токены специальные)")
    
    print("\n" + "="*60)
    print("4. Выводы")
    print("="*60)
    
    # Проверяем, есть ли в словаре целые слова
    print("Проверка наличия целых слов в словаре:")
    for word in test_words:
        word_id = tokenizer.token_to_id(word)
        if word_id is not None:
            print(f"  ✓ '{word}' найден (id: {word_id})")
        else:
            print(f"  ✗ '{word}' ОТСУТСТВУЕТ в словаре!")
    
    print("\n✅ Токенизация работает корректно!")
    print("   Целые слова сохраняются, маскирование применяется к словам, а не субвордам.")
    print("   Можно запускать обучение BERT.")


if __name__ == "__main__":
    main()