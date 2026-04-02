#!/usr/bin/env python3
"""Проверка корректности маскирования и токенизации"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

import torch
from tokenizers import BertWordPieceTokenizer
from model import BERT, BERTForMLM
from train_tiny import mask_tokens, SimpleTokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Загружаем токенизатор
    tokenizer = BertWordPieceTokenizer("data/processed/tokenizer/vocab.txt", lowercase=False)
    
    # Тестовый текст
    text = "Философия есть учение о всеобщем."
    
    print("="*60)
    print("1. Токенизация текста")
    print("="*60)
    encoded = tokenizer.encode(text)
    print(f"Текст: {text}")
    print(f"Токены: {encoded.tokens}")
    print(f"ID: {encoded.ids}")
    print(f"Длина: {len(encoded.tokens)}")
    
    # Проверка специальных токенов
    print("\n" + "="*60)
    print("2. Специальные токены")
    print("="*60)
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for st in special_tokens:
        print(f"{st} -> {tokenizer.token_to_id(st)}")
    
    # Тест маскирования
    print("\n" + "="*60)
    print("3. Проверка маскирования")
    print("="*60)
    
    # Создаем тензор input_ids
    input_ids = torch.tensor([encoded.ids])
    simple_tokenizer = SimpleTokenizer(vocab_size=30000)
    
    masked_ids, labels = mask_tokens(input_ids.clone(), simple_tokenizer, mlm_probability=0.15)
    
    print(f"Исходные ID: {input_ids[0].tolist()}")
    print(f"Маскированные ID: {masked_ids[0].tolist()}")
    print(f"Labels: {labels[0].tolist()}")
    
    # Декодируем
    print("\nДекодированные токены:")
    for i, (orig, masked, label) in enumerate(zip(input_ids[0], masked_ids[0], labels[0])):
        orig_token = tokenizer.id_to_token(orig.item())
        masked_token = tokenizer.id_to_token(masked.item()) if masked != 0 else "[PAD]"
        label_token = tokenizer.id_to_token(label.item()) if label != -100 else "IGNORE"
        if label != -100:
            print(f"  Позиция {i}: было '{orig_token}' -> стало '{masked_token}' (должен предсказать '{label_token}')")
    
    # Проверка модели
    print("\n" + "="*60)
    print("4. Загрузка модели")
    print("="*60)
    
    checkpoint = torch.load("data/models/tiny_bert_20260402_103424/best_model.pt", map_location=device)
    
    bert = BERT(
        vocab_size=30000,
        hidden_size=384,
        num_layers=6,
        num_heads=12,
        intermediate_size=1536,
        max_position=512,
        dropout=0.1
    )
    model = BERTForMLM(bert, 30000)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Модель загружена")
    
    # Проверка предсказаний на простом предложении
    print("\n" + "="*60)
    print("5. Проверка предсказаний на маскированном предложении")
    print("="*60)
    
    test_text = "Философия есть [MASK] о всеобщем."
    print(f"Текст: {test_text}")
    
    encoded = tokenizer.encode(test_text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    
    print(f"Токены: {encoded.tokens}")
    print(f"Позиция маски: {encoded.tokens.index('[MASK]')}")
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs['logits']
    
    mask_pos = encoded.tokens.index('[MASK]')
    mask_logits = logits[0, mask_pos, :]
    top10 = torch.topk(mask_logits, 10)
    
    print("\nTop-10 предсказаний:")
    for i, idx in enumerate(top10.indices):
        token = tokenizer.id_to_token(idx.item())
        score = top10.values[i].item()
        print(f"  {i+1}. '{token}' (score: {score:.2f})")
    
    # Проверка: каков истинный токен на этой позиции?
    # В оригинале без маски должно быть "учение"
    print("\n" + "="*60)
    print("6. Что должно быть")
    print("="*60)
    original_encoded = tokenizer.encode("Философия есть учение о всеобщем.")
    print(f"Оригинальные токены: {original_encoded.tokens}")
    print(f"На позиции маски ожидается: {original_encoded.tokens[mask_pos]}")

if __name__ == "__main__":
    main()