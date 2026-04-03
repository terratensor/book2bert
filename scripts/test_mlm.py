#!/usr/bin/env python3
"""Тест MLM предсказаний на реальных примерах из militera корпуса"""

import torch
import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT, BERTForMLM
from tokenizers import BertWordPieceTokenizer

def is_english(text):
    """Проверяет, является ли текст английским"""
    english_chars = sum(1 for c in text if ord('a') <= ord(c) <= ord('z') or ord('A') <= ord(c) <= ord('Z'))
    total_chars = sum(1 for c in text if c.isalpha())
    if total_chars == 0:
        return False
    return english_chars / total_chars > 0.7

def load_real_examples(sentences_dir, num_examples=10):
    """Загружает реальные предложения из militera корпуса (только русские)"""
    files = list(Path(sentences_dir).glob("*.jsonl"))
    random.shuffle(files)
    examples = []
    
    for filepath in files[:10]:  # берем первые 10 файлов
        with open(filepath, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines[:5]:  # из каждого файла 5 строк
                data = json.loads(line)
                text = data['text']
                # Проверяем: русский текст, не слишком короткий, не слишком длинный
                if (not is_english(text) and 
                    len(text.split()) > 8 and 
                    len(text) < 300 and
                    any('а' <= c <= 'я' or 'А' <= c <= 'Я' for c in text)):
                    examples.append(text)
                if len(examples) >= num_examples:
                    return examples[:num_examples]
    
    return examples[:num_examples]

def create_masked_example(text):
    """Маскирует одно случайное знаменательное слово"""
    words = text.split()
    if len(words) < 4:
        return text, None
    
    # Стоп-слова (не маскируем)
    stop_words = {}
    
    # Находим позиции знаменательных слов
    content_positions = []
    for i, w in enumerate(words):
        w_clean = w.strip('.,!?;:()[]«»"\'')
        if len(w_clean) > 2 and w_clean.lower() not in stop_words and w_clean[0].isalpha():
            content_positions.append(i)
    
    if not content_positions:
        return text, None
    
    pos = random.choice(content_positions)
    original_word = words[pos]
    words[pos] = "[MASK]"
    return ' '.join(words), original_word

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Пути
    model_path = "data/models/tiny_bert_militera_v2/best_model.pt"
    tokenizer_path = "data/processed/tokenizer_militera"
    sentences_dir = "data/processed/sentences_militera"
    
    # Загружаем токенизатор
    tokenizer = BertWordPieceTokenizer(
        str(Path(tokenizer_path) / "vocab.txt"),
        lowercase=False
    )
    
    # Создаем модель
    bert = BERT(
        vocab_size=50000,
        hidden_size=384,
        num_layers=6,
        num_heads=12,
        intermediate_size=1536,
        max_position=512,
        dropout=0.1
    )
    model = BERTForMLM(bert, vocab_size=50000)
    
    # Загружаем веса
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully\n")
    
    # Загружаем реальные примеры
    print("Loading real examples from militera corpus...")
    examples = load_real_examples(sentences_dir, 8)
    
    print("="*60)
    print("MLM Predictions on military-historical texts (after epoch 1)")
    print("="*60 + "\n")
    
    for original in examples:
        masked_text, original_word = create_masked_example(original)
        if masked_text is None or original_word is None:
            continue
            
        print(f"Original: {original[:120]}...")
        print(f"Masked:   {masked_text}")
        
        encoded = tokenizer.encode(masked_text)
        input_ids = torch.tensor([encoded.ids]).to(device)
        
        # Находим позицию маски
        mask_token_id = tokenizer.token_to_id("[MASK]")
        mask_positions = [i for i, token_id in enumerate(encoded.ids) if token_id == mask_token_id]
        
        if not mask_positions:
            print("  No mask found in tokens\n")
            continue
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs['logits']
        
        for pos in mask_positions:
            pos_logits = logits[0, pos, :]
            top_k = torch.topk(pos_logits, 10)
            
            tokens = []
            for idx in top_k.indices:
                token = tokenizer.id_to_token(idx.item())
                # Фильтруем мусор
                if token not in ['.', ',', ':', ';', '!', '?', '—', '-', '«', '»', '(', ')', 
                                 '[', ']', '"', "'", '[CLS]', '[SEP]', '[PAD]', '[UNK]', '``', "''"]:
                    tokens.append(token)
            
            print(f"  Masked word: '{original_word}' → Predictions: {tokens[:8]}")
        print()

if __name__ == "__main__":
    main()