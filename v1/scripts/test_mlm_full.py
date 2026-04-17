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


def load_real_examples(sentences_dir, num_examples=10):
    """Загружает реальные предложения из full корпуса"""
    files = list(Path(sentences_dir).glob("*.jsonl"))
    random.shuffle(files)
    examples = []
    
    for filepath in files[:20]:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines[:10]:
                data = json.loads(line)
                text = data['text']
                if 20 < len(text) < 300:
                    examples.append(text)
                if len(examples) >= num_examples:
                    return examples[:num_examples]
    
    return examples[:num_examples]


def create_masked_example(text):
    """Маскирует одно случайное слово"""
    words = text.split()
    if len(words) < 4:
        return text, None
    
    # Находим позиции для маскирования (любые слова, кроме совсем коротких)
    candidates = [i for i, w in enumerate(words) if len(w.strip('.,!?;:()[]«»"\'`')) > 1]
    if not candidates:
        return text, None
    
    pos = random.choice(candidates)
    original_word = words[pos].strip('.,!?;:()[]«»"\'`')
    words[pos] = "[MASK]"
    return ' '.join(words), original_word


def get_top_predictions(logits, tokenizer, top_k=10):
    """Получает топ предсказаний, очищая от пунктуации"""
    top_k_values, top_k_indices = torch.topk(logits, top_k * 3)
    tokens = []
    seen = set()
    
    for idx in top_k_indices:
        token = tokenizer.id_to_token(idx.item())
        
        # Пропускаем специальные токены
        if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
            continue
        
        # Пропускаем пунктуацию
        if token in ['.', ',', ':', ';', '!', '?', '—', '-', '«', '»', '(', ')', '[', ']', '"', "'", '``', "''", '…']:
            continue
        
        # Очищаем от остаточной пунктуации
        clean_token = token.strip('.,!?;:()[]«»"\'`…')
        
        if clean_token and len(clean_token) > 0 and clean_token not in seen:
            seen.add(clean_token)
            tokens.append(clean_token)
            
            if len(tokens) >= top_k:
                break
    
    return tokens


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Пути
    model_path = "data/models/small_bert_10pct/checkpoints/last_checkpoint.pt"
    tokenizer_path = "data/processed/tokenizer_full"
    sentences_dir = "data/processed/sentences_full"
    
    # Загружаем токенизатор
    tokenizer = BertWordPieceTokenizer(
        str(Path(tokenizer_path) / "vocab.txt"),
        lowercase=False
    )
    
    # Создаем модель
    bert = BERT(
        vocab_size=120000,
        hidden_size=512,
        num_layers=12,
        num_heads=8,
        intermediate_size=2048,
        max_position=512,
        dropout=0.1
    )
    model = BERTForMLM(bert, vocab_size=120000)
    
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
    print("MLM Predictions on military-historical texts (after epoch 5)")
    print("="*60 + "\n")
    
    for original in examples:
        masked_text, original_word = create_masked_example(original)
        if masked_text is None or original_word is None:
            continue
        
        print(f"Original: {original[:100]}...")
        print(f"Masked:   {masked_text}")
        
        # Токенизация
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
        
        # Берем первую маску
        pos = mask_positions[0]
        pos_logits = logits[0, pos, :]
        predictions = get_top_predictions(pos_logits, tokenizer, top_k=10)
        
        # Проверяем, угадало ли исходное слово
        original_clean = original_word.lower()
        predicted_clean = [p.lower() for p in predictions]
        
        if original_clean in predicted_clean:
            match_marker = "✓"
        else:
            match_marker = "✗"
        
        print(f"  Masked word: '{original_word}' {match_marker} → Predictions: {predictions[:8]}")
        print()

if __name__ == "__main__":
    main()