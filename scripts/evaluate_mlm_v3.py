#!/usr/bin/env python3
"""
Оценка MLM предсказаний для v3 модели.
Поддерживает как философские, так и военные примеры.
"""

import torch
import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT, BERTForMLM
from tokenizers import BertWordPieceTokenizer

def load_model(model_dir, device="cuda"):
    """Загружает модель из директории"""
    tokenizer = BertWordPieceTokenizer(
        str(Path(model_dir).parent.parent / "processed" / "tokenizer_militera_v3" / "vocab.txt"),
        lowercase=False
    )
    
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
    
    checkpoint_path = Path(model_dir) / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer

def predict_masked(model, tokenizer, text, device="cuda", top_k=10):
    """Предсказывает маскированные токены"""
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    
    mask_id = tokenizer.token_to_id("[MASK]")
    mask_positions = [i for i, tid in enumerate(encoded.ids) if tid == mask_id]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs['logits']
    
    predictions = []
    for pos in mask_positions:
        pos_logits = logits[0, pos, :]
        top_k_values, top_k_indices = torch.topk(pos_logits, top_k)
        
        tokens = []
        for idx in top_k_indices:
            token = tokenizer.id_to_token(idx.item())
            # Фильтруем знаки препинания
            if token not in ['.', ',', ':', ';', '!', '?', '—', '-', '«', '»', '(', ')', '[', ']', '"', "'", '[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                tokens.append(token)
        
        predictions.append(tokens)
    
    return predictions, encoded.tokens

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default='both', choices=['philosophy', 'military', 'both'])
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_dir}")
    
    model, tokenizer = load_model(args.model_dir, device)
    print("Model loaded successfully\n")
    
    # Философские примеры
    philosophy_texts = [
        "Философия есть [MASK] о всеобщем.",
        "Мера — это [MASK], в которых объект сохраняет устойчивость.",
        "Триединство: [MASK], информация и мера.",
        "Человек, который среди земных невзгод обходится без [MASK], подобен тому, кто шагает с непокрытой головой под проливным дождем."
    ]
    
    # Военные примеры (из реального корпуса)
    military_texts = [
        "Генерал [MASK] командовал дивизией в трудных условиях.",
        "Солдаты [MASK] Сталинград, тем не менее не могли простить ему происшедшего.",
        "Север Испании был [MASK] занят фашистами.",
        "Таким был Рубен Розенфельд, простой [MASK], один из тысяч героев."
    ]
    
    if args.mode in ['philosophy', 'both']:
        print("="*60)
        print("PHILOSOPHICAL TEXTS")
        print("="*60 + "\n")
        
        for text in philosophy_texts:
            print(f"Input: {text}")
            predictions, tokens = predict_masked(model, tokenizer, text, device)
            for i, preds in enumerate(predictions):
                print(f"  Mask {i+1}: {preds[:10]}")
            print()
    
    if args.mode in ['military', 'both']:
        print("="*60)
        print("MILITARY TEXTS")
        print("="*60 + "\n")
        
        for text in military_texts:
            print(f"Input: {text}")
            predictions, tokens = predict_masked(model, tokenizer, text, device)
            for i, preds in enumerate(predictions):
                print(f"  Mask {i+1}: {preds[:10]}")
            print()

if __name__ == "__main__":
    main()
