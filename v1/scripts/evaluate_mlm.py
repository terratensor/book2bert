#!/usr/bin/env python3
"""
Оценка MLM предсказаний для обученной модели.
Автоматически определяет параметры модели из конфига.
"""

import torch
import json
import sys
from pathlib import Path

# Добавляем путь к training
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT, BERTForMLM
from tokenizers import BertWordPieceTokenizer


def find_latest_model(models_dir="data/models"):
    """Находит последнюю модель (с наибольшей датой в имени)"""
    models_dir = Path(models_dir)
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("tiny_bert_")]
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found in {models_dir}")
    
    # Сортируем по имени (которое содержит timestamp) и берем последний
    latest = sorted(model_dirs)[-1]
    return latest


def load_model(model_dir, device="cuda"):
    """Загружает модель из директории с чекпоинтом и конфигом"""
    
    # Загружаем конфиг
    config_path = Path(model_dir) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    vocab_size = model_config['vocab_size']
    hidden_size = model_config['hidden_size']
    num_layers = model_config['num_layers']
    num_heads = model_config['num_heads']
    intermediate_size = model_config.get('intermediate_size', hidden_size * 4)
    
    print(f"Loading model with config:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")
    print(f"  intermediate_size: {intermediate_size}")
    
    # Загружаем токенизатор
    tokenizer_path = "data/processed/tokenizer"
    tokenizer = BertWordPieceTokenizer(
        str(Path(tokenizer_path) / "vocab.txt"),
        lowercase=False
    )
    
    # Создаем модель
    bert = BERT(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position=512,
        dropout=0.1
    )
    model = BERTForMLM(bert, vocab_size=vocab_size)
    
    # Загружаем лучшую модель
    best_model_path = Path(model_dir) / "best_model.pt"
    if not best_model_path.exists():
        # Если нет best_model, берем последний чекпоинт
        checkpoints = list(Path(model_dir).glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            best_model_path = sorted(checkpoints)[-1]
            print(f"best_model.pt not found, using {best_model_path.name}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer, config


def predict_masked(model, tokenizer, text, mask_token="[MASK]", device="cuda", top_k=20):
    """Предсказывает маскированные токены"""
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    
    mask_id = tokenizer.token_to_id(mask_token)
    mask_positions = [i for i, token_id in enumerate(encoded.ids) if token_id == mask_id]
    
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
            tokens.append(token)
        predictions.append(tokens)
    
    return predictions, encoded.tokens


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None, 
                        help='Path to model directory (e.g., data/models/tiny_bert_20260402_130630)')
    parser.add_argument('--top_k', type=int, default=20, help='Number of top predictions to show')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Определяем директорию модели
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = find_latest_model()
    
    print(f"\nLoading model from: {model_dir}")
    
    # Загружаем модель
    model, tokenizer, config = load_model(model_dir, device)
    print("\n✓ Model loaded successfully")
    
    # Тестовые примеры
    test_texts = [
        "Философия есть [MASK] о всеобщем.",
        "Мера — это [MASK], в которых объект сохраняет устойчивость.",
        "Триединство: [MASK], информация и мера.",
        "Человек, который среди земных невзгод обходится без [MASK], подобен тому, кто шагает с непокрытой головой под проливным дождем."
    ]
    
    print("\n" + "="*60)
    print("MLM Predictions")
    print("="*60 + "\n")
    
    for text in test_texts:
        print(f"📝 Input: {text}")
        predictions, tokens = predict_masked(model, tokenizer, text, device=device, top_k=args.top_k)
        
        # Показываем токенизацию для контекста
        print(f"   Tokens: {tokens}")
        
        for i, preds in enumerate(predictions):
            # Фильтруем знаки препинания для чистоты вывода
            filtered = [p for p in preds if p not in ['.', ',', ':', ';', '!', '?', '—', '-', '«', '»', '(', ')', '[', ']', '"', "'"]]
            print(f"   🎯 Mask {i+1}: {filtered[:10]} (top-10 meaningful)")
        print()


if __name__ == "__main__":
    main()