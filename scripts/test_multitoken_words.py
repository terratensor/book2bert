#!/usr/bin/env python3
"""
Тестирование предсказания слов, состоящих из нескольких токенов.

BERT предсказывает токены, а не целые слова. Если слово разбито на несколько токенов
(например, "возвратиться" → ["возврат", "##иться"]), модель должна предсказать
каждый токен отдельно. Этот скрипт проверяет, насколько хорошо модель
восстанавливает многотокенные слова целиком.
"""

import torch
import sys
import random
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT, BERTForMLM
from tokenizers import BertWordPieceTokenizer


def load_model(model_dir, device="cuda"):
    """Загружает модель и токенизатор из директории"""
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


def is_multitoken_word(tokenizer, word):
    """
    Проверяет, разбивается ли слово на несколько токенов.
    Исключает случай, когда слово является одним токеном.
    """
    encoded = tokenizer.encode(word)
    tokens = encoded.tokens
    
    # Убираем [CLS] и [SEP], которые добавляет токенизатор
    # При вызове tokenizer.encode(word) он добавляет [CLS] и [SEP]
    # Нас интересуют только токены самого слова
    if len(tokens) >= 3:  # [CLS] + токены слова + [SEP]
        word_tokens = tokens[1:-1]
        return len(word_tokens) >= 2, word_tokens
    return False, []


def find_word_in_text(text, word, tokenizer):
    """
    Находит слово в тексте и возвращает позиции его токенов.
    """
    encoded = tokenizer.encode(text)
    full_tokens = encoded.tokens
    full_ids = encoded.ids
    
    # Токенизируем слово отдельно и убираем [CLS]/[SEP]
    word_encoded = tokenizer.encode(word)
    word_tokens = word_encoded.tokens[1:-1]  # Убираем [CLS] и [SEP]
    word_ids = word_encoded.ids[1:-1]
    
    if not word_tokens:
        return [], []
    
    # Ищем последовательность word_tokens в full_tokens
    positions = []
    for i in range(len(full_tokens) - len(word_tokens) + 1):
        if full_tokens[i:i+len(word_tokens)] == word_tokens:
            positions = list(range(i, i+len(word_tokens)))
            break
    
    return positions, full_tokens


def test_single_word(model, tokenizer, text, word, device, top_k=10):
    """
    Тестирует предсказание одного слова (может быть одно- или многотокенным).
    """
    # Проверяем, является ли слово многотокенным
    is_multitoken, word_tokens = is_multitoken_word(tokenizer, word)
    
    # Находим позиции слова в тексте
    positions, full_tokens = find_word_in_text(text, word, tokenizer)
    
    if not positions:
        return {
            'success': False,
            'error': f'Word "{word}" not found in text',
            'word': word,
            'word_tokens': word_tokens if is_multitoken else [word]
        }
    
    # Токенизируем текст
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    original_tokens = encoded.tokens
    
    # Создаем маску: закрываем все позиции слова
    masked_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    
    for pos in positions:
        masked_ids[0, pos] = tokenizer.token_to_id("[MASK]")
        labels[0, pos] = input_ids[0, pos]
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=masked_ids)
        logits = outputs['logits']
    
    # Предсказания для каждой позиции
    predicted_tokens = []
    token_predictions = []
    
    for i, pos in enumerate(positions):
        pos_logits = logits[0, pos, :]
        top_k_values, top_k_indices = torch.topk(pos_logits, top_k)
        
        predicted_id = top_k_indices[0].item()
        predicted_token = tokenizer.id_to_token(predicted_id)
        predicted_tokens.append(predicted_token)
        
        token_predictions.append({
            'position': i,
            'original_token': original_tokens[pos],
            'predicted_token': predicted_token,
            'top_k': [
                {'token': tokenizer.id_to_token(idx.item()), 'score': val.item()}
                for idx, val in zip(top_k_indices, top_k_values)
            ]
        })
    
    # Склеиваем предсказанные токены в слово
    predicted_word = ''.join(predicted_tokens).replace('##', '')
    original_word = word
    
    return {
        'success': predicted_word == original_word,
        'is_multitoken': is_multitoken,
        'word': original_word,
        'word_tokens': word_tokens if is_multitoken else [word],
        'predicted_tokens': predicted_tokens,
        'predicted_word': predicted_word,
        'token_predictions': token_predictions,
        'positions': positions,
        'full_tokens': full_tokens
    }


def print_detailed_result(result):
    """Красиво выводит результат теста"""
    print(f"\n{'='*60}")
    print(f"Word: '{result['word']}'")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
        return
    
    is_multitoken = result.get('is_multitoken', False)
    if is_multitoken:
        print(f"Type: MULTITOKEN (разбито на {len(result['word_tokens'])} токенов)")
        print(f"Original tokens: {result['word_tokens']}")
    else:
        print(f"Type: SINGLE TOKEN (целый токен)")
    
    print(f"Predicted tokens: {result['predicted_tokens']}")
    print(f"Predicted word: '{result['predicted_word']}'")
    print(f"Success: {'✅' if result['success'] else '❌'}")
    
    if not result['success'] and result.get('token_predictions'):
        print("\n  Token-by-token analysis:")
        for pred in result['token_predictions']:
            print(f"    Position {pred['position']}:")
            print(f"      Original: '{pred['original_token']}'")
            print(f"      Predicted: '{pred['predicted_token']}'")
            print(f"      Top-5 alternatives:")
            for i, alt in enumerate(pred['top_k'][:5]):
                print(f"        {i+1}. '{alt['token']}' (score: {alt['score']:.4f})")


def demo_mode(model, tokenizer, device):
    """Демонстрационный режим с предопределенными примерами"""
    print("="*60)
    print("DEMO MODE: Testing predefined words")
    print("="*60)
    
    test_cases = [
        {
            'text': 'Генерал Шкуро оседлал коня и поскакал в атаку.',
            'word': 'оседлал'
        },
        {
            'text': 'Солдаты ориентировались по карте в незнакомой местности.',
            'word': 'ориентировались'
        },
        {
            'text': 'Командир возвратился в штаб после разведки.',
            'word': 'возвратился'
        },
        {
            'text': 'Противник предпринял отчаянную контратаку.',
            'word': 'предпринял'
        }
    ]
    
    for tc in test_cases:
        result = test_single_word(model, tokenizer, tc['text'], tc['word'], device)
        print_detailed_result(result)


def full_mode(model, tokenizer, device, num_samples=10):
    """Полный режим: поиск и тестирование случайных многотокенных слов"""
    print("="*60)
    print(f"FULL MODE: Testing random multitoken words")
    print("="*60)
    
    # Загружаем тексты из корпуса
    import json
    texts = []
    sentences_dir = Path("data/processed/sentences_militera_v3")
    
    if sentences_dir.exists():
        files = list(sentences_dir.glob("*.jsonl"))[:5]  # Берем 5 файлов для скорости
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 500:  # Увеличиваем до 500 строк на файл
                        break
                    if line.strip():
                        data = json.loads(line)
                        texts.append(data["text"])
    
    if not texts:
        print("No texts found, using demo texts")
        texts = [
            "Генерал Шкуро оседлал коня и поскакал в атаку.",
            "Солдаты ориентировались по карте в незнакомой местности.",
            "Командир возвратился в штаб после разведки.",
            "Противник предпринял отчаянную контратаку."
        ]
    
    # Собираем многотокенные слова
    multitoken_words = []
    for text in texts:
        # Находим все слова в тексте
        words = re.findall(r'[А-Яа-яЁё]+', text)
        for word in words:
            is_multi, tokens = is_multitoken_word(tokenizer, word)
            if is_multi:
                # Проверяем, что слово действительно есть в тексте как отдельное слово
                if word in text:
                    multitoken_words.append({
                        'word': word,
                        'tokens': tokens,
                        'text': text
                    })
    
    # Убираем дубликаты
    unique_words = {}
    for item in multitoken_words:
        if item['word'] not in unique_words:
            unique_words[item['word']] = item
    
    print(f"Found {len(unique_words)} unique multitoken words")
    
    if not unique_words:
        print("No multitoken words found, testing single token words as fallback")
        # Fallback: тестируем обычные слова
        for text in texts[:5]:
            words = re.findall(r'[А-Яа-яЁё]+', text)
            for word in words[:num_samples]:
                result = test_single_word(model, tokenizer, text, word, device)
                if 'error' not in result:
                    print_detailed_result(result)
        return
    
    # Случайная выборка
    sample = random.sample(list(unique_words.values()), min(num_samples, len(unique_words)))
    
    results = []
    for item in sample:
        print(f"\nTesting word: '{item['word']}' in context: {item['text'][:80]}...")
        result = test_single_word(model, tokenizer, item['text'], item['word'], device)
        if 'error' in result:
            print(f"  Skipped: {result['error']}")
            continue
        results.append(result)
        print_detailed_result(result)
    
    # Статистика
    if results:
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total tested: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Accuracy: {successful/len(results):.2%}")
    else:
        print("\nNo valid tests completed")

def interactive_mode(model, tokenizer, device):
    """Интерактивный режим"""
    print("="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter a sentence with a word to mask.")
    print("The script will mask that word and try to predict it.")
    print("Type 'exit' to quit\n")
    
    while True:
        print("\n" + "-"*40)
        sentence = input("Enter a sentence: ").strip()
        if sentence.lower() == 'exit':
            break
        if not sentence:
            continue
        
        word = input("Enter the word to mask: ").strip()
        if not word:
            continue
        
        if word not in sentence:
            print(f"Word '{word}' not found in sentence!")
            continue
        
        result = test_single_word(model, tokenizer, sentence, word, device)
        print_detailed_result(result)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test word predictions')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default='demo', 
                        choices=['demo', 'full', 'interactive'])
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_dir}")
    
    model, tokenizer = load_model(args.model_dir, device)
    print("Model loaded successfully\n")
    
    if args.mode == 'demo':
        demo_mode(model, tokenizer, device)
    elif args.mode == 'full':
        full_mode(model, tokenizer, device, args.num_samples)
    elif args.mode == 'interactive':
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
