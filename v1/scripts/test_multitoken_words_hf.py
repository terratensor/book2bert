#!/usr/bin/env python3
"""
Тестирование предсказания слов, состоящих из нескольких токенов.
Для моделей Hugging Face (rubert-tiny, distilrubert-tiny и др.)

Аналогичен scripts/test_multitoken_words.py, но использует transformers API.
"""

import torch
import sys
import random
import re
import json
from pathlib import Path

from transformers import AutoModelForMaskedLM, AutoTokenizer


class HFMultitokenTester:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.model_name = model_name
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        print(f"  Vocabulary size: {len(self.tokenizer)}")
        print(f"  Model parameters: {self.model.num_parameters():,}")
    
    def get_word_tokens(self, word):
        """Получить токены для одного слова (без [CLS]/[SEP])"""
        encoded = self.tokenizer(word, add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        # Убираем [CLS] и [SEP]
        if tokens and tokens[0] == '[CLS]':
            tokens = tokens[1:]
        if tokens and tokens[-1] == '[SEP]':
            tokens = tokens[:-1]
        return tokens
    
    def is_multitoken_word(self, word):
        """Проверяет, разбивается ли слово на несколько токенов"""
        tokens = self.get_word_tokens(word)
        return len(tokens) >= 2, tokens
    
    def find_word_in_text(self, text, word):
        """Находит слово в тексте и возвращает позиции его токенов"""
        # Токенизируем текст
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Токенизируем слово
        word_tokens = self.get_word_tokens(word)
        
        if not word_tokens:
            return [], []
        
        # Ищем последовательность word_tokens в tokens
        positions = []
        for i in range(len(tokens) - len(word_tokens) + 1):
            if tokens[i:i+len(word_tokens)] == word_tokens:
                positions = list(range(i, i+len(word_tokens)))
                break
        
        return positions, tokens
    
    def test_single_word(self, text, word, top_k=10):
        """Тестирует предсказание одного слова"""
        is_multitoken, word_tokens = self.is_multitoken_word(word)
        positions, full_tokens = self.find_word_in_text(text, word)
        
        if not positions:
            return {
                'success': False,
                'error': f'Word "{word}" not found in text',
                'word': word,
                'word_tokens': word_tokens if is_multitoken else [word]
            }
        
        # Создаем текст с масками
        masked_tokens = full_tokens.copy()
        for pos in positions:
            masked_tokens[pos] = self.tokenizer.mask_token
        
        # Восстанавливаем текст из токенов
        masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
        
        # Получаем предсказания
        inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Предсказания для каждой позиции
        predicted_tokens = []
        token_predictions = []
        
        for i, pos in enumerate(mask_positions):
            pos_logits = logits[0, pos, :]
            top_k_values, top_k_indices = torch.topk(pos_logits, top_k)
            
            predicted_id = top_k_indices[0].item()
            predicted_token = self.tokenizer.decode([predicted_id])
            predicted_tokens.append(predicted_token)
            
            token_predictions.append({
                'position': i,
                'original_token': full_tokens[positions[i]],
                'predicted_token': predicted_token,
                'top_k': [
                    {'token': self.tokenizer.decode([idx.item()]), 'score': val.item()}
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
            'token_predictions': token_predictions
        }
    
    def load_sentences_from_corpus(self, sentences_dir, max_files=5, max_lines_per_file=200):
        """Загружает предложения из корпуса militera"""
        texts = []
        sentences_path = Path(sentences_dir)
        
        if not sentences_path.exists():
            print(f"Corpus not found: {sentences_dir}")
            return []
        
        files = list(sentences_path.glob("*.jsonl"))[:max_files]
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_lines_per_file:
                        break
                    if line.strip():
                        data = json.loads(line)
                        texts.append(data["text"])
        
        return texts


def print_detailed_result(result):
    """Красиво выводит результат теста"""
    print(f"\n{'='*60}")
    print(f"Word: '{result['word']}'")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
        return
    
    if result['is_multitoken']:
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


def demo_mode(tester, device):
    """Демонстрационный режим с предопределенными примерами"""
    print("="*60)
    print(f"DEMO MODE: Testing predefined words on {tester.model_name}")
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
        result = tester.test_single_word(tc['text'], tc['word'])
        print_detailed_result(result)


def full_mode(tester, device, num_samples=10):
    """Полный режим: поиск и тестирование случайных многотокенных слов из корпуса"""
    print("="*60)
    print(f"FULL MODE: Testing random multitoken words on {tester.model_name}")
    print("="*60)
    
    # Загружаем тексты из корпуса
    texts = tester.load_sentences_from_corpus("data/processed/sentences_militera_v3")
    
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
        words = re.findall(r'[А-Яа-яЁё]+', text)
        for word in words:
            is_multi, tokens = tester.is_multitoken_word(word)
            if is_multi and len(word) > 5:
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
        print("No multitoken words found")
        return
    
    # Случайная выборка
    sample = random.sample(list(unique_words.values()), min(num_samples, len(unique_words)))
    
    results = []
    for item in sample:
        print(f"\n--- Testing: '{item['word']}' ---")
        print(f"Context: {item['text'][:80]}...")
        result = tester.test_single_word(item['text'], item['word'])
        if 'error' in result:
            print(f"  Skipped: {result['error']}")
            continue
        results.append(result)
        print(f"  Original tokens: {result['word_tokens']}")
        print(f"  Predicted tokens: {result['predicted_tokens']}")
        print(f"  Predicted word: '{result['predicted_word']}'")
        print(f"  Success: {'✅' if result['success'] else '❌'}")
    
    if results:
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print(f"SUMMARY: {successful}/{len(results)} successful ({successful/len(results)*100:.1f}%)")
        print(f"{'='*60}")


def interactive_mode(tester, device):
    """Интерактивный режим"""
    print("="*60)
    print(f"INTERACTIVE MODE on {tester.model_name}")
    print("="*60)
    print("Enter a sentence and a word to mask.\n")
    
    while True:
        print("\n" + "-"*40)
        text = input("Sentence: ").strip()
        if text.lower() == 'exit':
            break
        if not text:
            continue
        
        word = input("Word to mask: ").strip()
        if not word:
            continue
        
        if word not in text:
            print(f"Word '{word}' not found in sentence!")
            continue
        
        result = tester.test_single_word(text, word)
        print_detailed_result(result)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test multitoken words with HF models')
    parser.add_argument('--model', type=str, default="cointegrated/rubert-tiny",
                        help='Hugging Face model name')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'full', 'interactive'])
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tester = HFMultitokenTester(args.model, device)
    
    if args.mode == 'demo':
        demo_mode(tester, device)
    elif args.mode == 'full':
        full_mode(tester, device, args.num_samples)
    elif args.mode == 'interactive':
        interactive_mode(tester, device)


if __name__ == "__main__":
    main()