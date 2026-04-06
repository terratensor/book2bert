#!/usr/bin/env python3
"""
Тестирование предсказания отдельных токенов внутри многотокенных слов.
Маскируется один токен (начало, середина или конец слова), а не всё слово целиком.
"""

import torch
import sys
import random
import re
import json
from pathlib import Path

from transformers import AutoModelForMaskedLM, AutoTokenizer


class HFSingleTokenTester:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.model_name = model_name
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        print(f"  Vocabulary size: {len(self.tokenizer)}")
    
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
    
    def find_word_in_text(self, text, word):
        """Находит слово в тексте и возвращает позиции его токенов"""
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
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
    
    def test_masked_position(self, text, word, mask_pos, top_k=10):
        """
        Тестирует предсказание для одной позиции в слове.
        mask_pos: 0 = первый токен, 1 = второй, -1 = последний
        """
        positions, full_tokens = self.find_word_in_text(text, word)
        
        if not positions:
            return {
                'success': False,
                'error': f'Word "{word}" not found in text',
                'word': word
            }
        
        # Определяем индекс токена для маскирования
        if mask_pos < 0:
            token_idx = positions[-1]
            pos_name = "last"
        else:
            token_idx = positions[min(mask_pos, len(positions)-1)]
            pos_name = f"position_{mask_pos}"
        
        original_token = full_tokens[token_idx]
        
        # Создаем текст с маской
        masked_tokens = full_tokens.copy()
        masked_tokens[token_idx] = self.tokenizer.mask_token
        masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
        
        # Получаем предсказания
        inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Предсказания для маскированной позиции
        pos_logits = logits[0, mask_positions[0], :]
        top_k_values, top_k_indices = torch.topk(pos_logits, top_k)
        
        predictions = []
        for idx in top_k_indices:
            token = self.tokenizer.decode([idx.item()])
            predictions.append(token)
        
        predicted_token = predictions[0]
        is_correct = predicted_token == original_token
        
        return {
            'word': word,
            'masked_token': original_token,
            'mask_position': pos_name,
            'predicted_token': predicted_token,
            'predictions': predictions,
            'is_correct': is_correct
        }


def load_test_words_from_corpus(sentences_dir, min_token_count=2, num_words=20):
    """Загружает многотокенные слова из корпуса"""
    # Используем наш токенизатор для поиска (через HF)
    from transformers import AutoTokenizer
    temp_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    
    texts = []
    sentences_path = Path(sentences_dir)
    
    if sentences_path.exists():
        files = list(sentences_path.glob("*.jsonl"))[:3]
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 300:
                        break
                    if line.strip():
                        data = json.loads(line)
                        texts.append(data["text"])
    
    words = []
    for text in texts:
        found_words = re.findall(r'[А-Яа-яЁё]{6,}', text)
        for word in found_words:
            tokens = temp_tokenizer.tokenize(word)
            if len(tokens) >= min_token_count and word not in words:
                words.append(word)
                if len(words) >= num_words:
                    break
        if len(words) >= num_words:
            break
    
    return words[:num_words]


def print_result(result):
    """Выводит результат теста"""
    status = "✅" if result['is_correct'] else "❌"
    print(f"\n  {status} Word: '{result['word']}'")
    print(f"     Masked token: '{result['masked_token']}' ({result['mask_position']})")
    print(f"     Predicted: '{result['predicted_token']}'")
    print(f"     Top-5: {result['predictions'][:5]}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Hugging Face model name')
    parser.add_argument('--corpus_dir', type=str, 
                        default="data/processed/sentences_militera_v3",
                        help='Path to sentences corpus')
    parser.add_argument('--num_words', type=int, default=10)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Загружаем тестовые слова из корпуса
    print(f"\nLoading test words from corpus...")
    test_words = load_test_words_from_corpus(args.corpus_dir, min_token_count=2, num_words=args.num_words)
    print(f"Found {len(test_words)} multitoken words: {test_words}")
    
    # Загружаем модель
    tester = HFSingleTokenTester(args.model, device)
    
    print("\n" + "="*80)
    print(f"TESTING SINGLE TOKEN MASKING WITHIN WORDS")
    print(f"Model: {args.model}")
    print("="*80)
    
    results = []
    
    for word in test_words:
        word_tokens = tester.get_word_tokens(word)
        token_count = len(word_tokens)
        
        print(f"\n--- Word: '{word}' (tokens: {word_tokens}) ---")
        
        # Тестируем первый токен
        if token_count >= 1:
            result = tester.test_masked_position("Это слово " + word + " встречается в тексте.", word, 0)
            print_result(result)
            results.append(result)
        
        # Тестируем последний токен
        if token_count >= 2:
            result = tester.test_masked_position("Это слово " + word + " встречается в тексте.", word, -1)
            print_result(result)
            results.append(result)
        
        # Тестируем средний токен (если есть)
        if token_count >= 3:
            result = tester.test_masked_position("Это слово " + word + " встречается в тексте.", word, token_count // 2)
            print_result(result)
            results.append(result)
    
    # Статистика
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total tests: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {correct/total*100:.1f}%")
    
    # Анализ по позициям
    first_correct = sum(1 for r in results if r['mask_position'] == 'position_0' and r['is_correct'])
    last_correct = sum(1 for r in results if r['mask_position'] == 'last' and r['is_correct'])
    first_total = sum(1 for r in results if r['mask_position'] == 'position_0')
    last_total = sum(1 for r in results if r['mask_position'] == 'last')
    
    if first_total > 0:
        print(f"\nFirst token accuracy: {first_correct/first_total*100:.1f}% ({first_correct}/{first_total})")
    if last_total > 0:
        print(f"Last token accuracy: {last_correct/last_total*100:.1f}% ({last_correct}/{last_total})")


if __name__ == "__main__":
    main()