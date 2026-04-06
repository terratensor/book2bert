#!/usr/bin/env python3
"""
Тестирование MLM на токенах, которые являются частями многотокенных слов.
Берутся реальные тексты из корпуса, находятся слова, разбитые на несколько токенов,
и маскируется один из этих токенов (любой). Модель должна предсказать его,
имея контекст всего предложения, включая соседние части того же слова.
"""

import torch
import sys
import random
import re
import json
from pathlib import Path

from transformers import AutoModelForMaskedLM, AutoTokenizer


class MultitokenMLMTester:
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
        
        for i in range(len(tokens) - len(word_tokens) + 1):
            if tokens[i:i+len(word_tokens)] == word_tokens:
                return list(range(i, i+len(word_tokens))), tokens
        
        return [], []
    
    def test_masked_token(self, text, word, token_index, top_k=10):
        """
        Маскирует указанный токен в слове и предсказывает его.
        token_index: 0 = первый токен, 1 = второй, -1 = последний
        """
        positions, full_tokens = self.find_word_in_text(text, word)
        
        if not positions:
            return {
                'success': False,
                'error': f'Word "{word}" not found',
                'word': word
            }
        
        if token_index < 0:
            pos_idx = positions[-1]
            pos_name = "last"
        else:
            pos_idx = positions[min(token_index, len(positions)-1)]
            pos_name = f"pos_{token_index}"
        
        original_token = full_tokens[pos_idx]
        
        # Создаем маскированный текст
        masked_tokens = full_tokens.copy()
        masked_tokens[pos_idx] = self.tokenizer.mask_token
        masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
        
        # Предсказание
        inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (inputs['input_ids'][0] == mask_token_id).nonzero(as_tuple=True)[0]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        pos_logits = logits[0, mask_positions[0], :]
        top_k_values, top_k_indices = torch.topk(pos_logits, top_k)
        
        predictions = [self.tokenizer.decode([idx.item()]) for idx in top_k_indices]
        predicted_token = predictions[0]
        
        return {
            'word': word,
            'word_tokens': [full_tokens[p] for p in positions],
            'masked_token': original_token,
            'mask_position': pos_name,
            'predicted_token': predicted_token,
            'predictions': predictions,
            'is_correct': predicted_token == original_token
        }


def load_sentences_with_multitoken_words(sentences_dir, max_sentences=50):
    """Загружает предложения из корпуса, содержащие многотокенные слова"""
    from transformers import AutoTokenizer
    temp_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    
    sentences_with_words = []
    sentences_path = Path(sentences_dir)
    
    if not sentences_path.exists():
        print(f"Corpus not found: {sentences_dir}")
        return []
    
    files = list(sentences_path.glob("*.jsonl"))[:5]
    
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data["text"]
                if len(text) < 50 or len(text) > 500:
                    continue
                
                words = re.findall(r'[А-Яа-яЁё]{4,}', text)
                multitoken_words = []
                for word in words:
                    tokens = temp_tokenizer.tokenize(word)
                    if len(tokens) >= 2:
                        multitoken_words.append((word, tokens))
                
                if multitoken_words:
                    sentences_with_words.append({
                        'text': text,
                        'words': multitoken_words,
                        'book_id': data.get('book_id', 'unknown')
                    })
                    if len(sentences_with_words) >= max_sentences:
                        return sentences_with_words
    
    return sentences_with_words


def print_result(result, context_text=None):
    """Выводит результат теста"""
    status = "✅" if result['is_correct'] else "❌"
    print(f"\n  {status} Word: '{result['word']}'")
    print(f"     Word tokens: {result['word_tokens']}")
    print(f"     Masked: '{result['masked_token']}' ({result['mask_position']})")
    print(f"     Predicted: '{result['predicted_token']}'")
    print(f"     Top-5: {result['predictions'][:5]}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--corpus_dir', type=str, 
                        default="data/processed/sentences_militera_v3")
    parser.add_argument('--num_examples', type=int, default=15)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"\nLoading sentences with multitoken words from corpus...")
    sentences = load_sentences_with_multitoken_words(args.corpus_dir, args.num_examples)
    print(f"Found {len(sentences)} sentences with multitoken words")
    
    if not sentences:
        print("No sentences found, exiting")
        return
    
    tester = MultitokenMLMTester(args.model, device)
    
    print("\n" + "="*80)
    print(f"MLM ON MULTITOKEN WORDS (masking one token within a word)")
    print(f"Model: {args.model}")
    print("="*80)
    
    results = []
    
    for sent_data in sentences:
        text = sent_data['text']
        words = sent_data['words']
        
        print(f"\n--- Sentence: {text[:100]}...")
        
        for word, tokens in words[:2]:  # максимум 2 слова на предложение
            token_count = len(tokens)
            
            # Тестируем первый токен
            result = tester.test_masked_token(text, word, 0)
            if 'error' not in result:
                print_result(result)
                results.append(result)
            
            # Тестируем последний токен
            if token_count >= 2:
                result = tester.test_masked_token(text, word, -1)
                if 'error' not in result:
                    print_result(result)
                    results.append(result)
            
            # Тестируем средний (если есть)
            if token_count >= 3:
                result = tester.test_masked_token(text, word, token_count // 2)
                if 'error' not in result:
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


if __name__ == "__main__":
    main()