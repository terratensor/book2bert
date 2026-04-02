#!/usr/bin/env python3
"""
Обучение WordPiece токенизатора для BERT.
Поддерживает переменный размер словаря.
"""

import os
import json
import argparse
from tokenizers import BertWordPieceTokenizer
from glob import glob
from tqdm import tqdm

def collect_corpus(data_dir, output_file):
    """Собирает все предложения из JSONL в один текстовый файл"""
    print(f"Collecting corpus from {data_dir}...")
    
    files = glob(os.path.join(data_dir, "*.jsonl"))
    total_sentences = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filepath in tqdm(files, desc="Processing files"):
            with open(filepath, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if line.strip():
                        data = json.loads(line)
                        text = data['text']
                        out_f.write(text + '\n')
                        total_sentences += 1
    
    print(f"Corpus collected: {total_sentences} sentences")
    return total_sentences

def train_tokenizer(corpus_file, output_dir, vocab_size=30000):
    """Обучает WordPiece токенизатор с заданным vocab_size"""
    print(f"\nTraining tokenizer with vocab_size={vocab_size}...")
    
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
        wordpieces_prefix="##"
    )
    
    tokenizer.train(
        files=[corpus_file],
        vocab_size=vocab_size,  # ← теперь используем аргумент
        min_frequency=2,        # ← снижаем порог, чтобы получить больше слов
        limit_alphabet=1000,
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]"
        ]
    )
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    
    # Сохраняем конфигурацию
    config = {
        "vocab_size": vocab_size,
        "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        "model_type": "bert-wordpiece",
        "min_frequency": 2
    }
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Tokenizer saved to {output_dir}")
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
    
    return tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences-dir', type=str, default='data/processed/sentences')
    parser.add_argument('--output-dir', type=str, default='data/processed/tokenizer')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--min-frequency', type=int, default=2, help='Minimum frequency for token')
    args = parser.parse_args()
    
    # 1. Собираем корпус
    corpus_file = "data/processed/corpus.txt"
    total = collect_corpus(args.sentences_dir, corpus_file)
    
    # 2. Обучаем токенизатор
    tokenizer = train_tokenizer(corpus_file, args.output_dir, args.vocab_size)
    
    # 3. Сохраняем статистику
    stats = {
        "total_sentences": total,
        "vocab_size": len(tokenizer.get_vocab()),
        "corpus_file": corpus_file,
        "min_frequency": args.min_frequency
    }
    with open(os.path.join(args.output_dir, "stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 4. Проверяем наличие целых слов
    print("\n=== Проверка словаря ===")
    test_words = ["учение", "граница", "материя", "философия", "методология", "сознание"]
    vocab = tokenizer.get_vocab()
    
    for word in test_words:
        if word in vocab:
            print(f"  ✓ '{word}' найден (id: {vocab[word]})")
        else:
            print(f"  ✗ '{word}' отсутствует")
    
    print("\n=== Done ===")

if __name__ == "__main__":
    main()