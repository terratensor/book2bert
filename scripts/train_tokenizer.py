#!/usr/bin/env python3
"""
Обучение WordPiece токенизатора для BERT.
Использует библиотеку tokenizers от Hugging Face.
"""

import os
import json
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
from glob import glob

def collect_corpus(data_dir, output_file):
    """
    Собирает все предложения из JSONL файлов в один текстовый файл.
    """
    print(f"Collecting corpus from {data_dir}...")
    
    files = glob(os.path.join(data_dir, "*.jsonl"))
    total_sentences = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if line.strip():
                        data = json.loads(line)
                        text = data['text']
                        # Записываем каждое предложение на новой строке
                        out_f.write(text + '\n')
                        total_sentences += 1
                        
                        # Прогресс
                        if total_sentences % 10000 == 0:
                            print(f"  Processed {total_sentences} sentences...")
    
    print(f"Corpus collected: {total_sentences} sentences")
    return total_sentences

def train_tokenizer(corpus_file, output_dir, vocab_size=30000):
    """
    Обучает WordPiece токенизатор.
    """
    print(f"\nTraining tokenizer with vocab_size={vocab_size}...")
    
    # Создаем токенизатор
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,  # Сохраняем регистр (важно для русского)
        wordpieces_prefix="##"
    )
    
    # Обучаем
    tokenizer.train(
        files=[corpus_file],
        vocab_size=vocab_size,
        min_frequency=5,
        limit_alphabet=1000,
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]"
        ]
    )
    
    # Сохраняем
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    
    # Сохраняем конфигурацию
    config = {
        "vocab_size": vocab_size,
        "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        "model_type": "bert-wordpiece"
    }
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Tokenizer saved to {output_dir}")
    
    # Выводим статистику
    vocab = tokenizer.get_vocab()
    print(f"\nVocabulary size: {len(vocab)}")
    
    # Примеры токенизации
    print("\n=== Tokenization examples ===")
    test_texts = [
        "Философия есть учение о всеобщем.",
        "Мера — это границы, в которых объект сохраняет устойчивость.",
        "Триединство: материя, информация и мера."
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens
        print(f"\nText: {text}")
        print(f"Tokens ({len(tokens)}): {tokens[:20]}...")
        print(f"Token IDs: {encoded.ids[:20]}...")
    
    return tokenizer

def main():
    # Пути
    sentences_dir = "data/processed/sentences"
    corpus_file = "data/processed/corpus.txt"
    tokenizer_dir = "data/processed/tokenizer"
    
    # 1. Собираем корпус
    total = collect_corpus(sentences_dir, corpus_file)
    
    # 2. Обучаем токенизатор
    tokenizer = train_tokenizer(corpus_file, tokenizer_dir, vocab_size=30000)
    
    # 3. Сохраняем статистику
    stats = {
        "total_sentences": total,
        "vocab_size": len(tokenizer.get_vocab()),
        "corpus_file": corpus_file
    }
    with open(os.path.join(tokenizer_dir, "stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n=== Done ===")
    print(f"Tokenizer ready at {tokenizer_dir}")
    print(f"Corpus file: {corpus_file}")

if __name__ == "__main__":
    main()