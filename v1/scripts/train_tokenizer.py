#!/usr/bin/env python3
"""
Обучение токенизатора в streaming режиме.
Не загружает весь корпус в память.
"""

import os
import sys
import json
import argparse
from tokenizers import BertWordPieceTokenizer
from glob import glob
from tqdm import tqdm
from pathlib import Path


# Добавляем путь к scripts для импорта filter_utils
sys.path.insert(0, str(Path(__file__).parent))
from v1.scripts.filter_utils import filter_cjk_thai

def stream_sentences(sentences_dir):
    files = glob(os.path.join(sentences_dir, "*.jsonl"))
    for filepath in tqdm(files, desc="Processing files"):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    text = data['text']
                    text = filter_cjk_thai(text)  # ← фильтруем
                    if text.strip():
                        yield text + '\n'

def train_tokenizer_streaming(sentences_dir, output_dir, vocab_size=30000, min_frequency=2):
    """
    Обучает токенизатор в streaming режиме.
    Сначала записывает корпус во временный файл, затем обучает.
    """
    import tempfile
    
    print(f"Streaming sentences from {sentences_dir} to temporary file...")
    
    # Создаем временный файл
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp_file:
        tmp_path = tmp_file.name
        total_sentences = 0
        
        for sentence in stream_sentences(sentences_dir):
            tmp_file.write(sentence)
            total_sentences += 1
            if total_sentences % 100000 == 0:
                print(f"  Written {total_sentences} sentences...")
    
    print(f"Corpus collected: {total_sentences} sentences, temp file: {tmp_path}")
    
    # Обучаем токенизатор
    print(f"\nTraining tokenizer with vocab_size={vocab_size}...")
    
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
        wordpieces_prefix="##"
    )
    
    tokenizer.train(
        files=[tmp_path],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        limit_alphabet=1000,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    
    # Сохраняем конфиг
    config = {
        "vocab_size": vocab_size,
        "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        "model_type": "bert-wordpiece",
        "min_frequency": min_frequency
    }
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Удаляем временный файл
    os.unlink(tmp_path)
    
    print(f"Tokenizer saved to {output_dir}")
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
    
    return tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--vocab-size', type=int, default=50000)
    parser.add_argument('--min-frequency', type=int, default=2)
    args = parser.parse_args()
    
    tokenizer = train_tokenizer_streaming(
        args.sentences_dir,
        args.output_dir,
        args.vocab_size,
        args.min_frequency
    )
    
    print("\n=== Done ===")

if __name__ == "__main__":
    main()