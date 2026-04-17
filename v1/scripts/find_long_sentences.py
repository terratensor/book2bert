#!/usr/bin/env python3
"""
Находит самые длинные предложения в корпусе (по токенам).
"""

import json
import argparse
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
import heapq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences-dir', type=str, required=True)
    parser.add_argument('--tokenizer-path', type=str, required=True)
    parser.add_argument('--num-sentences', type=int, default=20)
    parser.add_argument('--min-length', type=int, default=500, help='Минимальная длина в токенах для поиска')
    args = parser.parse_args()
    
    # Загружаем токенизатор
    tokenizer = BertWordPieceTokenizer(
        str(Path(args.tokenizer_path) / "vocab.txt"),
        lowercase=False
    )
    
    print(f"Tokenizer loaded, vocab_size={tokenizer.get_vocab_size()}")
    print(f"Searching for sentences longer than {args.min_length} tokens...")
    
    # Находим все JSONL файлы
    files = list(Path(args.sentences_dir).glob("*.jsonl"))
    print(f"Found {len(files)} files")
    
    long_sentences = []  # heap для топ-N
    
    for filepath in tqdm(files[:100], desc="Scanning files (first 100)"):  # ограничим для скорости
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    text = data["text"]
                    
                    # Токенизируем
                    encoded = tokenizer.encode(text)
                    token_len = len(encoded.ids)
                    
                    if token_len > args.min_length:
                        heapq.heappush(long_sentences, (
                            token_len,
                            data.get("book_id", "unknown"),
                            data.get("position", 0),
                            text[:500] + "..." if len(text) > 500 else text
                        ))
                        
                        # Оставляем только топ args.num_sentences
                        if len(long_sentences) > args.num_sentences:
                            heapq.heappop(long_sentences)
    
    # Сортируем по убыванию
    long_sentences = sorted(long_sentences, reverse=True)
    
    print("\n" + "="*80)
    print(f"TOP {len(long_sentences)} LONGEST SENTENCES (by token count)")
    print("="*80)
    
    for i, (token_len, book_id, pos, text) in enumerate(long_sentences):
        print(f"\n--- #{i+1} ---")
        print(f"Book ID: {book_id}")
        print(f"Position: {pos}")
        print(f"Token length: {token_len}")
        print(f"Character length: {len(text)}")
        print(f"Text preview: {text[:300]}...")
        print("-"*40)
    
    # Сохраняем в файл для анализа
    output_file = "data/processed/long_sentences_analysis.txt"
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"LONG SENTENCES ANALYSIS\n")
        f.write(f"Min token length: {args.min_length}\n")
        f.write(f"Found: {len(long_sentences)} sentences\n")
        f.write("="*80 + "\n\n")
        
        for i, (token_len, book_id, pos, text) in enumerate(long_sentences):
            f.write(f"\n--- #{i+1} ---\n")
            f.write(f"Book ID: {book_id}\n")
            f.write(f"Position: {pos}\n")
            f.write(f"Token length: {token_len}\n")
            f.write(f"Character length: {len(text)}\n")
            f.write(f"Full text:\n{text}\n")
            f.write("-"*40 + "\n")
    
    print(f"\nFull analysis saved to: {output_file}")

if __name__ == "__main__":
    main()