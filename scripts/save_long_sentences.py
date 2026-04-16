#!/usr/bin/env python3
"""
Сохранение аномально длинных предложений (>1000 символов) для анализа
"""

import json
from pathlib import Path
from tqdm import tqdm
import argparse

def save_long_sentences(sentences_dir, output_file, min_length=1000, max_files=None):
    """Сохраняет длинные предложения в JSONL файл"""
    
    files = list(Path(sentences_dir).glob("*.jsonl"))
    if max_files:
        files = files[:max_files]
    
    print(f"Scanning {len(files)} files...")
    
    long_sentences = []
    total_sentences = 0
    
    for filepath in tqdm(files, desc="Processing"):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                total_sentences += 1
                data = json.loads(line)
                text = data.get('text', '')
                if len(text) >= min_length:
                    long_sentences.append(data)
    
    print(f"\nTotal sentences: {total_sentences:,}")
    print(f"Long sentences (>={min_length}): {len(long_sentences):,} ({len(long_sentences)/total_sentences*100:.4f}%)")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in long_sentences:
            f.write(json.dumps(sent, ensure_ascii=False) + '\n')
    
    print(f"Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences-dir', type=str, 
                        default='/mnt/archive/book2bert/data/processed/sentences_full',
                        help='директория с JSONL файлами')
    parser.add_argument('--output', type=str, 
                        default='data/analysis/long_sentences.jsonl',
                        help='выходной JSONL файл')
    parser.add_argument('--min-length', type=int, default=1000,
                        help='минимальная длина для сохранения')
    parser.add_argument('--max-files', type=int, default=None,
                        help='ограничить количество файлов (для теста)')
    args = parser.parse_args()
    
    save_long_sentences(args.sentences_dir, args.output, args.min_length, args.max_files)

if __name__ == "__main__":
    main()