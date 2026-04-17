#!/usr/bin/env python3
"""Сбор статистики по датасету предложений."""

import json
import os
from collections import Counter

def analyze_sentences(data_dir):
    """Анализирует все JSONL файлы в директории."""
    
    stats = {
        'total_sentences': 0,
        'total_chars': 0,
        'books': set(),
        'sentences_per_book': Counter(),
        'char_lengths': []
    }
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
    print(f"Found {len(files)} files")
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        book_id = filename.replace('.jsonl', '')
        sentence_count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sentence_count += 1
                    stats['total_sentences'] += 1
                    char_len = len(data['text'])
                    stats['total_chars'] += char_len
                    stats['char_lengths'].append(char_len)
                    
        stats['books'].add(book_id)
        stats['sentences_per_book'][book_id] = sentence_count
    
    return stats

def main():
    data_dir = 'data/processed/sentences'
    stats = analyze_sentences(data_dir)
    
    print("\n=== DATASET STATISTICS ===")
    print(f"Total books: {len(stats['books'])}")
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Average characters per sentence: {stats['total_chars'] / stats['total_sentences']:.1f}")
    
    # Распределение длин предложений
    lengths = stats['char_lengths']
    lengths.sort()
    
    print("\n=== Sentence length distribution (characters) ===")
    print(f"Min: {min(lengths)}")
    print(f"25th percentile: {lengths[int(len(lengths)*0.25)]}")
    print(f"Median: {lengths[len(lengths)//2]}")
    print(f"75th percentile: {lengths[int(len(lengths)*0.75)]}")
    print(f"Max: {max(lengths)}")
    
    # Количество предложений по книгам
    print("\n=== Sentences per book ===")
    counts = list(stats['sentences_per_book'].values())
    counts.sort()
    print(f"Min: {min(counts)}")
    print(f"Max: {max(counts)}")
    print(f"Average: {sum(counts) / len(counts):.1f}")

if __name__ == '__main__':
    main()
