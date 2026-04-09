# scripts/test_vocab_size_real.py
#!/usr/bin/env python3
"""
Реальное тестирование vocab_size: сколько уникальных слов из корпуса есть в словаре
"""

from tokenizers import BertWordPieceTokenizer
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import re
from pathlib import Path

def extract_unique_words(corpus_file, max_lines=500000):
    """Извлекает уникальные слова из корпуса (без токенизации)"""
    words = set()
    word_pattern = re.compile(r'[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)*')
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            # Извлекаем слова (буквенные последовательности)
            found = word_pattern.findall(line.lower())
            words.update(found)
    
    return words

def train_tokenizer(corpus_file, vocab_size):
    """Обучает токенизатор"""
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(
        files=[corpus_file],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    return tokenizer

def get_vocab_words(tokenizer):
    """Возвращает множество слов в словаре (без субвордов ##)"""
    vocab_words = set()
    for i in range(tokenizer.get_vocab_size()):
        token = tokenizer.id_to_token(i)
        if token and not token.startswith('##'):
            vocab_words.add(token.lower())
    return vocab_words

def test_vocab_size(corpus_file, vocab_size, test_words):
    """Тестирует один размер словаря"""
    print(f"  Training vocab_size={vocab_size}...")
    tokenizer = train_tokenizer(corpus_file, vocab_size)
    vocab_words = get_vocab_words(tokenizer)
    
    # Считаем, сколько тестовых слов есть в словаре
    found = sum(1 for w in test_words if w in vocab_words)
    return found / len(test_words) if test_words else 0, len(vocab_words)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help='Файл корпуса для обучения')
    parser.add_argument('--test-corpus', type=str, help='Отдельный корпус для извлечения слов')
    parser.add_argument('--vocab-sizes', type=str, default="10000,20000,30000,50000,70000,100000,120000,150000")
    parser.add_argument('--max-test-words', type=int, default=50000, help='Максимум тестовых слов')
    args = parser.parse_args()
    
    vocab_sizes = [int(x) for x in args.vocab_sizes.split(',')]
    
    # Извлекаем тестовые слова (из отдельного корпуса или из того же, но ограниченно)
    test_corpus = args.test_corpus if args.test_corpus else args.corpus
    print(f"Извлечение уникальных слов из {test_corpus}...")
    all_words = extract_unique_words(test_corpus, max_lines=10000000)
    
    # Берём самые частотные слова (или случайные)
    test_words = list(all_words)[:args.max_test_words]
    print(f"Тестовых слов: {len(test_words):,}")
    
    results = []
    
    for size in vocab_sizes:
        coverage, vocab_size = test_vocab_size(args.corpus, size, test_words)
        results.append({'vocab_size': size, 'coverage': coverage, 'actual_vocab': vocab_size})
        print(f"  Coverage: {coverage:.2%}")
    
    # Результаты
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ (покрытие уникальных слов)")
    print("="*60)
    print(df.to_string(index=False))
    
    # График
    plt.figure(figsize=(12, 8))
    plt.plot(df['vocab_size'], df['coverage'] * 100, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Размер словаря (vocab_size)', fontsize=12)
    plt.ylabel(f'Покрытие {len(test_words):,} уникальных слов (%)', fontsize=12)
    plt.title('Зависимость покрытия уникальных слов от размера словаря', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    output_path = Path('data/analysis/vocab_coverage_real.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nГрафик сохранён: {output_path}")

if __name__ == "__main__":
    main()