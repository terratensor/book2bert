# scripts/test_coverage_with_subwords.py
#!/usr/bin/env python3
"""
Измерение реального покрытия корпуса с учётом субвордов.
Слово считается "покрытым", если после токенизации в нём нет токена [UNK].
"""

from tokenizers import BertWordPieceTokenizer
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def test_vocab_size(corpus_file, vocab_size, max_lines=50000):
    """Обучает токенизатор и считает покрытие (с учётом субвордов)"""
    
    print(f"  Training vocab_size={vocab_size}...")
    
    # Обучаем токенизатор
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(
        files=[corpus_file],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    # Считаем покрытие на том же корпусе (или отдельном)
    total_tokens = 0
    unknown_tokens = 0
    unk_id = tokenizer.token_to_id("[UNK]")
    
    print(f"  Calculating coverage on {max_lines:,} lines...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=max_lines)):
            if i >= max_lines:
                break
            if line.strip():
                encoded = tokenizer.encode(line.strip())
                unknown_tokens += encoded.ids.count(unk_id)
                total_tokens += len(encoded.ids)
    
    coverage = 1 - unknown_tokens / total_tokens if total_tokens > 0 else 0
    return coverage, tokenizer.get_vocab_size()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--vocab-sizes', type=str, 
                       default="10000,20000,30000,50000,70000,100000,120000,150000")
    parser.add_argument('--max-lines', type=int, default=100000)
    args = parser.parse_args()
    
    vocab_sizes = [int(x) for x in args.vocab_sizes.split(',')]
    
    print(f"Корпус: {args.corpus}")
    print(f"Тестируемые размеры: {vocab_sizes}")
    print(f"Строк для оценки: {args.max_lines:,}\n")
    
    results = []
    
    for size in vocab_sizes:
        coverage, actual = test_vocab_size(args.corpus, size, args.max_lines)
        results.append({
            'vocab_size': size,
            'actual_vocab': actual,
            'coverage': coverage
        })
        print(f"  Coverage: {coverage:.4%}\n")
    
    # Результаты
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ (покрытие токенов с учётом субвордов)")
    print("="*60)
    print(df.to_string(index=False))
    
    # График
    plt.figure(figsize=(12, 8))
    plt.plot(df['vocab_size'], df['coverage'] * 100, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Размер словаря (vocab_size)', fontsize=12)
    plt.ylabel('Покрытие токенов корпуса (%)', fontsize=12)
    plt.title('Реальное покрытие с учётом субвордов', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(90, 100)
    
    # Точка перегиба
    for i in range(1, len(df)):
        if df['coverage'].iloc[i] - df['coverage'].iloc[i-1] < 0.001:
            optimal = df['vocab_size'].iloc[i]
            plt.axvline(x=optimal, color='r', linestyle='--', alpha=0.7)
            plt.annotate(f'Оптимум: {optimal}',
                        xy=(optimal, df['coverage'].iloc[i] * 100),
                        xytext=(optimal + 10000, df['coverage'].iloc[i] * 100 - 1),
                        arrowprops=dict(arrowstyle='->', color='red'))
            break
    
    output_path = Path('data/analysis/coverage_with_subwords.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\nГрафик сохранён: {output_path}")
    print(f"\nРекомендуемый vocab_size: {optimal if 'optimal' in locals() else '100,000'}")

if __name__ == "__main__":
    main()