#!/usr/bin/env python3
"""
Точный расчёт токен-статистики для корпуса.
Использует обученный токенизатор для подсчёта распределения токенов на предложение.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True,
                        help='Путь к файлу корпуса (corpus_clean.txt)')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Путь к директории токенизатора (tokenizer_clean)')
    parser.add_argument('--sample-size', type=int, default=1000000,
                        help='Количество предложений для анализа')
    parser.add_argument('--output', type=str, default='data/analysis/token_stats.json',
                        help='Выходной JSON файл')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Размер батча для токенизации')
    args = parser.parse_args()

    # Загружаем токенизатор
    tokenizer = BertWordPieceTokenizer(
        str(Path(args.tokenizer) / "vocab.txt"),
        lowercase=False
    )
    print(f"Tokenizer loaded, vocab_size={tokenizer.get_vocab_size()}")

    # Читаем корпус и берём выборку
    print(f"Reading corpus and sampling {args.sample_size:,} lines...")
    lines = []
    with open(args.corpus, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.sample_size:
                break
            line = line.strip()
            if line:
                lines.append(line)

    print(f"Loaded {len(lines):,} sentences")

    # Токенизируем батчами
    token_counts = []
    total_tokens = 0

    print("Tokenizing in batches...")
    for i in tqdm(range(0, len(lines), args.batch_size)):
        batch = lines[i:i+args.batch_size]
        encoded_batch = tokenizer.encode_batch(batch)

        for encoded in encoded_batch:
            # Убираем [CLS] и [SEP], которые добавляет токенизатор
            ids = encoded.ids
            cls_id = tokenizer.token_to_id("[CLS]")
            sep_id = tokenizer.token_to_id("[SEP]")
            real_tokens = [id for id in ids if id not in (cls_id, sep_id)]

            token_count = len(real_tokens)
            token_counts.append(token_count)
            total_tokens += token_count

    token_counts = np.array(token_counts)
    total_sentences = len(token_counts)

    # Вычисляем статистику
    stats = {
        "total_sentences": total_sentences,
        "total_tokens": int(total_tokens),
        "avg_tokens_per_sentence": float(total_tokens / total_sentences),
        "min_tokens": int(np.min(token_counts)),
        "max_tokens": int(np.max(token_counts)),
        "std_tokens": float(np.std(token_counts)),
        "percentiles": {
            "1": int(np.percentile(token_counts, 1)),
            "5": int(np.percentile(token_counts, 5)),
            "10": int(np.percentile(token_counts, 10)),
            "25": int(np.percentile(token_counts, 25)),
            "50": int(np.percentile(token_counts, 50)),
            "75": int(np.percentile(token_counts, 75)),
            "90": int(np.percentile(token_counts, 90)),
            "95": int(np.percentile(token_counts, 95)),
            "99": int(np.percentile(token_counts, 99)),
            "99.5": int(np.percentile(token_counts, 99.5)),
            "99.9": int(np.percentile(token_counts, 99.9)),
        }
    }

    # Добавляем распределение по бакетам
    buckets = [8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 384, 512]
    bucket_counts = {}
    for bucket in buckets:
        bucket_counts[bucket] = int(np.sum(token_counts <= bucket))

    stats["cumulative_coverage"] = {
        str(bucket): {
            "count": bucket_counts[bucket],
            "percent": round(bucket_counts[bucket] / total_sentences * 100, 2)
        }
        for bucket in buckets
    }

    # Сохраняем результат
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Выводим в консоль
    print("\n" + "="*60)
    print("TOKEN STATISTICS")
    print("="*60)
    print(f"Total sentences: {stats['total_sentences']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Avg tokens/sentence: {stats['avg_tokens_per_sentence']:.2f}")
    print(f"Min tokens: {stats['min_tokens']}")
    print(f"Max tokens: {stats['max_tokens']}")
    print(f"Std tokens: {stats['std_tokens']:.2f}")

    print("\nPercentiles:")
    for p, val in stats['percentiles'].items():
        print(f"  {p}%: {val}")

    print("\nCumulative coverage (sentences <= N tokens):")
    for bucket, data in stats['cumulative_coverage'].items():
        bucket_int = int(bucket)
        print(f"  ≤{bucket_int:3d} tokens: {data['count']:>10,} sentences ({data['percent']:5.2f}%)")

    print(f"\nSaved to: {output_path}")

    # Рекомендация по max_seq_length
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)

    target_coverage = [99.0, 99.5, 99.9]
    for target in target_coverage:
        for bucket, data in stats['cumulative_coverage'].items():
            if data['percent'] >= target:
                print(f"  For {target}% coverage: max_seq_length = {bucket}")
                break

    print("\n" + "="*60)


if __name__ == "__main__":
    main()