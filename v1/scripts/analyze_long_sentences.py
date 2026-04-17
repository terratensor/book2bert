#!/usr/bin/env python3
"""
Анализ длинных предложений в корпусе.
Оптимизированная версия с батчевой токенизацией.
"""

import json
import argparse
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
import csv
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences-dir', type=str, required=True)
    parser.add_argument('--tokenizer-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='data/analysis')
    parser.add_argument('--min-tokens', type=int, default=400, help='Минимальная длина для анализа')
    parser.add_argument('--batch-size', type=int, default=1000, help='Размер батча для токенизации')
    parser.add_argument('--max-files', type=int, default=None, help='Ограничить количество файлов')
    args = parser.parse_args()
    
    # Загружаем токенизатор
    tokenizer = BertWordPieceTokenizer(
        str(Path(args.tokenizer_path) / "vocab.txt"),
        lowercase=False
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV файл для всех длинных предложений
    csv_path = output_dir / f"long_sentences_{args.min_tokens}.csv"
    txt_path = output_dir / f"long_sentences_full_{args.min_tokens}.txt"
    
    # Находим все JSONL файлы
    files = list(Path(args.sentences_dir).glob("*.jsonl"))
    if args.max_files:
        files = files[:args.max_files]
    
    print(f"Found {len(files)} files")
    print(f"Analyzing sentences longer than {args.min_tokens} tokens...")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {csv_path}")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file, \
         open(txt_path, 'w', encoding='utf-8') as txt_file:
        
        writer = csv.writer(csv_file)
        writer.writerow([
            'book_id', 'position', 'token_length', 'char_length',
            'genre', 'author', 'title', 'source_file'
        ])
        
        txt_file.write("="*80 + "\n")
        txt_file.write(f"LONG SENTENCES ANALYSIS (min_tokens={args.min_tokens})\n")
        txt_file.write("="*80 + "\n\n")
        
        total_sentences = 0
        long_count = 0
        
        # Статистика по файлам
        file_stats = defaultdict(lambda: {'total': 0, 'long': 0})
        
        for filepath in tqdm(files, desc="Processing files"):
            # Читаем все предложения из файла
            sentences = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        sentences.append(data)
            
            if not sentences:
                continue
            
            # Батчевая токенизация
            texts = [s["text"] for s in sentences]
            encoded_batch = tokenizer.encode_batch(texts)
            
            for i, encoded in enumerate(encoded_batch):
                total_sentences += 1
                data = sentences[i]
                text = data["text"]
                token_len = len(encoded.ids)
                char_len = len(text)
                
                file_stats[filepath.name]['total'] += 1
                
                if token_len >= args.min_tokens:
                    long_count += 1
                    file_stats[filepath.name]['long'] += 1
                    
                    book_id = data.get("book_id", "unknown")
                    position = data.get("position", 0)
                    genre = data.get("genre", "Unknown")
                    author = data.get("author", "Unknown")
                    title = data.get("title", "Unknown")
                    
                    # CSV запись
                    writer.writerow([
                        book_id, position, token_len, char_len,
                        genre, author, title, filepath.name
                    ])
                    
                    # Текстовый файл
                    txt_file.write(f"\n{'='*80}\n")
                    txt_file.write(f"Book ID: {book_id}\n")
                    txt_file.write(f"Position: {position}\n")
                    txt_file.write(f"Source file: {filepath.name}\n")
                    txt_file.write(f"Title: {title}\n")
                    txt_file.write(f"Author: {author}\n")
                    txt_file.write(f"Genre: {genre}\n")
                    txt_file.write(f"Token length: {token_len}\n")
                    txt_file.write(f"Character length: {char_len}\n")
                    txt_file.write(f"\n--- FULL TEXT ---\n")
                    txt_file.write(text)
                    txt_file.write(f"\n{'='*80}\n")
                    
                    # Обрезанная версия
                    if token_len > 512:
                        truncated_tokens = encoded.ids[:512]
                        truncated_text = tokenizer.decode(truncated_tokens)
                        txt_file.write(f"\n--- TRUNCATED TO 512 TOKENS ---\n")
                        txt_file.write(truncated_text)
                        txt_file.write(f"\n... (original length: {token_len} tokens)\n")
        
        # Общая статистика
        print(f"\n{'='*60}")
        print(f"STATISTICS")
        print(f"{'='*60}")
        print(f"Total sentences analyzed: {total_sentences:,}")
        print(f"Sentences >={args.min_tokens} tokens: {long_count:,}")
        print(f"Percentage: {long_count/total_sentences*100:.4f}%")
        
        # Статистика по файлам с наибольшим количеством длинных предложений
        print(f"\n{'='*60}")
        print(f"TOP 10 FILES WITH MOST LONG SENTENCES")
        print(f"{'='*60}")
        
        sorted_files = sorted(file_stats.items(), key=lambda x: x[1]['long'], reverse=True)[:10]
        for filename, stats in sorted_files:
            pct = stats['long'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {filename}: {stats['long']} / {stats['total']} ({pct:.2f}%)")
        
        print(f"\nOutput files:")
        print(f"  CSV: {csv_path}")
        print(f"  Full text: {txt_path}")

if __name__ == "__main__":
    main()