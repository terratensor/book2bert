#!/usr/bin/env python3
# scripts/sample_dataset.py
"""
Создание выборки 10% из JSONL файлов без загрузки всего в память
"""

import random
import sys
from pathlib import Path
from tqdm import tqdm

def sample_file(input_path, output_path, sample_ratio=0.1, seed=42):
    """Создаёт выборку из одного JSONL файла (streaming)"""
    random.seed(seed)
    
    # Первый проход: считаем количество строк
    total_lines = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    if total_lines == 0:
        return 0
    
    # Количество строк в выборке
    sample_size = max(1, int(total_lines * sample_ratio))
    
    # Алгоритм reservoir sampling (выборка без загрузки в память)
    sample_lines = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc=Path(input_path).name)):
            if i < sample_size:
                sample_lines.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    sample_lines[j] = line
    
    # Записываем выборку
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(sample_lines)
    
    return len(sample_lines)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--sample-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    random.seed(args.seed)
    
    for split in ['train', 'val']:
        input_split = input_dir / split
        output_split = output_dir / split
        output_split.mkdir(parents=True, exist_ok=True)
        
        files = list(input_split.glob("*.jsonl"))
        print(f"\nProcessing {split}: {len(files)} files")
        
        total_samples = 0
        for input_file in tqdm(files, desc=split):
            output_file = output_split / input_file.name
            samples = sample_file(input_file, output_file, args.sample_ratio, args.seed)
            total_samples += samples
        
        print(f"  {split}: {total_samples:,} examples")
    
    print("\nDone!")

if __name__ == "__main__":
    main()