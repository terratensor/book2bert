#!/usr/bin/env python3
"""
Построение датасета в streaming режиме.
Исправленная версия:
- Нет двойного [CLS] и [SEP]
- NSP не используется (можно включить параметром)
"""

import os
import json
import argparse
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
import random

def stream_sentences_by_book(sentences_dir):
    """Генератор, читающий предложения по книгам."""
    files = list(Path(sentences_dir).glob("*.jsonl"))
    random.shuffle(files)
    
    for filepath in files:
        book_id = filepath.stem
        sentences = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sentences.append({
                        "text": data["text"],
                        "genre": data.get("genre", "Unknown"),
                        "position": data.get("position", 0)
                    })
        
        sentences.sort(key=lambda x: x["position"])
        yield book_id, sentences

def encode_group(tokenizer, sentences, max_length=512):
    """Кодирует группу предложений с [SEP] между ними."""
    # Вставляем [SEP] между предложениями
    text = " [SEP] ".join(sentences)
    encoded = tokenizer.encode(text)
    tokens = encoded.ids
    
    # Обрезаем до max_length
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    input_ids = tokens
    attention_mask = [1] * len(input_ids)
    
    # Паддинг
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids.extend([tokenizer.token_to_id("[PAD]")] * padding_length)
        attention_mask.extend([0] * padding_length)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": [0] * max_length  # NSP пока не используем
    }

def group_sentences(sentences, max_tokens=512):
    """Группирует предложения в батчи по max_tokens."""
    groups = []
    current_group = []
    current_tokens = 0
    
    for s in sentences:
        # Приблизительная оценка токенов (символы / 3.5)
        token_estimate = max(1, len(s["text"]) // 3.5)
        
        if current_tokens + token_estimate > max_tokens - 2 and current_group:
            groups.append(current_group)
            current_group = [s["text"]]
            current_tokens = token_estimate
        else:
            current_group.append(s["text"])
            current_tokens += token_estimate
    
    if current_group:
        groups.append(current_group)
    
    return groups

def save_examples(output_dir, split, examples):
    """Сохраняет примеры в JSONL файл."""
    output_dir = Path(output_dir) / split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Используем несколько файлов для параллельной записи
    file_idx = random.randint(0, 99)
    output_file = output_dir / f"part_{file_idx:04d}.jsonl"
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences-dir', type=str, required=True)
    parser.add_argument('--tokenizer-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--val-split', type=float, default=0.05)
    parser.add_argument('--max-books', type=int, default=None)
    parser.add_argument('--use-nsp', action='store_true', help='Включить NSP (пока не реализовано)')
    args = parser.parse_args()
    
    # Загружаем токенизатор
    tokenizer = BertWordPieceTokenizer(
        str(Path(args.tokenizer_path) / "vocab.txt"),
        lowercase=False
    )
    print(f"Tokenizer loaded, vocab_size={tokenizer.get_vocab_size()}")
    
    # Создаем выходную директорию
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_nsp:
        print("WARNING: NSP not yet implemented. Use without --use-nsp for now.")
    
    train_count = 0
    val_count = 0
    
    for book_id, sentences in tqdm(stream_sentences_by_book(args.sentences_dir), desc="Processing books"):
        # Определяем split по хешу book_id
        is_val = (hash(book_id) % 100) < (args.val_split * 100)
        split = "val" if is_val else "train"
        
        # Группируем предложения
        groups = group_sentences(sentences, args.max_length)
        
        # Кодируем группы
        examples = []
        for group in groups:
            encoded = encode_group(tokenizer, group, args.max_length)
            encoded["book_id"] = book_id
            encoded["genre"] = sentences[0]["genre"] if sentences else "Unknown"
            examples.append(encoded)
        
        # Сохраняем
        save_examples(args.output_dir, split, examples)
        
        if split == "train":
            train_count += len(examples)
        else:
            val_count += len(examples)
        
        # Прогресс
        if (train_count + val_count) % 10000 == 0:
            print(f"  Progress: train={train_count}, val={val_count}")
        
        if args.max_books and (train_count + val_count) >= args.max_books:
            break
    
    print(f"\n=== Done ===")
    print(f"Train examples: {train_count}")
    print(f"Val examples: {val_count}")

if __name__ == "__main__":
    main()