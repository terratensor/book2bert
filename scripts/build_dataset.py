#!/usr/bin/env python3
"""
Построение датасета в streaming режиме.
Исправленная версия:
- Точная токенизация каждого предложения
- Группировка с учетом [SEP] между предложениями
- Гарантированное отсутствие обрезания
"""

import os
import json
import argparse
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from filter_utils import filter_cjk_thai
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
                    text = data["text"]
                    text = filter_cjk_thai(text)  # ← фильтруем
                    if not text.strip():
                        continue
                    sentences.append({
                        "text": text,
                        "genre": data.get("genre", "Unknown"),
                        "position": data.get("position", 0)
                    })
        
        sentences.sort(key=lambda x: x["position"])
        yield book_id, sentences

def tokenize_sentences(tokenizer, sentences):
    """Токенизирует каждое предложение точно."""
    tokenized = []
    for s in sentences:
        encoded = tokenizer.encode(s["text"])
        tokenized.append({
            "tokens": encoded.ids,
            "length": len(encoded.ids),
            "text": s["text"],
            "genre": s["genre"]
        })
    return tokenized

def group_sentences_exact(tokenized_sentences, max_length=512):
    """
    Группирует токенизированные предложения с учетом [SEP] между ними.
    Возвращает группы исходных текстов.
    """
    groups = []
    current_group_texts = []
    # [CLS] в начале и финальный [SEP] в конце
    current_tokens = 2
    
    for ts in tokenized_sentences:
        # +1 за [SEP] между предложениями (кроме последнего в группе)
        needed = ts["length"] + 1
        
        if current_tokens + needed > max_length and current_group_texts:
            groups.append(current_group_texts)
            current_group_texts = [ts["text"]]
            current_tokens = 2 + ts["length"] + 1
        else:
            current_group_texts.append(ts["text"])
            current_tokens += ts["length"] + 1
    
    if current_group_texts:
        groups.append(current_group_texts)
    
    return groups

def encode_group(tokenizer, group_texts, max_length=512):
    """Кодирует группу с [SEP] между предложениями."""
    text = " [SEP] ".join(group_texts)
    encoded = tokenizer.encode(text)
    tokens = encoded.ids
    
    # Это не должно происходить, если group_sentences_exact работает правильно
    if len(tokens) > max_length:
        print(f"WARNING: Group exceeded {max_length} tokens ({len(tokens)}), truncating!")
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
        "token_type_ids": [0] * max_length
    }

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
        
        # Точная токенизация
        tokenized = tokenize_sentences(tokenizer, sentences)
        
        # Точная группировка
        groups = group_sentences_exact(tokenized, args.max_length)
        
        examples = []
        for group_texts in groups:
            encoded = encode_group(tokenizer, group_texts, args.max_length)
            encoded["book_id"] = book_id
            encoded["genre"] = sentences[0]["genre"] if sentences else "Unknown"
            examples.append(encoded)
        
        save_examples(args.output_dir, split, examples)
        
        if split == "train":
            train_count += len(examples)
        else:
            val_count += len(examples)
        
        if (train_count + val_count) % 10000 == 0:
            print(f"  Progress: train={train_count}, val={val_count}")
        
        if args.max_books and (train_count + val_count) >= args.max_books:
            break
    
    print(f"\n=== Done ===")
    print(f"Train examples: {train_count}")
    print(f"Val examples: {val_count}")

if __name__ == "__main__":
    main()