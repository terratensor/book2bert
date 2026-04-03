#!/usr/bin/env python3
"""
Построение датасета в streaming режиме.
ТОЧНАЯ группировка через токенизатор (медленно, но гарантированно правильно).
"""

import os
import json
import argparse
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
import random

def stream_sentences_by_book(sentences_dir):
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
                    if not text.strip():
                        continue
                    sentences.append({
                        "text": text,
                        "genre": data.get("genre", "Unknown"),
                        "position": data.get("position", 0)
                    })
        
        sentences.sort(key=lambda x: x["position"])
        yield book_id, sentences

def group_sentences_exact(sentences, tokenizer, max_length=512):
    """
    Точная группировка. Для каждой проверки вызывает tokenizer.encode().
    Гарантирует, что группа не превысит max_length.
    """
    groups = []
    current_group = []
    
    for s in sentences:
        text = s["text"]
        # Пробуем добавить предложение
        test_group = current_group + [text]
        test_text = " [SEP] ".join(test_group)
        encoded = tokenizer.encode(test_text)
        
        if len(encoded.ids) <= max_length:
            # Влезает
            current_group = test_group
        else:
            # Не влезает — сохраняем текущую группу (без этого предложения)
            if current_group:
                groups.append(current_group)
            # Начинаем новую группу с этого предложения
            current_group = [text]
    
    if current_group:
        groups.append(current_group)
    
    return groups

def encode_group(tokenizer, group_texts, max_length=512):
    """Кодирует группу текстов с [SEP] между ними."""
    text = " [SEP] ".join(group_texts)
    encoded = tokenizer.encode(text)
    tokens = encoded.ids
    
    # Это не должно происходить, но оставляем защиту
    if len(tokens) > max_length:
        print(f"WARNING: Group exceeded {max_length} tokens ({len(tokens)}), truncating!")
        tokens = tokens[:max_length]
    
    input_ids = tokens
    attention_mask = [1] * len(input_ids)
    
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
    output_dir = Path(output_dir) / split
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    args = parser.parse_args()
    
    tokenizer = BertWordPieceTokenizer(
        str(Path(args.tokenizer_path) / "vocab.txt"),
        lowercase=False
    )
    print(f"Tokenizer loaded, vocab_size={tokenizer.get_vocab_size()}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_count = 0
    val_count = 0
    
    for book_id, sentences in tqdm(stream_sentences_by_book(args.sentences_dir), desc="Processing books"):
        is_val = (hash(book_id) % 100) < (args.val_split * 100)
        split = "val" if is_val else "train"
        
        # Точная группировка через токенизатор
        groups = group_sentences_exact(sentences, tokenizer, args.max_length)
        
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
        
        if args.max_books and len(sentences) >= args.max_books:
            break
    
    print(f"\n=== Done ===")
    print(f"Train examples: {train_count}")
    print(f"Val examples: {val_count}")

if __name__ == "__main__":
    main()