#!/usr/bin/env python3
"""
Построение датасета в streaming режиме.
Оптимизированная версия:
- Токенизация каждого предложения ОДИН раз
- Группировка без повторной токенизации
- Разбивка по \n на этапе загрузки
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

def tokenize_sentences_batch(tokenizer, sentences, batch_size=1000):
    tokenized = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        texts = [s["text"] for s in batch]
        
        encoded_batch = tokenizer.encode_batch(texts)
        
        for j, encoded in enumerate(encoded_batch):
            ids = encoded.ids
            # Удаляем [CLS] и [SEP], которые добавил токенизатор
            cls_id = tokenizer.token_to_id("[CLS]")
            sep_id = tokenizer.token_to_id("[SEP]")
            ids = [id for id in ids if id not in (cls_id, sep_id)]
            
            tokenized.append({
                "ids": ids,
                "length": len(ids),
                "text": batch[j]["text"],
                "genre": batch[j]["genre"]
            })
    
    return tokenized

def group_sentences_exact(tokenized_sentences, max_length=512):
    """
    Группирует ТОКЕНИЗИРОВАННЫЕ предложения (без вызова токенизатора!).
    Использует заранее посчитанные длины токенов.
    """
    groups = []
    current_group = []
    # [CLS] в начале + финальный [SEP] в конце = 2 токена
    current_tokens = 2
    
    for ts in tokenized_sentences:
        # +1 для [SEP] после предложения
        needed = ts["length"] + 1
        
        if current_tokens + needed > max_length and current_group:
            groups.append(current_group)
            current_group = [ts]
            current_tokens = 2 + ts["length"] + 1
        else:
            current_group.append(ts)
            current_tokens += needed
    
    if current_group:
        groups.append(current_group)
    
    return groups

def encode_group(tokenizer, group, max_length=512):
    """
    Кодирует группу ТОКЕНИЗИРОВАННЫХ предложений.
    Склеивает ID с [SEP] между ними.
    """
    # Собираем ID всех предложений, вставляя [SEP] между ними
    sep_id = tokenizer.token_to_id("[SEP]")
    all_ids = []
    
    for i, ts in enumerate(group):
        if i > 0:
            all_ids.append(sep_id)
        all_ids.extend(ts["ids"])
    
    # Добавляем финальный [SEP]
    all_ids.append(sep_id)
    
    # [CLS] в начало
    cls_id = tokenizer.token_to_id("[CLS]")
    input_ids = [cls_id] + all_ids
    
    # Обрезаем (защита, хотя не должно происходить)
    if len(input_ids) > max_length:
        print(f"WARNING: Group exceeded {max_length} tokens ({len(input_ids)}), truncating!")
        input_ids = input_ids[:max_length]
    
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
    """Сохраняет примеры в JSONL файл."""
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
    parser.add_argument('--batch-size', type=int, default=1000, help='Размер батча для токенизации')
    args = parser.parse_args()
    
    tokenizer = BertWordPieceTokenizer(
        str(Path(args.tokenizer_path) / "vocab.txt"),
        lowercase=False
    )
    print(f"Tokenizer loaded, vocab_size={tokenizer.get_vocab_size()}")
    print(f"Batch size: {args.batch_size}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_count = 0
    val_count = 0
    
    for book_id, sentences in tqdm(stream_sentences_by_book(args.sentences_dir), desc="Processing books"):
        is_val = (hash(book_id) % 100) < (args.val_split * 100)
        split = "val" if is_val else "train"
        
        # 1. Токенизируем все предложения книги (один раз, батчами)
        tokenized = tokenize_sentences_batch(tokenizer, sentences, args.batch_size)
        
        # 2. Группируем по токенам (без вызова токенизатора)
        groups = group_sentences_exact(tokenized, args.max_length)
        
        # 3. Кодируем группы
        examples = []
        for group in groups:
            encoded = encode_group(tokenizer, group, args.max_length)
            encoded["book_id"] = book_id
            encoded["genre"] = sentences[0]["genre"] if sentences else "Unknown"
            examples.append(encoded)
        
        # 4. Сохраняем
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