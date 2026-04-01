#!/usr/bin/env python3
"""
Построение датасета для BERT из предложений.
Группирует предложения в примеры по 512 токенов,
сохраняет в формате HuggingFace datasets.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

from tokenizers import BertWordPieceTokenizer
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm
import numpy as np


class BertDatasetBuilder:
    def __init__(self, tokenizer_path: str, max_length: int = 512):
        """
        tokenizer_path: путь к директории с vocab.txt (или полный путь к vocab.txt)
        max_length: максимальная длина последовательности
        """
        self.tokenizer = BertWordPieceTokenizer(
            os.path.join(tokenizer_path, "vocab.txt") if os.path.isdir(tokenizer_path) else tokenizer_path,
            lowercase=False
        )
        self.max_length = max_length
        self.cls_token_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        
        print(f"Tokenizer loaded:")
        print(f"  Vocab size: {self.tokenizer.get_vocab_size()}")
        print(f"  [CLS] id: {self.cls_token_id}")
        print(f"  [SEP] id: {self.sep_token_id}")
        print(f"  [PAD] id: {self.pad_token_id}")
    
    def load_sentences(self, sentences_dir: str) -> Dict[str, List[Dict]]:
        """
        Загружает все предложения из JSONL файлов, группируя по книгам.
        Возвращает: {book_id: [{"text": ..., "genre": ..., "position": ...}, ...]}
        """
        sentences_dir = Path(sentences_dir)
        books = defaultdict(list)
        
        jsonl_files = list(sentences_dir.glob("*.jsonl"))
        print(f"Found {len(jsonl_files)} JSONL files")
        
        for filepath in tqdm(jsonl_files, desc="Loading sentences"):
            book_id = filepath.stem
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        books[book_id].append({
                            "text": data["text"],
                            "genre": data.get("genre", "Unknown"),
                            "position": data.get("position", 0)
                        })
        
        # Сортируем предложения в каждой книге по позиции
        for book_id in books:
            books[book_id].sort(key=lambda x: x["position"])
        
        print(f"Loaded {len(books)} books, total sentences: {sum(len(v) for v in books.values())}")
        return books
    
    def group_into_examples(self, sentences: List[Dict]) -> List[List[str]]:
        """
        Группирует предложения в примеры по max_length токенов.
        Возвращает список групп (каждая группа — список предложений).
        """
        examples = []
        current_group = []
        current_tokens = 0
        
        # Резервируем 2 токена под [CLS] и [SEP]
        max_content_tokens = self.max_length - 2
        
        for sent_data in sentences:
            text = sent_data["text"]
            # Токенизируем предложение
            encoded = self.tokenizer.encode(text)
            token_count = len(encoded.ids)
            
            # Если одно предложение превышает лимит — обрезаем его
            if token_count > max_content_tokens:
                if current_group:
                    examples.append(current_group)
                    current_group = []
                    current_tokens = 0
                # Создаем пример из одного обрезанного предложения
                examples.append([text])
                continue
            
            # Проверяем, влезет ли предложение в текущую группу
            if current_tokens + token_count > max_content_tokens and current_group:
                # Сохраняем текущую группу
                examples.append(current_group)
                current_group = [text]
                current_tokens = token_count
            else:
                current_group.append(text)
                current_tokens += token_count
        
        # Последняя группа
        if current_group:
            examples.append(current_group)
        
        return examples
    
    def encode_example(self, sentences: List[str]) -> Dict:
        """
        Кодирует группу предложений в формат для BERT.
        Возвращает словарь с input_ids, attention_mask, token_type_ids.
        """
        # Объединяем предложения с пробелами
        text = " ".join(sentences)
        
        # Токенизируем
        encoded = self.tokenizer.encode(text)
        tokens = encoded.ids
        
        # Обрезаем до max_length - 2
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        
        # Формируем input_ids: [CLS] + tokens + [SEP]
        input_ids = [self.cls_token_id] + tokens + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)  # Все 0, так как у нас один сегмент
        
        # Паддинг
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            token_type_ids.extend([0] * padding_length)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    
    def build_dataset(self, sentences_dir: str, output_dir: str, val_split: float = 0.05):
        """
        Основной метод: загружает предложения, группирует, кодирует и сохраняет.
        """
        # 1. Загружаем предложения
        books = self.load_sentences(sentences_dir)
        
        # 2. Разделяем книги на train/val
        book_ids = list(books.keys())
        np.random.seed(42)
        np.random.shuffle(book_ids)
        
        val_size = int(len(book_ids) * val_split)
        val_books = set(book_ids[:val_size])
        train_books = set(book_ids[val_size:])
        
        print(f"Split: {len(train_books)} train books, {len(val_books)} val books")
        
        # 3. Обрабатываем книги
        train_examples = []
        val_examples = []
        
        for book_id in tqdm(book_ids, desc="Processing books"):
            sentences = books[book_id]
            groups = self.group_into_examples(sentences)
            
            examples = []
            for group in groups:
                encoded = self.encode_example(group)
                encoded["book_id"] = book_id
                encoded["genre"] = sentences[0]["genre"] if sentences else "Unknown"
                examples.append(encoded)
            
            if book_id in train_books:
                train_examples.extend(examples)
            else:
                val_examples.extend(examples)
        
        print(f"Created {len(train_examples)} train examples, {len(val_examples)} val examples")
        
        # 4. Сохраняем в HuggingFace datasets
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Преобразуем в формат datasets
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)
        
        # Сохраняем
        train_dataset.save_to_disk(str(output_path / "train"))
        val_dataset.save_to_disk(str(output_path / "val"))
        
        # Сохраняем метаданные
        metadata = {
            "num_train_examples": len(train_examples),
            "num_val_examples": len(val_examples),
            "max_length": self.max_length,
            "vocab_size": self.tokenizer.get_vocab_size(),
            "train_books": len(train_books),
            "val_books": len(val_books),
        }
        
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        print(f"  Train: {len(train_examples)} examples")
        print(f"  Val: {len(val_examples)} examples")
        
        return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Build BERT dataset from sentences")
    parser.add_argument("--sentences-dir", default="data/processed/sentences",
                        help="Directory with JSONL sentence files")
    parser.add_argument("--tokenizer-path", default="data/processed/tokenizer",
                        help="Path to tokenizer directory (with vocab.txt)")
    parser.add_argument("--output-dir", default="data/processed/dataset",
                        help="Output directory for dataset")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Validation split ratio")
    
    args = parser.parse_args()
    
    builder = BertDatasetBuilder(args.tokenizer_path, args.max_length)
    builder.build_dataset(args.sentences_dir, args.output_dir, args.val_split)


if __name__ == "__main__":
    main()