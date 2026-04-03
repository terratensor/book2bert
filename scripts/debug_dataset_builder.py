#!/usr/bin/env python3
"""
Отладка сборки датасета: разбор каждого шага.
"""

import json
from pathlib import Path
from tokenizers import BertWordPieceTokenizer

def main():
    # Пути (укажи свои)
    sentences_file = "data/processed/sentences_militera/933adf34-0bca-4f65-9409-fabfdc76be97.jsonl"  # тот файл, откуда пример
    tokenizer_path = "data/processed/tokenizer_militera"
    
    # Загружаем токенизатор
    tokenizer = BertWordPieceTokenizer(
        str(Path(tokenizer_path) / "vocab.txt"),
        lowercase=False
    )
    
    print("="*60)
    print("ШАГ 1: Загрузка предложений из JSONL")
    print("="*60)
    
    sentences = []
    with open(sentences_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:  # берем только первые 5 предложений
                break
            data = json.loads(line)
            sentences.append(data["text"])
            print(f"Предложение {i+1}: {data['text'][:100]}...")
    
    print("\n" + "="*60)
    print("ШАГ 2: Токенизация каждого предложения")
    print("="*60)
    
    tokenized_sentences = []
    for i, sent in enumerate(sentences):
        encoded = tokenizer.encode(sent)
        print(f"\nПредложение {i+1}:")
        print(f"  Текст: {sent[:80]}...")
        print(f"  Токены: {encoded.tokens[:20]}...")
        print(f"  ID: {encoded.ids[:20]}...")
        print(f"  Начинается с [CLS]? {encoded.tokens[0] == '[CLS]' if encoded.tokens else False}")
        print(f"  Заканчивается на [SEP]? {encoded.tokens[-1] == '[SEP]' if encoded.tokens else False}")
        tokenized_sentences.append(encoded)
    
    print("\n" + "="*60)
    print("ШАГ 3: Группировка предложений (как в build_dataset.py)")
    print("="*60)
    
    # Эмуляция group_sentences
    max_length = 512
    groups = []
    current_group = []
    current_tokens = 0
    
    for i, sent in enumerate(sentences):
        token_estimate = max(1, len(sent) // 3.5)
        print(f"Предложение {i+1}: {len(sent)} символов, оценка токенов ~{token_estimate}")
        
        if current_tokens + token_estimate > max_length - 2 and current_group:
            print(f"  → Сохраняем группу из {len(current_group)} предложений")
            groups.append(current_group)
            current_group = [sent]
            current_tokens = token_estimate
        else:
            current_group.append(sent)
            current_tokens += token_estimate
    
    if current_group:
        print(f"  → Сохраняем последнюю группу из {len(current_group)} предложений")
        groups.append(current_group)
    
    print(f"\nВсего групп: {len(groups)}")
    
    print("\n" + "="*60)
    print("ШАГ 4: Кодирование группы (как в encode_group)")
    print("="*60)
    
    for g_idx, group in enumerate(groups):
        print(f"\n--- Группа {g_idx+1} ---")
        print(f"Предложения в группе: {len(group)}")
        
        # Склеиваем текст
        text = " ".join(group)
        print(f"Склеенный текст: {text[:150]}...")
        
        # Токенизируем
        encoded = tokenizer.encode(text)
        print(f"Токенов после токенизации: {len(encoded.ids)}")
        
        # Обрезаем
        tokens = encoded.ids
        if len(tokens) > max_length - 2:
            tokens = tokens[:max_length - 2]
            print(f"Обрезано до {len(tokens)} токенов")
        
        # Добавляем [CLS] и [SEP]
        cls_id = tokenizer.token_to_id("[CLS]")
        sep_id = tokenizer.token_to_id("[SEP]")
        pad_id = tokenizer.token_to_id("[PAD]")
        
        input_ids = [cls_id] + tokens + [sep_id]
        print(f"После добавления [CLS] и [SEP]: {len(input_ids)} токенов")
        
        # Проверяем на двойные [CLS]
        if input_ids[0] == cls_id and input_ids[1] == cls_id:
            print("  ⚠️ ПРОБЛЕМА: Двойной [CLS]! Первый токен текста уже был [CLS]")
        
        # Проверяем на двойные [SEP]
        if input_ids[-1] == sep_id and input_ids[-2] == sep_id:
            print("  ⚠️ ПРОБЛЕМА: Двойной [SEP]! Последний токен текста уже был [SEP]")
        
        # Паддинг
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([pad_id] * padding_length)
            print(f"Добавлен паддинг: {padding_length} токенов [PAD]")
        
        print(f"Финальная длина: {len(input_ids)} (максимум {max_length})")
        print(f"Первые 10 ID: {input_ids[:10]}")
        print(f"Последние 10 ID: {input_ids[-10:]}")
        
        # Декодируем первые токены для наглядности
        first_tokens = [tokenizer.id_to_token(i) for i in input_ids[:10]]
        print(f"Первые 10 токенов: {first_tokens}")
        
        if g_idx == 0:
            print("\n" + "="*60)
            print("ВЫВОД: Проблема в том, что предложения в JSONL уже содержат [CLS] и [SEP]")
            print("Решение: При загрузке предложений из JSONL нужно убирать [CLS] и [SEP]")
            print("="*60)

if __name__ == "__main__":
    main()