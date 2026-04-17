# scripts/debug_full_tokens.py
import json
from pathlib import Path
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer("data/processed/tokenizer_militera/vocab.txt", lowercase=False)

# Берем первые 3 предложения из того же файла
sentences = []
with open("data/processed/sentences_militera/0a2a41ed-6746-4f42-a240-a269c8e137c1.jsonl", 'r') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        data = json.loads(line)
        sentences.append(data["text"])

print("Исходные предложения (как в JSONL):")
for i, s in enumerate(sentences):
    print(f"{i+1}: {s[:100]}...")

print("\n" + "="*60)
print("Токенизация каждого предложения отдельно:")
for i, s in enumerate(sentences):
    enc = tokenizer.encode(s)
    print(f"\nПредложение {i+1}:")
    print(f"  Токены: {enc.tokens}")
    print(f"  ID: {enc.ids}")

print("\n" + "="*60)
print("Склеиваем через пробел и токенизируем:")
text = " ".join(sentences)
print(f"Склеенный текст: {text[:200]}...")
enc = tokenizer.encode(text)
print(f"Токены: {enc.tokens}")
print(f"ID: {enc.ids}")
print(f"Длина: {len(enc.ids)}")