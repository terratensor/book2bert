# scripts/extract_test_examples.py
import json
import random
from pathlib import Path

def extract_random_sentences(sentences_dir, num_examples=10):
    """Извлекает случайные предложения из militera корпуса"""
    files = list(Path(sentences_dir).glob("*.jsonl"))
    sentences = []
    
    for filepath in random.sample(files, min(5, len(files))):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in random.sample(lines, min(2, len(lines))):
                data = json.loads(line)
                text = data['text']
                if len(text.split()) > 5:  # не слишком короткие
                    sentences.append(text)
            if len(sentences) >= num_examples:
                break
    
    return sentences[:num_examples]

def create_masked_examples(sentences):
    """Создает примеры с маскированием ключевых слов"""
    masked_examples = []
    for s in sentences:
        words = s.split()
        if len(words) < 4:
            continue
        # Маскируем случайное знаменательное слово
        for i, word in enumerate(words):
            if len(word) > 4 and word not in [',', '.', '!', '?', '—', '-']:
                masked_words = words.copy()
                masked_words[i] = "[MASK]"
                masked_examples.append(' '.join(masked_words))
                break
    return masked_examples

if __name__ == "__main__":
    sentences = extract_random_sentences("data/processed/sentences_militera", 10)
    print("=== Real sentences from militera corpus ===\n")
    for s in sentences:
        print(f"Original: {s}")
    print("\n=== Masked examples for testing ===\n")
    for ex in create_masked_examples(sentences):
        print(f"Input: {ex}")