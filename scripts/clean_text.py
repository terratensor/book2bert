import re

def clean_text(text: str) -> str:
    # Удаляем строки с ISBN, УДК, ББК
    text = re.sub(r'(?m)^.*\b(?:ISBN|УДК|ББК|ISSN|DOI)\b.*$', '', text)
    
    # Удаляем оглавления
    text = re.sub(r'([А-Яа-яЁё])(\d+)(?:\s|$)', r'\1', text)
    
    # Удаляем строки-таблицы (много цифр)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        digits = sum(1 for c in line if c.isdigit())
        if len(line) > 0 and digits / len(line) > 0.5:
            continue
        cleaned.append(line)
    
    return '\n'.join(cleaned)