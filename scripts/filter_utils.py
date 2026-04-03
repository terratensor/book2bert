#!/usr/bin/env python3
"""
Утилиты для фильтрации CJK и тайских символов.
"""

# Диапазоны CJK-символов
CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # Extension A
    (0x20000, 0x2A6DF), # Extension B
    (0x2A700, 0x2B73F), # Extension C
    (0x2B740, 0x2B81F), # Extension D
    (0x2B820, 0x2CEAF), # Extension E
    (0x2CEB0, 0x2EBEF), # Extension F
    (0x30000, 0x3134F), # Extension G
    (0x31350, 0x323AF), # Extension H
    (0x2E80, 0x2EFF),   # CJK Radicals
    (0x2F00, 0x2FDF),   # Kangxi Radicals
    (0x2FF0, 0x2FFF),   # Ideographic Description
    (0x3000, 0x303F),   # CJK Symbols
    (0x31C0, 0x31EF),   # CJK Strokes
    (0x3200, 0x32FF),   # Enclosed CJK
    (0x3300, 0x33FF),   # CJK Compatibility
    (0xF900, 0xFAFF),   # Compatibility Ideographs
    (0xFE30, 0xFE4F),   # Compatibility Forms
]

THAI_RANGE = (0x0E00, 0x0E7F)

def is_cjk(char: str) -> bool:
    """Проверяет, является ли символ CJK."""
    code = ord(char)
    for start, end in CJK_RANGES:
        if start <= code <= end:
            return True
    return False

def is_thai(char: str) -> bool:
    """Проверяет, является ли символ тайским."""
    code = ord(char)
    return THAI_RANGE[0] <= code <= THAI_RANGE[1]

def filter_cjk_thai(text: str) -> str:
    """Удаляет CJK и тайские символы из текста."""
    return ''.join(ch for ch in text if not is_cjk(ch) and not is_thai(ch))

def has_cjk_thai(text: str) -> bool:
    """Проверяет, содержит ли текст CJK или тайские символы."""
    for ch in text:
        if is_cjk(ch) or is_thai(ch):
            return True
    return False