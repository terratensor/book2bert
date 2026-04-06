package textutils

import (
	"regexp"
	"strings"
	"unicode"
)

// CleanText очищает текст от OCR-артефактов
func CleanText(text string) string {
	// 1. Удаляем управляющие символы
	text = regexp.MustCompile(`[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`).ReplaceAllString(text, "")

	// 2. Восстанавливаем разорванные слова "внима тельно" → "внимательно"
	text = fixBrokenWords(text)

	// 3. Восстанавливаем переносы слов "внима- \n тельно" → "внимательно"
	text = fixHyphenatedWords(text)

	// 4. Удаляем строки с метаданными (ISBN, УДК, ББК)
	text = regexp.MustCompile(`(?m)^.*\b(?:ISBN|УДК|ББК|ISSN|DOI|УДК|ББК)\b.*$`).ReplaceAllString(text, "")

	// 5. Удаляем нумерацию (строки, состоящие только из цифр и точки)
	text = regexp.MustCompile(`(?m)^\s*\d+\.?\s*$`).ReplaceAllString(text, "")

	// 6. Удаляем оглавления (цифры в конце строки)
	text = regexp.MustCompile(`([А-Яа-яЁё])(\d+)(?:\s|$)`).ReplaceAllString(text, "$1")

	// 7. Удаляем строки-таблицы (много цифр и знаков)
	lines := strings.Split(text, "\n")
	var cleanedLines []string
	for _, line := range lines {
		if isTableLine(line) {
			continue
		}
		if isBibliographicLine(line) {
			continue
		}
		if isGarbageLine(line) {
			continue
		}
		cleanedLines = append(cleanedLines, line)
	}
	text = strings.Join(cleanedLines, "\n")

	// 8. Удаляем специальные символы
	specialChars := regexp.MustCompile(`[▲►▼◄■□▪▫●○◦★☆♦✓✗→←↑↓]`)
	text = specialChars.ReplaceAllString(text, "")

	// 9. Удаляем повторяющиеся символы
	text = regexp.MustCompile(`(?m)^[=\-*_]{10,}$`).ReplaceAllString(text, "")

	// 10. Нормализуем пробелы
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}

// fixBrokenWords исправляет разрывы слов: "внима тельно" → "внимательно"
func fixBrokenWords(text string) string {
	// Буква + пробел + буква → буква + буква
	re := regexp.MustCompile(`(\p{L})[ \t]+(\p{L})`)
	for re.MatchString(text) {
		text = re.ReplaceAllString(text, "$1$2")
	}
	return text
}

// fixHyphenatedWords исправляет переносы слов: "внима- \n тельно" → "внимательно"
func fixHyphenatedWords(text string) string {
	// Слово с дефисом + перенос строки + продолжение
	re := regexp.MustCompile(`(\p{L}+)-\s*\n\s*(\p{L}+)`)
	text = re.ReplaceAllString(text, "$1$2")
	return text
}

// isTableLine проверяет, является ли строка таблицей (>50% цифр и пунктуации)
func isTableLine(line string) bool {
	if len(line) < 20 {
		return false
	}

	digitCount := 0
	punctCount := 0
	for _, r := range line {
		if unicode.IsDigit(r) {
			digitCount++
		} else if unicode.IsPunct(r) || r == ',' || r == '.' || r == ';' {
			punctCount++
		}
	}

	total := len([]rune(line))
	if total == 0 {
		return false
	}
	ratio := float64(digitCount+punctCount) / float64(total)
	return ratio > 0.5
}

// isBibliographicLine проверяет строки с библиографическими ссылками
func isBibliographicLine(line string) bool {
	// Паттерны: "М Ѵ.І48, 151; Я 1.85", "Бдх П.3.45", "Вдх 5.3"
	patterns := []string{
		`[А-ЯЁ]\s+[ѴІ]\.\s*[А-ЯЁ]?\d+`, // М Ѵ.І48
		`[А-ЯЁ]{2,3}\s+[А-ЯЁ]?\.\d+`,   // Бдх П.3.45
		`[А-ЯЁ]\.\s*\d+\.\d+`,          // Я 1.85
	}
	for _, pattern := range patterns {
		if matched, _ := regexp.MatchString(pattern, line); matched {
			return true
		}
	}
	return false
}

// isGarbageLine проверяет строки с битыми символами (>30% не-буквенных символов)
func isGarbageLine(line string) bool {
	if len(line) < 10 {
		return false
	}

	letterCount := 0
	for _, r := range line {
		if unicode.IsLetter(r) {
			letterCount++
		}
	}

	total := len([]rune(line))
	if total == 0 {
		return false
	}
	ratio := float64(letterCount) / float64(total)
	return ratio < 0.3 // менее 30% букв → мусор
}

// IsAcceptableChar проверяет, можно ли оставить символ
func IsAcceptableChar(r rune) bool {
	// Разрешённые диапазоны
	switch {
	case r == ' ' || r == '\n' || r == '\r' || r == '\t':
		return true
	case r >= 0x20 && r <= 0x7E: // ASCII
		return true
	case r >= 0x0400 && r <= 0x052F: // Кириллица + расширенная
		return true
	case r >= 0x2E00 && r <= 0x2E7F: // Дополнительная пунктуация
		return true
	case r == '—' || r == '–' || r == '…': // Тире, многоточие
		return true
	default:
		return false
	}
}

// FilterNonRussian удаляет символы, не относящиеся к русскому/английскому
func FilterNonRussian(text string) string {
	var result []rune
	for _, r := range text {
		if IsAcceptableChar(r) {
			result = append(result, r)
		}
	}
	return string(result)
}
