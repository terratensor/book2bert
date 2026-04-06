package textutils

import (
	"regexp"
	"strings"
	"unicode"
)

// CleanText удаляет артефакты OCR, мусор, метаданные
func CleanText(text string) string {
	// 1. Удаляем управляющие символы (кроме \n, \r, \t)
	text = regexp.MustCompile(`[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`).ReplaceAllString(text, "")

	// 2. Удаляем строки с ISBN, УДК, ББК
	text = regexp.MustCompile(`(?m)^.*\b(?:ISBN|УДК|ББК|ISSN|DOI)\b.*$`).ReplaceAllString(text, "")

	// 3. Удаляем оглавления (цифры в конце строки, когда перед ними нет точки)
	//    "Введение7" → "Введение"
	text = regexp.MustCompile(`([А-Яа-яЁё])(\d+)(?:\s|$)`).ReplaceAllString(text, "$1")

	// 4. Удаляем строки, содержащие более 50% цифр и знаков пунктуации (таблицы)
	lines := strings.Split(text, "\n")
	var cleanedLines []string
	for _, line := range lines {
		if isTableLine(line) {
			continue
		}
		cleanedLines = append(cleanedLines, line)
	}
	text = strings.Join(cleanedLines, "\n")

	// 5. Удаляем специальные символы (стрелки, блоки, геометрические фигуры)
	specialChars := regexp.MustCompile(`[▲►▼◄■□▪▫●○◦★☆♦✓✗→←↑↓]`)
	text = specialChars.ReplaceAllString(text, "")

	// 6. Удаляем повторяющиеся символы (тире, звёздочки, равно)
	text = regexp.MustCompile(`(?m)^[=\-*_]{10,}$`).ReplaceAllString(text, "")

	// 7. Нормализуем пробелы
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}

// isTableLine проверяет, является ли строка таблицей (много цифр и знаков)
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

	// Если цифры и пунктуация составляют >50% строки — это таблица
	total := len([]rune(line))
	if total == 0 {
		return false
	}
	ratio := float64(digitCount+punctCount) / float64(total)
	return ratio > 0.5
}
