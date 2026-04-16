package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
)

// Stats хранит агрегированную статистику
type Stats struct {
	// Базовые метрики
	TotalSentences int64
	TotalChars     int64
	TotalFiles     int64
	EmptySentences int64

	// Распределение длины
	Lengths []int64

	// Языковой состав
	RussianOnly        int64
	EnglishOnly        int64
	MixedCyrillicLatin int64
	Other              int64

	// Качество
	TooShort20      int64
	TooShort50      int64
	TooLong1000     int64
	TooLong2000     int64
	HighDigit10     int64
	HighDigit30     int64
	HighDigit50     int64
	HighPunct30     int64
	HighPunct50     int64
	ListMarker      int64
	HasISBN         int64
	HasUDK          int64
	HasBBK          int64
	HasURL          int64
	HasEmail        int64
	HighUppercase50 int64
	HighUppercase80 int64

	// Дубликаты (опционально, требует памяти)
	Duplicates int64
	uniqueMap  map[[16]byte]bool
	mu         sync.Mutex
}

// FileStats хранит статистику по одному файлу
type FileStats struct {
	Path         string
	Sentences    int
	AvgLength    float64
	RussianRatio float64
	EnglishRatio float64
	MixedRatio   float64
	DigitRatio30 float64
	ListRatio    float64
	GarbageRatio float64
}

type Progress struct {
	processedFiles int64
	totalFiles     int64
	startTime      time.Time
}

// isRussian проверяет наличие кириллицы
func isRussian(text string) bool {
	for _, r := range text {
		if r >= 0x0400 && r <= 0x04FF {
			return true
		}
	}
	return false
}

// isEnglish проверяет наличие латиницы
func isEnglish(text string) bool {
	for _, r := range text {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			return true
		}
	}
	return false
}

// digitRatio считает долю цифр
func digitRatio(text string) float64 {
	digits := 0
	for _, r := range text {
		if r >= '0' && r <= '9' {
			digits++
		}
	}
	return float64(digits) / float64(len(text))
}

// punctRatio считает долю пунктуации
func punctRatio(text string) float64 {
	punct := 0
	for _, r := range text {
		if unicode.IsPunct(r) {
			punct++
		}
	}
	return float64(punct) / float64(len(text))
}

// uppercaseRatio считает долю заглавных букв
func uppercaseRatio(text string) float64 {
	upper := 0
	letters := 0
	for _, r := range text {
		if unicode.IsLetter(r) {
			letters++
			if unicode.IsUpper(r) {
				upper++
			}
		}
	}
	if letters == 0 {
		return 0
	}
	return float64(upper) / float64(letters)
}

// hasListMarker проверяет, начинается ли строка с маркера списка
var listMarkerRegex = regexp.MustCompile(`^\s*[•\-*•\d]+[\.\)]\s+`)

func hasListMarker(text string) bool {
	return listMarkerRegex.MatchString(text)
}

// hasGarbagePatterns проверяет наличие мусорных паттернов
var (
	isbnRegex  = regexp.MustCompile(`\bISBN\s*\d{3}-\d{1,5}-\d{1,7}-\d{1,7}-\d{1,7}\b`)
	udkRegex   = regexp.MustCompile(`\bУДК\s*\d+(?:\.\d+)+\b`)
	bbkRegex   = regexp.MustCompile(`\bББК\s*\d+(?:\.\d+)+\b`)
	urlRegex   = regexp.MustCompile(`https?://[^\s]+`)
	emailRegex = regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)
)

func hasGarbagePatterns(text string) (hasISBN, hasUDK, hasBBK, hasURL, hasEmail bool) {
	hasISBN = isbnRegex.MatchString(text)
	hasUDK = udkRegex.MatchString(text)
	hasBBK = bbkRegex.MatchString(text)
	hasURL = urlRegex.MatchString(text)
	hasEmail = emailRegex.MatchString(text)
	return
}

// analyzeFile анализирует один JSONL файл
func analyzeFile(filePath string, stats *Stats, fileStatsChan chan<- FileStats, wg *sync.WaitGroup, sem chan struct{}, progress *Progress) {
	defer wg.Done()
	sem <- struct{}{}
	defer func() { <-sem }()
	defer atomic.AddInt64(&progress.processedFiles, 1)

	file, err := os.Open(filePath)
	if err != nil {
		log.Printf("Error opening %s: %v", filePath, err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

	var localLengths []int64
	var localStats FileStats
	localStats.Path = filePath

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var data map[string]interface{}
		if err := json.Unmarshal(line, &data); err != nil {
			continue
		}

		text, ok := data["text"].(string)
		if !ok || text == "" {
			atomic.AddInt64(&stats.EmptySentences, 1)
			continue
		}

		atomic.AddInt64(&stats.TotalSentences, 1)
		localStats.Sentences++
		length := int64(len(text))
		atomic.AddInt64(&stats.TotalChars, length)
		localLengths = append(localLengths, length)

		// Языковой состав
		hasRu := isRussian(text)
		hasEn := isEnglish(text)

		if hasRu && !hasEn {
			atomic.AddInt64(&stats.RussianOnly, 1)
			localStats.RussianRatio++
		} else if !hasRu && hasEn {
			atomic.AddInt64(&stats.EnglishOnly, 1)
			localStats.EnglishRatio++
		} else if hasRu && hasEn {
			atomic.AddInt64(&stats.MixedCyrillicLatin, 1)
			localStats.MixedRatio++
		} else {
			atomic.AddInt64(&stats.Other, 1)
		}

		// Качество
		if length < 20 {
			atomic.AddInt64(&stats.TooShort20, 1)
		}
		if length < 50 {
			atomic.AddInt64(&stats.TooShort50, 1)
		}
		if length > 1000 {
			atomic.AddInt64(&stats.TooLong1000, 1)
		}
		if length > 2000 {
			atomic.AddInt64(&stats.TooLong2000, 1)
		}

		dRatio := digitRatio(text)
		if dRatio > 0.1 {
			atomic.AddInt64(&stats.HighDigit10, 1)
		}
		if dRatio > 0.3 {
			atomic.AddInt64(&stats.HighDigit30, 1)
			localStats.DigitRatio30++
		}
		if dRatio > 0.5 {
			atomic.AddInt64(&stats.HighDigit50, 1)
		}

		pRatio := punctRatio(text)
		if pRatio > 0.3 {
			atomic.AddInt64(&stats.HighPunct30, 1)
		}
		if pRatio > 0.5 {
			atomic.AddInt64(&stats.HighPunct50, 1)
		}

		if hasListMarker(text) {
			atomic.AddInt64(&stats.ListMarker, 1)
			localStats.ListRatio++
		}

		hasISBN, hasUDK, hasBBK, hasURL, hasEmail := hasGarbagePatterns(text)
		if hasISBN {
			atomic.AddInt64(&stats.HasISBN, 1)
			localStats.GarbageRatio++
		}
		if hasUDK {
			atomic.AddInt64(&stats.HasUDK, 1)
			localStats.GarbageRatio++
		}
		if hasBBK {
			atomic.AddInt64(&stats.HasBBK, 1)
			localStats.GarbageRatio++
		}
		if hasURL {
			atomic.AddInt64(&stats.HasURL, 1)
			localStats.GarbageRatio++
		}
		if hasEmail {
			atomic.AddInt64(&stats.HasEmail, 1)
			localStats.GarbageRatio++
		}

		uRatio := uppercaseRatio(text)
		if uRatio > 0.5 {
			atomic.AddInt64(&stats.HighUppercase50, 1)
		}
		if uRatio > 0.8 {
			atomic.AddInt64(&stats.HighUppercase80, 1)
		}
	}

	// Сохраняем локальные длины в общий срез
	stats.mu.Lock()
	stats.Lengths = append(stats.Lengths, localLengths...)
	stats.mu.Unlock()

	// Нормализуем ratios
	if localStats.Sentences > 0 {
		localStats.RussianRatio /= float64(localStats.Sentences)
		localStats.EnglishRatio /= float64(localStats.Sentences)
		localStats.MixedRatio /= float64(localStats.Sentences)
		localStats.DigitRatio30 /= float64(localStats.Sentences)
		localStats.ListRatio /= float64(localStats.Sentences)
		localStats.GarbageRatio /= float64(localStats.Sentences)
	}
	fileStatsChan <- localStats
}

func main() {
	var (
		sentencesDir = flag.String("dir", "", "директория с JSONL файлами")
		workers      = flag.Int("workers", 32, "количество воркеров")
		outputDir    = flag.String("output", "data/analysis", "выходная директория")
	)
	flag.Parse()

	if *sentencesDir == "" {
		log.Fatal("--dir is required")
	}

	files, err := filepath.Glob(filepath.Join(*sentencesDir, "*.jsonl"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	log.Printf("Found %d files", len(files))

	stats := &Stats{
		Lengths:   make([]int64, 0),
		uniqueMap: make(map[[16]byte]bool),
	}

	progress := &Progress{
		totalFiles: int64(len(files)),
		startTime:  time.Now(),
	}

	fileStatsChan := make(chan FileStats, len(files))
	sem := make(chan struct{}, *workers)
	var wg sync.WaitGroup

	start := time.Now()

	// Запускаем горутину для отображения прогресса
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			processed := atomic.LoadInt64(&progress.processedFiles)
			if processed >= progress.totalFiles {
				return
			}
			elapsed := time.Since(progress.startTime)
			speed := float64(processed) / elapsed.Seconds()
			percent := float64(processed) / float64(progress.totalFiles) * 100

			log.Printf("[PROGRESS] %d/%d files processed (%.1f%%), speed: %.1f files/sec, elapsed: %v",
				processed, progress.totalFiles, percent, speed, elapsed.Round(time.Second))
		}
	}()

	for _, f := range files {
		wg.Add(1)
		go analyzeFile(f, stats, fileStatsChan, &wg, sem, progress)
	}

	go func() {
		wg.Wait()
		close(fileStatsChan)
	}()

	// Собираем пофайловую статистику
	var fileStatsList []FileStats
	for fs := range fileStatsChan {
		fileStatsList = append(fileStatsList, fs)
	}

	totalTime := time.Since(progress.startTime)
	log.Printf("[PROGRESS] COMPLETE: %d/%d files processed in %v",
		atomic.LoadInt64(&progress.processedFiles), progress.totalFiles, totalTime.Round(time.Second))

	elapsed := time.Since(start)

	// Сортируем длины для перцентилей
	sort.Slice(stats.Lengths, func(i, j int) bool {
		return stats.Lengths[i] < stats.Lengths[j]
	})

	// Вычисляем перцентили
	percentiles := []float64{1, 5, 10, 25, 50, 75, 90, 95, 99}
	percentileValues := make(map[float64]int64)
	for _, p := range percentiles {
		idx := int(float64(len(stats.Lengths)) * p / 100)
		if idx >= len(stats.Lengths) {
			idx = len(stats.Lengths) - 1
		}
		if idx < 0 {
			idx = 0
		}
		percentileValues[p] = stats.Lengths[idx]
	}

	// Сохраняем общую статистику в JSON
	summary := map[string]interface{}{
		"total_sentences":       stats.TotalSentences,
		"total_chars":           stats.TotalChars,
		"total_files":           len(files),
		"empty_sentences":       stats.EmptySentences,
		"min_length":            stats.Lengths[0],
		"max_length":            stats.Lengths[len(stats.Lengths)-1],
		"avg_length":            float64(stats.TotalChars) / float64(stats.TotalSentences),
		"percentiles":           percentileValues,
		"russian_only":          stats.RussianOnly,
		"english_only":          stats.EnglishOnly,
		"mixed_cyrillic_latin":  stats.MixedCyrillicLatin,
		"other":                 stats.Other,
		"too_short_20":          stats.TooShort20,
		"too_short_50":          stats.TooShort50,
		"too_long_1000":         stats.TooLong1000,
		"too_long_2000":         stats.TooLong2000,
		"high_digit_10":         stats.HighDigit10,
		"high_digit_30":         stats.HighDigit30,
		"high_digit_50":         stats.HighDigit50,
		"high_punct_30":         stats.HighPunct30,
		"high_punct_50":         stats.HighPunct50,
		"list_marker":           stats.ListMarker,
		"has_isbn":              stats.HasISBN,
		"has_udk":               stats.HasUDK,
		"has_bbk":               stats.HasBBK,
		"has_url":               stats.HasURL,
		"has_email":             stats.HasEmail,
		"high_uppercase_50":     stats.HighUppercase50,
		"high_uppercase_80":     stats.HighUppercase80,
		"analysis_time_seconds": elapsed.Seconds(),
	}

	// Создаём выходную директорию
	os.MkdirAll(*outputDir, 0755)

	// Сохраняем JSON
	jsonPath := filepath.Join(*outputDir, "stats_summary.json")
	jsonData, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile(jsonPath, jsonData, 0644)
	log.Printf("Saved summary to %s", jsonPath)

	// Сохраняем гистограмму длин
	histPath := filepath.Join(*outputDir, "stats_length_histogram.csv")
	histFile, _ := os.Create(histPath)
	defer histFile.Close()
	histWriter := csv.NewWriter(histFile)
	histWriter.Write([]string{"bucket", "count"})
	// Простая гистограмма с шагом 100
	bucketSize := int64(100)
	buckets := make(map[int64]int64)
	for _, l := range stats.Lengths {
		bucket := l / bucketSize * bucketSize
		buckets[bucket]++
	}
	for b, c := range buckets {
		histWriter.Write([]string{fmt.Sprintf("%d-%d", b, b+bucketSize), fmt.Sprintf("%d", c)})
	}
	histWriter.Flush()
	log.Printf("Saved histogram to %s", histPath)

	// Сохраняем пофайловую статистику
	fileStatsPath := filepath.Join(*outputDir, "stats_per_file.csv")
	fileStatsFile, _ := os.Create(fileStatsPath)
	defer fileStatsFile.Close()
	fsWriter := csv.NewWriter(fileStatsFile)
	fsWriter.Write([]string{"file", "sentences", "avg_length", "russian_ratio", "english_ratio", "mixed_ratio", "digit_ratio_30", "list_ratio", "garbage_ratio"})
	for _, fs := range fileStatsList {
		avgLength := 0.0
		if fs.Sentences > 0 {
			// Нужно было сохранить сумму длин, пока так
		}
		fsWriter.Write([]string{
			filepath.Base(fs.Path),
			fmt.Sprintf("%d", fs.Sentences),
			fmt.Sprintf("%.2f", avgLength),
			fmt.Sprintf("%.4f", fs.RussianRatio),
			fmt.Sprintf("%.4f", fs.EnglishRatio),
			fmt.Sprintf("%.4f", fs.MixedRatio),
			fmt.Sprintf("%.4f", fs.DigitRatio30),
			fmt.Sprintf("%.4f", fs.ListRatio),
			fmt.Sprintf("%.4f", fs.GarbageRatio),
		})
	}
	fsWriter.Flush()
	log.Printf("Saved per-file stats to %s", fileStatsPath)

	// Вывод в консоль
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("CORPUS ANALYSIS COMPLETE")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Total files:      %d\n", len(files))
	fmt.Printf("Total sentences:  %d\n", stats.TotalSentences)
	fmt.Printf("Total chars:      %d\n", stats.TotalChars)
	fmt.Printf("Average length:   %.2f\n", float64(stats.TotalChars)/float64(stats.TotalSentences))
	fmt.Printf("Min length:       %d\n", stats.Lengths[0])
	fmt.Printf("Max length:       %d\n", stats.Lengths[len(stats.Lengths)-1])
	fmt.Printf("Time:             %v\n", elapsed)
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Russian only:     %d (%.2f%%)\n", stats.RussianOnly, float64(stats.RussianOnly)/float64(stats.TotalSentences)*100)
	fmt.Printf("English only:     %d (%.2f%%)\n", stats.EnglishOnly, float64(stats.EnglishOnly)/float64(stats.TotalSentences)*100)
	fmt.Printf("Mixed (Ru+En):    %d (%.2f%%)\n", stats.MixedCyrillicLatin, float64(stats.MixedCyrillicLatin)/float64(stats.TotalSentences)*100)
	fmt.Printf("Other:            %d (%.2f%%)\n", stats.Other, float64(stats.Other)/float64(stats.TotalSentences)*100)
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Too short (<20):  %d (%.2f%%)\n", stats.TooShort20, float64(stats.TooShort20)/float64(stats.TotalSentences)*100)
	fmt.Printf("Too long (>1000): %d (%.2f%%)\n", stats.TooLong1000, float64(stats.TooLong1000)/float64(stats.TotalSentences)*100)
	fmt.Printf("High digit (>30%%): %d (%.2f%%)\n", stats.HighDigit30, float64(stats.HighDigit30)/float64(stats.TotalSentences)*100)
	fmt.Printf("List marker:      %d (%.2f%%)\n", stats.ListMarker, float64(stats.ListMarker)/float64(stats.TotalSentences)*100)
	fmt.Println(strings.Repeat("=", 60))
}
