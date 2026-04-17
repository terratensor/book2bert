package main

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
)

// SafeWriter оборачивает gzip.Writer с мьютексом
type SafeWriter struct {
	writer *gzip.Writer
	mu     sync.Mutex
}

func (sw *SafeWriter) Write(data []byte) (int, error) {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	return sw.writer.Write(data)
}

func (sw *SafeWriter) Close() error {
	sw.mu.Lock()
	defer sw.mu.Unlock()
	return sw.writer.Close()
}

// FilterStats хранит статистику фильтрации
type FilterStats struct {
	TotalFiles         int64
	TotalSentences     int64
	KeptSentences      int64
	ProcessedSentences int64 // для round-robin распределения
	FilteredOut        map[string]int64
	mu                 sync.Mutex
}

var (
	isbnRegex       = regexp.MustCompile(`\bISBN\s*\d{3}-\d{1,5}-\d{1,7}-\d{1,7}-\d{1,7}\b`)
	udkRegex        = regexp.MustCompile(`\bУДК\s*\d+(?:\.\d+)+\b`)
	bbkRegex        = regexp.MustCompile(`\bББК\s*\d+(?:\.\d+)+\b`)
	urlRegex        = regexp.MustCompile(`https?://[^\s]+`)
	emailRegex      = regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)
	listMarkerRegex = regexp.MustCompile(`^\s*[•\-*•\d]+[\.\)]\s+`)
)

func isRussian(text string) bool {
	for _, r := range text {
		if r >= 0x0400 && r <= 0x04FF {
			return true
		}
	}
	return false
}

func minLengthFilter(text string) bool {
	return len([]rune(text)) >= 20
}

func maxLengthFilter(text string) bool {
	return len([]rune(text)) <= 2000
}

func languageFilter(text string) bool {
	return isRussian(text)
}

func listMarkerFilter(text string) bool {
	return !listMarkerRegex.MatchString(text)
}

func garbageFilter(text string) bool {
	if isbnRegex.MatchString(text) {
		return false
	}
	if udkRegex.MatchString(text) {
		return false
	}
	if bbkRegex.MatchString(text) {
		return false
	}
	if urlRegex.MatchString(text) {
		return false
	}
	if emailRegex.MatchString(text) {
		return false
	}
	return true
}

func digitRatioFilter(text string) bool {
	runes := []rune(text)
	if len(runes) == 0 {
		return false
	}
	digits := 0
	for _, r := range runes {
		if r >= '0' && r <= '9' {
			digits++
		}
	}
	return float64(digits)/float64(len(runes)) <= 0.3
}

func punctRatioFilter(text string) bool {
	runes := []rune(text)
	if len(runes) == 0 {
		return false
	}
	punct := 0
	for _, r := range runes {
		if unicode.IsPunct(r) {
			punct++
		}
	}
	return float64(punct)/float64(len(runes)) <= 0.5
}

func uppercaseRatioFilter(text string) bool {
	runes := []rune(text)
	if len(runes) == 0 {
		return false
	}
	upper := 0
	letters := 0
	for _, r := range runes {
		if unicode.IsLetter(r) {
			letters++
			if unicode.IsUpper(r) {
				upper++
			}
		}
	}
	if letters == 0 {
		return true
	}
	return float64(upper)/float64(letters) <= 0.8
}

func isGoodSentence(text string, stats *FilterStats) bool {
	filters := []struct {
		name   string
		filter func(string) bool
	}{
		{"min_length", minLengthFilter},
		{"max_length", maxLengthFilter},
		{"language", languageFilter},
		{"list_marker", listMarkerFilter},
		{"garbage", garbageFilter},
		{"digit_ratio", digitRatioFilter},
		{"punct_ratio", punctRatioFilter},
		{"uppercase_ratio", uppercaseRatioFilter},
	}

	for _, f := range filters {
		if !f.filter(text) {
			stats.mu.Lock()
			stats.FilteredOut[f.name]++
			stats.mu.Unlock()
			return false
		}
	}
	return true
}

func ProcessFile(inputPath string, outputWriters []*SafeWriter, stats *FilterStats, wg *sync.WaitGroup, sem chan struct{}) {
	defer wg.Done()
	sem <- struct{}{}
	defer func() { <-sem }()

	file, err := os.Open(inputPath)
	if err != nil {
		log.Printf("Error opening %s: %v", inputPath, err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

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
			continue
		}

		// Увеличиваем общий счётчик предложений
		atomic.AddInt64(&stats.TotalSentences, 1)

		if isGoodSentence(text, stats) {
			// Используем отдельный счётчик для round-robin распределения
			idx := atomic.AddInt64(&stats.ProcessedSentences, 1) % int64(len(outputWriters))
			writer := outputWriters[idx]
			dataBytes, err := json.Marshal(data)
			if err != nil {
				continue
			}
			writer.Write(dataBytes)
			writer.Write([]byte("\n"))
			atomic.AddInt64(&stats.KeptSentences, 1)
		}
	}

	atomic.AddInt64(&stats.TotalFiles, 1)
}

func main() {
	var (
		inputDir   = flag.String("input", "", "входная директория с JSONL файлами")
		outputDir  = flag.String("output", "", "выходная директория для отфильтрованных файлов")
		workers    = flag.Int("workers", 32, "количество параллельных воркеров")
		numOutputs = flag.Int("num-outputs", 10000, "количество выходных файлов")
	)
	flag.Parse()

	if *inputDir == "" || *outputDir == "" {
		log.Fatal("--input and --output are required")
	}

	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output dir: %v", err)
	}

	// Создаём выходные writers
	outputWriters := make([]*SafeWriter, *numOutputs)
	outputFiles := make([]*os.File, *numOutputs)
	for i := 0; i < *numOutputs; i++ {
		filename := filepath.Join(*outputDir, fmt.Sprintf("part_%05d.jsonl.gz", i))
		f, err := os.Create(filename)
		if err != nil {
			log.Fatalf("Failed to create %s: %v", filename, err)
		}
		outputFiles[i] = f
		outputWriters[i] = &SafeWriter{
			writer: gzip.NewWriter(f),
		}
	}
	defer func() {
		for i := 0; i < *numOutputs; i++ {
			outputWriters[i].Close()
			outputFiles[i].Close()
		}
	}()

	inputFiles, err := filepath.Glob(filepath.Join(*inputDir, "*.jsonl"))
	if err != nil {
		log.Fatalf("Glob failed: %v", err)
	}
	log.Printf("Found %d input files", len(inputFiles))

	stats := &FilterStats{
		FilteredOut: make(map[string]int64),
	}

	sem := make(chan struct{}, *workers)
	var wg sync.WaitGroup

	start := time.Now()

	// Прогресс-бар
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			processed := atomic.LoadInt64(&stats.TotalFiles)
			kept := atomic.LoadInt64(&stats.KeptSentences)
			total := atomic.LoadInt64(&stats.TotalSentences)
			if total == 0 {
				continue
			}
			percent := float64(processed) / float64(len(inputFiles)) * 100
			log.Printf("[PROGRESS] %d/%d files (%.1f%%), kept: %d/%d sentences (%.1f%%)",
				processed, len(inputFiles), percent, kept, total, float64(kept)/float64(total)*100)
		}
	}()

	for _, f := range inputFiles {
		wg.Add(1)
		go ProcessFile(f, outputWriters, stats, &wg, sem)
	}

	wg.Wait()

	elapsed := time.Since(start)

	// Вывод статистики
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("FILTERING COMPLETE")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Time:           %v\n", elapsed.Round(time.Second))
	fmt.Printf("Input files:    %d\n", stats.TotalFiles)
	fmt.Printf("Total sentences: %d\n", stats.TotalSentences)
	fmt.Printf("Kept sentences:  %d (%.2f%%)\n", stats.KeptSentences, float64(stats.KeptSentences)/float64(stats.TotalSentences)*100)
	fmt.Printf("Filtered out:    %d (%.2f%%)\n", stats.TotalSentences-stats.KeptSentences, float64(stats.TotalSentences-stats.KeptSentences)/float64(stats.TotalSentences)*100)
	fmt.Println(strings.Repeat("-", 60))
	fmt.Println("Filter reasons:")
	// Сортируем для красивого вывода
	for name, count := range stats.FilteredOut {
		fmt.Printf("  %-15s: %d (%.2f%%)\n", name, count, float64(count)/float64(stats.TotalSentences)*100)
	}
	fmt.Println(strings.Repeat("=", 60))
}
