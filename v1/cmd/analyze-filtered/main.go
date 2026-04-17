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
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Stats struct {
	TotalSentences int64
	TotalChars     int64
	Lengths        []int64
	mu             sync.Mutex
}

func analyzeFile(filePath string, stats *Stats, wg *sync.WaitGroup, sem chan struct{}) {
	defer wg.Done()
	sem <- struct{}{}
	defer func() { <-sem }()

	file, err := os.Open(filePath)
	if err != nil {
		log.Printf("Error opening %s: %v", filePath, err)
		return
	}
	defer file.Close()

	var scanner *bufio.Scanner
	if strings.HasSuffix(filePath, ".gz") {
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			log.Printf("Error creating gzip reader for %s: %v", filePath, err)
			return
		}
		defer gzReader.Close()
		scanner = bufio.NewScanner(gzReader)
	} else {
		scanner = bufio.NewScanner(file)
	}

	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

	var localLengths []int64

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

		length := int64(len([]rune(text)))
		atomic.AddInt64(&stats.TotalSentences, 1)
		atomic.AddInt64(&stats.TotalChars, length)
		localLengths = append(localLengths, length)
	}

	stats.mu.Lock()
	stats.Lengths = append(stats.Lengths, localLengths...)
	stats.mu.Unlock()
}

func main() {
	var (
		inputDir = flag.String("input", "", "директория с JSONL.gz файлами")
		workers  = flag.Int("workers", 32, "количество воркеров")
	)
	flag.Parse()

	if *inputDir == "" {
		log.Fatal("--input is required")
	}

	files, err := filepath.Glob(filepath.Join(*inputDir, "*.jsonl.gz"))
	if err != nil {
		log.Fatalf("Glob failed: %v", err)
	}
	log.Printf("Found %d files", len(files))

	stats := &Stats{
		Lengths: make([]int64, 0),
	}

	sem := make(chan struct{}, *workers)
	var wg sync.WaitGroup

	start := time.Now()

	for _, f := range files {
		wg.Add(1)
		go analyzeFile(f, stats, &wg, sem)
	}

	wg.Wait()

	elapsed := time.Since(start)

	// Сортируем длины
	sort.Slice(stats.Lengths, func(i, j int) bool {
		return stats.Lengths[i] < stats.Lengths[j]
	})

	// Вычисляем перцентили
	percentiles := []float64{1, 5, 10, 25, 50, 75, 90, 95, 99}
	percentileValues := make(map[string]int64)
	for _, p := range percentiles {
		idx := int(float64(len(stats.Lengths)) * p / 100)
		if idx >= len(stats.Lengths) {
			idx = len(stats.Lengths) - 1
		}
		if idx < 0 {
			idx = 0
		}
		percentileValues[fmt.Sprintf("%.0f", p)] = stats.Lengths[idx]
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("FILTERED CORPUS ANALYSIS")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Files:           %d\n", len(files))
	fmt.Printf("Total sentences: %d\n", stats.TotalSentences)
	fmt.Printf("Total chars:     %d\n", stats.TotalChars)
	fmt.Printf("Avg length:      %.2f\n", float64(stats.TotalChars)/float64(stats.TotalSentences))
	fmt.Printf("Min length:      %d\n", stats.Lengths[0])
	fmt.Printf("Max length:      %d\n", stats.Lengths[len(stats.Lengths)-1])
	fmt.Printf("Time:            %v\n", elapsed.Round(time.Second))
	fmt.Println(strings.Repeat("-", 60))
	fmt.Println("Percentiles:")
	for p, val := range percentileValues {
		fmt.Printf("  %s%%: %d\n", p, val)
	}
	fmt.Println(strings.Repeat("=", 60))
}
