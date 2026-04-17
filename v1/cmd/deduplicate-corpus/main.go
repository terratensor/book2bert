package main

import (
	"bufio"
	"compress/gzip"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

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

type DedupStats struct {
	TotalFiles     int64
	TotalSentences int64
	KeptSentences  int64
	Duplicates     int64
}

func processFile(inputPath string, seen map[string]bool, seenMu *sync.Mutex, outputWriters []*SafeWriter, stats *DedupStats, wg *sync.WaitGroup, sem chan struct{}) {
	defer wg.Done()
	sem <- struct{}{}
	defer func() { <-sem }()

	file, err := os.Open(inputPath)
	if err != nil {
		log.Printf("Error opening %s: %v", inputPath, err)
		return
	}
	defer file.Close()

	var scanner *bufio.Scanner
	if strings.HasSuffix(inputPath, ".gz") {
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			log.Printf("Error creating gzip reader for %s: %v", inputPath, err)
			return
		}
		defer gzReader.Close()
		scanner = bufio.NewScanner(gzReader)
	} else {
		scanner = bufio.NewScanner(file)
	}

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

		atomic.AddInt64(&stats.TotalSentences, 1)

		// Вычисляем MD5 хэш текста
		hash := md5.Sum([]byte(text))
		hashStr := hex.EncodeToString(hash[:])

		seenMu.Lock()
		if seen[hashStr] {
			seenMu.Unlock()
			atomic.AddInt64(&stats.Duplicates, 1)
			continue
		}
		seen[hashStr] = true
		seenMu.Unlock()

		// Сохраняем предложение
		idx := atomic.AddInt64(&stats.KeptSentences, 1) % int64(len(outputWriters))
		writer := outputWriters[idx]
		dataBytes, _ := json.Marshal(data)
		writer.Write(dataBytes)
		writer.Write([]byte("\n"))
	}

	atomic.AddInt64(&stats.TotalFiles, 1)
}

func main() {
	var (
		inputDir   = flag.String("input", "", "входная директория с JSONL.gz файлами")
		outputDir  = flag.String("output", "", "выходная директория для дедуплицированных файлов")
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

	inputFiles, err := filepath.Glob(filepath.Join(*inputDir, "*.jsonl.gz"))
	if err != nil {
		log.Fatalf("Glob failed: %v", err)
	}
	log.Printf("Found %d input files", len(inputFiles))

	seen := make(map[string]bool)
	var seenMu sync.Mutex
	stats := &DedupStats{}

	sem := make(chan struct{}, *workers)
	var wg sync.WaitGroup

	start := time.Now()

	// Прогресс-бар
	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			processed := atomic.LoadInt64(&stats.TotalFiles)
			total := atomic.LoadInt64(&stats.TotalSentences)
			kept := atomic.LoadInt64(&stats.KeptSentences)
			duplicates := atomic.LoadInt64(&stats.Duplicates)
			if total == 0 {
				continue
			}
			percent := float64(processed) / float64(len(inputFiles)) * 100
			log.Printf("[PROGRESS] %d/%d files (%.1f%%), total: %d, kept: %d, dup: %d (%.1f%%)",
				processed, len(inputFiles), percent, total, kept, duplicates, float64(duplicates)/float64(total)*100)
		}
	}()

	for _, f := range inputFiles {
		wg.Add(1)
		go processFile(f, seen, &seenMu, outputWriters, stats, &wg, sem)
	}

	wg.Wait()

	elapsed := time.Since(start)

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("DEDUPLICATION COMPLETE")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Time:            %v\n", elapsed.Round(time.Second))
	fmt.Printf("Input files:     %d\n", stats.TotalFiles)
	fmt.Printf("Total sentences: %d\n", stats.TotalSentences)
	fmt.Printf("Kept sentences:  %d (%.2f%%)\n", stats.KeptSentences, float64(stats.KeptSentences)/float64(stats.TotalSentences)*100)
	fmt.Printf("Duplicates:      %d (%.2f%%)\n", stats.Duplicates, float64(stats.Duplicates)/float64(stats.TotalSentences)*100)
	fmt.Println(strings.Repeat("=", 60))
}
