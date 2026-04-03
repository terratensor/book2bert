package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/terratensor/book2bert/pkg/textutils"
)

var (
	sentencesDir = flag.String("sentences", "", "директория с JSONL файлами предложений")
	outputFile   = flag.String("output", "corpus.txt", "выходной файл корпуса")
	workers      = flag.Int("workers", 8, "количество воркеров")
	filterCJK    = flag.Bool("filter-cjk", true, "фильтровать CJK/тайские символы")
)

type Sentence struct {
	Text string `json:"text"`
}

func processFile(filePath string, ch chan<- string, stats *int64) error {
	file, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	// Увеличиваем буфер для длинных строк
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		// Быстрый парсинг: ищем "text":" ... "
		// Но проще использовать json.Unmarshal
		var s Sentence
		if err := json.Unmarshal(line, &s); err != nil {
			continue
		}

		text := s.Text
		if *filterCJK {
			text = textutils.FilterCJKThai(text)
		}

		if len(strings.TrimSpace(text)) > 0 {
			ch <- text + "\n"
			atomic.AddInt64(stats, 1)
		}
	}

	return scanner.Err()
}

func main() {
	flag.Parse()

	if *sentencesDir == "" {
		log.Fatal("--sentences is required")
	}

	log.Printf("=== Corpus Builder (Go) ===")
	log.Printf("Sentences dir: %s", *sentencesDir)
	log.Printf("Output file: %s", *outputFile)
	log.Printf("Workers: %d", *workers)
	log.Printf("Filter CJK: %v", *filterCJK)

	// Находим все JSONL файлы
	files, err := filepath.Glob(filepath.Join(*sentencesDir, "*.jsonl"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	log.Printf("Found %d JSONL files", len(files))

	// Создаем выходной файл
	outFile, err := os.Create(*outputFile)
	if err != nil {
		log.Fatalf("create output: %v", err)
	}
	defer outFile.Close()

	writer := bufio.NewWriterSize(outFile, 1024*1024)
	defer writer.Flush()

	// Канал для строк
	ch := make(chan string, 10000)
	var wg sync.WaitGroup
	var totalSentences int64

	// Запускаем воркеров для записи
	go func() {
		for line := range ch {
			writer.WriteString(line)
		}
	}()

	// Запускаем воркеров для обработки файлов
	sem := make(chan struct{}, *workers)
	for _, file := range files {
		sem <- struct{}{}
		wg.Add(1)
		go func(f string) {
			defer func() { <-sem; wg.Done() }()
			if err := processFile(f, ch, &totalSentences); err != nil {
				log.Printf("ERROR processing %s: %v", f, err)
			}
		}(file)
	}

	wg.Wait()
	close(ch)
	writer.Flush()

	log.Printf("Done! Total sentences: %d", totalSentences)
	log.Printf("Corpus saved to: %s", *outputFile)
}
