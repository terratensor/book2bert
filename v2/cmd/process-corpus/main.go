package main

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/terratensor/book2bert/v2/pkg/adapters/filerepo"
	segmenterAdapter "github.com/terratensor/book2bert/v2/pkg/adapters/segmenter"
	"github.com/terratensor/book2bert/v2/pkg/core/book"
	"github.com/terratensor/book2bert/v2/pkg/core/segmenter"
	"github.com/terratensor/book2bert/v2/pkg/textutils"
)

// BookMeta метаданные книги (отдельный файл)
type BookMeta struct {
	BookID     string `json:"book_id"`
	Title      string `json:"title,omitempty"`
	Author     string `json:"author,omitempty"`
	Genre      string `json:"genre,omitempty"`
	SourceFile string `json:"source_file"`
}

// BlockType тип блока текста
type BlockType int

const (
	NormalText BlockType = iota
	Table
	List
)

// detectBlockType определяет тип блока текста
func detectBlockType(text string) BlockType {
	lines := strings.Split(text, "\n")
	if len(lines) == 0 {
		return NormalText
	}

	var (
		shortLines      int
		digitLines      int
		listMarkerLines int
	)

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		runes := []rune(line)
		if len(runes) < 30 {
			shortLines++
		}

		digits := 0
		for _, r := range runes {
			if r >= '0' && r <= '9' {
				digits++
			}
		}
		if float64(digits)/float64(len(runes)) > 0.3 {
			digitLines++
		}

		if strings.HasPrefix(line, "•") || strings.HasPrefix(line, "-") ||
			(len(line) > 1 && line[0] >= '0' && line[0] <= '9' && (line[1] == '.' || line[1] == ')')) {
			listMarkerLines++
		}
	}

	total := len(lines)

	if float64(listMarkerLines)/float64(total) > 0.5 {
		return List
	}
	if float64(digitLines)/float64(total) > 0.5 {
		return Table
	}
	if float64(shortLines)/float64(total) > 0.5 {
		return Table
	}

	return NormalText
}

// processBlock обрабатывает блок в зависимости от типа
func processBlock(blockType BlockType, text string, seg segmenter.Segmenter, ctx context.Context) ([]string, error) {
	switch blockType {
	case NormalText:
		return seg.Segment(ctx, text)
	case Table:
		return []string{text}, nil
	case List:
		lines := strings.Split(text, "\n")
		var sentences []string
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line != "" {
				sentences = append(sentences, line)
			}
		}
		return sentences, nil
	default:
		return []string{text}, nil
	}
}

// parseFilename извлекает метаданные из имени файла
func parseFilename(filename string) (genre, author, title string) {
	basename := strings.TrimSuffix(filename, ".txt.gz")
	basename = strings.TrimSuffix(basename, ".txt")

	if idx := strings.Index(basename, " — "); idx != -1 {
		parts := strings.SplitN(basename, " — ", 2)
		if len(parts) == 2 {
			left := parts[0]
			right := parts[1]
			if underscoreIdx := strings.Index(left, "_"); underscoreIdx != -1 {
				genre = left[:underscoreIdx]
				author = left[underscoreIdx+1:]
			} else {
				author = left
			}
			title = right
			return
		}
	}

	author = "Unknown"
	title = basename
	return
}

// processFile обрабатывает один файл
func processFile(
	filePath string,
	seg segmenter.Segmenter,
	repo book.Repository,
	metaChan chan<- BookMeta,
	stats *ProcessStats,
	wg *sync.WaitGroup,
	sem chan struct{},
) {
	defer wg.Done()
	sem <- struct{}{}
	defer func() { <-sem }()

	file, err := os.Open(filePath)
	if err != nil {
		log.Printf("[ERROR] open %s: %v", filePath, err)
		atomic.AddInt64(&stats.Errors, 1)
		return
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		log.Printf("[ERROR] gzip %s: %v", filePath, err)
		atomic.AddInt64(&stats.Errors, 1)
		return
	}
	defer gzReader.Close()

	content, err := io.ReadAll(gzReader)
	if err != nil {
		log.Printf("[ERROR] read %s: %v", filePath, err)
		atomic.AddInt64(&stats.Errors, 1)
		return
	}

	text, err := textutils.ToUTF8(content)
	if err != nil {
		text = string(content)
	}
	text = textutils.NormalizeText(text)

	genre, author, title := parseFilename(filepath.Base(filePath))
	if title == "" {
		title = strings.TrimSuffix(filepath.Base(filePath), ".txt.gz")
		title = strings.TrimSuffix(title, ".txt")
	}

	bookID := uuid.New().String()

	// Отправляем метаданные (omitempty уберёт пустые поля)
	metaChan <- BookMeta{
		BookID:     bookID,
		Title:      title,
		Author:     author,
		Genre:      genre,
		SourceFile: filePath,
	}

	blocks := strings.Split(text, "\n\n")
	ctx := context.Background()

	var allSentences []string

	for _, block := range blocks {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}

		blockType := detectBlockType(block)
		sentences, err := processBlock(blockType, block, seg, ctx)
		if err != nil {
			log.Printf("[ERROR] process block in %s: %v", filePath, err)
			continue
		}

		allSentences = append(allSentences, sentences...)
	}

	if len(allSentences) == 0 {
		atomic.AddInt64(&stats.Skipped, 1)
		return
	}

	// Сохраняем предложения (без метаданных)
	bookSentences := make([]book.Sentence, len(allSentences))
	for i, text := range allSentences {
		bookSentences[i] = book.Sentence{
			BookID:    bookID,
			Text:      text,
			Position:  i,
			CreatedAt: time.Now(),
		}
	}

	if err := repo.SaveSentences(context.Background(), bookSentences); err != nil {
		log.Printf("[ERROR] save sentences %s: %v", filePath, err)
		atomic.AddInt64(&stats.Errors, 1)
		return
	}

	atomic.AddInt64(&stats.Files, 1)
	atomic.AddInt64(&stats.Sentences, int64(len(allSentences)))
	log.Printf("[OK] %s | genre=%s author=%s sentences=%d", filepath.Base(filePath), genre, author, len(allSentences))
}

type ProcessStats struct {
	Files     int64
	Sentences int64
	Errors    int64
	Skipped   int64
}

func main() {
	var (
		corpusDir    = flag.String("corpus", "", "директория с txt/txt.gz файлами")
		outputDir    = flag.String("output", "", "выходная директория")
		segmenterURL = flag.String("segmenter", "http://localhost:8090", "URL сервиса сегментации")
		workers      = flag.Int("workers", 10, "количество воркеров")
	)
	flag.Parse()

	if *corpusDir == "" || *outputDir == "" {
		log.Fatal("--corpus and --output are required")
	}

	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}

	// Файл метаданных
	metaFile, err := os.Create(filepath.Join(*outputDir, "books_meta.jsonl"))
	if err != nil {
		log.Fatalf("create meta file: %v", err)
	}
	defer metaFile.Close()
	metaWriter := bufio.NewWriter(metaFile)
	defer metaWriter.Flush()

	metaChan := make(chan BookMeta, 100)

	go func() {
		for meta := range metaChan {
			data, err := json.Marshal(meta)
			if err != nil {
				log.Printf("ERROR marshalling meta: %v", err)
				continue
			}
			metaWriter.Write(data)
			metaWriter.Write([]byte("\n"))
		}
	}()

	repo, err := filerepo.NewJSONLRepository(*outputDir)
	if err != nil {
		log.Fatalf("create repository: %v", err)
	}
	defer repo.Close()

	seg := segmenterAdapter.NewHTTPClient(*segmenterURL, 120*time.Second)

	files, err := filepath.Glob(filepath.Join(*corpusDir, "*.txt.gz"))
	if err != nil {
		log.Fatalf("glob: %v", err)
	}
	log.Printf("Found %d files", len(files))

	stats := &ProcessStats{}
	sem := make(chan struct{}, *workers)
	var wg sync.WaitGroup

	start := time.Now()

	go func() {
		ticker := time.NewTicker(10 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			filesDone := atomic.LoadInt64(&stats.Files)
			sentencesDone := atomic.LoadInt64(&stats.Sentences)
			if filesDone == 0 {
				continue
			}
			percent := float64(filesDone) / float64(len(files)) * 100
			log.Printf("[PROGRESS] %d/%d files (%.1f%%), sentences: %d", filesDone, len(files), percent, sentencesDone)
		}
	}()

	for _, f := range files {
		wg.Add(1)
		go processFile(f, seg, repo, metaChan, stats, &wg, sem)
	}

	wg.Wait()
	close(metaChan)

	elapsed := time.Since(start)

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("PROCESSING COMPLETE")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Time:        %v\n", elapsed.Round(time.Second))
	fmt.Printf("Files:       %d\n", stats.Files)
	fmt.Printf("Sentences:   %d\n", stats.Sentences)
	fmt.Printf("Errors:      %d\n", stats.Errors)
	fmt.Printf("Skipped:     %d\n", stats.Skipped)
	fmt.Println(strings.Repeat("=", 60))
}
