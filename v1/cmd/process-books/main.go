package main

import (
	"context"
	"flag"
	"log"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/terratensor/book2bert/pkg/adapters/filerepo"
	"github.com/terratensor/book2bert/pkg/app"
	"github.com/terratensor/book2bert/pkg/core/book"
	"github.com/terratensor/book2bert/pkg/textutils"
	segmenterAdapter "github.com/terratensor/book2bert/v1/pkg/adapters/segmenter"
)

var (
	booksDir     = flag.String("books", "data/sample/texts", "директория с txt файлами книг")
	outputDir    = flag.String("output", "data/processed/sentences", "директория для сохранения предложений")
	segmenterURL = flag.String("segmenter", "http://localhost:8090", "URL сервиса сегментации")
)

func main() {
	flag.Parse()

	// 1. Создаем репозиторий
	repo, err := filerepo.NewJSONLRepository(*outputDir)
	if err != nil {
		log.Fatalf("create repository: %v", err)
	}
	defer repo.Close()

	// 2. Создаем клиент сегментатора
	seg := segmenterAdapter.NewHTTPClient(*segmenterURL, 60*time.Second)

	// 3. Создаем use case
	uc := app.NewProcessBooksUseCase(seg, repo)

	// 4. Читаем все txt файлы из директории
	files, err := filepath.Glob(filepath.Join(*booksDir, "*.txt"))
	if err != nil {
		log.Fatalf("glob books: %v", err)
	}

	if len(files) == 0 {
		log.Fatalf("no txt files found in %s", *booksDir)
	}

	log.Printf("Found %d files to process", len(files))

	ctx := context.Background()
	stats := &processStats{}

	for i, file := range files {
		log.Printf("[%d/%d] Processing %s", i+1, len(files), filepath.Base(file))

		// Читаем файл с автоопределением кодировки
		text, err := textutils.ReadFileWithEncoding(file)
		if err != nil {
			log.Printf("  ERROR reading file: %v", err)
			stats.errors++
			continue
		}

		// Пропускаем пустые файлы
		if len(strings.TrimSpace(text)) == 0 {
			log.Printf("  WARNING: empty file, skipping")
			stats.skipped++
			continue
		}

		// Извлекаем название из имени файла (без расширения)
		baseName := strings.TrimSuffix(filepath.Base(file), ".txt")

		// Создаем книгу
		b := &book.Book{
			ID:     uuid.New().String(),
			Title:  baseName,
			Author: "Unknown",
			Genre:  "Unknown",
			Source: file,
			Text:   text,
		}

		// Обрабатываем
		if err := uc.Process(ctx, b); err != nil {
			log.Printf("  ERROR processing: %v", err)
			stats.errors++
			continue
		}

		// Подсчитываем предложения (примерно)
		sentenceCount := strings.Count(text, ".") + strings.Count(text, "!") + strings.Count(text, "?")
		stats.processed++
		stats.totalSentences += sentenceCount

		log.Printf("  OK: processed, ~%d sentences", sentenceCount)
	}

	// Выводим статистику
	log.Printf("=== Summary ===")
	log.Printf("Processed: %d files", stats.processed)
	log.Printf("Errors: %d", stats.errors)
	log.Printf("Skipped: %d", stats.skipped)
	log.Printf("Total sentences (approx): %d", stats.totalSentences)
	log.Println("Done")
}

type processStats struct {
	processed      int
	errors         int
	skipped        int
	totalSentences int
}
