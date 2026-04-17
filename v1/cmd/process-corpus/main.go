package main

import (
	"compress/gzip"
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/terratensor/book2bert/v1/pkg/adapters/filerepo"
	segmenterAdapter "github.com/terratensor/book2bert/v1/pkg/adapters/segmenter"
	"github.com/terratensor/book2bert/v1/pkg/app"
	"github.com/terratensor/book2bert/v1/pkg/core/book"
	"github.com/terratensor/book2bert/v1/pkg/core/segmenter"
	"github.com/terratensor/book2bert/v1/pkg/textutils"
)

var (
	corpusDir    = flag.String("corpus", "", "директория с txt/txt.gz файлами")
	outputDir    = flag.String("output", "data/processed/sentences", "директория для сохранения предложений")
	segmenterURL = flag.String("segmenter", "http://localhost:8090", "URL сервиса сегментации")
	workers      = flag.Int("workers", 5, "количество параллельных воркеров (горутин)")
	extensions   = flag.String("extensions", ".txt,.txt.gz", "расширения файлов для обработки (через запятую)")
)

// FileTask задача для воркера
type FileTask struct {
	Path     string
	Filename string
}

// ProcessStats статистика обработки
type ProcessStats struct {
	Total     int64
	Processed int64
	Errors    int64
	Skipped   int64
	Sentences int64
}

// parseMetadataFromFilename извлекает жанр, автора и название из имени файла
func parseMetadataFromFilename(filename string) (genre, author, title string) {
	basename := strings.TrimSuffix(filename, ".txt.gz")
	basename = strings.TrimSuffix(basename, ".txt")

	// Паттерн 1: {жанр}_{автор} — {название}
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

	// Паттерн 2: militera формат "Автор — Название, год"
	if idx := strings.Index(basename, " — "); idx != -1 {
		parts := strings.SplitN(basename, " — ", 2)
		if len(parts) == 2 {
			author = parts[0]
			title = parts[1]
			// Убираем год в конце (если есть)
			title = regexp.MustCompile(`,\s*\d{4}$`).ReplaceAllString(title, "")
			title = regexp.MustCompile(`\s*\(\d{4}\)$`).ReplaceAllString(title, "")
			return
		}
	}

	// Паттерн 3: просто имя файла
	author = "Unknown"
	title = basename
	return
}

// readGZFile читает содержимое .txt.gz файла
func readGZFile(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return "", err
	}
	defer gzReader.Close()

	data, err := io.ReadAll(gzReader)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

// readTXTFile читает содержимое .txt файла
func readTXTFile(filePath string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// processFile обрабатывает один файл
func processFile(ctx context.Context, task FileTask, seg segmenter.Segmenter, repo book.Repository, stats *ProcessStats, cjkLogFile *os.File) {
	defer atomic.AddInt64(&stats.Processed, 1)

	// Определяем тип файла и читаем содержимое
	var content string
	var err error

	if strings.HasSuffix(task.Path, ".gz") {
		content, err = readGZFile(task.Path)
	} else {
		content, err = readTXTFile(task.Path)
	}

	if err != nil {
		log.Printf("[ERROR] read file %s: %v", task.Filename, err)
		atomic.AddInt64(&stats.Errors, 1)
		return
	}

	// Конвертируем кодировку (Windows-1251 → UTF-8)
	text, err := textutils.ToUTF8([]byte(content))
	if err != nil {
		text = content
	}
	text = textutils.NormalizeText(text)
	// Удаляем мусор
	text = textutils.CleanText(text)
	// Удаляем не-русский текст
	text = textutils.FilterNonRussian(text)

	// Извлекаем метаданные из имени файла (сначала!)
	genre, author, title := parseMetadataFromFilename(task.Filename)
	if title == "" {
		title = strings.TrimSuffix(task.Filename, ".txt.gz")
		title = strings.TrimSuffix(title, ".txt")
	}

	// Проверяем наличие CJK/тайских символов ДО фильтрации
	hadCJK := textutils.HasCJKThai(text)

	// Фильтруем CJK и тайские символы
	text = textutils.FilterCJKThai(text)

	// Если были CJK символы — логируем (теперь метаданные уже есть)
	if hadCJK {
		cjkLogFile.WriteString(fmt.Sprintf("%s\t%s\t%s\t%s\n",
			task.Filename, title, author, genre))
	}

	// Пропускаем пустые файлы
	if len(strings.TrimSpace(text)) == 0 {
		log.Printf("[SKIP] %s: empty after filtering (had CJK: %v)", task.Filename, hadCJK)
		atomic.AddInt64(&stats.Skipped, 1)
		return
	}

	// Создаем книгу
	b := &book.Book{
		ID:     uuid.New().String(),
		Title:  title,
		Author: author,
		Genre:  genre,
		Source: task.Path,
		Text:   text,
	}

	// Создаем use case и обрабатываем
	uc := app.NewProcessBooksUseCase(seg, repo)
	if err := uc.Process(ctx, b); err != nil {
		log.Printf("[ERROR] process %s: %v", task.Filename, err)
		atomic.AddInt64(&stats.Errors, 1)
		return
	}

	// Примерный подсчет предложений
	sentenceCount := strings.Count(text, ".") + strings.Count(text, "!") + strings.Count(text, "?")
	atomic.AddInt64(&stats.Sentences, int64(sentenceCount))

	log.Printf("[OK] %s | genre=%s author=%s sentences=%d | CJK: %v", task.Filename, genre, author, sentenceCount, hadCJK)
}

// collectFiles собирает все файлы с нужными расширениями
func collectFiles(rootDir string, extensions []string) ([]FileTask, error) {
	var tasks []FileTask

	err := filepath.Walk(rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}

		// Проверяем расширение
		for _, ext := range extensions {
			if strings.HasSuffix(path, ext) {
				tasks = append(tasks, FileTask{
					Path:     path,
					Filename: filepath.Base(path),
				})
				break
			}
		}
		return nil
	})

	return tasks, err
}

func main() {
	flag.Parse()

	if *corpusDir == "" {
		log.Fatal("--corpus is required")
	}

	// Парсим расширения
	extList := strings.Split(*extensions, ",")
	for i, ext := range extList {
		extList[i] = strings.TrimSpace(ext)
	}

	log.Printf("=== Corpus Processor v3 (with CJK filtering) ===")
	log.Printf("Corpus dir: %s", *corpusDir)
	log.Printf("Output dir: %s", *outputDir)
	log.Printf("Segmenter URL: %s", *segmenterURL)
	log.Printf("Workers: %d", *workers)
	log.Printf("Extensions: %v", extList)

	// Создаем CJK лог файл
	cjkLogPath := filepath.Join(*outputDir, "cjk_filtered.log")
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("create output dir: %v", err)
	}
	cjkLogFile, err := os.Create(cjkLogPath)
	if err != nil {
		log.Fatalf("create CJK log: %v", err)
	}
	defer cjkLogFile.Close()

	// Заголовок CSV для CJK лога
	cjkLogFile.WriteString("filename\ttitle\tauthor\tgenre\n")

	// Собираем файлы
	log.Printf("Collecting files...")
	tasks, err := collectFiles(*corpusDir, extList)
	if err != nil {
		log.Fatalf("collect files: %v", err)
	}
	log.Printf("Found %d files", len(tasks))

	if len(tasks) == 0 {
		log.Fatal("No files found")
	}

	// Создаем репозиторий
	repo, err := filerepo.NewJSONLRepository(*outputDir)
	if err != nil {
		log.Fatalf("create repository: %v", err)
	}
	defer repo.Close()

	// Создаем клиент сегментатора (один на всех)
	seg := segmenterAdapter.NewHTTPClient(*segmenterURL, 120*time.Second)

	// Статистика
	var stats ProcessStats
	stats.Total = int64(len(tasks))

	// Создаем очередь задач
	taskQueue := make(chan FileTask, len(tasks))

	// Контекст с отменой
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Запускаем воркеров
	var wg sync.WaitGroup
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			log.Printf("[Worker %d] started", workerID)
			for task := range taskQueue {
				select {
				case <-ctx.Done():
					return
				default:
					processFile(ctx, task, seg, repo, &stats, cjkLogFile)
				}
			}
			log.Printf("[Worker %d] finished", workerID)
		}(i)
	}

	// Отправляем задачи
	for _, task := range tasks {
		taskQueue <- task
	}
	close(taskQueue)

	// Ждем завершения воркеров
	wg.Wait()

	// Выводим статистику
	log.Printf("\n=== Summary ===")
	log.Printf("Total files: %d", stats.Total)
	log.Printf("Processed: %d", stats.Processed)
	log.Printf("Errors: %d", stats.Errors)
	log.Printf("Skipped: %d", stats.Skipped)
	log.Printf("Total sentences (approx): %d", stats.Sentences)
	log.Printf("CJK log saved to: %s", cjkLogPath)
	log.Println("Done!")
}
