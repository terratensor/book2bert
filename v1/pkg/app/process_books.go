package app

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/terratensor/book2bert/pkg/core/book"
	"github.com/terratensor/book2bert/pkg/core/segmenter"
)

// ProcessBooksUseCase сценарий обработки книг: разбивка на предложения
type ProcessBooksUseCase struct {
	segmenter segmenter.Segmenter
	repo      book.Repository
}

// NewProcessBooksUseCase создает новый use case
func NewProcessBooksUseCase(seg segmenter.Segmenter, repo book.Repository) *ProcessBooksUseCase {
	return &ProcessBooksUseCase{
		segmenter: seg,
		repo:      repo,
	}
}

// Process обрабатывает книгу: сегментирует текст, разбивает по \n, сохраняет
func (uc *ProcessBooksUseCase) Process(ctx context.Context, b *book.Book) error {
	// 1. Сохраняем метаданные книги
	if err := uc.repo.SaveBook(ctx, b); err != nil {
		return fmt.Errorf("save book: %w", err)
	}

	// 2. Сегментируем текст
	sentences, err := uc.segmenter.Segment(ctx, b.Text)
	if err != nil {
		return fmt.Errorf("segment text: %w", err)
	}

	// 3. Разбиваем каждое предложение по \n
	var allSentences []string
	for _, s := range sentences {
		lines := strings.Split(s, "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line != "" {
				allSentences = append(allSentences, line)
			}
		}
	}

	// 4. Преобразуем в доменные объекты
	bookSentences := make([]book.Sentence, len(allSentences))
	for i, text := range allSentences {
		bookSentences[i] = book.Sentence{
			BookID:    b.ID,
			Title:     b.Title,
			Author:    b.Author,
			Genre:     b.Genre,
			Text:      text,
			Position:  i,
			CreatedAt: time.Now(),
		}
	}

	// 5. Сохраняем предложения
	if err := uc.repo.SaveSentences(ctx, bookSentences); err != nil {
		return fmt.Errorf("save sentences: %w", err)
	}

	return nil
}
