package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/terratensor/book2bert/pkg/core/book"
	// потребуется
)

var (
	sentencesDir  = flag.String("sentences", "data/processed/sentences", "директория с JSONL предложениями")
	tokenizerPath = flag.String("tokenizer", "data/processed/tokenizer/vocab.txt", "путь к vocab.txt токенизатора")
	outputDir     = flag.String("output", "data/processed/dataset", "директория для датасета")
	maxTokens     = flag.Int("max_tokens", 512, "максимальное количество токенов в примере")
	mlmProb       = flag.Float64("mlm_prob", 0.15, "вероятность маскирования для MLM")
)

// Tokenizer интерфейс для токенизатора
type Tokenizer interface {
	Encode(text string) ([]int, []string, error)
}

// BertExample пример для обучения BERT
type BertExample struct {
	InputIDs      []int  `json:"input_ids"`
	AttentionMask []int  `json:"attention_mask"`
	TokenTypeIDs  []int  `json:"token_type_ids"`
	Labels        []int  `json:"labels"`  // для MLM
	IsNext        bool   `json:"is_next"` // для NSP
	BookID        string `json:"book_id"`
	Genre         string `json:"genre"`
}

// SentenceWithMeta предложение с метаданными
type SentenceWithMeta struct {
	Text     string
	BookID   string
	Title    string
	Author   string
	Genre    string
	Position int
}

func main() {
	flag.Parse()

	// 1. Загружаем токенизатор
	tokenizer, err := loadTokenizer(*tokenizerPath)
	if err != nil {
		log.Fatalf("load tokenizer: %v", err)
	}

	// 2. Создаем выходные директории
	trainDir := filepath.Join(*outputDir, "train")
	valDir := filepath.Join(*outputDir, "val")
	os.MkdirAll(trainDir, 0755)
	os.MkdirAll(valDir, 0755)

	// 3. Собираем все предложения по книгам
	books, err := loadSentencesByBook(*sentencesDir)
	if err != nil {
		log.Fatalf("load sentences: %v", err)
	}
	log.Printf("Loaded %d books", len(books))

	// 4. Обрабатываем каждую книгу
	trainExamples := 0
	valExamples := 0

	for bookID, sentences := range books {
		// 95/5 split по книгам (не по примерам)
		isTrain := hashBookID(bookID) < 95
		examples := buildExamples(sentences, tokenizer, *maxTokens)

		// Сохраняем примеры
		outputFile := filepath.Join(*outputDir, getSplitDir(isTrain), bookID+".jsonl")
		if err := saveExamples(outputFile, examples); err != nil {
			log.Printf("ERROR saving %s: %v", bookID, err)
			continue
		}

		if isTrain {
			trainExamples += len(examples)
		} else {
			valExamples += len(examples)
		}
	}

	log.Printf("Done: %d train examples, %d val examples", trainExamples, valExamples)
}

// loadTokenizer загружает токенизатор из vocab.txt
func loadTokenizer(vocabPath string) (Tokenizer, error) {
	// Читаем vocab
	file, err := os.Open(vocabPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	vocab := make(map[string]int)
	scanner := bufio.NewScanner(file)
	idx := 0
	for scanner.Scan() {
		token := strings.TrimSpace(scanner.Text())
		if token != "" {
			vocab[token] = idx
			idx++
		}
	}

	return &simpleTokenizer{vocab: vocab}, nil
}

// simpleTokenizer простая реализация токенизатора
type simpleTokenizer struct {
	vocab map[string]int
}

func (t *simpleTokenizer) Encode(text string) ([]int, []string, error) {
	// Упрощенная версия: разбиваем по пробелам и ищем в словаре
	// В реальном проекте нужно использовать полноценный WordPiece
	words := strings.Fields(text)
	ids := make([]int, 0, len(words))
	tokens := make([]string, 0, len(words))

	for _, w := range words {
		if id, ok := t.vocab[w]; ok {
			ids = append(ids, id)
			tokens = append(tokens, w)
		} else {
			// UNK токен
			ids = append(ids, t.vocab["[UNK]"])
			tokens = append(tokens, "[UNK]")
		}
	}
	return ids, tokens, nil
}

// loadSentencesByBook загружает предложения и группирует по книгам
func loadSentencesByBook(dir string) (map[string][]SentenceWithMeta, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.jsonl"))
	if err != nil {
		return nil, err
	}

	books := make(map[string][]SentenceWithMeta)

	for _, file := range files {
		bookID := strings.TrimSuffix(filepath.Base(file), ".jsonl")

		f, err := os.Open(file)
		if err != nil {
			log.Printf("WARN: cannot open %s: %v", file, err)
			continue
		}

		scanner := bufio.NewScanner(f)
		// Увеличиваем буфер для длинных строк
		scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}

			var s book.Sentence
			if err := json.Unmarshal([]byte(line), &s); err != nil {
				log.Printf("WARN: parse line in %s: %v", file, err)
				continue
			}

			books[bookID] = append(books[bookID], SentenceWithMeta{
				Text:     s.Text,
				BookID:   s.BookID,
				Title:    s.Title,
				Author:   s.Author,
				Genre:    s.Genre,
				Position: s.Position,
			})
		}
		f.Close()
	}

	return books, nil
}

// buildExamples группирует предложения в примеры по maxTokens
func buildExamples(sentences []SentenceWithMeta, tokenizer Tokenizer, maxTokens int) []BertExample {
	var examples []BertExample
	var currentSentences []SentenceWithMeta
	var currentTokens int

	for _, s := range sentences {
		// Получаем количество токенов в предложении
		ids, _, err := tokenizer.Encode(s.Text)
		if err != nil {
			// Пропускаем проблемные предложения
			continue
		}
		tokenCount := len(ids)

		// +2 для [CLS] и [SEP]
		if currentTokens+tokenCount+2 > maxTokens && len(currentSentences) > 0 {
			// Сохраняем текущий пример
			example := createExample(currentSentences, tokenizer, maxTokens)
			if example != nil {
				examples = append(examples, *example)
			}

			// Начинаем новый пример с текущего предложения
			currentSentences = []SentenceWithMeta{s}
			currentTokens = tokenCount
		} else {
			currentSentences = append(currentSentences, s)
			currentTokens += tokenCount
		}
	}

	// Последний пример
	if len(currentSentences) > 0 {
		example := createExample(currentSentences, tokenizer, maxTokens)
		if example != nil {
			examples = append(examples, *example)
		}
	}

	return examples
}

// createExample создает один обучающий пример из группы предложений
func createExample(sentences []SentenceWithMeta, tokenizer Tokenizer, maxTokens int) *BertExample {
	if len(sentences) == 0 {
		return nil
	}

	// Собираем текст
	text := ""
	for _, s := range sentences {
		text += s.Text + " "
	}
	text = strings.TrimSpace(text)

	// Токенизируем
	ids, _, err := tokenizer.Encode(text)
	if err != nil {
		return nil
	}

	// Обрезаем до maxTokens-2 (оставляем место для [CLS] и [SEP])
	if len(ids) > maxTokens-2 {
		ids = ids[:maxTokens-2]
	}

	// Формируем входные данные
	inputIDs := make([]int, 0, maxTokens)
	attentionMask := make([]int, 0, maxTokens)

	// [CLS]
	inputIDs = append(inputIDs, 2) // индекс [CLS] в словаре
	attentionMask = append(attentionMask, 1)

	// Токены предложения
	for _, id := range ids {
		inputIDs = append(inputIDs, id)
		attentionMask = append(attentionMask, 1)
	}

	// [SEP]
	inputIDs = append(inputIDs, 3) // индекс [SEP] в словаре
	attentionMask = append(attentionMask, 1)

	// Паддинг
	for len(inputIDs) < maxTokens {
		inputIDs = append(inputIDs, 0) // [PAD]
		attentionMask = append(attentionMask, 0)
	}

	// Для MLM метки (пока просто -100, позже добавим маскирование)
	labels := make([]int, maxTokens)
	for i := range labels {
		labels[i] = -100
	}

	return &BertExample{
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
		TokenTypeIDs:  make([]int, maxTokens), // все 0 для первого сегмента
		Labels:        labels,
		IsNext:        true, // пока все true, позже добавим NSP
		BookID:        sentences[0].BookID,
		Genre:         sentences[0].Genre,
	}
}

// saveExamples сохраняет примеры в JSONL
func saveExamples(path string, examples []BertExample) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	for _, ex := range examples {
		data, err := json.Marshal(ex)
		if err != nil {
			return err
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			return err
		}
	}
	return writer.Flush()
}

// hashBookID определяет, в какую часть попадет книга
func hashBookID(id string) int {
	// Простая хеш-функция для split 95/5
	h := 0
	for _, c := range id {
		h = (h*31 + int(c)) % 100
	}
	return h
}

func getSplitDir(isTrain bool) string {
	if isTrain {
		return "train"
	}
	return "val"
}
