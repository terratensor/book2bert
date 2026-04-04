# Глава 2: Подготовка данных — самый важный этап

> "Garbage in, garbage out" — качество модели определяется качеством данных.

## 2.1 Исходные данные

Наш militera корпус представляет собой коллекцию из 11,361 файла в формате `.txt.gz` (сжатые текстовые файлы). Каждый файл — одна книга.

```bash
$ ls -la /mnt/archive/corpus/militera_2023_11359_txt/
-rwxr-xr-x 1 audetv audetv 292735 "Гладков Т.К. Ковпак, 1973.txt.gz"
-rwxr-xr-x 1 audetv audetv 574788 "Гладков Т.К. Коротков, 2005.txt.gz"
-rwxr-xr-x 1 audetv audetv 224724 "Гладков Т.К. Легенда советской разведки – Н. Кузнецов, 2001.txt.gz"
...
```

**Проблемы исходных данных:**

| Проблема | Пример | Влияние |
|----------|--------|---------|
| Кодировка | Windows-1251 (не UTF-8) | Текст нечитаем без конвертации |
| OCR-артефакты | "внима тельно" (разрыв слов) | Искажает токенизацию |
| Списки без точек | Списки фамилий, библиография | Сегментатор не разбивает |
| CJK/тайские символы | Китайские иероглифы, тайские буквы | Шум в словаре |
| Метаданные в имени файла | `antique_Ончуков Николай — Заветные сказки.txt.gz` | Нужно извлекать жанр, автора, название |

## 2.2 Архитектура пайплайна

Мы выбрали **гексагональную архитектуру** (ports & adapters) для Go-части, чтобы отделить бизнес-логику от инфраструктуры.

```
┌─────────────────────────────────────────────────────────────────┐
│                      cmd/process-corpus                         │
│  (точка входа, координация)                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      app/ProcessBooksUseCase                     │
│  (бизнес-логика: сегментация → разбивка по \n → сохранение)     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │ Segmenter │  │ Repository│  │  Logger   │
    │ (порт)    │  │  (порт)   │  │  (порт)   │
    └───────────┘  └───────────┘  └───────────┘
            │             │             │
            ▼             ▼             ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │HTTPClient │  │ JSONLRepo │  │  LogFile  │
    │(адаптер)  │  │(адаптер)  │  │(адаптер)  │
    └───────────┘  └───────────┘  └───────────┘
```

**Почему Go?** 
- Многопоточность (горутины) для параллельной обработки файлов
- Быстрая работа с файловой системой
- Низкое потребление памяти

## 2.3 Шаг 1: Чтение и конвертация кодировки

Первая проблема: файлы в кодировке Windows-1251, а нам нужен UTF-8.

```go
// pkg/textutils/encoding.go
func ToUTF8(text []byte) (string, error) {
    // Проверяем, не UTF-8 ли уже
    if utf8.Valid(text) {
        return string(text), nil
    }
    
    // Пробуем декодировать из Windows-1251
    decoder := charmap.Windows1251.NewDecoder()
    utf8Text, err := decoder.Bytes(text)
    if err != nil {
        return "", err
    }
    return string(utf8Text), nil
}
```

**Почему это важно:** Без правильной конвертации мы получим кракозябры вместо текста.

```json
// Было (Windows-1251, неверно прочитано):
{"text": "��������: ����������������� � �����"}

// Стало (UTF-8):
{"text": "АНАЛИТИЧЕСКАЯ ЗАПИСКА\nЛюбовь к мудрости:\nот прошлого к будущему..."}
```

## 2.4 Шаг 2: Фильтрация CJK и тайских символов

В корпусе встречаются символы китайских, японских, корейских и тайских алфавитов. Это шум (OCR-ошибки, редкие вкрапления).

```go
// pkg/textutils/filter.go
func IsCJK(r rune) bool {
    code := int(r)
    // Основной диапазон CJK
    if code >= 0x4E00 && code <= 0x9FFF {
        return true
    }
    // Дополнительные диапазоны...
    return false
}

func FilterCJKThai(s string) string {
    var result []rune
    for _, r := range s {
        if !IsCJK(r) && !IsThai(r) {
            result = append(result, r)
        }
    }
    return string(result)
}
```

**Результат фильтрации:**
- Словарь стал чище (нет мусорных токенов)
- Размер корпуса уменьшился незначительно (0.0008% длинных предложений)

## 2.5 Шаг 3: Сегментация предложений

Для разбивки текста на предложения мы используем библиотеку `razdel` (Python). Почему не Go? Потому что `razdel` использует ML-модель, обученную на русском языке, и обеспечивает высокое качество.

**Проблема:** Вызывать Python из Go для каждого файла — медленно. Решение: **HTTP-сервис**.

```python
# services/segmenter/app.py
from flask import Flask, request, jsonify
from razdel import sentenize

app = Flask(__name__)

@app.route('/segment', methods=['POST'])
def segment():
    data = request.get_json()
    text = data.get('text', '')
    sentences = [s.text for s in sentenize(text)]
    return jsonify({'sentences': sentences})

@app.route('/segment_batch', methods=['POST'])
def segment_batch():
    data = request.get_json()
    texts = data.get('texts', [])
    results = []
    for text in texts:
        sentences = [s.text for s in sentenize(text)]
        results.append(sentences)
    return jsonify({'results': results})
```

Запускаем сервис:

```bash
cd services/segmenter
gunicorn -w 8 -b 0.0.0.0:8090 app:app
```

Go-клиент для вызова:

```go
// pkg/adapters/segmenter/http.go
func (c *HTTPClient) Segment(ctx context.Context, text string) ([]string, error) {
    reqBody := struct {
        Text string `json:"text"`
    }{Text: text}
    
    jsonBody, _ := json.Marshal(reqBody)
    resp, err := c.client.Post(c.baseURL+"/segment", "application/json", bytes.NewReader(jsonBody))
    // ... обработка ответа
}
```

## 2.6 Шаг 4: Разбивка по `\n` (списки и таблицы)

**Проблема:** Сегментатор не разбивает списки, потому что в них нет точек.

```text
// Пример списка (одна строка в исходном файле)
"Абаимов Павел Васильевич, 1904\nАбакаров Гаджи, 1899\nАбакумов Алексей Федорович, 1923"
```

`razdel` вернет **одно предложение** (всё целиком), а нужно **три отдельных** (каждая строка).

**Решение:** После сегментации разбиваем каждое предложение по `\n`.

```go
// pkg/app/process_books.go
func (uc *ProcessBooksUseCase) Process(ctx context.Context, b *book.Book) error {
    // 1. Сегментируем текст
    sentences, err := uc.segmenter.Segment(ctx, b.Text)
    
    // 2. Разбиваем каждое предложение по \n
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
    
    // 3. Сохраняем в JSONL
    // ...
}
```

## 2.7 Шаг 5: Сохранение в JSONL

Каждое предложение сохраняется в отдельный JSONL-файл (по одному на книгу):

```json
{"book_id":"0a2a41ed-6746-4f42-a240-a269c8e137c1","title":"Аношкин М. П. Особое задание","author":"Unknown","genre":"","text":"«Военная Литература»","position":0}
{"book_id":"0a2a41ed-6746-4f42-a240-a269c8e137c1","title":"Аношкин М. П. Особое задание","author":"Unknown","genre":"","text":"Проза войны","position":1}
{"book_id":"0a2a41ed-6746-4f42-a240-a269c8e137c1","title":"Аношкин М. П. Особое задание","author":"Unknown","genre":"","text":"«ЧП»","position":2}
```

**Поля:**
- `book_id` — уникальный идентификатор книги
- `title`, `author`, `genre` — метаданные (для future use)
- `text` — предложение (чистый текст, без специальных токенов)
- `position` — порядковый номер предложения в книге

## 2.8 Параллельная обработка

Go позволяет обрабатывать файлы параллельно с помощью горутин:

```go
// Запускаем воркеров
for i := 0; i < *workers; i++ {
    wg.Add(1)
    go func(workerID int) {
        defer wg.Done()
        for task := range taskQueue {
            processFile(ctx, task, seg, repo, &stats, cjkLogFile)
        }
    }(i)
}
```

Настройка `--workers 10` означает 10 параллельных горутин. Сервис сегментации должен быть готов к такой нагрузке (поэтому мы запускаем gunicorn с 8 воркерами).

## 2.9 Результат

```bash
=== Summary ===
Total files: 11361
Processed: 11361
Errors: 0
Skipped: 10
CJK log saved to: data/processed/sentences_militera_v3/cjk_filtered.log
```

**Итог:** 11,361 книга → 93,124,036 предложений, готовых к токенизации.

## 2.10 Промежуточные итоги

| Проблема | Решение |
|----------|---------|
| Кодировка Windows-1251 | Конвертация в UTF-8 |
| CJK/тайские символы | Фильтрация (удаление) |
| Разбивка на предложения | HTTP-сервис razdel |
| Списки без точек | Разбивка по `\n` после сегментации |
| Параллельная обработка | Горутины (10 воркеров) |
| Скорость | 11,361 файл за 2-3 часа |

**Ключевой вывод:** Качество данных напрямую зависит от каждого шага предобработки. Пропуск фильтрации CJK или разбивки по `\n` привел бы к замусоренному словарю и потере структуры списков.

---
[Далее: Глава 3 - Токенизация...]