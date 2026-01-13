# KREPS RAG System

Офлайн система для поиска по документам с использованием гибридного поиска (FAISS + BM25) и генерации ответов через Qwen.

## Что это

Система работает так:
1. Загружает PDF/TXT/MD файлы
2. Режет их на чанки (кусочки текста)
3. Индексирует через FAISS (семантика) и BM25 (ключевые слова)
4. По запросу ищет похожие чанки
5. Генерирует ответ через Qwen LLM

## Структура проекта

```
kREPS-rag/
├── src/                  # Код
│   ├── app.py           # CLI интерфейс
│   ├── ingest.py        # Загрузка документов
│   ├── chunking.py      # Разбивка на чанки
│   ├── embed.py         # Эмбеддинги через Ollama
│   ├── index_faiss.py   # Семантический поиск
│   ├── index_bm25.py    # Лексический поиск
│   ├── retrieve.py      # Гибридный поиск
│   └── answer.py        # Генерация ответа
├── data/
│   └── raw_docs/        # Сюда кидать документы
├── storage/             # Тут хранятся индексы
└── frontend.py          # Веб-интерфейс (Streamlit)
```

## Установка

1. Создаем виртуальное окружение:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

2. Ставим зависимости:
```bash
pip install -r requirements.txt
```

3. Проверяем что Ollama запущен:
```bash
ollama list
# Должны быть модели: nomic-embed-text и qwen2.5:3b
```

Если моделей нет:
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

## Как использовать

### 1. Добавить документы
Кидаем PDF/TXT/MD файлы в папку `data/raw_docs/`

### 2. Создать индекс
```bash
python src/app.py index
```

Это:
- Загрузит документы
- Разобьет на чанки
- Определит язык и тип документа
- Создаст FAISS и BM25 индексы

### 3. Задать вопрос

Через CLI:
```bash
python src/app.py query "Какие процедуры безопасности?"
```

Через веб-интерфейс:
```bash
python -m streamlit run frontend.py
```

### 4. Проверить статус
```bash
python src/app.py status
```

## Фичи

**Автоопределение метаданных:**
- Язык документа (en/ru)
- Тип (policy/report/manual)
- Год публикации

**Адаптивное разбиение:**
- Политики → 500 токенов (точный поиск)
- Отчеты → 900 токенов (больше контекста)
- Мануалы → 700 токенов

**Языковая приоритизация:**
- Русский запрос → приоритет русским документам
- Английский запрос → приоритет английским

**Гарантии качества:**
- Минимум источников (MIN_SOURCES)
- Порог релевантности
- Отказ при слабых доказательствах
- Ответ ТОЛЬКО из контекста

## Настройки

В `src/config.py`:
```python
CHUNK_SIZE_TOKENS = 700          # Размер чанка
CHUNK_OVERLAP_TOKENS = 100       # Перекрытие
SEMANTIC_WEIGHT = 0.7            # Вес семантики
LEXICAL_WEIGHT = 0.3             # Вес BM25
MIN_SOURCES = 2                  # Мин. источников
CONFIDENCE_THRESHOLD_HIGH = 0.75 # Порог уверенности
```

## Как работает поиск

1. **Детект языка запроса** (по кириллице/латинице)
2. **FAISS search** → топ-10 семантически похожих
3. **BM25 search** → топ-10 по ключевым словам
4. **Merge + нормализация** → гибридный скор
5. **Language boost** → +15% за совпадение языка
6. **Топ-5 чанков** → в промпт Qwen

## Примеры

```bash
# Индексация
python src/app.py index

# Запрос
python src/app.py query "Что такое безопасность?"

# Статус
python src/app.py status

# Веб-интерфейс
python -m streamlit run frontend.py
```

## Возможные проблемы

**ModuleNotFoundError: No module named 'faiss'**
```bash
pip install faiss-cpu
```

**ModuleNotFoundError: No module named 'rank_bm25'**
```bash
pip install rank-bm25
```

**Connection refused (Ollama)**
- Проверь что Ollama запущен: `ollama list`
- Проверь порт: `http://localhost:11434`

**Timeout при генерации**
- Увеличь timeout в `src/answer.py` (сейчас 450 сек)
- Или используй более легкую модель

## Результат запроса

```python
{
    "answer": "Ответ из документов",
    "confidence": "High",  # High/Medium/Low
    "sources": [
        {
            "document": "safety_policy.pdf",
            "page": 5,
            "section": "Procedures",
            "score": 0.87
        }
    ],
    "chunks": [...]  # Найденные чанки
}
```

## Зависимости

- Python 3.9+
- Ollama (для эмбеддингов и LLM)
- FAISS (векторный поиск)
- BM25 (лексический поиск)
- Streamlit (веб-интерфейс)

## Лицензия

Учебный проект
