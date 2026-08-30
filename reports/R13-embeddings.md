# R13 — multimodal embeddings

## Контракт

- Модель: `Qwen/Qwen3-VL-Embedding-2B`.
- Размерность: **2048**.
- Входы: `name`, `description`, `category` и изображения из R06 manifest.
- Итоговая таблица: `features/embeddings.parquet`.
- Одна строка соответствует одному товару.
- `label`, правила и финальный verdict при расчёте не используются.

Таблица содержит `text_embedding`, пары `image_embeddings` и
`joint_text_image_embeddings`, их нормализованные агрегаты, а также
`text_image_similarity` и `image_disagreement`.

## Возобновляемость

Каждый успешно рассчитанный text/image/joint-вектор атомарно сохраняется в
`cache/embeddings/`. Ключ кеша учитывает модель, инструкцию, размерность и
SHA-256 входного текста или изображения.

## Текущий результат

- Реализация и автоматические проверки подготовлены.
- Полный GPU-прогон и фактическое время будут добавлены после запуска Colab.
