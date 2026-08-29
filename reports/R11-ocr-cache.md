# R11 — кешируемый OCR

## Состояние

Код, схема кеша, возобновление и тесты реализованы. Проведён успешный
GPU smoke-тест на одном реальном изображении. Полный OCR-прогон пока не
выполнялся и R11 остаётся в статусе `In Progress`.

## Зафиксированный контракт

- Вход: `data/processed/image_manifest.parquet`, 49 456 изображений.
- Модель: `PaddlePaddle/PaddleOCR-VL-1.5`.
- Кеш: `cache/ocr/`, одна атомарная JSON-запись на изображение.
- Итог: `features/ocr_text.parquet`, одна строка на изображение товара.
- Выходы: `ocr_text_by_image`, `ocr_quality`, `source_image_id` и статусы.
- `label`, OCR Rules и итоговый verdict не используются.
- Ошибка отдельного изображения сохраняется и не останавливает обработку.
- Недоступность всего OCR backend останавливает запуск сразу.

## GPU smoke-тест

```yaml
gpu: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB
images: 1
max_pixels: 401408
max_new_tokens: 256
status: ok
ocr_errors: 0
ocr_seconds: 60.187
peak_observed_gpu_memory: approximately 3 GiB
```

Распознан русский и английский текст упаковки, включая прямое указание
`A Dietary Supplement`. При такой скорости полный последовательный прогон
занял бы около 34 дней, поэтому RTX 3050 подходит только для smoke-тестов.

## Команды

Короткая локальная проверка:

```bash
SHARED_MODELS_PATH="$PWD/artifacts/shared_models" \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python -m ecup.features.ocr \
  --limit 10 --max-pixels 401408 --max-new-tokens 256
```

Полный прогон необходимо выполнить на более производительной GPU. После него
отчёт автоматически заменится фактическими статусами, checksum и временем.
