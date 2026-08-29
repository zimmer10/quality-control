# R11 — кешируемый OCR

## Состояние

Код, схема кеша, возобновление и тесты реализованы. Полный OCR-прогон пока не
выполнялся: в локальном окружении отсутствуют OCR-зависимости и модель
`PaddlePaddle/PaddleOCR-VL-1.5`.

## Зафиксированный контракт

- Вход: `data/processed/image_manifest.parquet`, 49 456 изображений.
- Кеш: `cache/ocr/`, одна атомарная JSON-запись на изображение.
- Итог: `features/ocr_text.parquet`, одна строка на изображение товара.
- Выходы: `ocr_text_by_image`, `ocr_quality`, `source_image_id` и статусы.
- `label`, OCR Rules и итоговый verdict не используются.
- Ошибка отдельного изображения сохраняется и не останавливает обработку.
- Недоступность всего OCR backend останавливает запуск сразу.

## Команды

```bash
pip install -e '.[ocr]'
PYTHONPATH=src python -m ecup.features.ocr --limit 10
PYTHONPATH=src python -m ecup.features.ocr
```

После полного запуска этот отчёт автоматически заменяется фактическими
статусами, checksum и временем обработки.
