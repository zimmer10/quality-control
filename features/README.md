# Рассчитанные признаки

Планируемые файлы:

```text
features/text_rules.parquet
features/ocr_text.parquet
features/ocr_rules.parquet
features/embeddings.parquet
features/retrieval.parquet
```

Содержимое не добавляется в Git. Схемы признаков фиксируются в коде и отчётах.

`text_rules.parquet` создаётся командой:

```bash
PYTHONPATH=src python -m ecup.features.text
```

Контракт колонок зафиксирован в `src/ecup/features/text.py`, а статистика
полного прогона — в `reports/R09-text-rules.md`.
