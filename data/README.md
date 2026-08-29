# Локальные данные

Каталог не хранится в Git, кроме этого файла.

Ожидаемая структура:

```text
data/
├── raw/
│   ├── data.csv
│   └── images/
└── processed/
    ├── manifest.parquet
    ├── folds.parquet
    ├── duplicate_groups.parquet
    ├── label_conflicts.parquet
    ├── image_manifest.parquet
    ├── image_hashes.parquet
    ├── visual_duplicates.parquet
    └── visual_duplicate_stats.json
```

- `image_manifest.parquet` — одна строка на файл изображения, результат R06.
- `visual_duplicates.parquet` — кандидатные пары товаров и признак `auto_merge`, результат R07.
- `duplicate_groups.parquet` и `folds.parquet` пересобираются после подтверждения визуальных дублей.
- `manifest.parquet` — итоговая строка на товар с текстом, группой, числом изображений и fold.

`folds.parquet` является общим зафиксированным контрактом двух веток разработки.

