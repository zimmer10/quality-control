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
    └── image_manifest.parquet
```

`folds.parquet` является общим зафиксированным контрактом двух веток разработки.

