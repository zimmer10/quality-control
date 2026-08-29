# OOF-предсказания

## Готовые артефакты

`fast.parquet` создаётся этапом R10 командой:

```bash
python -m ecup.models.fast
```

В нём одна строка на каждый train-товар, базовые поля
`id, category, fold, group_id, true_label`, четыре выхода fast-моделей и
числовые Text Rule features R09. TF-IDF и классификаторы обучаются заново
внутри каждого train-fold; OCR и изображения не используются.

Полная схема, метрики и checksum зафиксированы в
[`reports/R10-fast-oof.md`](../reports/R10-fast-oof.md).

## Планируемые артефакты

Файлы следующих этапов:

```text
oof/expensive.parquet
oof/lora_v1.parquet
oof/lora_v2.parquet
oof/router_1.parquet
oof/router_2.parquet
oof/meta.parquet
```

Обязательные ключевые поля всех OOF-таблиц:

```text
id, category, fold, group_id, true_label
```

Одна строка соответствует одному train-товару. Содержимое не добавляется в Git.
