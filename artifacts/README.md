# Обученные артефакты

Каталог предназначен для:

```text
artifacts/
├── fast/
├── embeddings/
├── retrieval_index/
├── lora_v1/
├── lora_v2/
├── routers/
├── meta/
├── thresholds.json
└── explanation_templates.json
```

Веса и индексы не добавляются в Git. Для каждого эксперимента в `reports/experiments.md` указываются конфигурация и commit.

## Fast text models

R10 создаёт два локальных inference bundle:

```text
artifacts/fast/bad.joblib
artifacts/fast/flammable.joblib
artifacts/fast/manifest.json
```

Каждый bundle содержит собственные word/character TF-IDF vectorizers,
два `LinearSVC` и `LogisticRegression`, обученные на всех данных своей
категории. `manifest.json` хранит схему выходов, checksum OOF и checksum
каждой модели. В Git эти файлы не добавляются.
