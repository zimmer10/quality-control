# Журнал экспериментов

Каждый существенный эксперимент добавляется отдельным разделом.

## Шаблон

### EXX — название

```yaml
roadmap_id: RXX
git_commit: TODO
command: TODO
config: TODO
seed: 42
```

| Метрика | Значение |
|---|---:|
| F1 BAD | — |
| F1 Flammable | — |
| Mean F1 | — |
| Время | — |

Артефакты:

```text
TODO
```

Решение: `оставить / отклонить / повторить`.

Комментарий: TODO.

## Завершённые инфраструктурные задачи

### R02 — метрика соревнования

- Реализован бинарный F1 для меток `0/1`.
- F1 рассчитывается отдельно для категорий `БАД` и `Легковоспламеняющиеся`.
- Итоговая метрика является средним двух категорий.
- Добавлены тесты граничных условий и проверки входных данных.

## Завершённые эксперименты

### E01 — Fast text OOF baseline

```yaml
roadmap_id: R10
git_commit: 61216dc
command: python -m ecup.models.fast
config: configs/fast_models.yaml
seed: 42
```

| Метрика | Значение |
|---|---:|
| F1 БАД | 0.934082 |
| F1 Легковоспламеняющиеся | 0.786127 |
| Mean F1 | 0.860105 |
| OOF fit + predict | 228.314 s |
| Final fit | 46.355 s |
| Final inference | 2.512 ms/row |

Артефакты:

```text
oof/fast.parquet
artifacts/fast/bad.joblib
artifacts/fast/flammable.joblib
artifacts/fast/manifest.json
reports/R10-fast-oof.md
```

Решение: `оставить` как воспроизводимый baseline и вход Router/meta-model.

Комментарий: повторный OOF-прогон подтвердил логическую SHA-256
`719a745cb6ce04f262e6d5968705e9ce429b8ecb278c1103f69bc8cdd3bfafa8`.
