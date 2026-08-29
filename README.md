# E-CUP «Контроль качества»

Репозиторий решения соревнования по классификации товаров категорий «БАД» и «Легковоспламеняющиеся» с формированием объяснения.

## Текущий статус

Завершены этапы R01–R09: метрика, аудит данных, финальные группы и
folds, image manifest, визуальные дубли, evidence schema и Text Rules.
Обучение моделей ещё не начато.

## Документы

- [`COMPETITION_CONDITIONS.md`](COMPETITION_CONDITIONS.md) — технически значимые условия соревнования.
- [`ROADMAP.md`](ROADMAP.md) — задачи, зависимости и контрольные точки.
- [`PIPELINE.md`](PIPELINE.md) — целевая архитектура и порядок построения решения.
- [`pipeline.drawio`](pipeline.drawio) — схемы обучения и инференса.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — правила работы с задачами, ветками и Pull Request.
- [`reports/experiments.md`](reports/experiments.md) — журнал экспериментов.

## Основные каталоги

```text
configs/       конфигурации этапов
src/ecup/      исходный код
tests/         автоматические проверки
reports/       небольшие отчёты
data/          локальные данные
cache/         OCR и embeddings
features/      рассчитанные признаки
annotations/   LoRA-разметка
oof/           OOF-предсказания
artifacts/     обученные модели и индексы
```

Большие данные и артефакты не хранятся в Git.

## Установка для разработки

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Проверка

```bash
pytest
```
