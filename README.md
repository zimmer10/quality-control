# E-CUP «Контроль качества»

Репозиторий решения соревнования по классификации товаров категорий «БАД» и «Легковоспламеняющиеся» с формированием объяснения.

## Текущий статус

Создан каркас проекта и правила совместной разработки. Обучение моделей ещё не начато.

## Документы

- [`COMPETITION_CONDITIONS.md`](COMPETITION_CONDITIONS.md) — технически значимые условия соревнования.
- [`ROADMAP.md`](ROADMAP.md) — задачи, зависимости и контрольные точки.
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

