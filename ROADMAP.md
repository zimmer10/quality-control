# Roadmap разработки

## Правила работы

- GitHub Issues/Project хранит текущий статус задач.
- Этот файл фиксирует порядок работ и зависимости.
- Свободную задачу может взять любой разработчик.
- Задача берётся в работу только после завершения её зависимостей.
- После существенного этапа проводится отдельная проверка результатов.

Статусы GitHub Project:

```text
Backlog → Ready → In Progress → Review → Done
                         ↘ Blocked
```

## Задачи

| ID | Задача | Зависимости | Рекомендуемая область | Статус |
|---|---|---|---|---|
| R01 | Создать структуру проекта и окружение | — | любая | Done |
| R02 | Реализовать метрику соревнования | R01 | data | Done |
| R03 | Провести аудит `data.csv` | R01 | data | Done |
| R04 | Найти дубли и создать `group_id` | R03 | data | Backlog |
| R05 | Создать и зафиксировать `folds.parquet` | R02, R04 | data | Backlog |
| R06 | Проверить изображения и создать image manifest | R01 | multimodal | Backlog |
| R07 | Найти визуальные дубли | R06 | multimodal | Backlog |
| R08 | Зафиксировать evidence schema | R03 | shared | Backlog |
| R09 | Реализовать Text Rules | R05, R08 | data | Backlog |
| R10 | Получить `oof/fast.parquet` | R05, R09 | data | Backlog |
| R11 | Реализовать кешируемый OCR | R06 | multimodal | Backlog |
| R12 | Реализовать OCR Rules | R08, R11 | multimodal | Backlog |
| R13 | Посчитать multimodal embeddings | R06 | multimodal | Backlog |
| R14 | Реализовать fold-safe retrieval | R05, R13 | multimodal | Backlog |
| R15 | Получить `oof/expensive.parquet` | R12, R13, R14 | multimodal | Backlog |
| R16 | Собрать LoRA dataset v1 | R08, R09, R12 | multimodal | Backlog |
| R17 | Обучить LoRA v1 и получить OOF | R16 | multimodal | Backlog |
| R18 | Выполнить hard-example mining | R10, R15, R17 | shared | Backlog |
| R19 | Собрать LoRA dataset v2 | R18 | shared | Backlog |
| R20 | Обучить LoRA v2 и получить OOF | R19 | multimodal | Backlog |
| R21 | Обучить Router 1 | R10, R15 | integration | Backlog |
| R22 | Обучить Router 2 | R10, R15, R20 | integration | Backlog |
| R23 | Реализовать routing simulation | R21, R22 | integration | Backlog |
| R24 | Обучить meta-model и подобрать thresholds | R23 | integration | Backlog |
| R25 | Реализовать шаблонные объяснения | R08, R24 | shared | Backlog |
| R26 | Собрать полный `run.py` | R20, R24, R25 | inference | Backlog |
| R27 | Собрать Docker и `metadata.json` | R26 | inference | Backlog |
| R28 | Провести полный timed dry run | R27 | shared | Backlog |

## Контрольные точки

| Gate | Проверяемый результат |
|---|---|
| G1 | `folds.parquet`: дубли, баланс и отсутствие пересечений групп |
| G2 | `oof/fast.parquet`: воспроизводимый text baseline |
| G3 | OCR cache: формат, качество и возобновляемость |
| G4 | `oof/expensive.parquet`: OOF-классификаторы и fold-safe retrieval |
| G5 | `annotations/lora_v1.jsonl`: корректность evidence и targets |
| G6 | `oof/lora_v1.parquet`: реальные ошибки первой LoRA |
| G7 | `oof/lora_v2.parquet`: честное сравнение v1 и v2 |
| G8 | OOF Router 1/2: корректные входы и targets |
| G9 | `oof/meta.parquet`: итоговая OOF-метрика и routing simulation |
| G10 | Docker: автономность, формат ответа и лимит времени |

