# E-CUP «Контроль качества»: техническое описание решения

## 1. Назначение документа

Этот документ описывает целевую архитектуру решения для классификации товаров двух категорий:

1. БАД.
2. Легковоспламеняющиеся товары.

Для каждого товара система должна:

- определить `label`;
- сформировать итоговый вердикт `бан` или `не бан`;
- указать факты, на которых основано решение;
- сформировать объяснение требуемой длины;
- уложиться в ограничение по времени инференса.

Архитектура строится как каскад. Простые товары обрабатываются быстрыми текстовыми моделями, товары, для которых нужны фотографии, направляются в OCR и multimodal embeddings, а наиболее сложные случаи в Qwen3-VL-2B с обученным LoRA-адаптером.

Каждый обучаемый компонент оценивается через OOF-предсказания. Компонент включается в финальную систему только при измеримом приросте качества или экономии времени.

---

## 2. Основные термины

### 2.1. Label

`Label` - правильный класс товара в обучающем датасете:

```text
label = 1 → товар соответствует проверяемому признаку → бан
label = 0 → товар не соответствует проверяемому признаку → не бан
```

Необходимо различать:

```text
true label      → правильный ответ организаторов;
predicted label → ответ нашей системы.
```

### 2.2. Feature

`Feature`, или признак, - числовое или категориальное описание товара, используемое моделью.

Примеры:

```text
text_probability = 0.82
has_bad_marker = 1
fuel_included = false
nearest_positive_distance = 0.14
has_lora = 0
```

### 2.3. Evidence

`Evidence` - конкретный найденный факт вместе с источником.

```json
{
  "field": "fuel_included",
  "value": false,
  "source": "description",
  "text": "баллон приобретается отдельно"
}
```

Evidence отвечает на вопросы:

- какой факт найден;
- какое значение подтверждено;
- где факт найден;
- какой фрагмент текста или какое изображение его подтверждает.

### 2.4. Evidence schema

`Evidence schema` - заранее заданный список фактов, которые система должна уметь извлекать.

Для легковоспламеняющихся товаров возможны поля:

```text
object_type
fuel_present
fuel_included
requires_external_fuel
independent_ignition_source
flammable_item_in_kit
explicit_without_fuel
```

Для БАД используются другие поля:

```text
explicit_bad_marker
dietary_supplement_marker
explicit_not_bad
sport_nutrition
marker_source
```

Для неоднозначных фактов используется три состояния:

```text
true    → факт подтверждён;
false   → подтверждено обратное;
unknown → информации недостаточно.
```

### 2.5. Rule engine

`Rule engine` - программный модуль, который:

1. Ищет формулировки и отрицания в тексте.
2. Заполняет evidence поля.
3. Создаёт числовые `Rule features`.
4. Определяет применимое правило `rule_id`.
5. Может формировать предварительный verdict для однозначных случаев.

Пример:

```python
if requires_external_fuel and fuel_included is False:
    rule_id = "DEVICE_WITHOUT_INCLUDED_FUEL"
    rule_verdict = 0
```

### 2.6. Fold

`Fold` - одна часть обучающего датасета. При пяти folds датасет делится на пять непересекающихся частей.

### 2.7. OOF

`OOF`, или `Out-of-Fold`, - предсказание для обучающего товара, сделанное моделью, которая не обучалась на этом товаре и его дублях.

Это таблица честных предсказаний и признаков.

### 2.8. Router

`Router` - небольшой классификатор или набор правил, который решает, нужно ли запускать следующий, более дорогой уровень обработки.

### 2.9. Meta-model

`Meta-model` - финальный классификатор, объединяющий доступные результаты текстовых моделей, rules, OCR, embeddings, retrieval и LoRA.

---

## 3. Входные и выходные данные

### 3.1. Входная карточка товара

```json
{
  "id": 12345,
  "category": "Легковоспламеняющиеся",
  "name": "Газовая горелка туристическая",
  "description": "Работает от внешнего баллона. Баллон приобретается отдельно.",
  "images": ["12345_1.jpg", "12345_2.jpg"],
  "label": 0
}
```

Поле `label` присутствует только в обучающих данных.

### 3.2. Выход системы

Внутренний результат:

```json
{
  "id": 12345,
  "probability": 0.11,
  "label": 0,
  "rule_id": "DEVICE_WITHOUT_INCLUDED_FUEL",
  "evidence": [
    {
      "field": "fuel_included",
      "value": false,
      "source": "description",
      "text": "баллон приобретается отдельно"
    }
  ]
}
```

Пользовательский результат:

```text
Газовая горелка работает от внешнего баллона, который не входит в комплект. Горючее содержимое в товаре не подтверждено. <не бан>
```

---

## 4. Общая архитектура
* в файле drawio

## 5. Раздельные модели для двух категорий

Категория товара известна заранее. Для неё выбирается отдельный набор классификаторов и правил:

```text
category = БАД
→ модели, evidence schema, rules и thresholds для БАД

category = Легковоспламеняющиеся
→ модели, evidence schema, rules и thresholds для легковоспламеняющихся
```

Причина разделения: категории определяются разными признаками и имеют разное распределение классов.

Для БАД важны:

- явная маркировка БАД;
- `dietary supplement`;
- отрицание принадлежности к БАД;
- признаки спортивного питания.

Для легковоспламеняющихся важны:

- наличие топлива;
- комплектность;
- внешний или встроенный источник топлива;
- самостоятельность горючего предмета.

Отдельными должны быть как минимум:

- TF-IDF-классификаторы;
- embedding-классификаторы;
- meta-model;
- калибратор;
- threshold;
- evidence schema и правила.

---

## 6. Этап 1. Проверка и подготовка данных

Необходимо проверить:

- точные и почти точные дубли;
- совпадающие `name + description`;
- одинаковые и близкие изображения;
- конфликтующие label;
- пропуски текста и изображений;
- повреждённые изображения;
- HTML, повторяющийся текст и мусорные символы;
- баланс классов по категориям.

Для каждого товара формируются:

```text
normalized_name
normalized_description
group_id
duplicate_type
label_conflict
image_count
fold
```

Одинаковые или тесно связанные карточки получают один `group_id`.

---

## 7. Этап 2. StratifiedGroupKFold

Используем `StratifiedGroupKFold`

Он одновременно:

1. Не разделяет одну группу дублей между train и validation.
2. Старается сохранить соотношение `label=0/1` в каждом fold.

При пяти folds:

```text
Раунд 1: train = folds 2–5, validation = fold 1
Раунд 2: train = folds 1,3,4,5, validation = fold 2
...
Раунд 5: train = folds 1–4, validation = fold 5
```

Пять folds дают разумный баланс между стабильностью оценки и стоимостью обучения. Для дорогой LoRA допустимо использовать три folds, если пятиразовое обучение не укладывается в вычислительный бюджет.

Все дубли товара должны находиться в одном fold во всех ветках.

---

## 8. Этап 3. Формализация правил

Формализация правил — перевод правил соревнования с человеческого языка в:

- evidence schema;
- словари и шаблоны выражений;
- обработку отрицаний;
- программные условия `if/else`;
- `rule_id`;
- числовые Rule features.

Пример:

```text
«баллон приобретается отдельно»
→ requires_external_fuel = true
→ fuel_included = false
→ rule_id = DEVICE_WITHOUT_INCLUDED_FUEL
```

Rule engine не заменяет модели. Он создаёт проверяемые признаки и особенно важен для отрицаний и условий комплектации.

---

## 9. Этап 4. Быстрая ветка

Быстрая ветка работает без изображений и запускается для каждого товара.

### 9.1. Text Rule features

Используются только:

```text
name + description
```

Примеры:

```text
text_has_bad_marker
text_has_explicit_not_bad
text_has_sport_nutrition
text_requires_external_fuel
text_fuel_included
text_has_negation
```

OCR-признаки в быструю ветку попадать не должны.

### 9.2. TF-IDF

TF-IDF преобразует текст в разреженный числовой вектор.

Используются два представления:

```text
Word TF-IDF      → слова и последовательности слов;
Character TF-IDF → последовательности символов.
```

Word TF-IDF хорошо распознаёт точные фразы, character TF-IDF устойчив к опечаткам, окончаниям и вариантам написания.

### 9.3. Классификаторы

Поверх TF-IDF обучаются:

- `LinearSVC` — сильный классификатор для разреженного текста;
- `Logistic Regression` — источник вероятности и дополнительного сигнала.

Результаты:

```text
word_svc_margin
char_svc_margin
text_logreg_probability
text_models_disagree
```

### 9.4. OOF быстрых моделей

Для каждого validation-fold сохраняются предсказания моделей, обученных на остальных folds.

Пример строки:

```json
{
  "id": 101,
  "fold": 1,
  "true_label": 0,
  "word_svc_margin": -1.21,
  "char_svc_margin": -0.73,
  "text_probability": 0.24,
  "text_requires_external_fuel": 1,
  "text_fuel_included": 0
}
```

OOF быстрых моделей используется для meta-model, Router 1, Router 2, hard-example mining и настройки порогов.

---

## 10. Этап 5. Дорогая мультимодальная ветка

На обучении дорогая ветка рассчитывается для всех товаров, чтобы измерить её полезность и обучить router. На инференсе она запускается только по решению Router 1.

### 10.1. OCR

OCR обрабатывает каждое изображение и извлекает текст с упаковки.

OCR не принимает финальный verdict. Его результат используется как дополнительный текст и источник evidence.

Сохраняются:

```text
ocr_text_by_image
ocr_quality
source_image_id
```

### 10.2. OCR Rule features

Rule engine повторно применяется после OCR, теперь к:

```text
name + description + OCR-текст
```

Примеры:

```text
ocr_has_bad_marker
ocr_has_dietary_supplement
ocr_has_fuel_warning
ocr_fuel_included
text_ocr_disagreement
```

OCR Rule features относятся к дорогой ветке.

### 10.3. Multimodal embeddings

`Qwen3-VL-Embedding-2B` преобразует текст и изображения в числовые векторы.

Планируемые представления:

```text
text_embedding
image_embedding_i
joint_text_image_embedding_i
aggregated_image_embedding
```

Поверх embeddings обучается отдельный классификатор, например Logistic Regression или небольшой MLP.

Результаты:

```text
embedding_probability
text_image_similarity
image_disagreement
```

### 10.4. Fold-safe retrieval

Retrieval ищет ближайшие размеченные товары по embeddings.

Во время OOF-валидации validation-товар ищет соседей только в train-части текущего fold.

Признаки:

```text
positive_share_k_neighbors
nearest_positive_distance
nearest_negative_distance
neighbor_margin
ood_distance
```

На финальном инференсе поиск выполняется по индексу всего обучающего датасета.

### 10.5. OOF дорогих моделей

Сюда входят:

- предсказания supervised-классификатора поверх embeddings;
- OCR-признаки;
- OCR Rule features;
- fold-safe retrieval-признаки;
- признаки конфликта текста и изображений.

Предобученные OCR и embedding-модель можно закэшировать для всех товаров, так как они не используют label. Однако supervised-классификаторы поверх embeddings должны выдавать OOF-предсказания, а retrieval обязан быть fold-safe.

---

## 11. Этап 6. Создание LoRA dataset

LoRA dataset — расширенный датасет вида:

```text
товар + изображения + инструкция
→ evidence + rule_id + verdict
```

### 11.1. Вход LoRA

```text
category
name
description
OCR-текст
1–5 изображений
фиксированная инструкция
```

### 11.2. Target LoRA

Пример для легковоспламеняющихся:

```json
{
  "fuel_present": "unknown",
  "fuel_included": false,
  "requires_external_fuel": true,
  "rule_id": "DEVICE_WITHOUT_INCLUDED_FUEL",
  "verdict": 0
}
```

Пример для БАД:

```json
{
  "explicit_bad_marker": true,
  "dietary_supplement_marker": false,
  "explicit_not_bad": false,
  "sport_nutrition": false,
  "marker_source": "description",
  "rule_id": "EXPLICIT_BAD_MARKER",
  "verdict": 1
}
```

### 11.3. Уровни качества разметки

```text
gold        → evidence проверен человеком;
silver_high → evidence автоматически получен однозначным правилом;
silver      → автоматический evidence с меньшей уверенностью;
label_only  → известен verdict, промежуточные поля могут быть unknown.
```

Silver-разметку автоматически создают rule engine и OCR. Сложные, конфликтующие и важные товары автоматически попадают в очередь на ручную проверку. Человек исправляет evidence, указывает источник и подтверждает target, после чего пример получает статус `gold`.

Исходный label определяет verdict, но не должен использоваться для выдумывания evidence. При недостатке данных значение остаётся `unknown`.

---

## 12. Этап 7. LoRA v1

Базовая модель:

```text
Qwen3-VL-2B-Instruct
```

LoRA — небольшая обучаемая надстройка. Основные веса Qwen замораживаются, обучаются дополнительные LoRA-параметры.

LoRA v1 учится:

- анализировать текст и изображения;
- заполнять evidence schema;
- выбирать `rule_id`;
- возвращать валидный JSON;
- выдавать verdict.

### 12.1. OOF LoRA v1

Для получения честных предсказаний создаётся несколько fold-адаптеров:

```text
обучение без fold 1 → предсказание fold 1;
обучение без fold 2 → предсказание fold 2;
...
```

Сохраняются:

```text
predicted_verdict
predicted_evidence
rule_id
json_valid
evidence_supported
confidence
latency
```

LoRA v1 не является финальной моделью. Она нужна для поиска реальных слабых мест.

---

## 13. Этап 8. Hard-example mining и LoRA v2

В анализ ошибок поступают:

```text
OOF быстрых моделей
+ OOF дорогих моделей
+ OOF LoRA v1
+ true label
```

Ищутся:

- уверенные false positives;
- уверенные false negatives;
- конфликты текста и изображений;
- ошибки evidence;
- невалидный JSON;
- близкие товары противоположных классов;
- повторяющиеся типы ошибок.

Выбранные примеры группируются, дедуплицируются и направляются на ручную проверку. После проверки они добавляются в LoRA dataset v2.

```text
LoRA dataset v1
+ проверенные hard examples
+ исправленные evidence
→ LoRA dataset v2
→ Qwen3-VL-2B + LoRA v2
```

LoRA v2 заменяет v1 в финальной системе.

### 13.1. OOF LoRA v2

LoRA v2 также должна получить честные OOF-предсказания. Они используются для:

- сравнения v1 и v2;
- обучения Router 2;
- обучения meta-model;
- финальной оценки качества evidence и verdict.

---

## 14. Этап 9. Selective routing

Router должен быть разделён на два последовательных компонента.

### 14.1. Router 1: нужны ли фотографии

Входные признаки:

```text
OOF быстрых моделей
```

Цель обучения формируется сравнением быстрых и дорогих OOF-результатов:

```text
images_helped = 1
```

если дорогая ветка исправила ошибку или дала значимое улучшение.

OOF дорогих моделей используется для создания target, но не как вход Router 1.

На инференсе Router 1 получает только признаки, доступные до запуска фотографий.

### 14.2. Router 2: нужна ли LoRA

Входные признаки:

```text
OOF быстрых моделей
+ OOF дорогих моделей
+ признаки конфликтов
```

Цель формируется сравнением результата без LoRA и OOF LoRA v2:

```text
lora_helped = 1
```

если LoRA исправила ошибку или дала значимое улучшение.

Результат LoRA используется только для формирования target. Он не является входным признаком Router 2.

### 14.3. Пороги router

Каждый router выдаёт вероятность полезности следующего уровня:

```text
P(images_will_help)
P(lora_will_help)
```

Routing thresholds выбираются по OOF с учётом:

- итогового Macro F1;
- доли направленных товаров;
- фактического времени обработки;
- ограничения GPU.

### 14.4. OOF router

Решения router, используемые для обучения meta-model, также должны быть OOF. Router обучается на одних meta-folds и выбирает маршруты для невиденного meta-fold.

---

## 15. Имитация routing и маски доступности

На обучении мы заранее рассчитали все дорогие признаки, но на инференсе router будет скрывать часть веток.

Поэтому перед обучением meta-model воспроизводятся OOF-решения Router 1 и Router 2.

### 15.1. Возможные маршруты

```text
Маршрут A: быстрые признаки;
Маршрут B: быстрые + OCR + embeddings + retrieval;
Маршрут C: быстрые + OCR + embeddings + retrieval + LoRA.
```

Незапущенные признаки заменяются на `NaN` или согласованное служебное значение.

### 15.2. Маски доступности

Добавляются признаки:

```text
has_ocr
has_embeddings
has_retrieval
has_lora
```

Пример:

```json
{
  "has_ocr": 1,
  "has_embeddings": 1,
  "has_retrieval": 1,
  "has_lora": 0
}
```

Маски позволяют отличить:

```text
модель выдала нулевой результат
```

от:

```text
модель вообще не запускалась.
```

Основной блок на схеме рекомендуется называть:

```text
Имитация маршрутов по OOF-решениям Router 1 и Router 2
```

---

## 16. Meta-model

Meta-model не получает исходные фотографии. Она получает числовые результаты доступных компонентов.

Пример входа:

```json
{
  "text_probability": 0.74,
  "char_svc_margin": 1.42,
  "text_has_bad_marker": 0,
  "ocr_has_bad_marker": 1,
  "embedding_probability": 0.81,
  "positive_share_neighbors": 0.8,
  "lora_verdict": 1,
  "lora_json_valid": 1,
  "has_ocr": 1,
  "has_embeddings": 1,
  "has_retrieval": 1,
  "has_lora": 1
}
```

Meta-model учится определять:

- каким моделям доверять в каждой категории;
- как учитывать конфликты;
- как интерпретировать отсутствующие признаки;
- когда rules важнее общей семантики;
- когда LoRA ненадёжна из-за невалидного JSON или отсутствующего evidence.

Для каждой категории обучается отдельная meta-model.

Начальный вариант — Logistic Regression. Более сложная модель допускается только при устойчивом OOF-приросте.

---

## 17. Калибровка и threshold

Meta-model выдаёт raw score. Калибратор преобразует его в более корректную вероятность:

```text
P(label=1)
```

После этого применяется отдельный threshold для каждой категории:

```python
label = int(probability >= category_threshold)
```

Threshold выбирается только по OOF-предсказаниям с максимизацией competition metric.

```text
threshold_BAD
threshold_FLAMMABLE
```

---

## 18. Формирование объяснения

Свободная генерация объяснения не требуется. После определения label выбирается шаблон по `rule_id` и evidence.

Пример:

```text
rule_id = DEVICE_WITHOUT_INCLUDED_FUEL
fuel_included = false
requires_external_fuel = true
```

Шаблон:

```text
Газовая горелка работает от внешнего баллона, который не входит в комплект. Горючее содержимое в товаре не подтверждено.
```

Требования:

- объяснение использует только подтверждённый evidence;
- объяснение согласовано с verdict;
- соблюдается ограничение длины;
- при отсутствии evidence используется безопасный общий шаблон без выдуманных фактов.

---

## 19. Полный процесс обучения

1. Загрузить исходный датасет.
2. Нормализовать текст и проверить изображения.
3. Найти дубли, конфликты и сформировать `group_id`.
4. Создать `StratifiedGroupKFold`.
5. Формализовать правила отдельно для двух категорий.
6. Посчитать Text Rule features.
7. Обучить TF-IDF-модели по folds и получить OOF быстрых моделей.
8. Посчитать OCR для всех train-изображений и закэшировать результат.
9. Получить OCR Rule features.
10. Посчитать multimodal embeddings.
11. Выполнить fold-safe retrieval.
12. Обучить supervised-классификаторы дорогой ветки по folds и получить OOF дорогих моделей.
13. Сформировать silver-разметку.
14. Отправить сложные случаи на ручную проверку и получить gold-разметку.
15. Собрать LoRA dataset v1.
16. Обучить LoRA v1 по folds и получить OOF LoRA v1.
17. Выполнить hard-example mining.
18. Проверить и исправить выбранные примеры.
19. Собрать LoRA dataset v2.
20. Обучить LoRA v2 по folds и получить OOF LoRA v2.
21. Обучить OOF Router 1.
22. Обучить OOF Router 2.
23. Имитировать маршруты и создать маски доступности.
24. Обучить отдельные meta-model для двух категорий.
25. Получить OOF meta-предсказания.
26. Выполнить калибровку и подобрать thresholds.
27. Зафиксировать архитектуру и гиперпараметры.
28. Переобучить финальные базовые модели и LoRA v2 на всём train.
29. Построить retrieval-индекс по всему train.
30. Провести полный timed dry run.

---

## 20. Полный процесс инференса

Для нового товара:

1. Определить известную категорию.
2. Нормализовать `name + description`.
3. Получить Text Rule features.
4. Получить word/char TF-IDF scores.
5. Передать быстрые признаки в Router 1.
6. Если Router 1 отклонил фотографии:
   - установить `has_ocr=0`, `has_embeddings=0`, `has_retrieval=0`, `has_lora=0`;
   - передать быстрые признаки в meta-model.
7. Если Router 1 разрешил фотографии:
   - параллельно запустить OCR и multimodal embeddings;
   - получить OCR Rule features;
   - выполнить retrieval по всему train-индексу;
   - установить маски дорогих компонентов в `1`;
   - передать признаки в Router 2.
8. Если Router 2 отклонил LoRA:
   - установить `has_lora=0`;
   - передать быстрые и дорогие признаки в meta-model.
9. Если Router 2 разрешил LoRA:
   - передать в Qwen3-VL-2B + LoRA v2 исходный текст, изображения, OCR и инструкцию;
   - распарсить JSON;
   - добавить LoRA-признаки и `has_lora=1`;
   - передать все признаки в meta-model.
10. Применить категорийную meta-model.
11. Откалибровать `P(label=1)`.
12. Сравнить вероятность с категорийным threshold.
13. Получить `label=0/1`.
14. Выбрать evidence и `rule_id`.
15. Сформировать объяснение и вердикт.

---

## 21. Связь обучения и инференса

| Обучение | Что получается | Использование на инференсе |
|---|---|---|
| TF-IDF по folds | OOF scores и финальные vectorizer/classifier | Быстрая оценка каждого товара |
| Text Rules | Быстрые rule features | Доступны до обработки фотографий |
| OCR train-изображений | OCR cache | Запускается по решению Router 1 |
| OCR Rules | Дорогие rule features | Используются после OCR |
| Embedding folds | OOF embedding scores | Запускаются по решению Router 1 |
| Fold-safe retrieval | Честные neighbor-признаки | На тесте используется весь train-индекс |
| LoRA v1 OOF | Реальные ошибки первой версии | В финале не используется |
| Hard-example mining | Исправленные сложные примеры | Улучшает LoRA v2 |
| LoRA v2 OOF | Честные verdict/evidence | На тесте LoRA v2 запускается через Router 2 |
| Router 1 OOF | Политика запуска изображений | Решает, нужны ли OCR и embeddings |
| Router 2 OOF | Политика запуска LoRA | Решает, нужна ли LoRA |
| Routing simulation | Датасет с реальными масками | Meta-model умеет работать с любым маршрутом |
| Meta OOF | Честные финальные scores | Финальное объединение сигналов |
| Calibration/threshold | Категорийные пороги | Превращение вероятности в label |

Главное требование: признаки, доступные meta-model на определённом маршруте обучения, должны точно совпадать с признаками того же маршрута на инференсе.

---

## 22. Защита от утечек

Запрещено:

- размещать дубли одного товара в разных folds;
- обучать модель на validation-товаре перед его OOF-предсказанием;
- искать retrieval-соседей validation-товара во всём датасете;
- передавать true label или правильный target в LoRA input;
- обучать meta-model на train-предсказаниях базовых моделей;
- использовать результат LoRA как вход Router 2 при решении, запускать ли LoRA;
- использовать OCR-признаки в быстрой ветке;
- использовать LoRA v1 в meta-model, если на инференсе запускается только v2;
- выбирать threshold по test или public leaderboard вместо OOF.

Разрешено заранее посчитать для всех train-товаров признаки замороженных моделей, не использующих label, например OCR-текст и frozen embeddings. Но любые supervised heads поверх них должны оцениваться OOF.

---

## 23. Метрики и критерии принятия компонентов

Основные метрики:

```text
F1_BAD
F1_FLAMMABLE
mean_F1 = (F1_BAD + F1_FLAMMABLE) / 2
```

Дополнительно контролируются:

- precision и recall по категориям;
- recall редкого положительного класса;
- JSON validity LoRA;
- evidence accuracy/F1 на gold-наборе;
- hallucinated evidence rate;
- доля товаров, направленных Router 1 и Router 2;
- время каждого каскада;
- полное время инференса;
- стабильность результата по folds.

Компонент включается в финал, если он:

- даёт устойчивый OOF-прирост;
- исправляет важный класс ошибок;
- не ухудшает существенно одну из категорий;
- или сохраняет качество при заметном сокращении времени.

---

## 24. Планируемые артефакты

```text
data/processed/manifest.parquet
data/processed/folds.parquet
data/processed/duplicate_groups.parquet
data/processed/label_conflicts.parquet

cache/ocr/*.parquet
cache/embeddings/*.npy

features/text_rules.parquet
features/ocr_rules.parquet
features/retrieval.parquet

annotations/lora_v1.jsonl
annotations/hard_examples.parquet
annotations/lora_v2.jsonl

oof/fast.parquet
oof/expensive.parquet
oof/lora_v1.parquet
oof/lora_v2.parquet
oof/router_1.parquet
oof/router_2.parquet
oof/meta.parquet

artifacts/bad/*
artifacts/flammable/*
artifacts/lora_v2/*
artifacts/retrieval_index/*
artifacts/thresholds.json
artifacts/explanation_templates.json
```

---

## 25. Краткое резюме системы

```text
1. Dataset содержит карточки товаров и правильные label.
2. StratifiedGroupKFold создаёт честные train/validation-разделения.
3. Быстрая ветка анализирует name + description.
4. Router 1 решает, нужны ли фотографии.
5. OCR читает упаковку, embeddings анализируют общий смысл, retrieval ищет похожие товары.
6. Router 2 решает, нужна ли LoRA.
7. LoRA v2 извлекает evidence и verdict для сложных случаев.
8. Маски показывают meta-model, какие компоненты запускались.
9. Meta-model объединяет доступные сигналы.
10. Калибровка и threshold превращают результат в label.
11. Шаблон формирует объяснение только из подтверждённого evidence.
```

Обучение и инференс связаны OOF-предсказаниями: обучение имитирует ровно те маршруты и наборы признаков, которые появятся при обработке новых товаров.
