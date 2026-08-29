# R08 — evidence schema

## Контракт

- Источник схемы: `configs/evidence.yaml`.
- Версия: **1**.
- Значение недостатка информации: `unknown`.
- `unknown` допустим в полях target, но не создаётся как подтверждённый evidence.
- Исходный `label` не входит в evidence и не может подтверждать факт.

## Источники

| Source | Нужен text | Нужен image_index |
|---|---:|---:|
| `name` | true | false |
| `description` | true | false |
| `ocr` | true | true |
| `image` | false | true |

## Категория: БАД

| Поле | Тип | Допустимые значения | Описание |
|---|---|---|---|
| `explicit_bad_marker` | `tri_state` | true / false / unknown | Прямое указание, что товар является БАД. |
| `dietary_supplement_marker` | `tri_state` | true / false / unknown | Маркировка dietary supplement. |
| `explicit_not_bad` | `tri_state` | true / false / unknown | Явное указание, что товар не является БАД. |
| `sport_nutrition` | `tri_state` | true / false / unknown | Прямое указание на спортивное питание. |
| `marker_source` | `enum` | name / description / ocr / image / unknown | Источник найденной маркировки БАД. |

Допустимые `rule_id`: `EXPLICIT_BAD_MARKER`, `DIETARY_SUPPLEMENT_MARKER`, `EXPLICIT_NOT_BAD`, `SPORT_NUTRITION`, `NO_BAD_MARKER`, `INSUFFICIENT_EVIDENCE`.

## Категория: Легковоспламеняющиеся

| Поле | Тип | Допустимые значения | Описание |
|---|---|---|---|
| `object_type` | `enum` | independent_ignition_source / combustible_content / device_requiring_external_fuel / device_with_embedded_ignition / flammable_component / kit / other / unknown | Тип объекта относительно правил воспламеняемости. |
| `fuel_present` | `tri_state` | true / false / unknown | В товаре присутствует горючее вещество или газ. |
| `fuel_included` | `tri_state` | true / false / unknown | Горючее содержимое входит в продаваемый комплект. |
| `requires_external_fuel` | `tri_state` | true / false / unknown | Для работы требуется внешний источник топлива. |
| `independent_ignition_source` | `tri_state` | true / false / unknown | Товар является самостоятельным источником воспламенения. |
| `flammable_item_in_kit` | `tri_state` | true / false / unknown | В комплект входит отдельный легковоспламеняющийся товар. |
| `explicit_without_fuel` | `tri_state` | true / false / unknown | Явно указано отсутствие топлива или горючего содержимого. |

Допустимые `rule_id`: `INDEPENDENT_IGNITION_SOURCE`, `FLAMMABLE_CONTENT`, `FLAMMABLE_ITEM_IN_KIT`, `DEVICE_WITHOUT_INCLUDED_FUEL`, `DEVICE_WITH_EMBEDDED_IGNITION`, `FLAMMABLE_COMPONENT_ONLY`, `NO_FLAMMABLE_CONTENT`, `INSUFFICIENT_EVIDENCE`.

## Примеры evidence

### БАД

```json
{
  "field": "explicit_bad_marker",
  "value": true,
  "source": "description",
  "text": "Биологически активная добавка к пище"
}
```

### Легковоспламеняющиеся

```json
{
  "field": "fuel_included",
  "value": false,
  "source": "description",
  "text": "Баллон приобретается отдельно"
}
```

## Примеры LoRA target

### БАД

```json
{
  "explicit_bad_marker": true,
  "dietary_supplement_marker": "unknown",
  "explicit_not_bad": "unknown",
  "sport_nutrition": "unknown",
  "marker_source": "description",
  "rule_id": "EXPLICIT_BAD_MARKER",
  "verdict": 1
}
```

### Легковоспламеняющиеся

```json
{
  "object_type": "device_requiring_external_fuel",
  "fuel_present": "unknown",
  "fuel_included": false,
  "requires_external_fuel": true,
  "independent_ignition_source": "unknown",
  "flammable_item_in_kit": "unknown",
  "explicit_without_fuel": true,
  "rule_id": "DEVICE_WITHOUT_INCLUDED_FUEL",
  "verdict": 0
}
```

## Границы R08

R08 фиксирует только имена, типы, источники и сериализацию. Словари, регулярные выражения, отрицания и условия `if/else` реализуются в R09 и R12.
