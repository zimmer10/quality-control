"""Reproducible dataset audit for the E-CUP training table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

EXPECTED_COLUMNS = ("id", "name", "description", "category", "label")
EXPECTED_CATEGORIES = ("БАД", "Легковоспламеняющиеся")
EXPECTED_LABELS = (0, 1)


@dataclass(frozen=True)
class ValidationSummary:
    """Structural checks that do not mutate the source table."""

    row_count: int
    schema: tuple[tuple[str, str], ...]
    unexpected_columns: tuple[str, ...]
    index_like_columns: tuple[str, ...]
    unique_id_count: int
    duplicate_id_rows: int
    unknown_categories: tuple[str, ...]
    invalid_labels: tuple[str, ...]


@dataclass(frozen=True)
class AuditResult:
    """All tables and counters needed to render the R03 report."""

    validation: ValidationSummary
    class_balance: pl.DataFrame
    missing_values: pl.DataFrame
    exact_duplicates: pl.DataFrame
    near_duplicates: pl.DataFrame


def load_dataset(path: str | Path) -> pl.DataFrame:
    """Load the competition CSV without discarding unexpected columns."""

    return pl.read_csv(Path(path), infer_schema_length=10_000)


def validate_required_columns(frame: pl.DataFrame) -> None:
    """Reject a table that cannot satisfy the shared dataset contract."""

    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def _is_zero_based_index(frame: pl.DataFrame, column: str) -> bool:
    values = frame.get_column(column).cast(pl.Int64, strict=False)
    if values.null_count() or len(values) != frame.height:
        return False
    return values.to_list() == list(range(frame.height))


def validation_summary(frame: pl.DataFrame) -> ValidationSummary:
    """Collect schema, identifier and categorical-domain diagnostics."""

    validate_required_columns(frame)
    unexpected = tuple(column for column in frame.columns if column not in EXPECTED_COLUMNS)
    index_like = tuple(
        column for column in unexpected if _is_zero_based_index(frame, column)
    )
    unknown_categories = tuple(
        str(value)
        for value in sorted(
            set(frame.get_column("category").drop_nulls().unique().to_list())
            - set(EXPECTED_CATEGORIES)
        )
    )
    invalid_labels = tuple(
        str(value)
        for value in sorted(
            set(frame.get_column("label").drop_nulls().unique().to_list())
            - set(EXPECTED_LABELS),
            key=str,
        )
    )

    return ValidationSummary(
        row_count=frame.height,
        schema=tuple((name, str(dtype)) for name, dtype in frame.schema.items()),
        unexpected_columns=unexpected,
        index_like_columns=index_like,
        unique_id_count=frame.get_column("id").n_unique(),
        duplicate_id_rows=int(frame.get_column("id").is_duplicated().sum()),
        unknown_categories=unknown_categories,
        invalid_labels=invalid_labels,
    )


def class_balance(frame: pl.DataFrame) -> pl.DataFrame:
    """Count labels and their shares independently inside each category."""

    validate_required_columns(frame)
    return (
        frame.group_by(["category", "label"])
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum().over("category")).alias(
                "share"
            )
        )
        .sort(["category", "label"])
    )


def missing_value_summary(
    frame: pl.DataFrame,
    columns: Sequence[str] = ("name", "description"),
) -> pl.DataFrame:
    """Separate actual nulls from non-null strings containing only whitespace."""

    rows: list[dict[str, int | str]] = []
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"missing column for null audit: {column}")
        values = pl.col(column).cast(pl.String)
        counts = frame.select(
            values.is_null().sum().alias("null_count"),
            (values.is_not_null() & values.str.strip_chars().eq(""))
            .sum()
            .alias("blank_count"),
        ).row(0, named=True)
        null_count = int(counts["null_count"])
        blank_count = int(counts["blank_count"])
        rows.append(
            {
                "column": column,
                "null_count": null_count,
                "blank_count": blank_count,
                "missing_total": null_count + blank_count,
            }
        )
    return pl.DataFrame(rows)


def normalized_text(column: str) -> pl.Expr:
    """Build a conservative fingerprint insensitive to case and punctuation."""

    return (
        pl.col(column)
        .cast(pl.String)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def exact_duplicate_groups(frame: pl.DataFrame) -> pl.DataFrame:
    """Find repeated raw name/description pairs inside one task category."""

    validate_required_columns(frame)
    return (
        frame.group_by(["category", "name", "description"], maintain_order=True)
        .agg(
            pl.len().alias("row_count"),
            pl.col("id").sort().alias("ids"),
            pl.col("label").unique().sort().alias("labels"),
        )
        .filter(pl.col("row_count") > 1)
        .with_columns(
            (pl.col("labels").list.len() > 1).alias("label_conflict")
        )
        .sort("row_count", descending=True)
    )


def near_duplicate_groups(frame: pl.DataFrame) -> pl.DataFrame:
    """Find raw text variants that collapse to one normalized fingerprint."""

    validate_required_columns(frame)
    prepared = frame.with_columns(
        normalized_text("name").alias("_normalized_name"),
        normalized_text("description").alias("_normalized_description"),
        pl.concat_str(
            [pl.col("name").fill_null(""), pl.col("description").fill_null("")],
            separator="\u241f",
        ).alias("_raw_signature"),
    )
    return (
        prepared.group_by(
            ["category", "_normalized_name", "_normalized_description"],
            maintain_order=True,
        )
        .agg(
            pl.len().alias("row_count"),
            pl.col("_raw_signature").n_unique().alias("raw_variants"),
            pl.col("id").sort().alias("ids"),
            pl.col("label").unique().sort().alias("labels"),
            pl.col("name").drop_nulls().unique().sort().alias("name_variants"),
        )
        .filter((pl.col("row_count") > 1) & (pl.col("raw_variants") > 1))
        .with_columns(
            (pl.col("labels").list.len() > 1).alias("label_conflict")
        )
        .sort("row_count", descending=True)
    )


def audit_dataset(frame: pl.DataFrame) -> AuditResult:
    """Run all R03 checks over an already loaded table."""

    return AuditResult(
        validation=validation_summary(frame),
        class_balance=class_balance(frame),
        missing_values=missing_value_summary(frame),
        exact_duplicates=exact_duplicate_groups(frame),
        near_duplicates=near_duplicate_groups(frame),
    )


def _display_column(name: str) -> str:
    return "<empty>" if name == "" else name


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _format_columns(columns: Sequence[str]) -> str:
    if not columns:
        return "нет"
    return ", ".join(f"`{_display_column(column)}`" for column in columns)


def _duplicate_counts(frame: pl.DataFrame) -> tuple[int, int, int, int]:
    if frame.is_empty():
        return 0, 0, 0, 0
    groups = frame.height
    rows = int(frame.get_column("row_count").sum())
    extra_rows = int((frame.get_column("row_count") - 1).sum())
    conflicts = int(frame.get_column("label_conflict").sum())
    return groups, rows, extra_rows, conflicts


def _conflict_examples(frame: pl.DataFrame, limit: int = 10) -> list[str]:
    conflicts = frame.filter(pl.col("label_conflict")).head(limit)
    if conflicts.is_empty():
        return ["Конфликтов не найдено."]

    lines = ["| Категория | Название | ID | Метки |", "|---|---|---|---|"]
    for row in conflicts.iter_rows(named=True):
        name = row.get("name")
        if name is None:
            variants = row.get("name_variants") or []
            name = variants[0] if variants else ""
        short_name = str(name)
        if len(short_name) > 80:
            short_name = f"{short_name[:77]}..."
        ids = ", ".join(str(value) for value in row["ids"][:8])
        if len(row["ids"]) > 8:
            ids = f"{ids}, ..."
        labels = ", ".join(str(value) for value in row["labels"])
        lines.append(
            "| "
            + " | ".join(
                _escape_markdown(value)
                for value in (row["category"], short_name, ids, labels)
            )
            + " |"
        )
    return lines


def render_report(result: AuditResult, input_path: str | Path) -> str:
    """Render a compact, reviewable Markdown report from an audit result."""

    validation = result.validation
    exact_groups, exact_rows, exact_extra, exact_conflicts = _duplicate_counts(
        result.exact_duplicates
    )
    near_groups, near_rows, near_extra, near_conflicts = _duplicate_counts(
        result.near_duplicates
    )

    lines = [
        "# R03 — аудит `data.csv`",
        "",
        "## Входные данные и схема",
        "",
        f"- Файл: `{input_path}`.",
        f"- Строк: **{validation.row_count}**.",
        f"- Уникальных `id`: **{validation.unique_id_count}**.",
        f"- Строк с повторяющимся `id`: **{validation.duplicate_id_rows}**.",
        f"- Неожиданные столбцы: {_format_columns(validation.unexpected_columns)}.",
        f"- Индексоподобные столбцы: {_format_columns(validation.index_like_columns)}.",
        f"- Неизвестные категории: {', '.join(validation.unknown_categories) or 'нет'}.",
        f"- Недопустимые метки: {', '.join(validation.invalid_labels) or 'нет'}.",
        "",
        "| Столбец | Тип Polars |",
        "|---|---|",
    ]
    lines.extend(
        f"| `{_display_column(name)}` | `{dtype}` |"
        for name, dtype in validation.schema
    )
    lines.extend(
        [
            "",
            "## Баланс классов",
            "",
            "| Категория | Label | Количество | Доля внутри категории |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result.class_balance.iter_rows(named=True):
        lines.append(
            f"| {_escape_markdown(row['category'])} | {row['label']} | "
            f"{row['count']} | {row['share']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Полярность `label`",
            "",
            "Согласно условию соревнования:",
            "",
            "- `label = 1` — товар качественный: соответствует правилам отнесения к указанной категории.",
            "- `label = 0` — товар некачественный: не соответствует правилам отнесения к указанной категории.",
            "",
            "Таким образом, положительный класс метрики — `label = 1`.",
            "",
            "## Пропуски",
            "",
            "| Столбец | Null | Пустая строка | Всего отсутствует |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result.missing_values.iter_rows(named=True):
        lines.append(
            f"| `{row['column']}` | {row['null_count']} | {row['blank_count']} | "
            f"{row['missing_total']} |"
        )

    lines.extend(
        [
            "",
            "## Дубли и конфликты",
            "",
            "Точный дубль — совпадение исходных `name` и `description` внутри одной категории.",
            "",
            f"- Групп точных дублей: **{exact_groups}**.",
            f"- Строк в группах: **{exact_rows}**.",
            f"- Повторных строк сверх первой: **{exact_extra}**.",
            f"- Групп с конфликтующими `label`: **{exact_conflicts}**.",
            "",
            "Кандидат в почти-дубли — разные исходные тексты, которые совпали после приведения к нижнему регистру, удаления пунктуации и нормализации пробелов.",
            "",
            f"- Групп кандидатов: **{near_groups}**.",
            f"- Строк в группах: **{near_rows}**.",
            f"- Повторных строк сверх первой: **{near_extra}**.",
            f"- Групп с конфликтующими `label`: **{near_conflicts}**.",
            "",
            "### Примеры конфликтов среди точных дублей",
            "",
        ]
    )
    lines.extend(_conflict_examples(result.exact_duplicates))
    lines.extend(
        [
            "",
            "### Примеры конфликтов среди кандидатов в почти-дубли",
            "",
        ]
    )
    lines.extend(_conflict_examples(result.near_duplicates))
    lines.extend(
        [
            "",
            "## Выводы для следующих этапов",
            "",
            "- Пустой индексный столбец не использовать как признак.",
            "- Пропуски `description` учитывать отдельным признаком и безопасно заполнять пустой строкой при текстовой обработке.",
            "- При построении folds учитывать сильный дисбаланс классов внутри категорий.",
            "- В R04 присвоить единый `group_id` точным и подтверждённым почти-дублям, затем не разделять одну группу между folds.",
            "- Конфликтующие метки не исправлять автоматически: сохранить их для отдельной проверки и анализа ошибок.",
            "",
            "R03 только фиксирует кандидатов и статистику; артефакты `group_id` создаются в R04.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the E-CUP training CSV")
    parser.add_argument("--input", default="data/raw/data.csv", type=Path)
    parser.add_argument("--output", default="reports/R03-audit.md", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_dataset(args.input)
    result = audit_dataset(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(result, args.input), encoding="utf-8")
    print(f"Audit complete: {result.validation.row_count} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

