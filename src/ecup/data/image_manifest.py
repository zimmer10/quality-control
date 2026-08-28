"""Build and audit a deterministic manifest of product images."""

from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
from PIL import Image

from ecup.data.audit import load_dataset

MANIFEST_COLUMNS = (
    "id",
    "image_index",
    "relative_path",
    "extension",
    "format",
    "width",
    "height",
    "mode",
    "file_size_bytes",
    "in_dataset",
    "status",
    "error",
)
MANIFEST_SCHEMA = {
    "id": pl.Int64,
    "image_index": pl.Int64,
    "relative_path": pl.String,
    "extension": pl.String,
    "format": pl.String,
    "width": pl.Int64,
    "height": pl.Int64,
    "mode": pl.String,
    "file_size_bytes": pl.Int64,
    "in_dataset": pl.Boolean,
    "status": pl.String,
    "error": pl.String,
}
SERVICE_NAMES = frozenset({"__MACOSX", ".DS_Store", "Thumbs.db"})


@dataclass(frozen=True)
class ImageArtifacts:
    """The row-level manifest and service paths excluded from it."""

    manifest: pl.DataFrame
    ignored_paths: tuple[str, ...]


@dataclass(frozen=True)
class ManifestSummary:
    """Counters used by validation, reports and downstream planning."""

    dataset_items: int
    image_files: int
    readable_files: int
    unreadable_files: int
    items_with_files: int
    items_with_readable_files: int
    missing_ids: tuple[int, ...]
    ids_without_readable_files: tuple[int, ...]
    orphan_ids: tuple[int, ...]
    orphan_files: int
    invalid_id_paths: int
    invalid_image_indices: int
    duplicate_slots: int


def validate_dataset_ids(frame: pl.DataFrame) -> set[int]:
    """Return unique, non-null product identifiers from data.csv."""

    if "id" not in frame.columns:
        raise ValueError("source dataset misses required column: id")
    ids = frame.get_column("id")
    if frame.is_empty():
        raise ValueError("source dataset is empty")
    if ids.null_count():
        raise ValueError("source id contains null values")
    if ids.n_unique() != frame.height:
        raise ValueError("source id values must be unique")
    try:
        normalized = {int(value) for value in ids.to_list()}
    except (TypeError, ValueError) as error:
        raise ValueError("source id values must be integers") from error
    if any(value < 0 for value in normalized):
        raise ValueError("source id values must be non-negative")
    return normalized


def _is_service_path(relative_path: Path) -> bool:
    return any(
        part in SERVICE_NAMES or part.startswith(".")
        for part in relative_path.parts
    )


def discover_image_files(
    images_root: str | Path,
) -> tuple[list[Path], tuple[str, ...]]:
    """Recursively discover files while excluding OS metadata."""

    root = Path(images_root)
    if not root.exists():
        raise FileNotFoundError(f"images root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"images root is not a directory: {root}")

    files: list[Path] = []
    ignored: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        relative_text = relative.as_posix()
        if _is_service_path(relative):
            ignored.append(relative_text)
        else:
            files.append(path)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    return files, tuple(sorted(ignored))


def _parse_non_negative_integer(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _inspect_image(
    path: Path,
    images_root: Path,
    dataset_ids: set[int],
) -> dict[str, object]:
    relative = path.relative_to(images_root)
    product_id = (
        _parse_non_negative_integer(relative.parts[0])
        if len(relative.parts) >= 2
        else None
    )
    image_index = _parse_non_negative_integer(path.stem)
    image_format: str | None = None
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    status = "ok"
    error_text: str | None = None

    try:
        with Image.open(path) as image:
            image_format = image.format
            width, height = image.size
            mode = image.mode
            image.load()
    except Exception as error:  # Pillow uses several format-specific errors.
        status = "error"
        error_text = f"{type(error).__name__}: {error}"

    try:
        file_size = path.stat().st_size
    except OSError as error:
        file_size = 0
        status = "error"
        error_text = f"{type(error).__name__}: {error}"

    return {
        "id": product_id,
        "image_index": image_index,
        "relative_path": relative.as_posix(),
        "extension": path.suffix.lower(),
        "format": image_format,
        "width": width,
        "height": height,
        "mode": mode,
        "file_size_bytes": file_size,
        "in_dataset": product_id in dataset_ids,
        "status": status,
        "error": error_text,
    }


def validate_manifest(manifest: pl.DataFrame, dataset_ids: set[int]) -> None:
    """Enforce the manifest schema and row-level invariants."""

    if manifest.columns != list(MANIFEST_COLUMNS):
        raise ValueError(f"manifest columns must be exactly: {MANIFEST_COLUMNS}")
    if dict(manifest.schema) != MANIFEST_SCHEMA:
        raise ValueError("manifest dtypes do not match the frozen schema")
    if manifest.get_column("relative_path").null_count():
        raise ValueError("relative_path contains null values")
    if manifest.get_column("relative_path").n_unique() != manifest.height:
        raise ValueError("relative_path values must be unique")

    invalid_statuses = set(manifest.get_column("status").unique().to_list()) - {
        "ok",
        "error",
    }
    if invalid_statuses:
        raise ValueError(f"unknown image statuses: {sorted(invalid_statuses)}")

    expected_membership = (
        manifest.get_column("id").fill_null(-1).is_in(sorted(dataset_ids)).to_list()
    )
    if expected_membership != manifest.get_column("in_dataset").to_list():
        raise ValueError("in_dataset does not match the source id set")

    readable = manifest.filter(pl.col("status") == "ok")
    incomplete_readable = readable.filter(
        pl.any_horizontal(
            pl.col("format").is_null(),
            pl.col("width").is_null(),
            pl.col("height").is_null(),
            pl.col("mode").is_null(),
        )
    )
    if incomplete_readable.height:
        raise ValueError("readable images must contain format and dimensions")
    if readable.filter(
        (pl.col("width") <= 0) | (pl.col("height") <= 0)
    ).height:
        raise ValueError("readable images must have positive dimensions")
    if manifest.filter(
        (pl.col("status") == "error") & pl.col("error").is_null()
    ).height:
        raise ValueError("unreadable images must contain an error message")


def build_image_manifest(
    images_root: str | Path,
    dataset_ids: Iterable[int],
    workers: int = 4,
) -> ImageArtifacts:
    """Inspect every non-service file and return a deterministic manifest."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    root = Path(images_root)
    normalized_ids = {int(value) for value in dataset_ids}
    files, ignored_paths = discover_image_files(root)

    def inspect(path: Path) -> dict[str, object]:
        return _inspect_image(path, root, normalized_ids)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(inspect, files))
    manifest = pl.DataFrame(rows, schema=MANIFEST_SCHEMA).select(MANIFEST_COLUMNS)
    manifest = manifest.sort(
        ["id", "image_index", "relative_path"],
        nulls_last=True,
    )
    validate_manifest(manifest, normalized_ids)
    return ImageArtifacts(manifest=manifest, ignored_paths=ignored_paths)


def summarize_manifest(
    manifest: pl.DataFrame,
    dataset_ids: Iterable[int],
) -> ManifestSummary:
    """Summarize coverage, path problems and decoding failures."""

    normalized_ids = {int(value) for value in dataset_ids}
    validate_manifest(manifest, normalized_ids)
    mapped = manifest.filter(pl.col("in_dataset"))
    readable_mapped = mapped.filter(pl.col("status") == "ok")
    ids_with_files = set(mapped.get_column("id").drop_nulls().to_list())
    ids_with_readable = set(
        readable_mapped.get_column("id").drop_nulls().to_list()
    )
    numeric_ids = set(manifest.get_column("id").drop_nulls().to_list())
    orphan_ids = tuple(sorted(numeric_ids - normalized_ids))
    duplicate_slots = (
        manifest.filter(
            pl.col("id").is_not_null() & pl.col("image_index").is_not_null()
        )
        .group_by(["id", "image_index"])
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .height
    )

    return ManifestSummary(
        dataset_items=len(normalized_ids),
        image_files=manifest.height,
        readable_files=manifest.filter(pl.col("status") == "ok").height,
        unreadable_files=manifest.filter(pl.col("status") == "error").height,
        items_with_files=len(ids_with_files),
        items_with_readable_files=len(ids_with_readable),
        missing_ids=tuple(sorted(normalized_ids - ids_with_files)),
        ids_without_readable_files=tuple(
            sorted(normalized_ids - ids_with_readable)
        ),
        orphan_ids=orphan_ids,
        orphan_files=manifest.filter(
            pl.col("id").is_not_null() & ~pl.col("in_dataset")
        ).height,
        invalid_id_paths=manifest.get_column("id").null_count(),
        invalid_image_indices=manifest.get_column("image_index").null_count(),
        duplicate_slots=duplicate_slots,
    )


def logical_manifest_checksum(manifest: pl.DataFrame) -> str:
    """Hash logical rows independently of Parquet metadata."""

    digest = hashlib.sha256()
    for row in manifest.select(MANIFEST_COLUMNS).iter_rows():
        digest.update(repr(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _count_table(frame: pl.DataFrame, column: str) -> list[str]:
    counts = (
        frame.group_by(column)
        .agg(pl.len().alias("count"))
        .sort(["count", column], descending=[True, False], nulls_last=True)
    )
    lines = [f"| {column} | Файлов |", "|---|---:|"]
    for row in counts.iter_rows(named=True):
        value = row[column] if row[column] is not None else "NULL"
        lines.append(f"| {value} | {row['count']} |")
    return lines


def _image_count_table(counts_per_item: pl.DataFrame) -> list[str]:
    distribution = (
        counts_per_item.group_by("image_count")
        .agg(pl.len().alias("items"))
        .sort("image_count")
    )
    lines = ["| Изображений | Товаров |", "|---:|---:|"]
    lines.extend(
        f"| {row['image_count']} | {row['items']} |"
        for row in distribution.iter_rows(named=True)
    )
    return lines


def _resolution_table(manifest: pl.DataFrame, limit: int = 15) -> list[str]:
    counts = (
        manifest.filter(pl.col("status") == "ok")
        .group_by(["width", "height"])
        .agg(pl.len().alias("count"))
        .sort(
            ["count", "width", "height"],
            descending=[True, False, False],
        )
        .head(limit)
    )
    lines = ["| Ширина | Высота | Файлов |", "|---:|---:|---:|"]
    lines.extend(
        f"| {row['width']} | {row['height']} | {row['count']} |"
        for row in counts.iter_rows(named=True)
    )
    return lines


def _examples(values: Sequence[object], limit: int = 20) -> str:
    if not values:
        return "—"
    rendered = ", ".join(str(value) for value in values[:limit])
    if len(values) > limit:
        rendered += f" и ещё {len(values) - limit}"
    return rendered


def _error_examples(manifest: pl.DataFrame, limit: int = 15) -> list[str]:
    rows = manifest.filter(pl.col("status") == "error").head(limit)
    if rows.is_empty():
        return ["Ошибок чтения не найдено."]
    lines = ["| Путь | Ошибка |", "|---|---|"]
    for row in rows.iter_rows(named=True):
        path = str(row["relative_path"]).replace("|", "\\|")
        error = str(row["error"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {path} | {error} |")
    return lines


def render_report(
    artifacts: ImageArtifacts,
    dataset_ids: Iterable[int],
    data_path: str | Path,
    images_root: str | Path,
) -> str:
    """Render the tracked R06 image audit report."""

    ids = {int(value) for value in dataset_ids}
    manifest = artifacts.manifest
    summary = summarize_manifest(manifest, ids)
    mapped = manifest.filter(pl.col("in_dataset"))
    counts_per_item = mapped.group_by("id").agg(pl.len().alias("image_count"))
    if counts_per_item.is_empty():
        count_min = count_median = count_mean = count_max = 0
    else:
        count_min = int(counts_per_item.get_column("image_count").min())
        count_median = float(
            counts_per_item.get_column("image_count").median()
        )
        count_mean = float(counts_per_item.get_column("image_count").mean())
        count_max = int(counts_per_item.get_column("image_count").max())
    total_gib = int(manifest.get_column("file_size_bytes").sum()) / 1024**3
    readable = manifest.filter(pl.col("status") == "ok")
    if readable.is_empty():
        dimension_range = "—"
    else:
        dimension_range = (
            f"{int(readable.get_column('width').min())}–"
            f"{int(readable.get_column('width').max())} px по ширине, "
            f"{int(readable.get_column('height').min())}–"
            f"{int(readable.get_column('height').max())} px по высоте"
        )


    lines = [
        "# R06 — аудит изображений и image manifest",
        "",
        "## Метод",
        "",
        "- Файлы рекурсивно сканируются в стабильном порядке.",
        "- id берётся из первого каталога пути, image_index — из имени файла.",
        "- Каждый файл полностью декодируется Pillow; проверки расширения недостаточно.",
        "- Служебные и скрытые пути (__MACOSX, .DS_Store) исключаются.",
        "- Ошибочные файлы остаются в manifest со status = error.",
        "",
        "## Покрытие",
        "",
        f"- Таблица товаров: {data_path}.",
        f"- Корень изображений: {images_root}.",
        f"- Товаров в data.csv: **{summary.dataset_items}**.",
        f"- Файлов в manifest: **{summary.image_files}**.",
        f"- Общий размер файлов: **{total_gib:.2f} GiB**.",
        f"- Успешно прочитано: **{summary.readable_files}**.",
        f"- Не прочитано: **{summary.unreadable_files}**.",
        f"- Товаров хотя бы с одним файлом: **{summary.items_with_files}**.",
        f"- Товаров с читаемым изображением: **{summary.items_with_readable_files}**.",
        f"- Товаров без файлов: **{len(summary.missing_ids)}**.",
        f"- Товаров без читаемых изображений: **{len(summary.ids_without_readable_files)}**.",
        f"- Числовых ID вне data.csv: **{len(summary.orphan_ids)}** "
        f"({summary.orphan_files} файлов).",
        f"- Файлов с некорректным каталогом ID: **{summary.invalid_id_paths}**.",
        f"- Файлов с нечисловым индексом: **{summary.invalid_image_indices}**.",
        f"- Повторяющихся пар id + image_index: **{summary.duplicate_slots}**.",
        f"- Исключено служебных файлов: **{len(artifacts.ignored_paths)}**.",
        "",
        "## Количество изображений на товар",
        "",
        f"- Минимум среди товаров с файлами: **{count_min}**.",
        f"- Медиана: **{count_median:.1f}**.",
        f"- Среднее: **{count_mean:.2f}**.",
        f"- Максимум: **{count_max}**.",
        "",
        "## Статусы",
        "",
    ]
    lines.extend(_count_table(manifest, "status"))
    lines.extend(["", "## Распределение числа изображений", ""])
    lines.extend(_image_count_table(counts_per_item))
    lines.extend(["", "## Расширения", ""])
    lines.extend(_count_table(manifest, "extension"))
    lines.extend(["", "## Форматы Pillow", ""])
    lines.extend(_count_table(manifest, "format"))
    lines.extend(["", "## Цветовые режимы", ""])
    lines.extend(_count_table(manifest, "mode"))
    lines.extend(["", "## Наиболее частые разрешения", ""])
    lines.extend([f"Диапазон размеров: **{dimension_range}**.", ""])
    lines.extend(_resolution_table(manifest))
    lines.extend(
        [
            "",
            "## Проблемные ID",
            "",
            f"- ID без файлов: {_examples(summary.missing_ids)}.",
            "- ID без читаемых изображений: "
            f"{_examples(summary.ids_without_readable_files)}.",
            f"- ID вне data.csv: {_examples(summary.orphan_ids)}.",
            "",
            "## Ошибки чтения",
            "",
        ]
    )
    lines.extend(_error_examples(manifest))
    lines.extend(
        [
            "",
            "## Схема image_manifest.parquet",
            "",
            "| Столбец | Тип Polars |",
            "|---|---|",
        ]
    )
    lines.extend(
        f"| {name} | {dtype} |" for name, dtype in manifest.schema.items()
    )
    lines.extend(
        [
            "",
            "## Воспроизводимость",
            "",
            "- Логическая SHA-256 manifest: "
            f"{logical_manifest_checksum(manifest)}.",
            "- В Git сохраняются код и отчёт; изображения и Parquet остаются локальными.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and audit the E-CUP image manifest"
    )
    parser.add_argument("--data", type=Path, default="data/raw/data.csv")
    parser.add_argument("--images", type=Path, default="data/raw/images")
    parser.add_argument(
        "--output",
        type=Path,
        default="data/processed/image_manifest.parquet",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default="reports/R06-images.md",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = load_dataset(args.data)
    dataset_ids = validate_dataset_ids(source)
    artifacts = build_image_manifest(args.images, dataset_ids, args.workers)
    report = render_report(
        artifacts,
        dataset_ids,
        args.data,
        args.images,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    artifacts.manifest.write_parquet(args.output, compression="zstd")
    args.report.write_text(report, encoding="utf-8")

    summary = summarize_manifest(artifacts.manifest, dataset_ids)
    print(
        f"Image manifest complete: {summary.image_files} files, "
        f"{summary.readable_files} readable, "
        f"{len(summary.missing_ids)} items without files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
