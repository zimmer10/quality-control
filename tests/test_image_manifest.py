from __future__ import annotations

import polars as pl
import pytest
from PIL import Image

from ecup.data.image_manifest import (
    MANIFEST_COLUMNS,
    build_image_manifest,
    logical_manifest_checksum,
    main,
    summarize_manifest,
    validate_dataset_ids,
)


def write_image(
    path,
    *,
    size: tuple[int, int] = (12, 7),
    image_format: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 40, 60)).save(
        path,
        format=image_format,
    )


def sample_source(ids: list[int] | None = None) -> pl.DataFrame:
    values = ids or [1, 2]
    return pl.DataFrame(
        {
            "": list(range(len(values))),
            "id": values,
            "name": [f"Product {value}" for value in values],
            "description": ["Description"] * len(values),
            "category": ["БАД"] * len(values),
            "label": [1] * len(values),
        }
    )


def rows_by_path(manifest: pl.DataFrame) -> dict[str, dict[str, object]]:
    return {
        row["relative_path"]: row
        for row in manifest.iter_rows(named=True)
    }


def test_manifest_maps_images_and_excludes_service_files(tmp_path) -> None:
    images = tmp_path / "images"
    write_image(images / "1" / "0.jpg", size=(12, 7))
    write_image(images / "1" / "1.png", size=(8, 9))
    write_image(images / "1" / "not_number.jpg")
    write_image(images / "99" / "0.jpg")
    write_image(images / "invalid" / "0.jpg")
    write_image(images / "root.jpg")
    write_image(images / "1" / ".hidden.jpg")
    write_image(images / "__MACOSX" / "metadata.jpg")
    (images / ".DS_Store").write_bytes(b"service")

    artifacts = build_image_manifest(images, {1, 2}, workers=2)
    manifest = artifacts.manifest
    rows = rows_by_path(manifest)

    assert manifest.columns == list(MANIFEST_COLUMNS)
    assert manifest.height == 6
    assert set(artifacts.ignored_paths) == {
        ".DS_Store",
        "1/.hidden.jpg",
        "__MACOSX/metadata.jpg",
    }
    assert rows["1/0.jpg"]["id"] == 1
    assert rows["1/0.jpg"]["image_index"] == 0
    assert rows["1/0.jpg"]["format"] == "JPEG"
    assert rows["1/0.jpg"]["width"] == 12
    assert rows["1/0.jpg"]["height"] == 7
    assert rows["1/0.jpg"]["in_dataset"] is True
    assert rows["1/1.png"]["format"] == "PNG"
    assert rows["99/0.jpg"]["in_dataset"] is False
    assert rows["invalid/0.jpg"]["id"] is None
    assert rows["root.jpg"]["id"] is None

    summary = summarize_manifest(manifest, {1, 2})
    assert summary.items_with_files == 1
    assert summary.missing_ids == (2,)
    assert summary.orphan_ids == (99,)
    assert summary.orphan_files == 1
    assert summary.invalid_id_paths == 2
    assert summary.invalid_image_indices == 2
    assert summary.unreadable_files == 0


def test_corrupt_file_is_retained_with_error_status(tmp_path) -> None:
    images = tmp_path / "images"
    broken = images / "1" / "0.jpg"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"not a jpeg")

    artifacts = build_image_manifest(images, {1}, workers=1)
    row = artifacts.manifest.row(0, named=True)
    summary = summarize_manifest(artifacts.manifest, {1})

    assert row["status"] == "error"
    assert row["error"]
    assert row["format"] is None
    assert row["width"] is None
    assert summary.missing_ids == ()
    assert summary.ids_without_readable_files == (1,)
    assert summary.unreadable_files == 1


def test_duplicate_image_slots_are_reported(tmp_path) -> None:
    images = tmp_path / "images"
    write_image(images / "1" / "0.jpg")
    write_image(images / "1" / "0.png")

    artifacts = build_image_manifest(images, {1}, workers=1)
    summary = summarize_manifest(artifacts.manifest, {1})

    assert summary.duplicate_slots == 1


def test_manifest_is_deterministic_across_worker_counts(tmp_path) -> None:
    images = tmp_path / "images"
    write_image(images / "2" / "1.jpg", size=(20, 10))
    write_image(images / "1" / "2.jpg", size=(10, 20))
    write_image(images / "1" / "0.jpg", size=(5, 5))

    sequential = build_image_manifest(images, {1, 2}, workers=1).manifest
    parallel = build_image_manifest(images, {1, 2}, workers=3).manifest

    assert sequential.equals(parallel)
    assert logical_manifest_checksum(sequential) == logical_manifest_checksum(
        parallel
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (pl.DataFrame({"name": ["x"]}), "misses required column"),
        (pl.DataFrame({"id": [1, 1]}), "must be unique"),
        (pl.DataFrame({"id": [-1]}), "must be non-negative"),
    ],
)
def test_invalid_dataset_ids_are_rejected(
    source: pl.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_dataset_ids(source)


def test_cli_writes_manifest_and_report(tmp_path) -> None:
    data_path = tmp_path / "data.csv"
    images = tmp_path / "images"
    output_path = tmp_path / "processed" / "image_manifest.parquet"
    report_path = tmp_path / "reports" / "R06-images.md"
    sample_source().write_csv(data_path)
    write_image(images / "1" / "0.jpg", size=(32, 24))

    exit_code = main(
        [
            "--data",
            str(data_path),
            "--images",
            str(images),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--workers",
            "1",
        ]
    )

    assert exit_code == 0
    manifest = pl.read_parquet(output_path)
    assert manifest.columns == list(MANIFEST_COLUMNS)
    assert manifest.height == 1
    assert manifest.row(0, named=True)["status"] == "ok"

    report = report_path.read_text(encoding="utf-8")
    assert "# R06 — аудит изображений и image manifest" in report
    assert "Товаров в data.csv: **2**" in report
    assert "Товаров без файлов: **1**" in report
    assert "Успешно прочитано: **1**" in report
