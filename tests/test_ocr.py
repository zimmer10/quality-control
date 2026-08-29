from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from PIL import Image

from ecup.data.image_manifest import MANIFEST_SCHEMA
from ecup.features.ocr import (
    OCRBackendUnavailableError,
    OCRPrediction,
    TransformersOCRBackend,
    load_ocr_config,
    main,
    normalize_ocr_text,
    ocr_output_checksum,
    run_ocr,
)


class FakeBackend:
    model_id = "PaddlePaddle/PaddleOCR-VL-1.5"
    cache_signature = "test-signature-v1"

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[str] = []

    def extract(self, image_path: Path) -> OCRPrediction:
        self.calls.append(image_path.name)
        if image_path.name == self.fail_on:
            raise RuntimeError("synthetic failure")
        if image_path.stem == "1":
            return OCRPrediction("   ", None)
        return OCRPrediction("  БАД\n 500 мг &amp; витамин  ", 0.9)


def sample_manifest() -> pl.DataFrame:
    rows = [
        {
            "id": 10,
            "image_index": 0,
            "relative_path": "10/0.jpg",
            "extension": ".jpg",
            "format": "JPEG",
            "width": 8,
            "height": 8,
            "mode": "RGB",
            "file_size_bytes": 100,
            "in_dataset": True,
            "status": "ok",
            "error": None,
        },
        {
            "id": 10,
            "image_index": 1,
            "relative_path": "10/1.jpg",
            "extension": ".jpg",
            "format": "JPEG",
            "width": 8,
            "height": 8,
            "mode": "RGB",
            "file_size_bytes": 101,
            "in_dataset": True,
            "status": "ok",
            "error": None,
        },
        {
            "id": 11,
            "image_index": 0,
            "relative_path": "11/0.jpg",
            "extension": ".jpg",
            "format": None,
            "width": None,
            "height": None,
            "mode": None,
            "file_size_bytes": 20,
            "in_dataset": True,
            "status": "error",
            "error": "UnidentifiedImageError: broken",
        },
        {
            "id": 999,
            "image_index": 0,
            "relative_path": "999/0.jpg",
            "extension": ".jpg",
            "format": "JPEG",
            "width": 8,
            "height": 8,
            "mode": "RGB",
            "file_size_bytes": 90,
            "in_dataset": False,
            "status": "ok",
            "error": None,
        },
    ]
    return pl.DataFrame(rows, schema=MANIFEST_SCHEMA)


def write_images(root: Path) -> None:
    for relative in (Path("10/0.jpg"), Path("10/1.jpg")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "white").save(path)


def test_canonical_config_and_normalization() -> None:
    config = load_ocr_config("configs/ocr.yaml")

    assert config.version == 1
    assert config.model == "PaddlePaddle/PaddleOCR-VL-1.5"
    assert config.resume is True
    assert normalize_ocr_text("  БАД\n 500&nbsp;мг  ") == "БАД 500 мг"


def test_run_writes_one_row_per_mapped_image_and_reuses_cache(
    tmp_path: Path,
) -> None:
    manifest = sample_manifest()
    images_root = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    write_images(images_root)
    first_backend = FakeBackend()

    first = run_ocr(manifest, images_root, cache_dir, first_backend)
    second_backend = FakeBackend(fail_on="0.jpg")
    second = run_ocr(manifest, images_root, cache_dir, second_backend)

    assert first.frame.height == 3
    assert first_backend.calls == ["0.jpg", "1.jpg"]
    assert first.frame.get_column("ocr_status").to_list() == [
        "ok",
        "no_text",
        "source_error",
    ]
    assert first.frame.row(0, named=True)["ocr_text_by_image"] == (
        "БАД 500 мг & витамин"
    )
    assert second_backend.calls == []
    assert second.cache_hits == 2
    assert first.frame.equals(second.frame)
    assert ocr_output_checksum(first.frame) == ocr_output_checksum(second.frame)


def test_ocr_errors_are_cached_but_retried(tmp_path: Path) -> None:
    manifest = sample_manifest().head(1)
    images_root = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    write_images(images_root)

    failed = run_ocr(
        manifest,
        images_root,
        cache_dir,
        FakeBackend(fail_on="0.jpg"),
    )
    recovered_backend = FakeBackend()
    recovered = run_ocr(
        manifest,
        images_root,
        cache_dir,
        recovered_backend,
        retry_errors=True,
    )

    assert failed.frame.row(0, named=True)["ocr_status"] == "ocr_error"
    assert "synthetic failure" in failed.frame.row(0, named=True)["ocr_error"]
    assert recovered_backend.calls == ["0.jpg"]
    assert recovered.frame.row(0, named=True)["ocr_status"] == "ok"


def test_invalid_quality_becomes_row_error(tmp_path: Path) -> None:
    class InvalidQualityBackend(FakeBackend):
        def extract(self, image_path: Path) -> OCRPrediction:
            return OCRPrediction("text", 2.0)

    manifest = sample_manifest().head(1)
    images_root = tmp_path / "images"
    write_images(images_root)

    run = run_ocr(
        manifest,
        images_root,
        tmp_path / "cache",
        InvalidQualityBackend(),
    )

    assert run.frame.row(0, named=True)["ocr_status"] == "ocr_error"


def test_unavailable_backend_stops_run_immediately(tmp_path: Path) -> None:
    class UnavailableBackend(FakeBackend):
        def extract(self, image_path: Path) -> OCRPrediction:
            raise OCRBackendUnavailableError("model is absent")

    manifest = sample_manifest().head(1)

    with pytest.raises(OCRBackendUnavailableError, match="model is absent"):
        run_ocr(
            manifest,
            tmp_path / "images",
            tmp_path / "cache",
            UnavailableBackend(),
        )


def test_cli_can_aggregate_a_complete_cache_without_heavy_dependencies(
    tmp_path: Path,
) -> None:
    canonical = load_ocr_config("configs/ocr.yaml")
    manifest = sample_manifest()
    images_root = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    output_path = tmp_path / "ocr.parquet"
    report_path = tmp_path / "report.md"
    manifest_path = tmp_path / "manifest.parquet"
    config_path = tmp_path / "ocr.yaml"
    write_images(images_root)
    manifest.write_parquet(manifest_path)
    backend = FakeBackend()
    config = replace(
        canonical,
        paths=replace(
            canonical.paths,
            images_root=images_root,
            images_manifest=manifest_path,
            cache_dir=cache_dir,
            output_path=output_path,
            report_path=report_path,
        ),
    )
    backend.cache_signature = TransformersOCRBackend(config).cache_signature
    run_ocr(manifest, images_root, cache_dir, backend)
    payload = {
        "version": config.version,
        "model": config.model,
        "backend": config.backend,
        "device": config.device,
        "prompt": config.prompt,
        "max_new_tokens": config.max_new_tokens,
        "max_pixels": config.max_pixels,
        "local_files_only": config.local_files_only,
        "images_root": str(config.paths.images_root),
        "images_manifest": str(config.paths.images_manifest),
        "cache_dir": str(config.paths.cache_dir),
        "output_path": str(config.paths.output_path),
        "report_path": str(config.paths.report_path),
        "resume": config.resume,
        "retry_errors": config.retry_errors,
    }
    import yaml

    config_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path)]) == 0
    assert pl.read_parquet(output_path).height == 3
    assert "# R11 — кешируемый OCR" in report_path.read_text(encoding="utf-8")
