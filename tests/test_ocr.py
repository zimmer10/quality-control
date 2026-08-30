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
    PaddleOCRBackend,
    load_ocr_config,
    main,
    normalize_ocr_text,
    ocr_output_checksum,
    run_ocr,
    select_ocr_manifest,
)


class FakeBackend:
    model_id = "PP-OCRv5-mobile-ru"
    cache_signature = "test-signature-v1"

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    def extract(self, image_path: Path) -> OCRPrediction:
        self.calls.append(image_path.name)
        if image_path.name == self.fail_on:
            raise RuntimeError("synthetic failure")
        if image_path.stem == "1":
            return OCRPrediction("   ", None)
        return OCRPrediction("  БАД\n 500 мг &amp; витамин  ", 0.9)

    def extract_batch(
        self,
        image_paths: list[Path],
    ) -> list[OCRPrediction]:
        self.batch_calls.append([path.name for path in image_paths])
        return [self.extract(path) for path in image_paths]


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

    assert config.version == 2
    assert config.model == "PP-OCRv5-mobile-ru"
    assert config.max_images_per_product is None
    assert config.resume is True
    assert normalize_ocr_text("  БАД\n 500&nbsp;мг  ") == "БАД 500 мг"


def test_selection_caps_images_per_product() -> None:
    selected = select_ocr_manifest(sample_manifest(), 1)

    assert selected.height == 2
    assert selected.get_column("relative_path").to_list() == [
        "10/0.jpg",
        "11/0.jpg",
    ]


def test_run_writes_one_row_per_mapped_image_and_reuses_cache(
    tmp_path: Path,
) -> None:
    manifest = sample_manifest()
    images_root = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    write_images(images_root)
    first_backend = FakeBackend()

    first = run_ocr(
        manifest,
        images_root,
        cache_dir,
        first_backend,
        batch_size=2,
    )
    second_backend = FakeBackend(fail_on="0.jpg")
    second = run_ocr(manifest, images_root, cache_dir, second_backend)

    assert first.frame.height == 3
    assert first_backend.calls == ["0.jpg", "1.jpg"]
    assert first_backend.batch_calls == [["0.jpg", "1.jpg"]]
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
    backend.cache_signature = PaddleOCRBackend(config).cache_signature
    run_ocr(manifest, images_root, cache_dir, backend)
    payload = {
        "version": config.version,
        "model": config.model,
        "backend": config.backend,
        "device": config.device,
        "detection_model": config.detection_model,
        "recognition_model": config.recognition_model,
        "precision": config.precision,
        "text_det_limit_side_len": config.text_det_limit_side_len,
        "text_rec_score_thresh": config.text_rec_score_thresh,
        "text_recognition_batch_size": config.text_recognition_batch_size,
        "inference_batch_size": config.inference_batch_size,
        "max_images_per_product": config.max_images_per_product,
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
