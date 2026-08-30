from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import pytest
import yaml

from ecup.data.image_manifest import MANIFEST_SCHEMA
from ecup.features.embeddings import (
    EMBEDDING_COLUMNS,
    EmbeddingBackendUnavailableError,
    SentenceTransformersBackend,
    aggregate_embeddings,
    compose_product_text,
    image_disagreement,
    load_embeddings_config,
    main,
    run_embeddings,
)


class FakeBackend:
    model_id = "Qwen/Qwen3-VL-Embedding-2B"
    cache_signature = "fake-embeddings-v1"

    def __init__(
        self,
        embedding_dim: int = 3,
        fail_images: set[str] | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.fail_images = fail_images or set()
        self.calls = {"text": 0, "image": 0, "joint": 0}
        self.seen_texts: list[str] = []

    def _padded(self, values: Sequence[float]) -> np.ndarray:
        result = np.zeros(self.embedding_dim, dtype=np.float32)
        result[: len(values)] = values
        return result

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        self.calls["text"] += len(texts)
        self.seen_texts.extend(texts)
        return np.stack([self._padded([1.0, 0.0, 0.0]) for _ in texts])

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        self.calls["image"] += len(image_paths)
        if any(path.name in self.fail_images for path in image_paths):
            raise RuntimeError("synthetic image failure")
        return np.stack(
            [
                self._padded(
                    [1.0, 0.0, 0.0]
                    if path.stem == "0"
                    else [0.0, 1.0, 0.0]
                )
                for path in image_paths
            ]
        )

    def encode_joint(
        self,
        texts: Sequence[str],
        image_paths: Sequence[Path],
    ) -> np.ndarray:
        self.calls["joint"] += len(image_paths)
        assert len(texts) == len(image_paths)
        self.seen_texts.extend(texts)
        return np.stack(
            [self._padded([1.0, 1.0, 0.0]) for _ in image_paths]
        )


def sample_source() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [10, 11],
            "name": ["Добавка", "Зажигалка"],
            "description": ["Описание A", "Описание B"],
            "category": ["БАД", "Легковоспламеняющиеся"],
            "label": [12345, 54321],
        },
        schema={
            "id": pl.Int64,
            "name": pl.String,
            "description": pl.String,
            "category": pl.String,
            "label": pl.Int64,
        },
    )


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
            "file_size_bytes": 5,
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
            "file_size_bytes": 5,
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
            "file_size_bytes": 0,
            "in_dataset": True,
            "status": "error",
            "error": "broken image",
        },
    ]
    return pl.DataFrame(rows, schema=MANIFEST_SCHEMA)


def write_images(root: Path) -> None:
    for relative in (Path("10/0.jpg"), Path("10/1.jpg")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fake-{path.stem}".encode("utf-8"))


def test_canonical_config_and_label_free_text() -> None:
    config = load_embeddings_config("configs/embeddings.yaml")
    text = compose_product_text(sample_source().row(0, named=True))

    assert config.version == 1
    assert config.model == "Qwen/Qwen3-VL-Embedding-2B"
    assert config.embedding_dim == 2048
    assert "Добавка" in text
    assert "12345" not in text
    assert "label" not in text.lower()


def test_aggregation_and_disagreement() -> None:
    left = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    right = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    aggregate = aggregate_embeddings([left, right])

    assert np.linalg.norm(aggregate) == pytest.approx(1.0)
    assert aggregate.tolist() == pytest.approx(
        [2**-0.5, 2**-0.5, 0.0]
    )
    assert image_disagreement([left]) == 0.0
    assert image_disagreement([left, right]) == pytest.approx(1.0)


def test_run_builds_product_rows_and_reuses_cache(tmp_path: Path) -> None:
    source = sample_source()
    manifest = sample_manifest()
    images_root = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    write_images(images_root)
    first_backend = FakeBackend()

    first = run_embeddings(
        source,
        manifest,
        images_root,
        cache_dir,
        first_backend,
        batch_size=2,
    )
    second_backend = FakeBackend(fail_images={"0.jpg", "1.jpg"})
    second = run_embeddings(
        source,
        manifest,
        images_root,
        cache_dir,
        second_backend,
        batch_size=2,
    )

    product = first.frame.filter(pl.col("id") == 10).row(0, named=True)
    missing = first.frame.filter(pl.col("id") == 11).row(0, named=True)
    assert first.frame.columns == list(EMBEDDING_COLUMNS)
    assert product["embedding_status"] == "ok"
    assert product["embedded_image_count"] == 2
    assert product["text_image_similarity"] == pytest.approx(2**-0.5)
    assert product["image_disagreement"] == pytest.approx(1.0)
    assert missing["embedding_status"] == "embedding_error"
    assert missing["embedded_image_count"] == 0
    assert "broken image" in missing["embedding_errors"][0]
    assert first.computed_embeddings == 6
    assert all("12345" not in text and "54321" not in text for text in first_backend.seen_texts)
    assert second.cache_hits == 6
    assert second.computed_embeddings == 0
    assert second_backend.calls == {"text": 0, "image": 0, "joint": 0}
    assert first.frame.equals(second.frame)


def test_failed_image_isolated_and_retried(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    write_images(images_root)
    source = sample_source().head(1)
    manifest = sample_manifest().head(2)

    failed = run_embeddings(
        source,
        manifest,
        images_root,
        cache_dir,
        FakeBackend(fail_images={"1.jpg"}),
        batch_size=2,
    )
    recovered_backend = FakeBackend()
    recovered = run_embeddings(
        source,
        manifest,
        images_root,
        cache_dir,
        recovered_backend,
        batch_size=2,
    )

    assert failed.frame.row(0, named=True)["embedding_status"] == "partial"
    assert failed.frame.row(0, named=True)["embedded_image_count"] == 1
    assert failed.failed_embeddings == 1
    assert recovered.frame.row(0, named=True)["embedding_status"] == "ok"
    assert recovered.frame.row(0, named=True)["embedded_image_count"] == 2
    assert recovered_backend.calls == {"text": 0, "image": 1, "joint": 0}


def test_unavailable_backend_stops_run(tmp_path: Path) -> None:
    class UnavailableBackend(FakeBackend):
        def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
            raise EmbeddingBackendUnavailableError("model is absent")

    with pytest.raises(EmbeddingBackendUnavailableError, match="model is absent"):
        run_embeddings(
            sample_source().head(1),
            sample_manifest().head(2),
            tmp_path / "images",
            tmp_path / "cache",
            UnavailableBackend(),
        )


def test_cli_aggregates_complete_cache_without_loading_model(
    tmp_path: Path,
) -> None:
    canonical = load_embeddings_config("configs/embeddings.yaml")
    source = sample_source()
    manifest = sample_manifest()
    images_root = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    data_path = tmp_path / "data.csv"
    manifest_path = tmp_path / "manifest.parquet"
    output_path = tmp_path / "embeddings.parquet"
    report_path = tmp_path / "report.md"
    config_path = tmp_path / "embeddings.yaml"
    write_images(images_root)
    source.write_csv(data_path)
    manifest.write_parquet(manifest_path)
    config = replace(
        canonical,
        embedding_dim=64,
        paths=replace(
            canonical.paths,
            data_path=data_path,
            images_root=images_root,
            images_manifest=manifest_path,
            cache_dir=cache_dir,
            output_path=output_path,
            report_path=report_path,
        ),
    )
    backend = FakeBackend(embedding_dim=64)
    backend.cache_signature = SentenceTransformersBackend(config).cache_signature
    run_embeddings(source, manifest, images_root, cache_dir, backend)
    payload = {
        "version": config.version,
        "model": config.model,
        "backend": config.backend,
        "device": config.device,
        "instruction": config.instruction,
        "embedding_dim": config.embedding_dim,
        "batch_size": config.batch_size,
        "local_files_only": config.local_files_only,
        "normalize": config.normalize,
        "resume": config.resume,
        "data_path": str(config.paths.data_path),
        "images_root": str(config.paths.images_root),
        "images_manifest": str(config.paths.images_manifest),
        "cache_dir": str(config.paths.cache_dir),
        "output_path": str(config.paths.output_path),
        "report_path": str(config.paths.report_path),
    }
    config_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path), "--progress-every", "0"]) == 0
    assert pl.read_parquet(output_path).height == 2
    assert "# R13 — multimodal embeddings" in report_path.read_text(
        encoding="utf-8"
    )
