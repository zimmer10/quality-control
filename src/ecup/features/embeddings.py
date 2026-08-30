"""Label-free multimodal embeddings with resumable per-input caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl
import yaml

from ecup.data.image_manifest import MANIFEST_COLUMNS, MANIFEST_SCHEMA

DEFAULT_CONFIG_PATH = Path("configs/embeddings.yaml")
CACHE_VERSION = 1
SOURCE_COLUMNS = ("id", "name", "description", "category")
EMBEDDING_COLUMNS = (
    "id",
    "category",
    "image_indices",
    "relative_paths",
    "text_embedding",
    "image_embeddings",
    "joint_text_image_embeddings",
    "aggregated_image_embedding",
    "aggregated_joint_embedding",
    "text_image_similarity",
    "image_disagreement",
    "source_image_count",
    "embedded_image_count",
    "embedding_status",
    "embedding_errors",
    "embedding_model",
    "embedding_dim",
)
EMBEDDING_SCHEMA = {
    "id": pl.Int64,
    "category": pl.String,
    "image_indices": pl.List(pl.Int64),
    "relative_paths": pl.List(pl.String),
    "text_embedding": pl.List(pl.Float32),
    "image_embeddings": pl.List(pl.List(pl.Float32)),
    "joint_text_image_embeddings": pl.List(pl.List(pl.Float32)),
    "aggregated_image_embedding": pl.List(pl.Float32),
    "aggregated_joint_embedding": pl.List(pl.Float32),
    "text_image_similarity": pl.Float32,
    "image_disagreement": pl.Float32,
    "source_image_count": pl.Int64,
    "embedded_image_count": pl.Int64,
    "embedding_status": pl.String,
    "embedding_errors": pl.List(pl.String),
    "embedding_model": pl.String,
    "embedding_dim": pl.Int64,
}
EMBEDDING_STATUSES = frozenset(
    {"ok", "missing_images", "partial", "embedding_error"}
)


@dataclass(frozen=True)
class EmbeddingPaths:
    data_path: Path
    images_root: Path
    images_manifest: Path
    cache_dir: Path
    output_path: Path
    report_path: Path


@dataclass(frozen=True)
class EmbeddingConfig:
    version: int
    model: str
    backend: str
    device: str
    instruction: str
    embedding_dim: int
    batch_size: int
    local_files_only: bool
    normalize: bool
    resume: bool
    paths: EmbeddingPaths


class EmbeddingBackend(Protocol):
    model_id: str
    embedding_dim: int
    cache_signature: str

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode product texts into a normalized 2-D float array."""

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        """Encode images into the same vector space."""

    def encode_joint(
        self,
        texts: Sequence[str],
        image_paths: Sequence[Path],
    ) -> np.ndarray:
        """Encode aligned text-image pairs into the same vector space."""


class EmbeddingBackendUnavailableError(RuntimeError):
    """The configured embedding runtime or model cannot be loaded."""


@dataclass(frozen=True)
class ImageWorkItem:
    product_id: int
    image_index: int
    relative_path: str
    absolute_path: Path
    text: str
    source_digest: str


@dataclass(frozen=True)
class EmbeddingRun:
    frame: pl.DataFrame
    total_products: int
    selected_products: int
    selected_images: int
    cache_hits: int
    computed_embeddings: int
    failed_embeddings: int
    elapsed_seconds: float


def _as_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    path: str,
) -> None:
    extra = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if extra:
        raise ValueError(f"{path} has unknown keys: {', '.join(extra)}")
    if missing:
        raise ValueError(f"{path} misses keys: {', '.join(missing)}")


def load_embeddings_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> EmbeddingConfig:
    """Load and strictly validate the versioned R13 configuration."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _as_mapping(payload, "embeddings")
    _exact_keys(
        root,
        {
            "version",
            "model",
            "backend",
            "device",
            "instruction",
            "embedding_dim",
            "batch_size",
            "local_files_only",
            "normalize",
            "resume",
            "data_path",
            "images_root",
            "images_manifest",
            "cache_dir",
            "output_path",
            "report_path",
        },
        "embeddings",
    )
    if root["version"] != 1 or type(root["version"]) is not int:
        raise ValueError("embeddings.version must be integer 1")
    for key in (
        "model",
        "backend",
        "device",
        "instruction",
        "data_path",
        "images_root",
        "images_manifest",
        "cache_dir",
        "output_path",
        "report_path",
    ):
        if not isinstance(root[key], str) or not root[key].strip():
            raise ValueError(f"embeddings.{key} must be a non-empty string")
    if root["backend"] != "sentence-transformers":
        raise ValueError("embeddings.backend must be sentence-transformers")
    if root["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("embeddings.device must be auto, cpu or cuda")
    for key in ("embedding_dim", "batch_size"):
        if type(root[key]) is not int or root[key] <= 0:
            raise ValueError(f"embeddings.{key} must be a positive integer")
    if not 64 <= root["embedding_dim"] <= 2048:
        raise ValueError("embeddings.embedding_dim must be within [64, 2048]")
    for key in ("local_files_only", "normalize", "resume"):
        if type(root[key]) is not bool:
            raise ValueError(f"embeddings.{key} must be boolean")
    return EmbeddingConfig(
        version=1,
        model=str(root["model"]),
        backend=str(root["backend"]),
        device=str(root["device"]),
        instruction=str(root["instruction"]),
        embedding_dim=int(root["embedding_dim"]),
        batch_size=int(root["batch_size"]),
        local_files_only=bool(root["local_files_only"]),
        normalize=bool(root["normalize"]),
        resume=bool(root["resume"]),
        paths=EmbeddingPaths(
            data_path=Path(str(root["data_path"])),
            images_root=Path(str(root["images_root"])),
            images_manifest=Path(str(root["images_manifest"])),
            cache_dir=Path(str(root["cache_dir"])),
            output_path=Path(str(root["output_path"])),
            report_path=Path(str(root["report_path"])),
        ),
    )


def compose_product_text(source: Mapping[str, object]) -> str:
    """Build inference-available model input without ever reading label."""

    missing = sorted(set(SOURCE_COLUMNS) - set(source))
    if missing:
        raise ValueError(f"product source misses columns: {', '.join(missing)}")

    def clean(value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split())

    return "\n".join(
        (
            f"Category: {clean(source['category'])}",
            f"Name: {clean(source['name'])}",
            f"Description: {clean(source['description'])}",
        )
    )


def validate_sources(source: pl.DataFrame, manifest: pl.DataFrame) -> None:
    missing_source = sorted(set(SOURCE_COLUMNS) - set(source.columns))
    if missing_source:
        raise ValueError(
            f"source data misses columns: {', '.join(missing_source)}"
        )
    if source.is_empty():
        raise ValueError("source data is empty")
    ids = source.get_column("id")
    if ids.null_count() or ids.n_unique() != source.height:
        raise ValueError("source id must be non-null and unique")
    if manifest.columns != list(MANIFEST_COLUMNS):
        raise ValueError("image manifest columns do not match R06 contract")
    if dict(manifest.schema) != MANIFEST_SCHEMA:
        raise ValueError("image manifest dtypes do not match R06 contract")
    if manifest.get_column("relative_path").n_unique() != manifest.height:
        raise ValueError("image manifest relative_path values must be unique")


def _model_source(model_id: str, local_files_only: bool) -> str:
    shared_root = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
    shared_path = shared_root / model_id
    if shared_path.exists():
        return str(shared_path)
    if local_files_only:
        raise EmbeddingBackendUnavailableError(
            f"embedding model is absent: {shared_path}. Set "
            "SHARED_MODELS_PATH or download the model locally."
        )
    return model_id


class SentenceTransformersBackend:
    """Lazy backend for the official Qwen3-VL SentenceTransformers API."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.model_id = config.model
        self.embedding_dim = config.embedding_dim
        signature = {
            "version": CACHE_VERSION,
            "backend": config.backend,
            "model": config.model,
            "instruction": config.instruction,
            "embedding_dim": config.embedding_dim,
            "normalize": config.normalize,
        }
        self.cache_signature = json.dumps(
            signature,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._config = config
        self._model: Any = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingBackendUnavailableError(
                "embedding dependencies are missing. Install with: "
                "pip install -e '.[embeddings]'"
            ) from error
        if self._config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._config.device
        if device == "cuda" and not torch.cuda.is_available():
            raise EmbeddingBackendUnavailableError(
                "CUDA was requested but is unavailable"
            )
        if device == "cuda":
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32
        source = _model_source(
            self._config.model,
            self._config.local_files_only,
        )
        try:
            self._model = SentenceTransformer(
                source,
                device=device,
                truncate_dim=self.embedding_dim,
                local_files_only=self._config.local_files_only,
                model_kwargs={"torch_dtype": dtype},
            )
        except Exception as error:
            raise EmbeddingBackendUnavailableError(
                f"failed to load embedding model {self.model_id}: {error}"
            ) from error
        self._device = device

    def _encode(self, inputs: Sequence[object]) -> np.ndarray:
        self._load()
        try:
            encoded = self._model.encode(
                list(inputs),
                prompt=self._config.instruction,
                batch_size=self._config.batch_size,
                normalize_embeddings=self._config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except EmbeddingBackendUnavailableError:
            raise
        except Exception as error:
            raise RuntimeError(f"embedding inference failed: {error}") from error
        return _validate_matrix(
            encoded,
            len(inputs),
            self.embedding_dim,
        )

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts))

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        inputs = [{"image": str(path)} for path in image_paths]
        return self._encode(inputs)

    def encode_joint(
        self,
        texts: Sequence[str],
        image_paths: Sequence[Path],
    ) -> np.ndarray:
        if len(texts) != len(image_paths):
            raise ValueError("joint text and image batches must be aligned")
        inputs = [
            {"text": text, "image": str(path)}
            for text, path in zip(texts, image_paths, strict=True)
        ]
        return self._encode(inputs)


def _validate_matrix(
    value: object,
    expected_rows: int,
    embedding_dim: int,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (expected_rows, embedding_dim):
        raise ValueError(
            "backend returned invalid embedding shape: "
            f"{matrix.shape}, expected {(expected_rows, embedding_dim)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("backend returned non-finite embeddings")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0):
        raise ValueError("backend returned a zero embedding")
    return matrix


def _normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("cannot normalize an empty or non-finite vector")
    return (value / norm).astype(np.float32)


def aggregate_embeddings(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Mean-pool aligned vectors and normalize the product representation."""

    if not vectors:
        raise ValueError("at least one embedding is required")
    matrix = np.stack(
        [np.asarray(vector, dtype=np.float32) for vector in vectors]
    )
    return _normalized(matrix.mean(axis=0))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(_normalized(left), _normalized(right)), -1, 1))


def image_disagreement(vectors: Sequence[np.ndarray]) -> float:
    """Return mean pairwise cosine distance; one image has no disagreement."""

    if len(vectors) <= 1:
        return 0.0
    matrix = np.stack([_normalized(vector) for vector in vectors])
    similarities = matrix @ matrix.T
    upper = similarities[np.triu_indices(len(vectors), k=1)]
    return float(np.clip(1.0 - upper.mean(), 0.0, 2.0))


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(
    kind: str,
    signature: str,
    source_digest: str,
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "kind": kind,
        "signature": signature,
        "source_digest": source_digest,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_dir: Path, kind: str, key: str) -> Path:
    return cache_dir / kind / key[:2] / f"{key}.npz"


def _read_cached_vector(
    path: Path,
    key: str,
    embedding_dim: int,
) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            stored_key = str(payload["cache_key"].item())
            vector = np.asarray(payload["embedding"], dtype=np.float32)
    except (OSError, ValueError, KeyError):
        return None
    if stored_key != key or vector.shape != (embedding_dim,):
        return None
    if not np.isfinite(vector).all() or float(np.linalg.norm(vector)) <= 0:
        return None
    return vector


def _write_cached_vector(path: Path, key: str, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            cache_key=np.asarray(key),
            embedding=np.asarray(vector, dtype=np.float32),
        )
    temporary.replace(path)


def _short_error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[:2000]


def _encode_safely(
    items: Sequence[Any],
    encoder: Callable[[Sequence[Any]], np.ndarray],
    embedding_dim: int,
) -> tuple[list[np.ndarray | None], list[str | None]]:
    """Use a fast batch first, then isolate individual malformed inputs."""

    if not items:
        return [], []
    try:
        matrix = _validate_matrix(encoder(items), len(items), embedding_dim)
        return [row.copy() for row in matrix], [None] * len(items)
    except EmbeddingBackendUnavailableError:
        raise
    except Exception:
        vectors: list[np.ndarray | None] = []
        errors: list[str | None] = []
        for item in items:
            try:
                matrix = _validate_matrix(encoder([item]), 1, embedding_dim)
                vectors.append(matrix[0].copy())
                errors.append(None)
            except EmbeddingBackendUnavailableError:
                raise
            except Exception as error:
                vectors.append(None)
                errors.append(_short_error(error))
        return vectors, errors


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _frame_from_rows(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=EMBEDDING_SCHEMA).select(EMBEDDING_COLUMNS)
    return (
        pl.DataFrame(rows, schema=EMBEDDING_SCHEMA, strict=False)
        .select(EMBEDDING_COLUMNS)
        .sort("id")
    )


def validate_embeddings_output(
    selected_source: pl.DataFrame,
    output: pl.DataFrame,
    embedding_dim: int,
) -> None:
    """Validate R13 schema, product alignment and vector dimensions."""

    if output.columns != list(EMBEDDING_COLUMNS):
        raise ValueError("embedding output columns do not match R13 contract")
    if dict(output.schema) != EMBEDDING_SCHEMA:
        raise ValueError("embedding output dtypes do not match R13 contract")
    if output.height != selected_source.height:
        raise ValueError("embedding output must contain one row per product")
    if output.get_column("id").null_count():
        raise ValueError("embedding output id contains null values")
    if output.get_column("id").n_unique() != output.height:
        raise ValueError("embedding output id values must be unique")
    expected = selected_source.select("id", "category").sort("id")
    actual = output.select("id", "category").sort("id")
    if not expected.equals(actual):
        raise ValueError("embedding output is misaligned with source data")
    statuses = set(output.get_column("embedding_status").to_list())
    if not statuses <= EMBEDDING_STATUSES:
        raise ValueError(f"unknown embedding statuses: {sorted(statuses)}")
    for row in output.iter_rows(named=True):
        paired = int(row["embedded_image_count"])
        if paired != len(row["image_indices"]):
            raise ValueError("embedded_image_count does not match image_indices")
        if paired != len(row["relative_paths"]):
            raise ValueError("embedded_image_count does not match paths")
        if paired != len(row["image_embeddings"]):
            raise ValueError("embedded_image_count does not match image vectors")
        if paired != len(row["joint_text_image_embeddings"]):
            raise ValueError("embedded_image_count does not match joint vectors")
        vector_fields = ["text_embedding"]
        if paired:
            vector_fields.extend(
                ["aggregated_image_embedding", "aggregated_joint_embedding"]
            )
        for field in vector_fields:
            if len(row[field]) not in {0, embedding_dim}:
                raise ValueError(f"{field} has an invalid dimension")
        for field in ("image_embeddings", "joint_text_image_embeddings"):
            if any(len(vector) != embedding_dim for vector in row[field]):
                raise ValueError(f"{field} contains an invalid dimension")


def run_embeddings(
    source: pl.DataFrame,
    manifest: pl.DataFrame,
    images_root: str | Path,
    cache_dir: str | Path,
    backend: EmbeddingBackend,
    *,
    resume: bool = True,
    batch_size: int = 4,
    limit: int | None = None,
    progress_every: int = 0,
) -> EmbeddingRun:
    """Compute text, image and joint representations without using labels."""

    validate_sources(source, manifest)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    selected = source.select(SOURCE_COLUMNS).sort("id")
    if limit is not None:
        selected = selected.head(limit)
    selected_ids = selected.get_column("id").to_list()
    selected_id_set = set(selected_ids)
    mapped_manifest = manifest.filter(
        pl.col("in_dataset")
        & pl.col("id").is_in(selected_ids)
    ).sort(["id", "image_index", "relative_path"])
    selected_manifest = mapped_manifest.filter(pl.col("status") == "ok")

    root = Path(images_root)
    cache_root = Path(cache_dir)
    started = time.perf_counter()
    texts: dict[int, str] = {}
    categories: dict[int, str] = {}
    text_vectors: dict[int, np.ndarray] = {}
    image_vectors: dict[str, np.ndarray] = {}
    joint_vectors: dict[str, np.ndarray] = {}
    errors: dict[int, list[str]] = {int(item): [] for item in selected_ids}
    cache_hits = 0
    computed = 0

    for row in selected.iter_rows(named=True):
        product_id = int(row["id"])
        texts[product_id] = compose_product_text(row)
        categories[product_id] = str(row["category"])
    for row in mapped_manifest.filter(pl.col("status") != "ok").iter_rows(
        named=True
    ):
        product_id = int(row["id"])
        errors[product_id].append(
            f"source {row['relative_path']}: "
            f"{row['error'] or 'image manifest error'}"
        )

    missing_text: list[tuple[int, str, str, Path]] = []
    for product_id in selected_ids:
        text = texts[int(product_id)]
        key = _cache_key("text", backend.cache_signature, _text_digest(text))
        path = _cache_path(cache_root, "text", key)
        vector = (
            _read_cached_vector(path, key, backend.embedding_dim)
            if resume
            else None
        )
        if vector is None:
            missing_text.append((int(product_id), text, key, path))
        else:
            text_vectors[int(product_id)] = vector
            cache_hits += 1

    for offset in range(0, len(missing_text), batch_size):
        batch = missing_text[offset : offset + batch_size]
        values = [item[1] for item in batch]
        vectors, batch_errors = _encode_safely(
            values,
            backend.encode_texts,
            backend.embedding_dim,
        )
        for item, vector, error in zip(
            batch, vectors, batch_errors, strict=True
        ):
            product_id, _, key, path = item
            computed += 1
            if vector is None:
                errors[product_id].append(f"text: {error}")
                continue
            text_vectors[product_id] = vector
            _write_cached_vector(path, key, vector)

    work: list[ImageWorkItem] = []
    for row in selected_manifest.iter_rows(named=True):
        product_id = int(row["id"])
        if product_id not in selected_id_set:
            continue
        relative_path = str(row["relative_path"])
        absolute_path = root / relative_path
        try:
            source_digest = _source_digest(absolute_path)
        except OSError as error:
            errors[product_id].append(
                f"{relative_path}: {_short_error(error)}"
            )
            continue
        work.append(
            ImageWorkItem(
                product_id=product_id,
                image_index=int(row["image_index"]),
                relative_path=relative_path,
                absolute_path=absolute_path,
                text=texts[product_id],
                source_digest=source_digest,
            )
        )

    missing_images: list[tuple[ImageWorkItem, str, Path]] = []
    missing_joint: list[tuple[ImageWorkItem, str, Path]] = []
    for item in work:
        image_key = _cache_key(
            "image", backend.cache_signature, item.source_digest
        )
        image_path = _cache_path(cache_root, "image", image_key)
        image_vector = (
            _read_cached_vector(
                image_path,
                image_key,
                backend.embedding_dim,
            )
            if resume
            else None
        )
        if image_vector is None:
            missing_images.append((item, image_key, image_path))
        else:
            image_vectors[item.relative_path] = image_vector
            cache_hits += 1

        joint_digest = hashlib.sha256(
            f"{item.source_digest}:{_text_digest(item.text)}".encode("utf-8")
        ).hexdigest()
        joint_key = _cache_key(
            "joint", backend.cache_signature, joint_digest
        )
        joint_path = _cache_path(cache_root, "joint", joint_key)
        joint_vector = (
            _read_cached_vector(
                joint_path,
                joint_key,
                backend.embedding_dim,
            )
            if resume
            else None
        )
        if joint_vector is None:
            missing_joint.append((item, joint_key, joint_path))
        else:
            joint_vectors[item.relative_path] = joint_vector
            cache_hits += 1

    for offset in range(0, len(missing_images), batch_size):
        batch = missing_images[offset : offset + batch_size]
        paths = [item[0].absolute_path for item in batch]
        vectors, batch_errors = _encode_safely(
            paths,
            backend.encode_images,
            backend.embedding_dim,
        )
        for item_with_cache, vector, error in zip(
            batch, vectors, batch_errors, strict=True
        ):
            item, key, path = item_with_cache
            computed += 1
            if vector is None:
                errors[item.product_id].append(
                    f"image {item.relative_path}: {error}"
                )
                continue
            image_vectors[item.relative_path] = vector
            _write_cached_vector(path, key, vector)

    for offset in range(0, len(missing_joint), batch_size):
        batch = missing_joint[offset : offset + batch_size]
        pairs = [(item[0].text, item[0].absolute_path) for item in batch]

        def encode_joint(values: Sequence[tuple[str, Path]]) -> np.ndarray:
            return backend.encode_joint(
                [value[0] for value in values],
                [value[1] for value in values],
            )

        vectors, batch_errors = _encode_safely(
            pairs,
            encode_joint,
            backend.embedding_dim,
        )
        for item_with_cache, vector, error in zip(
            batch, vectors, batch_errors, strict=True
        ):
            item, key, path = item_with_cache
            computed += 1
            if vector is None:
                errors[item.product_id].append(
                    f"joint {item.relative_path}: {error}"
                )
                continue
            joint_vectors[item.relative_path] = vector
            _write_cached_vector(path, key, vector)

    work_by_product: dict[int, list[ImageWorkItem]] = {
        int(item): [] for item in selected_ids
    }
    for item in work:
        work_by_product[item.product_id].append(item)
    source_image_counts = {
        int(row["id"]): int(row["source_image_count"])
        for row in mapped_manifest.group_by("id")
        .agg(pl.len().alias("source_image_count"))
        .iter_rows(named=True)
    }

    rows: list[dict[str, object]] = []
    for position, product_id_value in enumerate(selected_ids, start=1):
        product_id = int(product_id_value)
        product_work = work_by_product[product_id]
        paired = [
            item
            for item in product_work
            if item.relative_path in image_vectors
            and item.relative_path in joint_vectors
        ]
        product_images = [image_vectors[item.relative_path] for item in paired]
        product_joint = [joint_vectors[item.relative_path] for item in paired]
        text_vector = text_vectors.get(product_id)
        aggregate_image = (
            aggregate_embeddings(product_images) if product_images else None
        )
        aggregate_joint = (
            aggregate_embeddings(product_joint) if product_joint else None
        )
        source_image_count = source_image_counts.get(product_id, 0)
        product_errors = errors[product_id]
        if text_vector is None or (source_image_count and not paired):
            status = "embedding_error"
        elif product_errors or len(paired) < source_image_count:
            status = "partial"
        elif source_image_count == 0:
            status = "missing_images"
        else:
            status = "ok"
        rows.append(
            {
                "id": product_id,
                "category": categories[product_id],
                "image_indices": [item.image_index for item in paired],
                "relative_paths": [item.relative_path for item in paired],
                "text_embedding": (
                    text_vector.tolist() if text_vector is not None else []
                ),
                "image_embeddings": [
                    vector.tolist() for vector in product_images
                ],
                "joint_text_image_embeddings": [
                    vector.tolist() for vector in product_joint
                ],
                "aggregated_image_embedding": (
                    aggregate_image.tolist()
                    if aggregate_image is not None
                    else []
                ),
                "aggregated_joint_embedding": (
                    aggregate_joint.tolist()
                    if aggregate_joint is not None
                    else []
                ),
                "text_image_similarity": (
                    cosine_similarity(text_vector, aggregate_image)
                    if text_vector is not None and aggregate_image is not None
                    else None
                ),
                "image_disagreement": (
                    image_disagreement(product_images)
                    if product_images
                    else None
                ),
                "source_image_count": source_image_count,
                "embedded_image_count": len(paired),
                "embedding_status": status,
                "embedding_errors": product_errors,
                "embedding_model": backend.model_id,
                "embedding_dim": backend.embedding_dim,
            }
        )
        if progress_every and (
            position % progress_every == 0 or position == len(selected_ids)
        ):
            elapsed = time.perf_counter() - started
            print(
                f"Embedding progress: {position}/{len(selected_ids)}, "
                f"cached={cache_hits}, computed={computed}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    output = _frame_from_rows(rows)
    validate_embeddings_output(selected, output, backend.embedding_dim)
    failed = sum(len(value) for value in errors.values())
    return EmbeddingRun(
        frame=output,
        total_products=source.height,
        selected_products=selected.height,
        selected_images=mapped_manifest.height,
        cache_hits=cache_hits,
        computed_embeddings=computed,
        failed_embeddings=failed,
        elapsed_seconds=time.perf_counter() - started,
    )


def embeddings_output_checksum(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.select(EMBEDDING_COLUMNS).sort("id").iter_rows():
        digest.update(repr(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def render_embeddings_report(
    run: EmbeddingRun,
    config: EmbeddingConfig,
    output_path: str | Path,
) -> str:
    counts = (
        run.frame.group_by("embedding_status")
        .agg(pl.len().alias("count"))
        .sort("embedding_status")
    )
    lines = [
        "# R13 — multimodal embeddings",
        "",
        "## Контракт",
        "",
        f"- Модель: `{config.model}`.",
        f"- Размерность: **{config.embedding_dim}**.",
        f"- Image manifest: `{config.paths.images_manifest}`.",
        f"- Кеш: `{config.paths.cache_dir}`.",
        f"- Итоговая таблица: `{output_path}`.",
        f"- Товаров в исходных данных: **{run.total_products}**.",
        f"- Товаров в текущем запуске: **{run.selected_products}**.",
        f"- Изображений в текущем запуске: **{run.selected_images}**.",
        f"- Логическая SHA-256: `{embeddings_output_checksum(run.frame)}`.",
        "- Одна строка соответствует одному товару.",
        "- `label`, правила и финальный verdict при расчёте не используются.",
        "",
        "## Статусы",
        "",
        "| Статус | Товаров |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {row['embedding_status']} | {row['count']} |"
        for row in counts.iter_rows(named=True)
    )
    lines.extend(
        [
            "",
            "## Возобновляемость и время",
            "",
            f"- Получено из кеша: **{run.cache_hits}** векторов.",
            f"- Вычислено backend: **{run.computed_embeddings}** векторов.",
            f"- Ошибок вычисления: **{run.failed_embeddings}**.",
            f"- Время текущего запуска: **{run.elapsed_seconds:.3f} s**.",
            "- Успешные text/image/joint-векторы записываются атомарно.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute cacheable label-free multimodal embeddings"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_embeddings_config(args.config)
    for name, value in (
        ("batch_size", args.batch_size),
        ("embedding_dim", args.embedding_dim),
    ):
        if value is not None:
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            config = replace(config, **{name: value})
    data_path = args.data or config.paths.data_path
    manifest_path = args.manifest or config.paths.images_manifest
    images_root = args.images_root or config.paths.images_root
    cache_dir = args.cache_dir or config.paths.cache_dir
    output_path = args.output or config.paths.output_path
    report_path = args.report or config.paths.report_path
    source = pl.read_csv(data_path, infer_schema_length=10_000)
    manifest = pl.read_parquet(manifest_path)
    backend = SentenceTransformersBackend(config)
    run = run_embeddings(
        source,
        manifest,
        images_root,
        cache_dir,
        backend,
        resume=config.resume and not args.no_resume,
        batch_size=config.batch_size,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    run.frame.write_parquet(output_path, compression="zstd")
    report_path.write_text(
        render_embeddings_report(run, config, output_path),
        encoding="utf-8",
    )
    print(
        f"Embeddings complete: {run.selected_products} products, "
        f"{run.selected_images} images, {run.cache_hits} cache hits, "
        f"{run.failed_embeddings} errors, {run.elapsed_seconds:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
