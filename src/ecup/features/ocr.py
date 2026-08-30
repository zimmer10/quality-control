"""Fast label-free OCR with deterministic selection and resumable caching."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import polars as pl
import yaml

from ecup.data.image_manifest import MANIFEST_COLUMNS, MANIFEST_SCHEMA

DEFAULT_CONFIG_PATH = Path("configs/ocr.yaml")
CACHE_VERSION = 2
OCR_COLUMNS = (
    "source_image_id",
    "id",
    "image_index",
    "relative_path",
    "ocr_text_by_image",
    "ocr_quality",
    "ocr_status",
    "ocr_error",
    "ocr_model",
    "cache_key",
)
OCR_SCHEMA = {
    "source_image_id": pl.String,
    "id": pl.Int64,
    "image_index": pl.Int64,
    "relative_path": pl.String,
    "ocr_text_by_image": pl.String,
    "ocr_quality": pl.Float32,
    "ocr_status": pl.String,
    "ocr_error": pl.String,
    "ocr_model": pl.String,
    "cache_key": pl.String,
}
OCR_STATUSES = frozenset({"ok", "no_text", "source_error", "ocr_error"})
SUCCESS_STATUSES = frozenset({"ok", "no_text"})
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class OCRPaths:
    images_root: Path
    images_manifest: Path
    cache_dir: Path
    output_path: Path
    report_path: Path


@dataclass(frozen=True)
class OCRConfig:
    version: int
    model: str
    backend: str
    device: str
    detection_model: str
    recognition_model: str
    precision: str
    text_det_limit_side_len: int
    text_rec_score_thresh: float
    text_recognition_batch_size: int
    inference_batch_size: int
    max_images_per_product: int | None
    resume: bool
    retry_errors: bool
    paths: OCRPaths


@dataclass(frozen=True)
class OCRPrediction:
    text: str
    quality: float | None = None


class OCRBackend(Protocol):
    model_id: str
    cache_signature: str

    def extract(self, image_path: Path) -> OCRPrediction:
        """Extract visible text from one image."""


class OCRBackendUnavailableError(RuntimeError):
    """The configured OCR runtime or model cannot be loaded."""


@dataclass(frozen=True)
class OCRRun:
    frame: pl.DataFrame
    total_manifest_rows: int
    selected_rows: int
    selected_products: int
    cache_hits: int
    computed_rows: int
    source_errors: int
    ocr_errors: int
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


def load_ocr_config(path: str | Path = DEFAULT_CONFIG_PATH) -> OCRConfig:
    """Load and strictly validate the versioned R11 configuration."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = _as_mapping(payload, "ocr")
    _exact_keys(
        root,
        {
            "version",
            "model",
            "backend",
            "device",
            "detection_model",
            "recognition_model",
            "precision",
            "text_det_limit_side_len",
            "text_rec_score_thresh",
            "text_recognition_batch_size",
            "inference_batch_size",
            "max_images_per_product",
            "images_root",
            "images_manifest",
            "cache_dir",
            "output_path",
            "report_path",
            "resume",
            "retry_errors",
        },
        "ocr",
    )
    if root["version"] != 2 or type(root["version"]) is not int:
        raise ValueError("ocr.version must be integer 2")
    for key in (
        "model",
        "backend",
        "device",
        "detection_model",
        "recognition_model",
        "precision",
        "images_root",
        "images_manifest",
        "cache_dir",
        "output_path",
        "report_path",
    ):
        if not isinstance(root[key], str) or not root[key].strip():
            raise ValueError(f"ocr.{key} must be a non-empty string")
    if root["backend"] != "paddleocr":
        raise ValueError("ocr.backend must be paddleocr")
    if root["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("ocr.device must be auto, cpu or cuda")
    if root["precision"] not in {"fp32", "fp16"}:
        raise ValueError("ocr.precision must be fp32 or fp16")
    for key in (
        "text_det_limit_side_len",
        "text_recognition_batch_size",
        "inference_batch_size",
    ):
        if type(root[key]) is not int or root[key] <= 0:
            raise ValueError(f"ocr.{key} must be a positive integer")
    image_limit = root["max_images_per_product"]
    if image_limit is not None and (
        type(image_limit) is not int or image_limit <= 0
    ):
        raise ValueError(
            "ocr.max_images_per_product must be null or a positive integer"
        )
    score_threshold = root["text_rec_score_thresh"]
    if type(score_threshold) not in {int, float} or not 0 <= score_threshold <= 1:
        raise ValueError("ocr.text_rec_score_thresh must be within [0, 1]")
    for key in ("resume", "retry_errors"):
        if type(root[key]) is not bool:
            raise ValueError(f"ocr.{key} must be boolean")
    return OCRConfig(
        version=2,
        model=str(root["model"]),
        backend=str(root["backend"]),
        device=str(root["device"]),
        detection_model=str(root["detection_model"]),
        recognition_model=str(root["recognition_model"]),
        precision=str(root["precision"]),
        text_det_limit_side_len=int(root["text_det_limit_side_len"]),
        text_rec_score_thresh=float(score_threshold),
        text_recognition_batch_size=int(root["text_recognition_batch_size"]),
        inference_batch_size=int(root["inference_batch_size"]),
        max_images_per_product=image_limit,
        resume=bool(root["resume"]),
        retry_errors=bool(root["retry_errors"]),
        paths=OCRPaths(
            images_root=Path(str(root["images_root"])),
            images_manifest=Path(str(root["images_manifest"])),
            cache_dir=Path(str(root["cache_dir"])),
            output_path=Path(str(root["output_path"])),
            report_path=Path(str(root["report_path"])),
        ),
    )


def normalize_ocr_text(value: object) -> str:
    """Normalize model output without adding or interpreting evidence."""

    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", html.unescape(str(value)))
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def validate_image_manifest(manifest: pl.DataFrame) -> None:
    if manifest.columns != list(MANIFEST_COLUMNS):
        raise ValueError("image manifest columns do not match R06 contract")
    if dict(manifest.schema) != MANIFEST_SCHEMA:
        raise ValueError("image manifest dtypes do not match R06 contract")
    if manifest.get_column("relative_path").n_unique() != manifest.height:
        raise ValueError("image manifest relative_path values must be unique")


def select_ocr_manifest(
    manifest: pl.DataFrame,
    max_images_per_product: int | None,
) -> pl.DataFrame:
    """Choose readable images first and cap work deterministically per item."""

    validate_image_manifest(manifest)
    if max_images_per_product is not None and max_images_per_product <= 0:
        raise ValueError("max_images_per_product must be positive")
    selected = (
        manifest.filter(pl.col("in_dataset"))
        .with_columns(
            (pl.col("status") != "ok").cast(pl.Int8).alias("_source_error")
        )
        .sort(["id", "_source_error", "image_index", "relative_path"])
    )
    if max_images_per_product is not None:
        selected = selected.group_by("id", maintain_order=True).head(
            max_images_per_product
        )
    return selected.drop("_source_error").sort("relative_path")


class PaddleOCRBackend:
    """Fast PP-OCRv5 Mobile detector and East-Slavic recognizer."""

    def __init__(self, config: OCRConfig) -> None:
        self.model_id = config.model
        signature_payload = {
            "version": CACHE_VERSION,
            "backend": config.backend,
            "detection_model": config.detection_model,
            "recognition_model": config.recognition_model,
            "precision": config.precision,
            "text_det_limit_side_len": config.text_det_limit_side_len,
            "text_rec_score_thresh": config.text_rec_score_thresh,
        }
        self.cache_signature = json.dumps(
            signature_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._config = config
        self._pipeline: Any = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import paddle
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise OCRBackendUnavailableError(
                "failed to import PaddleOCR runtime; check dependency and "
                f"CUDA/NCCL compatibility: {error}"
            ) from error
        has_cuda = bool(
            paddle.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
        if self._config.device == "auto":
            device = "gpu:0" if has_cuda else "cpu"
        elif self._config.device == "cuda":
            if not has_cuda:
                raise OCRBackendUnavailableError(
                    "CUDA was requested but PaddlePaddle GPU is unavailable"
                )
            device = "gpu:0"
        else:
            device = "cpu"
        precision = (
            self._config.precision if device.startswith("gpu") else "fp32"
        )
        try:
            self._pipeline = PaddleOCR(
                text_detection_model_name=self._config.detection_model,
                text_recognition_model_name=self._config.recognition_model,
                text_recognition_batch_size=(
                    self._config.text_recognition_batch_size
                ),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=self._config.text_det_limit_side_len,
                text_det_limit_type="max",
                text_rec_score_thresh=self._config.text_rec_score_thresh,
                device=device,
                precision=precision,
            )
        except Exception as error:
            raise OCRBackendUnavailableError(
                f"failed to initialize {self.model_id}: {error}"
            ) from error

    def extract_batch(
        self,
        image_paths: Sequence[Path],
    ) -> list[OCRPrediction]:
        self._load()
        if not image_paths:
            return []
        try:
            results = list(
                self._pipeline.predict([str(path) for path in image_paths])
            )
        except Exception as error:
            raise RuntimeError(f"PaddleOCR inference failed: {error}") from error
        if len(results) != len(image_paths):
            raise RuntimeError(
                "PaddleOCR returned an unexpected number of results"
            )
        predictions: list[OCRPrediction] = []
        for result in results:
            payload = result.json
            if not isinstance(payload, Mapping):
                raise RuntimeError("PaddleOCR result.json must be a mapping")
            values = payload.get("res", payload)
            if not isinstance(values, Mapping):
                raise RuntimeError("PaddleOCR result payload is invalid")
            texts = values.get("rec_texts", [])
            scores = values.get("rec_scores", [])
            if isinstance(texts, (str, bytes)):
                raise RuntimeError("PaddleOCR rec_texts is invalid")
            try:
                text_values = list(texts)
            except TypeError as error:
                raise RuntimeError(
                    "PaddleOCR rec_texts is invalid"
                ) from error
            normalized = []
            for text in text_values:
                value = normalize_ocr_text(text)
                if value:
                    normalized.append(value)
            numeric_scores = []
            for score in scores:
                try:
                    value = float(score)
                except (TypeError, ValueError):
                    continue
                if 0.0 <= value <= 1.0:
                    numeric_scores.append(value)
            quality = (
                sum(numeric_scores) / len(numeric_scores)
                if numeric_scores
                else None
            )
            predictions.append(
                OCRPrediction(text="\n".join(normalized), quality=quality)
            )
        return predictions

    def extract(self, image_path: Path) -> OCRPrediction:
        return self.extract_batch([image_path])[0]


def _cache_key(row: Mapping[str, object], signature: str) -> str:
    payload = {
        "version": CACHE_VERSION,
        "signature": signature,
        "relative_path": row["relative_path"],
        "file_size_bytes": row["file_size_bytes"],
        "width": row["width"],
        "height": row["height"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / cache_key[:2] / f"{cache_key}.json"


def _read_cache(
    path: Path,
    cache_key: str,
    signature: str,
) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("version") != CACHE_VERSION
        or payload.get("cache_key") != cache_key
        or payload.get("cache_signature") != signature
    ):
        return None
    row = payload.get("row")
    if not isinstance(row, dict) or set(row) != set(OCR_COLUMNS):
        return None
    return row


def _write_cache(
    path: Path,
    cache_key: str,
    signature: str,
    row: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "cache_key": cache_key,
        "cache_signature": signature,
        "row": dict(row),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _result_row(
    source: Mapping[str, object],
    model_id: str,
    cache_key: str,
    status: str,
    text: str = "",
    quality: float | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "source_image_id": str(source["relative_path"]),
        "id": source["id"],
        "image_index": source["image_index"],
        "relative_path": source["relative_path"],
        "ocr_text_by_image": normalize_ocr_text(text),
        "ocr_quality": quality,
        "ocr_status": status,
        "ocr_error": error,
        "ocr_model": model_id,
        "cache_key": cache_key,
    }


def _frame_from_rows(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=OCR_SCHEMA).select(OCR_COLUMNS)
    return (
        pl.DataFrame(rows, schema=OCR_SCHEMA)
        .select(OCR_COLUMNS)
        .sort("relative_path")
    )


def validate_ocr_output(
    selected_manifest: pl.DataFrame,
    output: pl.DataFrame,
) -> None:
    """Validate the frozen R11 schema and alignment with R06."""

    if output.columns != list(OCR_COLUMNS):
        raise ValueError("OCR output columns do not match R11 contract")
    if dict(output.schema) != OCR_SCHEMA:
        raise ValueError("OCR output dtypes do not match R11 contract")
    if output.height != selected_manifest.height:
        raise ValueError("OCR output must contain one row per selected image")
    for column in ("source_image_id", "relative_path", "cache_key"):
        if output.get_column(column).null_count():
            raise ValueError(f"OCR output {column} contains null values")
        if output.get_column(column).n_unique() != output.height:
            raise ValueError(f"OCR output {column} values must be unique")
    statuses = set(output.get_column("ocr_status").to_list())
    if not statuses <= OCR_STATUSES:
        raise ValueError(f"unknown OCR statuses: {sorted(statuses)}")
    successful = output.filter(pl.col("ocr_status") == "ok")
    if successful.filter(pl.col("ocr_text_by_image") == "").height:
        raise ValueError("ok OCR rows must contain text")
    no_text = output.filter(pl.col("ocr_status") == "no_text")
    if no_text.filter(pl.col("ocr_text_by_image") != "").height:
        raise ValueError("no_text OCR rows must have empty text")
    failed = output.filter(
        pl.col("ocr_status").is_in(["source_error", "ocr_error"])
    )
    if failed.filter(pl.col("ocr_error").is_null()).height:
        raise ValueError("failed OCR rows must contain an error")
    invalid_quality = output.filter(
        pl.col("ocr_quality").is_not_null()
        & ~pl.col("ocr_quality").is_between(0.0, 1.0)
    )
    if invalid_quality.height:
        raise ValueError("ocr_quality must be null or within [0, 1]")
    expected = selected_manifest.select(
        "id", "image_index", "relative_path"
    ).sort("relative_path")
    actual = output.select("id", "image_index", "relative_path").sort(
        "relative_path"
    )
    if not expected.equals(actual):
        raise ValueError("OCR output is misaligned with image manifest")


def _batch_predictions(
    backend: OCRBackend,
    image_paths: Sequence[Path],
) -> list[OCRPrediction | Exception]:
    batch_method = getattr(backend, "extract_batch", None)
    if batch_method is None:
        results: list[OCRPrediction | Exception] = []
        for path in image_paths:
            try:
                results.append(backend.extract(path))
            except OCRBackendUnavailableError:
                raise
            except Exception as error:
                results.append(error)
        return results
    try:
        predictions = list(batch_method(image_paths))
        if len(predictions) != len(image_paths):
            raise RuntimeError("OCR batch result length mismatch")
        return predictions
    except OCRBackendUnavailableError:
        raise
    except Exception:
        results = []
        for path in image_paths:
            try:
                one = list(batch_method([path]))
                if len(one) != 1:
                    raise RuntimeError("OCR single result length mismatch")
                results.append(one[0])
            except OCRBackendUnavailableError:
                raise
            except Exception as error:
                results.append(error)
        return results


def run_ocr(
    manifest: pl.DataFrame,
    images_root: str | Path,
    cache_dir: str | Path,
    backend: OCRBackend,
    *,
    resume: bool = True,
    retry_errors: bool = True,
    limit: int | None = None,
    progress_every: int = 0,
    batch_size: int = 1,
    max_images_per_product: int | None = None,
) -> OCRRun:
    """Process selected images in batches with an atomic cache per image."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    selected = select_ocr_manifest(manifest, max_images_per_product)
    if limit is not None:
        selected = selected.head(limit)
    root = Path(images_root)
    cache_root = Path(cache_dir)
    rows_by_path: dict[str, dict[str, object]] = {}
    pending: list[tuple[dict[str, object], str, Path]] = []
    cache_hits = 0
    source_errors = 0
    started = time.perf_counter()

    for source in selected.iter_rows(named=True):
        key = _cache_key(source, backend.cache_signature)
        cache_path = _cache_path(cache_root, key)
        relative_path = str(source["relative_path"])
        if source["status"] != "ok":
            source_errors += 1
            rows_by_path[relative_path] = _result_row(
                source,
                backend.model_id,
                key,
                "source_error",
                error=str(source["error"] or "image manifest error"),
            )
            continue
        cached = (
            _read_cache(cache_path, key, backend.cache_signature)
            if resume
            else None
        )
        if cached is not None and (
            cached["ocr_status"] in SUCCESS_STATUSES or not retry_errors
        ):
            rows_by_path[relative_path] = cached
            cache_hits += 1
        else:
            pending.append((source, key, cache_path))

    completed = len(rows_by_path)
    last_reported = 0

    def show_progress(*, force: bool = False) -> None:
        nonlocal last_reported
        if progress_every and (
            force or completed - last_reported >= progress_every
        ):
            elapsed = time.perf_counter() - started
            print(
                f"OCR progress: {completed}/{selected.height}, "
                f"cached={cache_hits}, computed={completed - cache_hits - source_errors}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            last_reported = completed

    if completed and not pending:
        show_progress(force=True)
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        paths = [root / str(item[0]["relative_path"]) for item in batch]
        predictions = _batch_predictions(backend, paths)
        for item, prediction in zip(batch, predictions, strict=True):
            source, key, cache_path = item
            relative_path = str(source["relative_path"])
            if isinstance(prediction, Exception):
                row = _result_row(
                    source,
                    backend.model_id,
                    key,
                    "ocr_error",
                    error=(
                        f"{type(prediction).__name__}: {prediction}"
                    )[:2000],
                )
            else:
                text = normalize_ocr_text(prediction.text)
                quality = prediction.quality
                if quality is not None and not 0.0 <= quality <= 1.0:
                    row = _result_row(
                        source,
                        backend.model_id,
                        key,
                        "ocr_error",
                        error="ValueError: backend quality must be within [0, 1]",
                    )
                else:
                    row = _result_row(
                        source,
                        backend.model_id,
                        key,
                        "ok" if text else "no_text",
                        text=text,
                        quality=quality,
                    )
            _write_cache(cache_path, key, backend.cache_signature, row)
            rows_by_path[relative_path] = row
            completed += 1
        show_progress(force=completed == selected.height)

    output = _frame_from_rows(list(rows_by_path.values()))
    validate_ocr_output(selected, output)
    return OCRRun(
        frame=output,
        total_manifest_rows=manifest.height,
        selected_rows=selected.height,
        selected_products=selected.get_column("id").n_unique(),
        cache_hits=cache_hits,
        computed_rows=len(pending),
        source_errors=source_errors,
        ocr_errors=output.filter(pl.col("ocr_status") == "ocr_error").height,
        elapsed_seconds=time.perf_counter() - started,
    )


def ocr_output_checksum(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.select(OCR_COLUMNS).sort("relative_path").iter_rows():
        digest.update(repr(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def render_ocr_report(
    run: OCRRun,
    config: OCRConfig,
    output_path: str | Path,
) -> str:
    counts = (
        run.frame.group_by("ocr_status")
        .agg(pl.len().alias("count"))
        .sort("ocr_status")
    )
    lines = [
        "# R11 — кешируемый OCR",
        "",
        "## Контракт",
        "",
        f"- Backend: `{config.backend}`.",
        f"- Детектор: `{config.detection_model}`.",
        f"- Распознавание: `{config.recognition_model}`.",
        f"- Image manifest: `{config.paths.images_manifest}`.",
        f"- Кеш: `{config.paths.cache_dir}`.",
        f"- Итоговая таблица: `{output_path}`.",
        f"- Строк в исходном manifest: **{run.total_manifest_rows}**.",
        f"- Товаров в текущем запуске: **{run.selected_products}**.",
        f"- Изображений в текущем запуске: **{run.selected_rows}**.",
        "- Максимум изображений товара: **все**."
        if config.max_images_per_product is None
        else (
            "- Максимум изображений товара: "
            f"**{config.max_images_per_product}**."
        ),
        f"- Логическая SHA-256: `{ocr_output_checksum(run.frame)}`.",
        "- Label, OCR Rules и финальный verdict не используются.",
        "",
        "## Статусы",
        "",
        "| Статус | Изображений |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {row['ocr_status']} | {row['count']} |"
        for row in counts.iter_rows(named=True)
    )
    lines.extend(
        [
            "",
            "## Возобновляемость и время",
            "",
            f"- Получено из кеша: **{run.cache_hits}**.",
            f"- Обработано backend: **{run.computed_rows}**.",
            f"- Ошибок исходных изображений: **{run.source_errors}**.",
            f"- Ошибок OCR: **{run.ocr_errors}**.",
            f"- Время текущего запуска: **{run.elapsed_seconds:.3f} s**.",
            "- Каждый результат записывается атомарно сразу после изображения.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and cache fast label-free OCR for product images"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-images-per-product", type=int)
    parser.add_argument("--all-images", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_ocr_config(args.config)
    for name, value in (
        ("inference_batch_size", args.batch_size),
        ("max_images_per_product", args.max_images_per_product),
    ):
        if value is not None:
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            config = replace(config, **{name: value})
    manifest_path = args.manifest or config.paths.images_manifest
    images_root = args.images_root or config.paths.images_root
    cache_dir = args.cache_dir or config.paths.cache_dir
    output_path = args.output or config.paths.output_path
    report_path = args.report or config.paths.report_path
    manifest = pl.read_parquet(manifest_path)
    backend = PaddleOCRBackend(config)
    run = run_ocr(
        manifest,
        images_root,
        cache_dir,
        backend,
        resume=config.resume and not args.no_resume,
        retry_errors=config.retry_errors,
        limit=args.limit,
        progress_every=args.progress_every,
        batch_size=config.inference_batch_size,
        max_images_per_product=(
            None if args.all_images else config.max_images_per_product
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    run.frame.write_parquet(output_path, compression="zstd")
    report_path.write_text(
        render_ocr_report(run, config, output_path),
        encoding="utf-8",
    )
    print(
        f"OCR complete: {run.selected_rows} images, "
        f"{run.cache_hits} cached, {run.ocr_errors} OCR errors, "
        f"{run.elapsed_seconds:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
