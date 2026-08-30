"""Label-free OCR extraction with per-image resumable caching."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import polars as pl
import yaml
from PIL import Image

from ecup.data.image_manifest import MANIFEST_COLUMNS, MANIFEST_SCHEMA

DEFAULT_CONFIG_PATH = Path("configs/ocr.yaml")
CACHE_VERSION = 1
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
    prompt: str
    max_new_tokens: int
    max_pixels: int
    local_files_only: bool
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
            "prompt",
            "max_new_tokens",
            "max_pixels",
            "local_files_only",
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
    if root["version"] != 1 or type(root["version"]) is not int:
        raise ValueError("ocr.version must be integer 1")
    for key in (
        "model",
        "backend",
        "device",
        "prompt",
        "images_root",
        "images_manifest",
        "cache_dir",
        "output_path",
        "report_path",
    ):
        if not isinstance(root[key], str) or not root[key].strip():
            raise ValueError(f"ocr.{key} must be a non-empty string")
    if root["backend"] != "transformers":
        raise ValueError("ocr.backend must be transformers")
    if root["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("ocr.device must be auto, cpu or cuda")
    for key in ("max_new_tokens", "max_pixels"):
        if type(root[key]) is not int or root[key] <= 0:
            raise ValueError(f"ocr.{key} must be a positive integer")
    for key in ("local_files_only", "resume", "retry_errors"):
        if type(root[key]) is not bool:
            raise ValueError(f"ocr.{key} must be boolean")
    return OCRConfig(
        version=1,
        model=root["model"],
        backend=root["backend"],
        device=root["device"],
        prompt=root["prompt"],
        max_new_tokens=root["max_new_tokens"],
        max_pixels=root["max_pixels"],
        local_files_only=root["local_files_only"],
        resume=root["resume"],
        retry_errors=root["retry_errors"],
        paths=OCRPaths(
            images_root=Path(root["images_root"]),
            images_manifest=Path(root["images_manifest"]),
            cache_dir=Path(root["cache_dir"]),
            output_path=Path(root["output_path"]),
            report_path=Path(root["report_path"]),
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


def _model_source(model_id: str, local_files_only: bool) -> str:
    shared_root = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
    shared_path = shared_root / model_id
    if shared_path.exists():
        return str(shared_path)
    if local_files_only:
        raise OCRBackendUnavailableError(
            f"OCR model is absent: {shared_path}. Set SHARED_MODELS_PATH "
            "or download the model locally."
        )
    return model_id


class TransformersOCRBackend:
    """Lazy Hugging Face backend for the allowed PaddleOCR-VL model."""

    def __init__(self, config: OCRConfig) -> None:
        self.model_id = config.model
        signature_payload = {
            "backend": config.backend,
            "model": config.model,
            "prompt": config.prompt,
            "max_new_tokens": config.max_new_tokens,
            "max_pixels": config.max_pixels,
        }
        self.cache_signature = json.dumps(
            signature_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._config = config
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise OCRBackendUnavailableError(
                "OCR dependencies are missing. Install with: "
                "pip install -e '.[ocr]'"
            ) from error
        source = _model_source(
            self._config.model,
            self._config.local_files_only,
        )
        if self._config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._config.device
        if device == "cuda" and not torch.cuda.is_available():
            raise OCRBackendUnavailableError(
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
        try:
            self._processor = AutoProcessor.from_pretrained(
                source,
                local_files_only=self._config.local_files_only,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                source,
                local_files_only=self._config.local_files_only,
                dtype=dtype,
            ).to(device)
        except Exception as error:
            raise OCRBackendUnavailableError(
                f"failed to load OCR model {self._config.model}: {error}"
            ) from error
        self._model.eval()
        self._torch = torch
        self._device = device

    def extract(self, image_path: Path) -> OCRPrediction:
        self._load()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._config.prompt},
                ],
            }
        ]
        image_processor = self._processor.image_processor
        shortest_edge = getattr(image_processor, "min_pixels", None)
        if shortest_edge is None:
            shortest_edge = image_processor.size.shortest_edge
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": shortest_edge,
                    "longest_edge": self._config.max_pixels,
                }
            },
        )
        inputs = inputs.to(self._device)
        input_length = int(inputs["input_ids"].shape[1])
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                do_sample=False,
            )
        generated_only = generated[0][input_length:-1]
        text = self._processor.decode(generated_only)
        return OCRPrediction(text=normalize_ocr_text(text), quality=None)


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
    failed = output.filter(pl.col("ocr_status").is_in(["source_error", "ocr_error"]))
    if failed.filter(pl.col("ocr_error").is_null()).height:
        raise ValueError("failed OCR rows must contain an error")
    invalid_quality = output.filter(
        pl.col("ocr_quality").is_not_null()
        & ~pl.col("ocr_quality").is_between(0.0, 1.0)
    )
    if invalid_quality.height:
        raise ValueError("ocr_quality must be null or within [0, 1]")
    expected = selected_manifest.select(
        "id",
        "image_index",
        "relative_path",
    ).sort("relative_path")
    actual = output.select("id", "image_index", "relative_path").sort(
        "relative_path"
    )
    if not expected.equals(actual):
        raise ValueError("OCR output is misaligned with image manifest")


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
) -> OCRRun:
    """Process mapped images, writing an atomic cache record per image."""

    validate_image_manifest(manifest)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    selected = manifest.filter(pl.col("in_dataset")).sort("relative_path")
    if limit is not None:
        selected = selected.head(limit)
    root = Path(images_root)
    cache_root = Path(cache_dir)
    rows: list[dict[str, object]] = []
    cache_hits = 0
    computed = 0
    source_errors = 0
    started = time.perf_counter()

    def show_progress(position: int) -> None:
        if progress_every and (
            position % progress_every == 0 or position == selected.height
        ):
            elapsed = time.perf_counter() - started
            print(
                f"OCR progress: {position}/{selected.height}, "
                f"cached={cache_hits}, computed={computed}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    for position, source in enumerate(
        selected.iter_rows(named=True),
        start=1,
    ):
        key = _cache_key(source, backend.cache_signature)
        cache_path = _cache_path(cache_root, key)
        if source["status"] != "ok":
            source_errors += 1
            rows.append(
                _result_row(
                    source,
                    backend.model_id,
                    key,
                    "source_error",
                    error=str(source["error"] or "image manifest error"),
                )
            )
            show_progress(position)
            continue
        cached = (
            _read_cache(cache_path, key, backend.cache_signature)
            if resume
            else None
        )
        if cached is not None and (
            cached["ocr_status"] in SUCCESS_STATUSES or not retry_errors
        ):
            rows.append(cached)
            cache_hits += 1
            show_progress(position)
            continue
        computed += 1
        try:
            prediction = backend.extract(root / str(source["relative_path"]))
            text = normalize_ocr_text(prediction.text)
            quality = prediction.quality
            if quality is not None and not 0.0 <= quality <= 1.0:
                raise ValueError("backend quality must be within [0, 1]")
            row = _result_row(
                source,
                backend.model_id,
                key,
                "ok" if text else "no_text",
                text=text,
                quality=quality,
            )
        except OCRBackendUnavailableError:
            raise
        except Exception as error:
            message = f"{type(error).__name__}: {error}"[:2000]
            row = _result_row(
                source,
                backend.model_id,
                key,
                "ocr_error",
                error=message,
            )
        _write_cache(cache_path, key, backend.cache_signature, row)
        rows.append(row)
        show_progress(position)

    output = _frame_from_rows(rows)
    validate_ocr_output(selected, output)
    return OCRRun(
        frame=output,
        total_manifest_rows=manifest.height,
        selected_rows=selected.height,
        cache_hits=cache_hits,
        computed_rows=computed,
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
        f"- Модель: `{config.model}`.",
        f"- Image manifest: `{config.paths.images_manifest}`.",
        f"- Кеш: `{config.paths.cache_dir}`.",
        f"- Итоговая таблица: `{output_path}`.",
        f"- Строк в исходном manifest: **{run.total_manifest_rows}**.",
        f"- Строк в текущем запуске: **{run.selected_rows}**.",
        f"- Логическая SHA-256: `{ocr_output_checksum(run.frame)}`.",
        "- Одна строка соответствует одному изображению товара.",
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
        description="Extract and cache label-free OCR for product images"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_ocr_config(args.config)
    for name, value in (
        ("max_pixels", args.max_pixels),
        ("max_new_tokens", args.max_new_tokens),
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
    backend = TransformersOCRBackend(config)
    run = run_ocr(
        manifest,
        images_root,
        cache_dir,
        backend,
        resume=config.resume and not args.no_resume,
        retry_errors=config.retry_errors,
        limit=args.limit,
        progress_every=args.progress_every,
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
        f"{run.cache_hits} cached, {run.ocr_errors} OCR errors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
