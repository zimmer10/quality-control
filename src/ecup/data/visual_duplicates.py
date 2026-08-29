"""Detect exact and perceptually similar product images without label leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl
from PIL import Image, ImageOps

HASH_COLUMNS = (
    "id",
    "image_index",
    "relative_path",
    "file_sha256",
    "phash",
    "dhash",
    "width",
    "height",
    "status",
    "error",
)
HASH_SCHEMA = {
    "id": pl.Int64,
    "image_index": pl.Int64,
    "relative_path": pl.String,
    "file_sha256": pl.String,
    "phash": pl.String,
    "dhash": pl.String,
    "width": pl.Int64,
    "height": pl.Int64,
    "status": pl.String,
    "error": pl.String,
}
PAIR_COLUMNS = (
    "left_id",
    "right_id",
    "left_path",
    "right_path",
    "match_kind",
    "exact_image_pairs",
    "strong_image_pairs",
    "candidate_image_pairs",
    "phash_distance",
    "dhash_distance",
    "aspect_ratio_delta",
    "primary_match",
    "auto_merge",
)
PAIR_SCHEMA = {
    "left_id": pl.Int64,
    "right_id": pl.Int64,
    "left_path": pl.String,
    "right_path": pl.String,
    "match_kind": pl.String,
    "exact_image_pairs": pl.UInt32,
    "strong_image_pairs": pl.UInt32,
    "candidate_image_pairs": pl.UInt32,
    "phash_distance": pl.Int16,
    "dhash_distance": pl.Int16,
    "aspect_ratio_delta": pl.Float64,
    "primary_match": pl.Boolean,
    "auto_merge": pl.Boolean,
}


@dataclass(frozen=True)
class VisualThresholds:
    """Conservative thresholds for product-level visual evidence."""

    candidate_phash_distance: int = 6
    candidate_dhash_distance: int = 10
    candidate_aspect_delta: float = 0.05
    strong_phash_distance: int = 4
    strong_dhash_distance: int = 4
    strong_aspect_delta: float = 0.02
    very_strong_phash_distance: int = 1
    very_strong_dhash_distance: int = 1
    very_strong_aspect_delta: float = 0.01
    max_common_exact_products: int = 20


@dataclass
class _PairEvidence:
    left_id: int
    right_id: int
    exact_pairs: set[tuple[str, str]] = field(default_factory=set)
    common_exact_pairs: set[tuple[str, str]] = field(default_factory=set)
    strong_pairs: set[tuple[str, str]] = field(default_factory=set)
    candidate_pairs: set[tuple[str, str]] = field(default_factory=set)
    primary_exact: bool = False
    primary_strong: bool = False
    primary_very_strong: bool = False
    best_score: tuple[float, ...] | None = None
    best_left_path: str = ""
    best_right_path: str = ""
    best_phash_distance: int = 0
    best_dhash_distance: int = 0
    best_aspect_delta: float = 0.0

    def _oriented(
        self,
        left: dict[str, object],
        right: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        if int(left["id"]) == self.left_id:
            return left, right
        return right, left

    def add_exact(
        self,
        left: dict[str, object],
        right: dict[str, object],
        *,
        trusted: bool,
    ) -> None:
        left, right = self._oriented(left, right)
        path_pair = (str(left["relative_path"]), str(right["relative_path"]))
        self.candidate_pairs.add(path_pair)
        primary = left["image_index"] == 0 and right["image_index"] == 0
        if trusted:
            self.exact_pairs.add(path_pair)
            self.primary_exact = self.primary_exact or primary
        else:
            self.common_exact_pairs.add(path_pair)
        score = (-1.0, -1.0, 0.0, *path_pair)
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_left_path, self.best_right_path = path_pair
            self.best_phash_distance = 0
            self.best_dhash_distance = 0
            self.best_aspect_delta = 0.0

    def add_near(
        self,
        left: dict[str, object],
        right: dict[str, object],
        *,
        phash_distance: int,
        dhash_distance: int,
        aspect_delta: float,
        thresholds: VisualThresholds,
    ) -> None:
        left, right = self._oriented(left, right)
        path_pair = (str(left["relative_path"]), str(right["relative_path"]))
        self.candidate_pairs.add(path_pair)
        primary = left["image_index"] == 0 and right["image_index"] == 0
        strong = (
            phash_distance <= thresholds.strong_phash_distance
            and dhash_distance <= thresholds.strong_dhash_distance
            and aspect_delta <= thresholds.strong_aspect_delta
        )
        very_strong = (
            phash_distance <= thresholds.very_strong_phash_distance
            and dhash_distance <= thresholds.very_strong_dhash_distance
            and aspect_delta <= thresholds.very_strong_aspect_delta
        )
        if strong:
            self.strong_pairs.add(path_pair)
            self.primary_strong = self.primary_strong or primary
        if very_strong and primary:
            self.primary_very_strong = True
        score = (
            float(phash_distance),
            float(dhash_distance),
            aspect_delta,
            *path_pair,
        )
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_left_path, self.best_right_path = path_pair
            self.best_phash_distance = phash_distance
            self.best_dhash_distance = dhash_distance
            self.best_aspect_delta = aspect_delta

    def as_row(self) -> dict[str, object]:
        auto_merge = self.primary_exact or len(self.exact_pairs) >= 2
        if self.exact_pairs:
            match_kind = "exact" if auto_merge else "exact_candidate"
        elif self.common_exact_pairs:
            match_kind = "common_image"
        else:
            match_kind = "near_candidate"
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_path": self.best_left_path,
            "right_path": self.best_right_path,
            "match_kind": match_kind,
            "exact_image_pairs": len(self.exact_pairs),
            "strong_image_pairs": len(self.strong_pairs),
            "candidate_image_pairs": len(self.candidate_pairs),
            "phash_distance": self.best_phash_distance,
            "dhash_distance": self.best_dhash_distance,
            "aspect_ratio_delta": self.best_aspect_delta,
            "primary_match": (
                self.primary_exact
                or self.primary_strong
                or self.primary_very_strong
            ),
            "auto_merge": auto_merge,
        }


@dataclass
class _BKNode:
    value: int
    children: dict[int, "_BKNode"] = field(default_factory=dict)


class _BKTree:
    """Small deterministic BK-tree for Hamming-neighbor lookup."""

    def __init__(self, values: Iterable[int]) -> None:
        self.root: _BKNode | None = None
        for value in sorted(set(values)):
            self.add(value)

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = _BKNode(value)
            return
        node = self.root
        while True:
            distance = hamming_distance(value, node.value)
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value)
                return
            node = child

    def query(self, value: int, max_distance: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node.value)
            if distance <= max_distance:
                matches.append(node.value)
            lower = max(0, distance - max_distance)
            upper = distance + max_distance
            for edge_distance, child in node.children.items():
                if lower <= edge_distance <= upper:
                    stack.append(child)
        return sorted(matches)


def _dct_matrix(size: int) -> np.ndarray:
    indices = np.arange(size, dtype=np.float64)
    frequencies = indices[:, None]
    matrix = np.cos(math.pi * (2 * indices + 1) * frequencies / (2 * size))
    matrix[0] *= math.sqrt(1 / size)
    matrix[1:] *= math.sqrt(2 / size)
    return matrix


_DCT_32 = _dct_matrix(32)


def _bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    width = (bits.size + 3) // 4
    return f"{value:0{width}x}"


def _perceptual_hashes(image: Image.Image) -> tuple[str, str]:
    normalized = ImageOps.exif_transpose(image).convert("L")
    phash_pixels = np.asarray(
        normalized.resize((32, 32), Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    transformed = _DCT_32 @ phash_pixels @ _DCT_32.T
    low_frequency = transformed[:8, :8]
    phash = _bits_to_hex(low_frequency > np.median(low_frequency))

    dhash_pixels = np.asarray(
        normalized.resize((9, 8), Image.Resampling.LANCZOS),
        dtype=np.int16,
    )
    dhash = _bits_to_hex(dhash_pixels[:, 1:] > dhash_pixels[:, :-1])
    return phash, dhash


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_image(
    row: dict[str, object],
    images_root: Path,
) -> dict[str, object]:
    path = images_root / str(row["relative_path"])
    file_sha256: str | None = None
    phash: str | None = None
    dhash: str | None = None
    width: int | None = None
    height: int | None = None
    status = "ok"
    error_text: str | None = None
    try:
        file_sha256 = _file_sha256(path)
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            phash, dhash = _perceptual_hashes(image)
    except Exception as error:  # Pillow and filesystem errors vary by format.
        status = "error"
        error_text = f"{type(error).__name__}: {error}"
    return {
        "id": row["id"],
        "image_index": row["image_index"],
        "relative_path": row["relative_path"],
        "file_sha256": file_sha256,
        "phash": phash,
        "dhash": dhash,
        "width": width,
        "height": height,
        "status": status,
        "error": error_text,
    }


def _eligible_manifest_rows(image_manifest: pl.DataFrame) -> pl.DataFrame:
    required = {"id", "image_index", "relative_path", "in_dataset", "status"}
    missing = sorted(required - set(image_manifest.columns))
    if missing:
        raise ValueError(
            "image manifest misses required columns: " + ", ".join(missing)
        )
    eligible = image_manifest.filter(
        pl.col("in_dataset")
        & (pl.col("status") == "ok")
        & pl.col("id").is_not_null()
        & pl.col("image_index").is_not_null()
    ).select("id", "image_index", "relative_path")
    if eligible.get_column("relative_path").n_unique() != eligible.height:
        raise ValueError("eligible image paths must be unique")
    return eligible.sort("relative_path")


def validate_image_hashes(
    image_manifest: pl.DataFrame,
    hashes: pl.DataFrame,
) -> None:
    """Validate the cached image-hash contract against R06."""

    if hashes.columns != list(HASH_COLUMNS):
        raise ValueError(f"hash columns must be exactly: {HASH_COLUMNS}")
    if dict(hashes.schema) != HASH_SCHEMA:
        raise ValueError("image hash dtypes do not match the frozen schema")
    eligible = _eligible_manifest_rows(image_manifest)
    if hashes.height != eligible.height:
        raise ValueError("image hashes must contain every eligible image")
    if hashes.get_column("relative_path").n_unique() != hashes.height:
        raise ValueError("image hash paths must be unique")
    expected = eligible.select("id", "image_index", "relative_path")
    actual = hashes.select("id", "image_index", "relative_path").sort(
        "relative_path"
    )
    if not expected.equals(actual):
        raise ValueError("image hash rows do not match image manifest")
    statuses = set(hashes.get_column("status").unique().to_list())
    if not statuses <= {"ok", "error"}:
        raise ValueError(f"unknown hash statuses: {sorted(statuses)}")
    readable = hashes.filter(pl.col("status") == "ok")
    if readable.select(
        pl.any_horizontal(
            pl.col("file_sha256").is_null(),
            pl.col("phash").is_null(),
            pl.col("dhash").is_null(),
            pl.col("width").is_null(),
            pl.col("height").is_null(),
        ).any()
    ).item():
        raise ValueError("successful hash rows must be complete")


def build_image_hashes(
    image_manifest: pl.DataFrame,
    images_root: str | Path,
    workers: int = 4,
) -> pl.DataFrame:
    """Calculate exact, perceptual and difference hashes for all images."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    eligible = _eligible_manifest_rows(image_manifest)
    rows = eligible.iter_rows(named=True)
    root = Path(images_root)

    def calculate(row: dict[str, object]) -> dict[str, object]:
        return _hash_image(row, root)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashed_rows = list(executor.map(calculate, rows))
    hashes = (
        pl.DataFrame(hashed_rows, schema=HASH_SCHEMA)
        .select(HASH_COLUMNS)
        .sort("relative_path")
    )
    validate_image_hashes(image_manifest, hashes)
    return hashes


def hamming_distance(left: int | str, right: int | str) -> int:
    """Return bitwise Hamming distance for integer or hexadecimal hashes."""

    left_value = int(left, 16) if isinstance(left, str) else left
    right_value = int(right, 16) if isinstance(right, str) else right
    return (left_value ^ right_value).bit_count()


def _aspect_delta(left: dict[str, object], right: dict[str, object]) -> float:
    left_ratio = int(left["width"]) / int(left["height"])
    right_ratio = int(right["width"]) / int(right["height"])
    return abs(left_ratio - right_ratio) / max(left_ratio, right_ratio)


def _pair_key(
    left: dict[str, object],
    right: dict[str, object],
) -> tuple[int, int] | None:
    left_id = int(left["id"])
    right_id = int(right["id"])
    if left_id == right_id:
        return None
    return min(left_id, right_id), max(left_id, right_id)


def _evidence_for(
    evidence: dict[tuple[int, int], _PairEvidence],
    key: tuple[int, int],
) -> _PairEvidence:
    if key not in evidence:
        evidence[key] = _PairEvidence(*key)
    return evidence[key]


def _add_exact_evidence(
    records: list[dict[str, object]],
    evidence: dict[tuple[int, int], _PairEvidence],
    thresholds: VisualThresholds,
) -> set[tuple[str, str]]:
    exact_image_pairs: set[tuple[str, str]] = set()
    groups: dict[str, list[dict[str, object]]] = {}
    for row in records:
        groups.setdefault(str(row["file_sha256"]), []).append(row)

    for group in groups.values():
        product_ids = sorted({int(row["id"]) for row in group})
        if len(product_ids) < 2:
            continue
        trusted = len(product_ids) <= thresholds.max_common_exact_products
        if trusted:
            image_pairs = combinations(group, 2)
        else:
            representatives: dict[int, dict[str, object]] = {}
            for row in group:
                representatives.setdefault(int(row["id"]), row)
            ordered = [representatives[item_id] for item_id in product_ids]
            image_pairs = (
                (ordered[0], candidate)
                for candidate in ordered[1:]
            )
        for left, right in image_pairs:
            key = _pair_key(left, right)
            if key is None:
                continue
            oriented_paths = tuple(
                sorted(
                    (
                        str(left["relative_path"]),
                        str(right["relative_path"]),
                    )
                )
            )
            exact_image_pairs.add(oriented_paths)
            _evidence_for(evidence, key).add_exact(
                left,
                right,
                trusted=trusted,
            )
    return exact_image_pairs


def _add_near_evidence(
    records: list[dict[str, object]],
    evidence: dict[tuple[int, int], _PairEvidence],
    exact_image_pairs: set[tuple[str, str]],
    thresholds: VisualThresholds,
) -> None:
    by_phash: dict[int, list[dict[str, object]]] = {}
    for row in records:
        by_phash.setdefault(int(str(row["phash"]), 16), []).append(row)
    tree = _BKTree(by_phash)

    for left_hash in sorted(by_phash):
        for right_hash in tree.query(
            left_hash,
            thresholds.candidate_phash_distance,
        ):
            if right_hash < left_hash:
                continue
            if right_hash == left_hash:
                image_pairs = combinations(by_phash[left_hash], 2)
            else:
                image_pairs = product(
                    by_phash[left_hash],
                    by_phash[right_hash],
                )
            phash_distance = hamming_distance(left_hash, right_hash)
            for left, right in image_pairs:
                key = _pair_key(left, right)
                if key is None:
                    continue
                if left["file_sha256"] == right["file_sha256"]:
                    continue
                path_pair = tuple(
                    sorted(
                        (
                            str(left["relative_path"]),
                            str(right["relative_path"]),
                        )
                    )
                )
                if path_pair in exact_image_pairs:
                    continue
                aspect_delta = _aspect_delta(left, right)
                if aspect_delta > thresholds.candidate_aspect_delta:
                    continue
                dhash_distance = hamming_distance(
                    str(left["dhash"]),
                    str(right["dhash"]),
                )
                if dhash_distance > thresholds.candidate_dhash_distance:
                    continue
                _evidence_for(evidence, key).add_near(
                    left,
                    right,
                    phash_distance=phash_distance,
                    dhash_distance=dhash_distance,
                    aspect_delta=aspect_delta,
                    thresholds=thresholds,
                )


def validate_visual_pairs(
    pairs: pl.DataFrame,
    dataset_ids: Iterable[int],
) -> None:
    """Validate one deterministic product-level row per candidate pair."""

    if pairs.columns != list(PAIR_COLUMNS):
        raise ValueError(f"pair columns must be exactly: {PAIR_COLUMNS}")
    if dict(pairs.schema) != PAIR_SCHEMA:
        raise ValueError("visual pair dtypes do not match the frozen schema")
    if pairs.select(["left_id", "right_id"]).n_unique() != pairs.height:
        raise ValueError("visual product pairs must be unique")
    if pairs.filter(pl.col("left_id") >= pl.col("right_id")).height:
        raise ValueError("visual pairs must satisfy left_id < right_id")
    known_ids = set(int(value) for value in dataset_ids)
    pair_ids = set(pairs.get_column("left_id").to_list()) | set(
        pairs.get_column("right_id").to_list()
    )
    if not pair_ids <= known_ids:
        raise ValueError("visual pairs contain ids outside the dataset")
    allowed_kinds = {
        "exact",
        "exact_candidate",
        "near_candidate",
        "common_image",
    }
    kinds = set(pairs.get_column("match_kind").unique().to_list())
    if not kinds <= allowed_kinds:
        raise ValueError(f"unknown visual match kinds: {sorted(kinds)}")


def build_visual_pairs(
    hashes: pl.DataFrame,
    thresholds: VisualThresholds = VisualThresholds(),
) -> pl.DataFrame:
    """Create conservative product-level exact and near-duplicate evidence."""

    records = hashes.filter(pl.col("status") == "ok").iter_rows(named=True)
    rows = list(records)
    evidence: dict[tuple[int, int], _PairEvidence] = {}
    exact_image_pairs = _add_exact_evidence(rows, evidence, thresholds)
    _add_near_evidence(
        rows,
        evidence,
        exact_image_pairs,
        thresholds,
    )
    pair_rows = [
        evidence[key].as_row()
        for key in sorted(evidence)
    ]
    pairs = (
        pl.DataFrame(pair_rows, schema=PAIR_SCHEMA)
        .select(PAIR_COLUMNS)
        .sort(["left_id", "right_id"])
    )
    validate_visual_pairs(
        pairs,
        hashes.get_column("id").drop_nulls().unique().to_list(),
    )
    return pairs


def image_hash_checksum(hashes: pl.DataFrame) -> str:
    """Hash logical image-hash rows independently of Parquet metadata."""

    digest = hashlib.sha256()
    for row in hashes.select(HASH_COLUMNS).sort("relative_path").iter_rows():
        digest.update(repr(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def visual_pair_checksum(pairs: pl.DataFrame) -> str:
    """Hash logical pair rows independently of Parquet metadata."""

    digest = hashlib.sha256()
    for row in pairs.select(PAIR_COLUMNS).sort(
        ["left_id", "right_id"]
    ).iter_rows():
        digest.update(repr(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find exact and perceptually similar E-CUP product images"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default="data/processed/image_manifest.parquet",
    )
    parser.add_argument("--images", type=Path, default="data/raw/images")
    parser.add_argument(
        "--hashes-output",
        type=Path,
        default="data/processed/image_hashes.parquet",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default="data/processed/visual_duplicates.parquet",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default="data/processed/visual_duplicate_stats.json",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument("--reuse-hashes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_manifest = pl.read_parquet(args.manifest)

    hash_started = time.perf_counter()
    if args.reuse_hashes and args.hashes_output.exists():
        hashes = pl.read_parquet(args.hashes_output)
        validate_image_hashes(image_manifest, hashes)
        previous_stats = (
            json.loads(args.stats_output.read_text(encoding="utf-8"))
            if args.stats_output.exists()
            else {}
        )
        hashing_seconds = float(previous_stats.get("hashing_seconds", 0.0))
    else:
        hashes = build_image_hashes(
            image_manifest,
            args.images,
            workers=args.workers,
        )
        hashing_seconds = time.perf_counter() - hash_started

    matching_started = time.perf_counter()
    pairs = build_visual_pairs(hashes)
    matching_seconds = time.perf_counter() - matching_started

    args.hashes_output.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    hashes.write_parquet(args.hashes_output, compression="zstd")
    pairs.write_parquet(args.pairs_output, compression="zstd")

    stats = {
        "image_count": hashes.height,
        "hash_errors": hashes.filter(pl.col("status") == "error").height,
        "candidate_product_pairs": pairs.height,
        "auto_merge_pairs": pairs.filter(pl.col("auto_merge")).height,
        "match_kind_counts": {
            str(row["match_kind"]): int(row["count"])
            for row in (
                pairs.group_by("match_kind")
                .agg(pl.len().alias("count"))
                .sort("match_kind")
                .iter_rows(named=True)
            )
        },
        "hashing_seconds": round(hashing_seconds, 3),
        "matching_seconds": round(matching_seconds, 3),
        "hash_checksum": image_hash_checksum(hashes),
        "pair_checksum": visual_pair_checksum(pairs),
    }
    args.stats_output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Visual duplicates complete: {hashes.height} images, "
        f"{pairs.height} candidate product pairs, "
        f"{pairs.filter(pl.col('auto_merge')).height} auto-merge pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
