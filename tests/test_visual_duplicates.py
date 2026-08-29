from __future__ import annotations

import json

import polars as pl
from PIL import Image, ImageDraw

from ecup.data.image_manifest import build_image_manifest
from ecup.data.visual_duplicates import (
    HASH_COLUMNS,
    PAIR_COLUMNS,
    VisualThresholds,
    build_image_hashes,
    build_visual_pairs,
    hamming_distance,
    image_hash_checksum,
    main,
)


def write_pattern(
    path,
    *,
    variant: int,
    image_format: str | None = None,
    quality: int = 95,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (96, 72), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    x = 8 + variant * 7
    draw.rectangle((x, 8, x + 28, 42), fill=(20, 70 + variant * 20, 180))
    draw.ellipse((50, 20 + variant, 82, 52 + variant), fill=(210, 40, 30))
    draw.line((0, 70 - variant, 95, 4 + variant), fill=(0, 0, 0), width=3)
    image.save(path, format=image_format, quality=quality)


def image_manifest(images, ids: set[int]) -> pl.DataFrame:
    return build_image_manifest(images, ids, workers=1).manifest


def pair_rows(pairs: pl.DataFrame) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (row["left_id"], row["right_id"]): row
        for row in pairs.iter_rows(named=True)
    }


def test_image_hashes_are_deterministic_across_workers(tmp_path) -> None:
    images = tmp_path / "images"
    write_pattern(images / "1" / "0.jpg", variant=0)
    write_pattern(images / "2" / "0.png", variant=2)
    manifest = image_manifest(images, {1, 2})

    sequential = build_image_hashes(manifest, images, workers=1)
    parallel = build_image_hashes(manifest, images, workers=2)

    assert sequential.columns == list(HASH_COLUMNS)
    assert sequential.equals(parallel)
    assert image_hash_checksum(sequential) == image_hash_checksum(parallel)
    assert sequential.get_column("status").to_list() == ["ok", "ok"]


def test_primary_exact_match_is_merged_but_secondary_match_is_candidate(
    tmp_path,
) -> None:
    images = tmp_path / "images"
    primary = images / "1" / "0.jpg"
    write_pattern(primary, variant=0)
    (images / "2").mkdir(parents=True)
    (images / "2" / "0.jpg").write_bytes(primary.read_bytes())

    secondary = images / "3" / "1.jpg"
    write_pattern(secondary, variant=3)
    (images / "4").mkdir(parents=True)
    (images / "4" / "1.jpg").write_bytes(secondary.read_bytes())

    manifest = image_manifest(images, {1, 2, 3, 4})
    hashes = build_image_hashes(manifest, images, workers=2)
    pairs = build_visual_pairs(hashes)
    rows = pair_rows(pairs)

    assert pairs.columns == list(PAIR_COLUMNS)
    assert rows[(1, 2)]["match_kind"] == "exact"
    assert rows[(1, 2)]["primary_match"] is True
    assert rows[(1, 2)]["auto_merge"] is True
    assert rows[(3, 4)]["match_kind"] == "exact_candidate"
    assert rows[(3, 4)]["primary_match"] is False
    assert rows[(3, 4)]["auto_merge"] is False


def test_reencoded_primary_image_remains_a_review_candidate(tmp_path) -> None:
    images = tmp_path / "images"
    write_pattern(
        images / "5" / "0.png",
        variant=1,
        image_format="PNG",
    )
    write_pattern(
        images / "6" / "0.jpg",
        variant=1,
        image_format="JPEG",
        quality=70,
    )
    manifest = image_manifest(images, {5, 6})
    hashes = build_image_hashes(manifest, images, workers=1)

    assert hashes.get_column("file_sha256").n_unique() == 2
    pairs = build_visual_pairs(hashes)
    row = pair_rows(pairs)[(5, 6)]

    assert row["match_kind"] == "near_candidate"
    assert row["primary_match"] is True
    assert row["auto_merge"] is False


def test_large_shared_exact_group_is_not_automatically_merged(tmp_path) -> None:
    images = tmp_path / "images"
    source = images / "1" / "0.jpg"
    write_pattern(source, variant=0)
    for item_id in (2, 3):
        target = images / str(item_id) / "0.jpg"
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())

    manifest = image_manifest(images, {1, 2, 3})
    hashes = build_image_hashes(manifest, images, workers=1)
    thresholds = VisualThresholds(max_common_exact_products=2)
    pairs = build_visual_pairs(hashes, thresholds)
    rows = pair_rows(pairs)

    assert set(rows) == {(1, 2), (1, 3)}
    assert all(row["match_kind"] == "common_image" for row in rows.values())
    assert all(row["auto_merge"] is False for row in rows.values())


def test_hamming_distance_accepts_hexadecimal_hashes() -> None:
    assert hamming_distance("0f", "00") == 4
    assert hamming_distance(0b1010, 0b0011) == 2


def test_cli_writes_hashes_pairs_and_stats(tmp_path) -> None:
    images = tmp_path / "images"
    first = images / "1" / "0.jpg"
    write_pattern(first, variant=0)
    (images / "2").mkdir(parents=True)
    (images / "2" / "0.jpg").write_bytes(first.read_bytes())

    manifest_path = tmp_path / "image_manifest.parquet"
    hashes_path = tmp_path / "image_hashes.parquet"
    pairs_path = tmp_path / "visual_duplicates.parquet"
    stats_path = tmp_path / "visual_duplicate_stats.json"
    image_manifest(images, {1, 2}).write_parquet(manifest_path)

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--images",
            str(images),
            "--hashes-output",
            str(hashes_path),
            "--pairs-output",
            str(pairs_path),
            "--stats-output",
            str(stats_path),
            "--workers",
            "1",
        ]
    )

    assert exit_code == 0
    assert pl.read_parquet(hashes_path).columns == list(HASH_COLUMNS)
    pairs = pl.read_parquet(pairs_path)
    assert pairs.columns == list(PAIR_COLUMNS)
    assert pairs.row(0, named=True)["auto_merge"] is True
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["image_count"] == 2
    assert stats["auto_merge_pairs"] == 1

    stats["hashing_seconds"] = 12.5
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    reuse_exit_code = main(
        [
            "--manifest", str(manifest_path),
            "--images", str(images),
            "--hashes-output", str(hashes_path),
            "--pairs-output", str(pairs_path),
            "--stats-output", str(stats_path),
            "--workers", "1",
            "--reuse-hashes",
        ]
    )

    assert reuse_exit_code == 0
    reused_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert reused_stats["hashing_seconds"] == 12.5
