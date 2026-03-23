from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm

DATAFILE_MAGIC: int = 20240520
DATAFILE_VERSION: int = 1
HEADER_INTS: int = 256
SHARD_SIZE_TOKENS: int = 100_000_000


@dataclass(frozen=True)
class PatternRule:
    """Stores one weighted regex rule used during document scoring."""

    name: str
    pattern: str
    weight: float


@dataclass(frozen=True)
class MixConfig:
    """Stores curation hyperparameters and positive/negative scoring rule sets."""

    name: str
    source_train_shards: int
    target_train_tokens: int
    sample_decode_tokens: int
    min_score: float
    positive_rules: tuple[PatternRule, ...]
    negative_rules: tuple[PatternRule, ...]


class CuratedShardWriter:
    """Writes selected tokens into challenge-format train shards at fixed token size."""

    def __init__(self, output_dir: Path):
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer: np.ndarray = np.empty((SHARD_SIZE_TOKENS,), dtype=np.uint16)
        self.fill: int = 0
        self.shard_idx: int = 0
        self.tokens_written: int = 0

    def _flush(self) -> None:
        if self.fill == 0:
            return
        write_datafile(
            self.output_dir / f"fineweb_train_{self.shard_idx:06d}.bin",
            self.buffer[: self.fill],
        )
        self.tokens_written += self.fill
        self.fill = 0
        self.shard_idx += 1

    def add_doc_tokens(self, doc_tokens: np.ndarray) -> None:
        pos: int = 0
        while pos < doc_tokens.size:
            take: int = min(SHARD_SIZE_TOKENS - self.fill, doc_tokens.size - pos)
            self.buffer[self.fill : self.fill + take] = doc_tokens[pos : pos + take]
            self.fill += take
            pos += take
            if self.fill == SHARD_SIZE_TOKENS:
                self._flush()

    def finalize(self) -> None:
        self._flush()


def write_datafile(path: Path, tokens: np.ndarray) -> None:
    """Writes one challenge shard with header metadata and uint16 payload."""

    header: np.ndarray = np.zeros((HEADER_INTS,), dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = int(tokens.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(tokens.astype("<u2", copy=False).tobytes())


def load_shard_tokens(path: Path) -> np.ndarray:
    """Loads and validates a challenge shard file and returns its token array."""

    header: np.ndarray = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS:
        raise ValueError(f"invalid header size for shard {path}")
    if int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
        raise ValueError(f"invalid shard magic/version for {path}")
    token_count: int = int(header[2])
    tokens: np.ndarray = np.fromfile(
        path,
        dtype="<u2",
        count=token_count,
        offset=HEADER_INTS * np.dtype("<i4").itemsize,
    )
    if tokens.size != token_count:
        raise ValueError(f"short read for shard {path}")
    return tokens


def iter_doc_bounds(tokens: np.ndarray, bos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns aligned start/end indices for BOS-delimited documents."""

    starts: np.ndarray = np.flatnonzero(tokens == bos_id)
    if starts.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    ends: np.ndarray = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = tokens.size
    return starts.astype(np.int64, copy=False), ends.astype(np.int64, copy=False)


def parse_mix_config(path: Path) -> MixConfig:
    """Parses a mix JSON config into typed scoring rules and target knobs."""

    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    def parse_rules(items: list[dict[str, Any]]) -> tuple[PatternRule, ...]:
        return tuple(
            PatternRule(
                name=str(item["name"]),
                pattern=str(item["pattern"]),
                weight=float(item["weight"]),
            )
            for item in items
        )

    return MixConfig(
        name=str(payload["name"]),
        source_train_shards=int(payload["source_train_shards"]),
        target_train_tokens=int(payload["target_train_tokens"]),
        sample_decode_tokens=int(payload.get("sample_decode_tokens", 512)),
        min_score=float(payload.get("min_score", 1.0)),
        positive_rules=parse_rules(list(payload.get("positive_patterns", []))),
        negative_rules=parse_rules(list(payload.get("negative_patterns", []))),
    )


def score_text(text: str, config: MixConfig) -> tuple[float, list[str], list[str]]:
    """Computes weighted score and matched rule names for one decoded document snippet."""

    score: float = 0.0
    positive_hits: list[str] = []
    negative_hits: list[str] = []
    for rule in config.positive_rules:
        if re.search(rule.pattern, text, flags=re.IGNORECASE):
            score += rule.weight
            positive_hits.append(rule.name)
    for rule in config.negative_rules:
        if re.search(rule.pattern, text, flags=re.IGNORECASE):
            score -= rule.weight
            negative_hits.append(rule.name)
    return score, positive_hits, negative_hits


def score_shard_worker(
    shard_path: str,
    shard_index: int,
    tokenizer_path: str,
    mix_config_path: str,
) -> dict[str, Any]:
    """Scores all documents in one shard and returns candidate metadata for manifest writing."""

    config: MixConfig = parse_mix_config(Path(mix_config_path))
    tokenizer: spm.SentencePieceProcessor = spm.SentencePieceProcessor(
        model_file=tokenizer_path
    )
    bos_id: int = int(tokenizer.bos_id())
    tokens: np.ndarray = load_shard_tokens(Path(shard_path))
    starts, ends = iter_doc_bounds(tokens, bos_id)

    candidates: list[dict[str, Any]] = []
    rule_hit_counts: dict[str, int] = {
        **{rule.name: 0 for rule in config.positive_rules},
        **{rule.name: 0 for rule in config.negative_rules},
    }

    for start, end in zip(starts, ends, strict=True):
        doc_tokens: np.ndarray = tokens[int(start) : int(end)]
        decode_tokens: np.ndarray = doc_tokens[
            1 : 1 + max(config.sample_decode_tokens, 1)
        ]
        decoded_text: str = tokenizer.decode(
            decode_tokens.astype(np.int32).tolist()
        ).lower()
        score, positive_hits, negative_hits = score_text(decoded_text, config)
        if score < config.min_score:
            continue
        for name in positive_hits:
            rule_hit_counts[name] += 1
        for name in negative_hits:
            rule_hit_counts[name] += 1
        candidates.append(
            {
                "shard_path": shard_path,
                "shard_index": shard_index,
                "doc_start": int(start),
                "doc_end": int(end),
                "num_tokens": int(end - start),
                "score": float(score),
            }
        )

    return {
        "shard_path": shard_path,
        "shard_index": shard_index,
        "scanned_tokens": int(tokens.size),
        "docs_seen": int(starts.size),
        "candidates": candidates,
        "rule_hit_counts": rule_hit_counts,
    }


def prescore_manifest(
    input_dataset_dir: Path,
    tokenizer_path: Path,
    mix_config_path: Path,
    manifest_path: Path,
    stats_path: Path,
    workers: int,
) -> dict[str, Any]:
    """Runs CPU-parallel shard scoring and writes one global candidate manifest sorted by score."""

    config: MixConfig = parse_mix_config(mix_config_path)
    source_train_files: list[Path] = sorted(
        input_dataset_dir.glob("fineweb_train_*.bin")
    )[: config.source_train_shards]
    if not source_train_files:
        raise FileNotFoundError(f"no source train shards in {input_dataset_dir}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    max_workers: int = max(1, workers)
    candidates: list[dict[str, Any]] = []
    total_scanned_tokens: int = 0
    total_docs_seen: int = 0
    total_docs_candidates: int = 0
    rule_hit_counts: dict[str, int] = {
        **{rule.name: 0 for rule in config.positive_rules},
        **{rule.name: 0 for rule in config.negative_rules},
    }

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                score_shard_worker,
                str(path),
                idx,
                str(tokenizer_path),
                str(mix_config_path),
            )
            for idx, path in enumerate(source_train_files)
        ]
        for future in as_completed(futures):
            result: dict[str, Any] = future.result()
            total_scanned_tokens += int(result["scanned_tokens"])
            total_docs_seen += int(result["docs_seen"])
            shard_candidates: list[dict[str, Any]] = list(result["candidates"])
            total_docs_candidates += len(shard_candidates)
            candidates.extend(shard_candidates)
            for rule_name, count in dict(result["rule_hit_counts"]).items():
                rule_hit_counts[rule_name] = rule_hit_counts.get(rule_name, 0) + int(
                    count
                )

    candidates.sort(
        key=lambda row: (
            -float(row["score"]),
            int(row["shard_index"]),
            int(row["doc_start"]),
        )
    )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    stats: dict[str, Any] = {
        "mode": "prescore",
        "mix_name": config.name,
        "input_dataset_dir": str(input_dataset_dir),
        "manifest_path": str(manifest_path),
        "source_train_shards": len(source_train_files),
        "workers": max_workers,
        "scanned_tokens": total_scanned_tokens,
        "docs_seen": total_docs_seen,
        "docs_candidates": total_docs_candidates,
        "candidate_ratio": float(total_docs_candidates / max(total_docs_seen, 1)),
        "rule_hit_counts": rule_hit_counts,
        "min_score": config.min_score,
        "target_train_tokens": config.target_train_tokens,
    }
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats


def copy_val_shards(source_dir: Path, output_dir: Path) -> int:
    """Copies validation shard files into the curated output dataset directory."""

    count: int = 0
    for shard_path in sorted(source_dir.glob("fineweb_val_*.bin")):
        (output_dir / shard_path.name).write_bytes(shard_path.read_bytes())
        count += 1
    return count


def build_from_manifest(
    input_dataset_dir: Path,
    manifest_path: Path,
    output_dataset_dir: Path,
    mix_config_path: Path,
    stats_path: Path,
    max_cached_shards: int,
) -> dict[str, Any]:
    """Selects top manifest entries up to target tokens and materializes curated train/val shards."""

    config: MixConfig = parse_mix_config(mix_config_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    selected: list[dict[str, Any]] = []
    selected_tokens: int = 0
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row: dict[str, Any] = json.loads(line)
            selected.append(row)
            selected_tokens += int(row["num_tokens"])
            if selected_tokens >= config.target_train_tokens:
                break

    if not selected:
        raise RuntimeError("manifest produced zero selected documents for build")

    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dataset_dir.glob("fineweb_train_*.bin"):
        stale.unlink()
    for stale in output_dataset_dir.glob("fineweb_val_*.bin"):
        stale.unlink()

    writer: CuratedShardWriter = CuratedShardWriter(output_dataset_dir)
    shard_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get_tokens(path_str: str) -> np.ndarray:
        cached: np.ndarray | None = shard_cache.get(path_str)
        if cached is not None:
            shard_cache.move_to_end(path_str)
            return cached
        tokens = load_shard_tokens(Path(path_str))
        shard_cache[path_str] = tokens
        while len(shard_cache) > max(1, max_cached_shards):
            shard_cache.popitem(last=False)
        return tokens

    for row in selected:
        tokens: np.ndarray = get_tokens(str(row["shard_path"]))
        start: int = int(row["doc_start"])
        end: int = int(row["doc_end"])
        writer.add_doc_tokens(tokens[start:end])

    writer.finalize()
    val_files_copied: int = copy_val_shards(input_dataset_dir, output_dataset_dir)

    stats: dict[str, Any] = {
        "mode": "build",
        "mix_name": config.name,
        "manifest_path": str(manifest_path),
        "output_dataset_dir": str(output_dataset_dir),
        "selected_docs": len(selected),
        "selected_tokens": selected_tokens,
        "target_train_tokens": config.target_train_tokens,
        "train_files_written": writer.shard_idx,
        "val_files_copied": val_files_copied,
        "max_cached_shards": max(1, max_cached_shards),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats


def run_full_pipeline(
    input_dataset_dir: Path,
    tokenizer_path: Path,
    mix_config_path: Path,
    manifest_path: Path,
    output_dataset_dir: Path,
    stats_path: Path,
    workers: int,
    max_cached_shards: int,
) -> dict[str, Any]:
    """Runs prescore and build sequentially and writes a combined stats payload."""

    prescore_stats_path: Path = stats_path.with_name(f"{stats_path.stem}.prescore.json")
    build_stats_path: Path = stats_path.with_name(f"{stats_path.stem}.build.json")
    prescore_stats: dict[str, Any] = prescore_manifest(
        input_dataset_dir=input_dataset_dir,
        tokenizer_path=tokenizer_path,
        mix_config_path=mix_config_path,
        manifest_path=manifest_path,
        stats_path=prescore_stats_path,
        workers=workers,
    )
    build_stats: dict[str, Any] = build_from_manifest(
        input_dataset_dir=input_dataset_dir,
        manifest_path=manifest_path,
        output_dataset_dir=output_dataset_dir,
        mix_config_path=mix_config_path,
        stats_path=build_stats_path,
        max_cached_shards=max_cached_shards,
    )
    combined: dict[str, Any] = {
        "mode": "full",
        "prescore": prescore_stats,
        "build": build_stats,
        "manifest_path": str(manifest_path),
        "output_dataset_dir": str(output_dataset_dir),
        "mix_config_path": str(mix_config_path),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return combined


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for prescore/build/full curated data pipeline modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prescore", "build", "full"), default="full")
    parser.add_argument("--input-dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--mix-config", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dataset-dir", type=Path, required=True)
    parser.add_argument("--stats-out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--max-cached-shards", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Runs selected pipeline mode and emits JSON stats for orchestration and debugging."""

    args = parse_args()
    if args.mode == "prescore":
        result = prescore_manifest(
            input_dataset_dir=args.input_dataset_dir,
            tokenizer_path=args.tokenizer_path,
            mix_config_path=args.mix_config,
            manifest_path=args.manifest_path,
            stats_path=args.stats_out,
            workers=args.workers,
        )
    elif args.mode == "build":
        result = build_from_manifest(
            input_dataset_dir=args.input_dataset_dir,
            manifest_path=args.manifest_path,
            output_dataset_dir=args.output_dataset_dir,
            mix_config_path=args.mix_config,
            stats_path=args.stats_out,
            max_cached_shards=args.max_cached_shards,
        )
    else:
        result = run_full_pipeline(
            input_dataset_dir=args.input_dataset_dir,
            tokenizer_path=args.tokenizer_path,
            mix_config_path=args.mix_config,
            manifest_path=args.manifest_path,
            output_dataset_dir=args.output_dataset_dir,
            stats_path=args.stats_out,
            workers=args.workers,
            max_cached_shards=args.max_cached_shards,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
