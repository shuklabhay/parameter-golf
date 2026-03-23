from __future__ import annotations

import argparse
import json
import re
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
    """Stores one regex weight rule used to score decoded document text."""

    name: str
    pattern: re.Pattern[str]
    weight: float


@dataclass(frozen=True)
class MixConfig:
    """Stores curation knobs and scoring rules loaded from a JSON mix config file."""

    name: str
    source_train_shards: int
    target_train_tokens: int
    sample_decode_tokens: int
    min_score: float
    positive_rules: tuple[PatternRule, ...]
    negative_rules: tuple[PatternRule, ...]


@dataclass(frozen=True)
class DataShard:
    """Stores one shard's decoded token array loaded from challenge binary format."""

    path: Path
    tokens: np.ndarray


class CuratedShardWriter:
    """Writes selected train tokens into challenge-format shard files of fixed token size."""

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
    """Writes one challenge-format shard file with header metadata and uint16 token payload."""

    header: np.ndarray = np.zeros((HEADER_INTS,), dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = int(tokens.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(tokens.astype("<u2", copy=False).tobytes())


def load_shard(path: Path) -> DataShard:
    """Loads and validates one challenge-format shard file and returns its token array."""

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
    return DataShard(path=path, tokens=tokens)


def parse_mix_config(path: Path) -> MixConfig:
    """Parses a JSON config into compiled scoring rules and fixed curation parameters."""

    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    def build_rules(items: list[dict[str, Any]]) -> tuple[PatternRule, ...]:
        rules: list[PatternRule] = []
        for item in items:
            rules.append(
                PatternRule(
                    name=str(item["name"]),
                    pattern=re.compile(str(item["pattern"]), re.IGNORECASE),
                    weight=float(item["weight"]),
                )
            )
        return tuple(rules)

    return MixConfig(
        name=str(payload["name"]),
        source_train_shards=int(payload["source_train_shards"]),
        target_train_tokens=int(payload["target_train_tokens"]),
        sample_decode_tokens=int(payload.get("sample_decode_tokens", 512)),
        min_score=float(payload.get("min_score", 1.0)),
        positive_rules=build_rules(list(payload.get("positive_patterns", []))),
        negative_rules=build_rules(list(payload.get("negative_patterns", []))),
    )


def score_text(text: str, config: MixConfig) -> float:
    """Scores one decoded document string using weighted positive and negative regex hits."""

    score: float = 0.0
    for rule in config.positive_rules:
        if rule.pattern.search(text):
            score += rule.weight
    for rule in config.negative_rules:
        if rule.pattern.search(text):
            score -= rule.weight
    return score


def iter_doc_bounds(tokens: np.ndarray, bos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Builds aligned start/end indices for BOS-delimited documents inside one token array."""

    starts: np.ndarray = np.flatnonzero(tokens == bos_id)
    if starts.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    ends: np.ndarray = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = tokens.size
    return starts.astype(np.int64, copy=False), ends.astype(np.int64, copy=False)


def copy_val_shards(source_dir: Path, output_dir: Path) -> int:
    """Copies validation shards byte-for-byte into the curated dataset directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    count: int = 0
    for shard_path in sorted(source_dir.glob("fineweb_val_*.bin")):
        data: bytes = shard_path.read_bytes()
        (output_dir / shard_path.name).write_bytes(data)
        count += 1
    return count


def build_curated_dataset(
    input_dataset_dir: Path,
    tokenizer_path: Path,
    output_dataset_dir: Path,
    config: MixConfig,
) -> dict[str, Any]:
    """Builds a curated train prefix and copied val split from source shards under one output dataset path."""

    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"missing tokenizer at {tokenizer_path}")
    if not input_dataset_dir.is_dir():
        raise FileNotFoundError(f"missing source dataset dir {input_dataset_dir}")

    tokenizer: spm.SentencePieceProcessor = spm.SentencePieceProcessor(
        model_file=str(tokenizer_path)
    )
    bos_id: int = int(tokenizer.bos_id())
    if bos_id < 0:
        raise ValueError("tokenizer must define a valid BOS id")

    source_train_files: list[Path] = sorted(
        input_dataset_dir.glob("fineweb_train_*.bin")
    )[: config.source_train_shards]
    if not source_train_files:
        raise FileNotFoundError("no source train shards were found")

    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dataset_dir.glob("fineweb_train_*.bin"):
        stale.unlink()
    for stale in output_dataset_dir.glob("fineweb_val_*.bin"):
        stale.unlink()

    writer: CuratedShardWriter = CuratedShardWriter(output_dataset_dir)
    docs_seen: int = 0
    docs_kept: int = 0
    scanned_train_tokens: int = 0
    kept_train_tokens: int = 0
    hit_counts: dict[str, int] = {
        **{rule.name: 0 for rule in config.positive_rules},
        **{rule.name: 0 for rule in config.negative_rules},
    }

    for shard_path in source_train_files:
        shard: DataShard = load_shard(shard_path)
        starts, ends = iter_doc_bounds(shard.tokens, bos_id)
        scanned_train_tokens += int(shard.tokens.size)

        for start, end in zip(starts, ends, strict=True):
            docs_seen += 1
            doc_tokens: np.ndarray = shard.tokens[int(start) : int(end)]
            decode_tokens: np.ndarray = doc_tokens[
                1 : 1 + max(config.sample_decode_tokens, 1)
            ]
            decoded_text: str = tokenizer.decode(
                decode_tokens.astype(np.int32).tolist()
            )
            lowered_text: str = decoded_text.lower()
            score: float = score_text(lowered_text, config)
            if score < config.min_score:
                continue

            for rule in config.positive_rules:
                if rule.pattern.search(lowered_text):
                    hit_counts[rule.name] += 1
            for rule in config.negative_rules:
                if rule.pattern.search(lowered_text):
                    hit_counts[rule.name] += 1

            writer.add_doc_tokens(doc_tokens)
            docs_kept += 1
            kept_train_tokens += int(doc_tokens.size)
            if kept_train_tokens >= config.target_train_tokens:
                break

        if kept_train_tokens >= config.target_train_tokens:
            break

    writer.finalize()
    val_files_copied: int = copy_val_shards(input_dataset_dir, output_dataset_dir)
    if docs_kept == 0:
        raise RuntimeError("curation selected zero documents")

    stats: dict[str, Any] = {
        "mix_name": config.name,
        "input_dataset_dir": str(input_dataset_dir),
        "output_dataset_dir": str(output_dataset_dir),
        "source_train_shards_scanned": len(source_train_files),
        "target_train_tokens": config.target_train_tokens,
        "scanned_train_tokens": scanned_train_tokens,
        "kept_train_tokens": kept_train_tokens,
        "docs_seen": docs_seen,
        "docs_kept": docs_kept,
        "kept_doc_ratio": float(docs_kept / max(docs_seen, 1)),
        "train_files_written": writer.shard_idx,
        "val_files_copied": val_files_copied,
        "sample_decode_tokens": config.sample_decode_tokens,
        "min_score": config.min_score,
        "rule_hit_counts": hit_counts,
    }
    return stats


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments required to build one curated training prefix dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dataset-dir", type=Path, required=True)
    parser.add_argument("--mix-config", type=Path, required=True)
    parser.add_argument("--stats-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Builds a curated prefix dataset and writes one JSON stats report for downstream run orchestration."""

    args = parse_args()
    mix_config: MixConfig = parse_mix_config(args.mix_config)
    stats: dict[str, Any] = build_curated_dataset(
        input_dataset_dir=args.input_dataset_dir,
        tokenizer_path=args.tokenizer_path,
        output_dataset_dir=args.output_dataset_dir,
        config=mix_config,
    )
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
