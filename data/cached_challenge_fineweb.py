import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
ROOT = Path(__file__).resolve().parent


def root_dirs(output_root: Path | None) -> tuple[Path, Path, Path]:
    """Returns resolved root, datasets, and tokenizers directories for download output."""

    resolved_root = output_root.resolve() if output_root is not None else ROOT
    return resolved_root, resolved_root / "datasets", resolved_root / "tokenizers"


def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(
        f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>"
    )


def local_path_for_remote(relative_path: str, output_root: Path | None) -> Path:
    resolved_root, datasets_dir, tokenizers_dir = root_dirs(output_root)
    remote_path = Path(relative_path)
    if REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (REMOTE_ROOT_PREFIX,):
        remote_path = remote_path.relative_to(REMOTE_ROOT_PREFIX)
    if remote_path.parts[:1] == ("datasets",):
        return datasets_dir.joinpath(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return tokenizers_dir.joinpath(*remote_path.parts[1:])
    return resolved_root / remote_path


def get(relative_path: str, output_root: Path | None) -> None:
    destination = local_path_for_remote(relative_path, output_root)
    if destination.exists():
        return
    if destination.is_symlink():
        destination.unlink()

    remote_path = Path(relative_path)
    cached_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path.name,
            subfolder=(
                remote_path.parent.as_posix()
                if remote_path.parent != Path(".")
                else None
            ),
            repo_type="dataset",
        )
    )
    # HF cache entries may be snapshot symlinks. Resolve to the underlying blob so we
    # always materialize a real file in data/, not a broken relative symlink.
    cached_source = cached_path.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached_source, destination)
    except OSError:
        shutil.copy2(cached_source, destination)


def manifest_path(output_root: Path | None) -> Path:
    return local_path_for_remote(f"{REMOTE_ROOT_PREFIX}/manifest.json", output_root)


def load_manifest(*, skip_manifest_download: bool, output_root: Path | None) -> dict:
    path = manifest_path(output_root)
    if not path.is_file():
        if skip_manifest_download:
            raise FileNotFoundError(
                f"manifest.json is required for manifest-driven shard counts but is not present locally at {path}"
            )
        get(f"{REMOTE_ROOT_PREFIX}/manifest.json", output_root)
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_paths_for_tokenizer(tokenizer_entry: dict) -> list[str]:
    artifacts = []
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            artifacts.append(str(value))
    if not artifacts:
        raise ValueError(
            f"tokenizer entry is missing downloadable artifacts: {tokenizer_entry}"
        )
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download challenge FineWeb shards from Hugging Face"
    )
    parser.add_argument(
        "train_shards_positional",
        nargs="?",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=80,
        help="Number of training shards to download for the selected variant. Defaults to 80.",
    )
    parser.add_argument(
        "--variant",
        default="sp1024",
        help="Tokenizer family to download, for example sp1024, sp4096, or byte260.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip downloading manifest.json.",
    )
    parser.add_argument(
        "--with-docs",
        action="store_true",
        help="Also download docs_selected.jsonl and its sidecar for tokenizer retraining or dataset re-export.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional root directory where datasets/tokenizers/manifest should be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = (
        Path(args.output_root).expanduser() if args.output_root is not None else None
    )
    dataset_dir = dataset_dir_for_variant(args.variant)
    train_shards = (
        args.train_shards_positional
        if args.train_shards_positional is not None
        else args.train_shards
    )
    if train_shards < 0:
        raise ValueError("train_shards must be non-negative")

    manifest = load_manifest(
        skip_manifest_download=args.skip_manifest, output_root=output_root
    )
    dataset_entry = next(
        (x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir), None
    )
    if dataset_entry is None:
        raise ValueError(
            f"dataset {dataset_dir} not found in {REMOTE_ROOT_PREFIX}/manifest.json"
        )
    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train"))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val"))
    if train_shards > max_train_shards:
        raise ValueError(
            f"{args.variant} only has {max_train_shards} training shards on {REPO_ID}, requested {train_shards}"
        )
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next(
        (x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name),
        None,
    )
    if tokenizer_entry is None:
        raise ValueError(
            f"tokenizer {tokenizer_name} not found in {REMOTE_ROOT_PREFIX}/manifest.json"
        )

    if args.with_docs:
        get(f"{REMOTE_ROOT_PREFIX}/docs_selected.jsonl", output_root)
        get(f"{REMOTE_ROOT_PREFIX}/docs_selected.source_manifest.json", output_root)

    dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{dataset_dir}"
    for i in range(val_shards):
        get(f"{dataset_prefix}/fineweb_val_{i:06d}.bin", output_root)
    for i in range(train_shards):
        get(f"{dataset_prefix}/fineweb_train_{i:06d}.bin", output_root)

    for artifact_path in artifact_paths_for_tokenizer(tokenizer_entry):
        get(f"{REMOTE_ROOT_PREFIX}/{artifact_path}", output_root)


if __name__ == "__main__":
    main()
