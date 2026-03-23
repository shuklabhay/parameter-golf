from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from run_result_parser import parse_training_log, write_result_files

APP_NAME: str = "parameter-golf-my-exp"
MODAL_ENVIRONMENT: str = "parametergolf"
VOLUME_NAME: str = "parametergolf_fineweb_sp1024_full"
PROJECT_ROOT: Path = Path("/root/project")
VOLUME_ROOT: Path = Path("/vol/parameter-golf-data")
RUNS_ROOT: Path = VOLUME_ROOT / "runs"
DATASET_DIR: Path = VOLUME_ROOT / "datasets" / "fineweb10B_sp1024"
CURATED_DATASETS_ROOT: Path = VOLUME_ROOT / "datasets_curated"
TOKENIZER_PATH: Path = VOLUME_ROOT / "tokenizers" / "fineweb_1024_bpe.model"
DEFAULT_MIX_CONFIG: Path = PROJECT_ROOT / "data" / "val_like_mix_v1.json"

app: modal.App = modal.App(APP_NAME)
image: modal.Image = modal.Image.debian_slim(
    python_version="3.11"
).pip_install_from_requirements("requirements.txt")
image = image.add_local_dir(".", remote_path="/root/project")
data_volume: modal.Volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=True,
    environment_name=MODAL_ENVIRONMENT,
)


def _run_streaming_command(
    cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path
) -> int:
    """Runs one subprocess while streaming combined stdout/stderr into a file."""

    with log_path.open("w", encoding="utf-8") as log_file:
        proc: subprocess.Popen[str] = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is None:
            raise RuntimeError("subprocess stdout pipe was not created")
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return proc.wait()


def _count_files(pattern: str) -> int:
    """Counts files in the mounted volume that match one glob pattern string."""

    return len(list(VOLUME_ROOT.glob(pattern)))


def _validate_dataset_dir(dataset_dir: Path) -> dict[str, Any]:
    """Validates one dataset directory and returns shard counts plus tokenizer path metadata."""

    train_file_count: int = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_file_count: int = len(list(dataset_dir.glob("fineweb_val_*.bin")))
    if train_file_count <= 0:
        raise FileNotFoundError(f"no train shards found in {dataset_dir}")
    if val_file_count <= 0:
        raise FileNotFoundError(f"no val shards found in {dataset_dir}")
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(f"tokenizer missing at {TOKENIZER_PATH}")
    return {
        "dataset_dir": str(dataset_dir),
        "train_file_count": train_file_count,
        "val_file_count": val_file_count,
        "tokenizer_path": str(TOKENIZER_PATH),
    }


def _resolve_mix_name(mix_config_path: Path) -> str:
    """Reads a mix JSON config and returns its configured dataset name string."""

    payload: dict[str, Any] = json.loads(mix_config_path.read_text(encoding="utf-8"))
    mix_name: str = str(payload.get("name", "fineweb10B_sp1024_val_like_web_mix_v1"))
    if not mix_name:
        raise ValueError(f"mix name missing in {mix_config_path}")
    return mix_name


def _run_experiment_impl(
    *,
    experiment_rel_dir: str,
    script_name: str,
    run_id: str | None,
    dataset_dir: Path,
    max_wallclock_seconds: int,
    train_batch_tokens: int | None,
    run_profile: str,
) -> dict[str, Any]:
    """Runs one training script with explicit dataset and timing overrides and returns parsed artifacts."""

    resolved_run_id: str = (
        run_id
        or f"modal_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    experiment_dir: Path = (PROJECT_ROOT / experiment_rel_dir).resolve()
    script_path: Path = (experiment_dir / script_name).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"experiment script not found: {script_path}")

    if max_wallclock_seconds <= 0:
        raise ValueError("max_wallclock_seconds must be positive")
    if train_batch_tokens is not None and train_batch_tokens <= 0:
        raise ValueError("train_batch_tokens must be positive when provided")

    run_dir: Path = (RUNS_ROOT / resolved_run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path: Path = run_dir / "train.log"

    env: dict[str, str] = dict(os.environ)
    env.update(
        {
            "RUN_ID": resolved_run_id,
            "DATA_PATH": str(dataset_dir),
            "TOKENIZER_PATH": str(TOKENIZER_PATH),
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
        }
    )
    if train_batch_tokens is not None:
        env["TRAIN_BATCH_TOKENS"] = str(train_batch_tokens)

    command: list[str] = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        str(script_path),
    ]
    exit_code: int = _run_streaming_command(
        command, cwd=run_dir, env=env, log_path=log_path
    )

    log_text: str = log_path.read_text(encoding="utf-8")
    parsed = parse_training_log(log_text)
    parsed_result_files: dict[str, Any] = write_result_files(
        run_dir, resolved_run_id, parsed
    )

    train_full_log_path: Path = run_dir / "logs" / f"{resolved_run_id}.txt"
    copied_train_full_log_path: Path = run_dir / "train_full.log"
    if train_full_log_path.is_file():
        copied_train_full_log_path.write_bytes(train_full_log_path.read_bytes())

    local_artifacts: dict[str, str] = {}
    for artifact_name in ("final_model.pt", "final_model.int8.ptz"):
        artifact_path: Path = run_dir / artifact_name
        if artifact_path.is_file():
            local_artifacts[artifact_name] = str(artifact_path)

    parsed_result_files["run_id"] = resolved_run_id
    parsed_result_files["run_dir"] = str(run_dir)
    parsed_result_files["train_log_path"] = str(log_path)
    parsed_result_files["train_full_log_path"] = (
        str(copied_train_full_log_path)
        if copied_train_full_log_path.is_file()
        else None
    )
    parsed_result_files["local_artifacts"] = local_artifacts
    parsed_result_files["exit_code"] = exit_code
    parsed_result_files["run_profile"] = run_profile
    parsed_result_files["dataset_dir"] = str(dataset_dir)
    parsed_result_files["max_wallclock_seconds"] = max_wallclock_seconds
    parsed_result_files["train_batch_tokens"] = train_batch_tokens

    data_volume.commit()
    if exit_code != 0:
        raise RuntimeError(json.dumps(parsed_result_files, sort_keys=True))
    return parsed_result_files


@app.function(
    image=image,
    volumes={VOLUME_ROOT: data_volume},
    timeout=60 * 60 * 6,
    cpu=4,
    include_source=True,
)
def prepare_data_full_sp1024(train_shards: int = 80) -> dict[str, Any]:
    """Downloads sp1024 challenge data into the mounted Modal volume and returns dataset checks."""

    if train_shards <= 0:
        raise ValueError("train_shards must be positive")

    cmd: list[str] = [
        "python",
        str(PROJECT_ROOT / "data" / "cached_challenge_fineweb.py"),
        "--variant",
        "sp1024",
        "--train-shards",
        str(train_shards),
        "--output-root",
        str(VOLUME_ROOT),
    ]
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"dataset download failed with exit code {proc.returncode}")

    manifest_path: Path = VOLUME_ROOT / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest file at {manifest_path}")

    train_file_count: int = _count_files(
        "datasets/fineweb10B_sp1024/fineweb_train_*.bin"
    )
    val_file_count: int = _count_files("datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    if train_file_count < train_shards:
        raise RuntimeError(
            f"expected at least {train_shards} train shards, found {train_file_count}"
        )
    if val_file_count <= 0:
        raise RuntimeError("no validation shards found")
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(f"tokenizer missing at {TOKENIZER_PATH}")

    data_volume.commit()
    return {
        "manifest_path": str(manifest_path),
        "train_file_count": train_file_count,
        "val_file_count": val_file_count,
        "tokenizer_path": str(TOKENIZER_PATH),
        "dataset_dir": str(DATASET_DIR),
    }


@app.function(
    image=image,
    volumes={VOLUME_ROOT: data_volume},
    timeout=60 * 60 * 6,
    cpu=8,
    memory=65536,
    include_source=True,
)
def prepare_curated_data_sp1024(
    mix_config_rel_path: str = "data/val_like_mix_v1.json",
    output_dataset_name: str | None = None,
) -> dict[str, Any]:
    """Builds one curated sp1024 train prefix dataset in-volume and returns curation stats."""

    mix_config_path: Path = (PROJECT_ROOT / mix_config_rel_path).resolve()
    if not mix_config_path.is_file():
        raise FileNotFoundError(f"mix config not found: {mix_config_path}")

    resolved_dataset_name: str = output_dataset_name or _resolve_mix_name(
        mix_config_path
    )
    if not resolved_dataset_name:
        raise ValueError("output dataset name must be non-empty")

    output_dataset_dir: Path = CURATED_DATASETS_ROOT / resolved_dataset_name
    stats_dir: Path = VOLUME_ROOT / "curation_stats"
    stats_path: Path = stats_dir / f"{resolved_dataset_name}.json"

    cmd: list[str] = [
        "python",
        str(PROJECT_ROOT / "data" / "build_curated_prefix.py"),
        "--input-dataset-dir",
        str(DATASET_DIR),
        "--tokenizer-path",
        str(TOKENIZER_PATH),
        "--output-dataset-dir",
        str(output_dataset_dir),
        "--mix-config",
        str(mix_config_path),
        "--stats-out",
        str(stats_path),
    ]
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"curated data build failed with exit code {proc.returncode}"
        )

    if not stats_path.is_file():
        raise FileNotFoundError(f"missing curation stats file at {stats_path}")
    stats_payload: dict[str, Any] = json.loads(stats_path.read_text(encoding="utf-8"))

    dataset_checks: dict[str, Any] = _validate_dataset_dir(output_dataset_dir)
    data_volume.commit()
    return {
        "mix_config_path": str(mix_config_path),
        "output_dataset_name": resolved_dataset_name,
        "output_dataset_dir": str(output_dataset_dir),
        "stats_path": str(stats_path),
        "stats": stats_payload,
        "dataset_checks": dataset_checks,
    }


@app.function(
    image=image,
    volumes={VOLUME_ROOT: data_volume},
    gpu="H100",
    timeout=60 * 60,
    cpu=16,
    memory=65536,
    include_source=True,
)
def run_experiment(
    experiment_rel_dir: str = "records/my_exp/baseline",
    script_name: str = "train_gpt.py",
    run_id: str | None = None,
    dataset_rel_path: str = "datasets/fineweb10B_sp1024",
    max_wallclock_seconds: int = 600,
    train_batch_tokens: int | None = None,
) -> dict[str, Any]:
    """Runs one H100 experiment with configurable dataset/time overrides and parsed result outputs."""

    dataset_dir: Path = (VOLUME_ROOT / dataset_rel_path).resolve()
    _validate_dataset_dir(dataset_dir)
    return _run_experiment_impl(
        experiment_rel_dir=experiment_rel_dir,
        script_name=script_name,
        run_id=run_id,
        dataset_dir=dataset_dir,
        max_wallclock_seconds=max_wallclock_seconds,
        train_batch_tokens=train_batch_tokens,
        run_profile="h100",
    )


@app.function(
    image=image,
    volumes={VOLUME_ROOT: data_volume},
    gpu="A10G",
    timeout=60 * 60,
    cpu=16,
    memory=65536,
    include_source=True,
)
def run_experiment_probe(
    experiment_rel_dir: str = "records/my_exp/baseline",
    script_name: str = "train_gpt.py",
    run_id: str | None = None,
    dataset_rel_path: str = "datasets/fineweb10B_sp1024",
    max_wallclock_seconds: int = 180,
    train_batch_tokens: int | None = 131072,
) -> dict[str, Any]:
    """Runs one cheaper GPU probe experiment with short-wallclock defaults for fast dataset iteration."""

    dataset_dir: Path = (VOLUME_ROOT / dataset_rel_path).resolve()
    _validate_dataset_dir(dataset_dir)
    return _run_experiment_impl(
        experiment_rel_dir=experiment_rel_dir,
        script_name=script_name,
        run_id=run_id,
        dataset_dir=dataset_dir,
        max_wallclock_seconds=max_wallclock_seconds,
        train_batch_tokens=train_batch_tokens,
        run_profile="probe_a10g",
    )


@app.local_entrypoint()
def bootstrap_baseline(
    train_shards: int = 80,
    experiment_rel_dir: str = "records/my_exp/baseline",
    script_name: str = "train_gpt.py",
) -> None:
    """Prepares full data and runs one baseline H100 training job, then prints JSON outputs."""

    data_result: dict[str, Any] = prepare_data_full_sp1024.remote(
        train_shards=train_shards
    )
    run_result: dict[str, Any] = run_experiment.remote(
        experiment_rel_dir=experiment_rel_dir,
        script_name=script_name,
    )
    print(
        json.dumps({"data": data_result, "run": run_result}, indent=2, sort_keys=True)
    )


@app.local_entrypoint()
def run_data_probe(
    train_shards: int = 80,
    use_curated: bool = False,
    build_curated: bool = False,
    mix_config_rel_path: str = "data/val_like_mix_v1.json",
    curated_dataset_name: str | None = None,
    experiment_rel_dir: str = "records/my_exp/baseline",
    script_name: str = "train_gpt.py",
    max_wallclock_seconds: int = 180,
    train_batch_tokens: int = 131072,
    use_h100: bool = False,
) -> None:
    """Runs one short probe on curated or baseline data and prints both data and run metadata."""

    data_result: dict[str, Any] = prepare_data_full_sp1024.remote(
        train_shards=train_shards
    )
    curated_result: dict[str, Any] | None = None
    dataset_rel_path: str = "datasets/fineweb10B_sp1024"

    if build_curated or use_curated:
        curated_result = prepare_curated_data_sp1024.remote(
            mix_config_rel_path=mix_config_rel_path,
            output_dataset_name=curated_dataset_name,
        )
        dataset_rel_path = str(
            Path("datasets_curated") / curated_result["output_dataset_name"]
        )

    run_fn = run_experiment if use_h100 else run_experiment_probe
    run_result: dict[str, Any] = run_fn.remote(
        experiment_rel_dir=experiment_rel_dir,
        script_name=script_name,
        dataset_rel_path=dataset_rel_path,
        max_wallclock_seconds=max_wallclock_seconds,
        train_batch_tokens=train_batch_tokens,
    )
    print(
        json.dumps(
            {
                "data": data_result,
                "curated": curated_result,
                "run": run_result,
                "dataset_rel_path": dataset_rel_path,
            },
            indent=2,
            sort_keys=True,
        )
    )
