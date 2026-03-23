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
PROJECT_ROOT: Path = Path(__file__).resolve().parent
VOLUME_ROOT: Path = Path("/vol/parameter-golf-data")
RUNS_ROOT: Path = VOLUME_ROOT / "runs"
DATASET_DIR: Path = VOLUME_ROOT / "datasets" / "fineweb10B_sp1024"
TOKENIZER_PATH: Path = VOLUME_ROOT / "tokenizers" / "fineweb_1024_bpe.model"

app: modal.App = modal.App(APP_NAME)
image: modal.Image = modal.Image.debian_slim(
    python_version="3.11"
).pip_install_from_requirements("requirements.txt")
data_volume: modal.Volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=True,
    environment_name=MODAL_ENVIRONMENT,
)


def _run_streaming_command(
    cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path
) -> int:
    """Runs a subprocess while streaming combined stdout/stderr into the provided log path."""

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
            raise RuntimeError("Subprocess stdout pipe was not created")
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return proc.wait()


def _count_files(pattern: str) -> int:
    """Counts matching files for a glob pattern string and returns the resulting integer."""

    return len(list(VOLUME_ROOT.glob(pattern)))


@app.function(
    image=image,
    volumes={VOLUME_ROOT: data_volume},
    timeout=60 * 60 * 6,
    cpu=4,
)
def prepare_data_full_sp1024(train_shards: int = 80) -> dict[str, Any]:
    """Downloads full sp1024 challenge data into the mounted Modal volume and returns file counts."""

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
    env: dict[str, str] = dict(os.environ)
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Dataset download failed with exit code {proc.returncode}")

    manifest_path: Path = VOLUME_ROOT / "manifest.json"
    train_file_count: int = _count_files(
        "datasets/fineweb10B_sp1024/fineweb_train_*.bin"
    )
    val_file_count: int = _count_files("datasets/fineweb10B_sp1024/fineweb_val_*.bin")
    tokenizer_exists: bool = TOKENIZER_PATH.is_file()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest file at {manifest_path}")
    if train_file_count < train_shards:
        raise RuntimeError(
            f"Expected at least {train_shards} train shards, found {train_file_count}"
        )
    if val_file_count <= 0:
        raise RuntimeError("No validation shards found")
    if not tokenizer_exists:
        raise FileNotFoundError(f"Tokenizer missing at {TOKENIZER_PATH}")

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
    gpu=modal.gpu.H100(count=1),
    timeout=60 * 60,
    cpu=16,
    memory=65536,
)
def run_experiment(
    experiment_rel_dir: str = "records/my_exp/baseline",
    script_name: str = "train_gpt.py",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Runs one 1xH100 experiment script for a 10-minute cap and writes parsed result artifacts."""

    resolved_run_id: str = (
        run_id
        or f"modal_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    experiment_dir: Path = (PROJECT_ROOT / experiment_rel_dir).resolve()
    script_path: Path = (experiment_dir / script_name).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Experiment script not found: {script_path}")

    run_dir: Path = (RUNS_ROOT / resolved_run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path: Path = run_dir / "train.log"

    env: dict[str, str] = dict(os.environ)
    env.update(
        {
            "RUN_ID": resolved_run_id,
            "DATA_PATH": str(DATASET_DIR),
            "TOKENIZER_PATH": str(TOKENIZER_PATH),
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": "600",
        }
    )

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

    if exit_code != 0:
        parsed_result_files["exit_code"] = exit_code
        parsed_result_files["run_dir"] = str(run_dir)
        data_volume.commit()
        raise RuntimeError(json.dumps(parsed_result_files, sort_keys=True))

    local_artifacts: dict[str, str] = {}
    for artifact_name in ("final_model.pt", "final_model.int8.ptz"):
        artifact_path: Path = run_dir / artifact_name
        if artifact_path.is_file():
            local_artifacts[artifact_name] = str(artifact_path)

    parsed_result_files["run_id"] = resolved_run_id
    parsed_result_files["run_dir"] = str(run_dir)
    parsed_result_files["train_log_path"] = str(log_path)
    parsed_result_files["local_artifacts"] = local_artifacts
    parsed_result_files["exit_code"] = exit_code
    data_volume.commit()
    return parsed_result_files


@app.local_entrypoint()
def bootstrap_baseline(
    train_shards: int = 80,
    experiment_rel_dir: str = "records/my_exp/baseline",
    script_name: str = "train_gpt.py",
) -> None:
    """Prepares full data in volume, runs the baseline experiment, and prints a JSON summary."""

    data_result: dict[str, Any] = prepare_data_full_sp1024.remote(
        train_shards=train_shards
    )
    run_result: dict[str, Any] = run_experiment.remote(
        experiment_rel_dir=experiment_rel_dir, script_name=script_name
    )
    payload: dict[str, Any] = {"data": data_result, "run": run_result}
    print(json.dumps(payload, indent=2, sort_keys=True))
