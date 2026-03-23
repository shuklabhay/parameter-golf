This folder is an isolated experiment snapshot for Modal runs.

- `train_gpt.py` is copied from the root baseline and is the execution target.
- `run_config.json` captures the default 1xH100, 10-minute setup.
- `data_probe_plan.json` captures the cheap-GPU probe matrix and promotion rule for H100 confirmation.
- Runtime outputs are written to Modal volume run directories and parsed into `results.txt`, `metrics.json`, and `status.json`.
- `train.log` captures streamed console output and `train_full.log` mirrors the script-owned `logs/<run_id>.txt`.

Modal entrypoints:

- Baseline H100 run: `modal run modal_runner.py::bootstrap_baseline`
- Probe run (A10G, short wallclock): `modal run modal_runner.py::run_data_probe`
- Curated probe (build + run): `modal run modal_runner.py::run_data_probe --build-curated --use-curated`
