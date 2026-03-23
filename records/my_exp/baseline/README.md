This folder is an isolated experiment snapshot for Modal runs.

- `train_gpt.py` is copied from the root baseline and is the execution target.
- `run_config.json` captures the default 1xH100, 10-minute setup.
- Runtime outputs are written to Modal volume run directories and parsed into `results.txt`, `metrics.json`, and `status.json`.
