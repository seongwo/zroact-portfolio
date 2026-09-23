# Setup and run

Run commands from the repository root. Python 3.10+ is required by the source's type annotations. This repository provides a working CPU-only evaluation example; training and full video inference require resources that are not included.

## 1. CPU-only evaluation example

No pip installation, videos, GPU, model weights, or API credentials are required.

```bash
python3 benchmark2/scripts/evaluate_stage2_results.py \
  --gt-manifest docs/examples/mock_gt.jsonl \
  --results-csv docs/examples/mock_predictions.csv
```

Expected summary:

```text
Evaluated rows: 3
Missing GT rows: 0
Ignored GT rows: 1
Ignored prediction rows: 1
Accuracy: 2/3 (0.6667)
```

The fixture has one correct normal prediction, one unsafe→danger error, one correct danger prediction, and one excluded request. It is invented format-demonstration data, not CCTV data or a model result. The evaluator reports accuracy and confusion counts; it does not compute macro F1 or guarantee that every eligible GT request has a prediction. Real comparisons must also check unique request IDs and complete matched coverage.

## 2. Inspect training/evaluation commands

The following help commands require only the standard library and do not load a model:

```bash
python3 benchmark2/training/scripts/build_dataset.py --help
python3 benchmark2/training/scripts/train_lora.py --help
python3 benchmark2/training/scripts/evaluate_lora.py --help
python3 serving/run_job.py --help
```

`validate_dataset.py` imports Pillow before parsing arguments, so even its `--help` requires Pillow. That command was not runnable in the standard-library-only audit environment.

## 3. Reproduce the recorded LoRA experiment

This mode is **blocked in the public checkout** until the following original assets are supplied:

| Resource | Config / expected location |
|---|---|
| Frames and Stage 1 detections | `paths.data_root` in the selected training config |
| Frame-level risk labels | `paths.labeling_root` (currently an external sibling directory) |
| Original fixed video split | `benchmark2/training/splits/qwen35_08b_action_v0.json` |
| Matching Qwen model files | `paths.model_path` |
| Trained adapter for evaluation | Original checkpoint directory, passed with `--adapter-path` |

Preserve the original split assignments and record asset versions/hashes; do not regenerate a split and call it the same experiment. Store local config copies under ignored `local/`, while retaining the original config as the experiment record.

The project used separate Stage 1, Stage 2 inference, and Unsloth training environments. [Training requirements](../benchmark2/training/requirements.txt) pin part of the original training stack; they are not a fully locked, freshly validated environment. Root `requirements.txt` is a historical inference-environment snapshot and is incomplete for all public entry points. Use the [training guide](../benchmark2/training/README.md) as a record of the original procedure, with paths adapted to your installation.

After the resources and a compatible GPU environment are available, the source accepts:

```bash
python3 benchmark2/training/scripts/build_dataset.py \
  --config local/training.json
python3 benchmark2/training/scripts/validate_dataset.py \
  --config local/training.json
python3 benchmark2/training/scripts/train_lora.py \
  --config local/training.json --preflight \
  --limit-train 1 --limit-validation 1
```

Unlike `--help`, `--preflight` loads the model and LoRA modules. Start training with `--run` only after validating the dataset and loss mask. Select checkpoints on validation before evaluating the held-out test split. See [result provenance](RESULT_PROVENANCE.md) for the limits of the currently recorded comparison.

## 4. Full video inference and serving

The repository has a sequential pipeline, streaming prototypes, and an API wrapper. Full inference needs the external YOWOv3 tree (including the project's custom config/inference entry point), compatible checkpoint, Qwen weights, FFmpeg and model-specific environments. A generic upstream download alone may not supply the project-specific Stage 1 integration.

The direct sequential CLI exposes machine-path overrides. Once those dependencies exist, adapt this command to the installed environments:

```bash
python3 pipeline/main.py \
  --video /path/to/input.mp4 \
  --stage1-root /path/to/YOWOv3 \
  --stage1-config /path/to/custom_shufflenet.yaml \
  --stage1-pretrain-path /path/to/yowov3-checkpoint.pth \
  --conda-bin /path/to/miniconda3/bin/conda \
  --stage1-env yowov3 --stage2-env qwen35 \
  --vlm-model-path /path/to/Qwen3.5-2B \
  --prompt benchmark2/prompts/action_timev1.txt
```

This is a parameterized command reference, not a validated fresh-checkout smoke test. `--stage1-root` alone does not fix the separate hard-coded `--stage1-config` default.

**The committed API configuration is incomplete for the sequential wrapper:** `jobs_root` and `conda_bin` are missing. The wrapper also does not forward a custom Stage 1 config. `serving/run_job.py --config` accepts a local file, but `serving/app.py` reads `serving/config.json` directly. These are known follow-up fixes; the existing API examples in [serving notes](../serving/README.md) describe the original integration rather than a portable launch recipe.

Authentication is disabled when the config lacks `api_key`, and `/health` exposes configured filesystem paths. Keep prototype inspection local until config handling and the exposed fields are addressed. Do not put credentials in tracked config files.

## Validation scope

The documentation update was checked with Python AST parsing, JSON parsing, CLI help, and the synthetic evaluator. No package installation, dataset reconstruction, GPU training, VLM inference, latency measurement, or end-to-end API job was performed. Original result tables remain historical reported measurements.
