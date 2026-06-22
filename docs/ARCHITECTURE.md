# Project Architecture

ZroAct is organized around a two-stage pipeline.

```text
Input video
  -> 30fps frame extraction
  -> Stage 1 action detection
  -> Stage 2 request builder
  -> Qwen3.5 Vision risk classifier
  -> JSON logs, event logs, overlays, backend result
```

## Stage 1: Action Detection

Stage 1 uses YOWOv3 to detect action candidates from CCTV frames.

Runtime output is stored as frame-level JSON records. Each record contains:

- frame index,
- frame image reference,
- detected boxes,
- action class names,
- action scores.

For Stage 2, the system summarizes the top actions per selected frame. The current training setup uses action names only, not numeric confidence scores.

## Stage 2: VLM Risk Classification

Each Stage 2 request uses three images:

```text
t, t+10, t+20
```

At 30 fps, this covers roughly 0.67 seconds from the first selected frame to the last selected frame.

The request includes:

- `request_id`,
- `video_id`,
- selected image paths,
- selected frame indices,
- frame times,
- Stage 1 action summaries,
- sequence coverage metadata.

The model returns only:

```json
{"risk_state":"danger"}
```

## Training Architecture

Training code lives in `benchmark2/training`.

Main scripts:

| Script | Role |
|---|---|
| `build_dataset.py` | Build JSONL samples from images, Stage 1 labels, and risk labels. |
| `validate_dataset.py` | Check image paths, split leakage, schema, and class distribution. |
| `train_lora.py` | Train Qwen3.5 Vision LoRA with response-only loss. |
| `evaluate_lora.py` | Generate predictions and compute metrics/confusion matrix. |

Important training choices:

- video-level train/validation/test split,
- `unsafe` oversampling in train only,
- BF16 LoRA fine-tuning,
- response-only loss on assistant JSON,
- JSON/schema success tracked separately from classification accuracy.

## Serving Architecture

The serving layer lives in `serving`.

```text
Backend
  -> POST /jobs
  -> FastAPI saves upload
  -> background job runs pipeline
  -> result/log files are written
  -> backend polls status, result, logs, or events
```

Main files:

| File | Role |
|---|---|
| `serving/app.py` | FastAPI API surface. |
| `serving/run_job.py` | Single-job CLI/runtime wrapper. |
| `serving/config.json` | Local paths and runtime settings. |
| `serving/utils.py` | Job status, JSONL, and result helpers. |
| `serving/workers/` | Stage worker and daemon prototypes. |

## Pipeline Versions

| Path | Description |
|---|---|
| `pipeline/main.py` | Sequential subprocess-based integration path. |
| `pipeline_ver2/realtime_pipeline.py` | HTTP daemon based async streaming prototype. |
| `pipeline_ver2/plan.md` | Notes on bottlenecks and parallelization options. |

The realtime prototype keeps Stage 1 and Stage 2 services loaded and schedules Stage 2 requests as soon as the required Stage 1 frames are available.
