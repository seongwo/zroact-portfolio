# Setup and Run Notes

These notes describe the local execution shape. Exact paths may need to be changed on another machine.

## Environments

The project has historically used separate environments:

| Environment | Purpose |
|---|---|
| `yowov3` | Stage 1 action detection. |
| `qwen35` | Stage 2 VLM inference and serving. |
| `qwen35-lora` | LoRA training with Unsloth. |

## Stage 2 Training Environment

```bash
cd /home/capstone2/zroact-stage2
python -m pip install --upgrade pip
python -m pip install -r benchmark2/training/requirements.txt
```

The full training setup is documented in:

```text
benchmark2/training/README.md
```

## Build And Validate Dataset

```bash
cd /home/capstone2/zroact-stage2

python benchmark2/training/scripts/build_dataset.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json

python benchmark2/training/scripts/validate_dataset.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json
```

Generated JSONL files are ignored by git.

## Train LoRA

The training script requires `--run` so that training does not start accidentally.

```bash
python benchmark2/training/scripts/train_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --run
```

For Qwen3.5-2B:

```bash
python benchmark2/training/scripts/train_lora.py \
  --config benchmark2/training/configs/qwen35_2b_action_v2.json \
  --run
```

## Evaluate A Checkpoint

```bash
python benchmark2/training/scripts/evaluate_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --adapter-path benchmark2/training/outputs/qwen35_08b_action_v2/checkpoint-2794 \
  --split validation
```

## Run Serving API

```bash
cd /home/capstone2/zroact-stage2

/home/capstone2/miniconda3/bin/conda run -n qwen35 \
  uvicorn serving.app:app --host 0.0.0.0 --port 9000
```

Create a video job:

```bash
curl -X POST http://127.0.0.1:9000/jobs \
  -F "file=@/path/to/input.mp4"
```

Read results:

```bash
curl http://127.0.0.1:9000/jobs/{job_id}
curl http://127.0.0.1:9000/jobs/{job_id}/result
curl "http://127.0.0.1:9000/jobs/{job_id}/logs?after=0&limit=100"
curl http://127.0.0.1:9000/jobs/{job_id}/events
```

## Run One Job By CLI

```bash
/home/capstone2/miniconda3/bin/conda run -n qwen35 python serving/run_job.py \
  --video /path/to/input.mp4 \
  --job-id test_001
```

## Notes For A Fresh Machine

This repository alone is not enough to reproduce full results because it does not include:

- CCTV frame datasets,
- frame-level label files,
- Qwen model weights,
- YOWOv3 checkpoint weights,
- generated LoRA adapters.

That is intentional. The repository should show the system design and implementation, while large/private assets should be stored separately.
