# ZroAct Stage 2

ZroAct is a two-stage CCTV intrusion risk analysis project.

The project connects a video action detector with a vision-language model so that CCTV clips can be converted into structured risk states:

```json
{"risk_state":"normal|unsafe|danger"}
```

This repository is organized as a portfolio-ready engineering record. Large datasets, model weights, LoRA checkpoints, generated frames, and runtime outputs are intentionally excluded from git.

## What This Project Does

1. Stage 1 detects human action candidates from CCTV frames using YOWOv3.
2. Stage 2 receives three time-ordered CCTV images plus Stage 1 action summaries.
3. A Qwen3.5 Vision model classifies the request into `normal`, `unsafe`, or `danger`.
4. The serving layer exposes the pipeline as backend-facing video jobs.

## Risk States

| State | Meaning |
|---|---|
| `normal` | No intrusion-related behavior is visible. |
| `unsafe` | A person is approaching, waiting near, touching, or preparing around the boundary. |
| `danger` | Clear intrusion evidence such as climbing or crossing the boundary is visible. |

## Repository Map

| Path | Purpose |
|---|---|
| `benchmark2/` | Stage 2 datasets, prompts, evaluation scripts, and training code. |
| `benchmark2/training/` | Qwen3.5 LoRA dataset building, validation, training, and evaluation. |
| `pipeline/` | Original sequential Stage 1 + Stage 2 runtime pipeline. |
| `pipeline_ver2/` | HTTP daemon based streaming and parallel pipeline experiments. |
| `serving/` | FastAPI backend integration layer for upload-based video jobs. |
| `docs/` | Portfolio-friendly project documentation. |

More detail is available in:

- [Project Architecture](docs/ARCHITECTURE.md)
- [Repository Structure](docs/REPOSITORY_STRUCTURE.md)
- [Experiment Summary](docs/EXPERIMENTS_SUMMARY.md)
- [Setup and Run Notes](docs/SETUP_AND_RUN.md)
- [Portfolio Notes](docs/PORTFOLIO_NOTES.md)
- [Git Publishing Checklist](docs/GIT_PUBLISHING_CHECKLIST.md)

## Main Contributions Captured Here

- Built a Stage 2 VLM classification dataset from CCTV image sequences, Stage 1 action JSON, and frame-level risk labels.
- Designed request generation using three temporal frames: `t`, `t+10`, `t+20`.
- Implemented Qwen3.5 Vision LoRA training and evaluation with JSON-only output.
- Added dataset validation, split control, confusion matrix, and per-class metrics.
- Built a FastAPI job interface for backend integration.
- Prototyped a daemon-based realtime pipeline that keeps Stage 1 and Stage 2 workers loaded.

## Current Experimental Status

The strongest observed pattern is:

- `normal` and `danger` are learned relatively well.
- `unsafe` is the main bottleneck because it sits between normal behavior and clear intrusion.
- Qwen3.5-2B zero-shot is much stronger than Qwen3.5-0.8B zero-shot, but still struggles on `unsafe`.
- LoRA training clearly improves task formatting and class separation.

See [Experiment Summary](docs/EXPERIMENTS_SUMMARY.md) for the metrics that are safe to share without committing large output files.

## Git Hygiene

This working directory contains large local assets such as data, model weights, checkpoints, generated frames, and logs. They should not be committed.

Recommended first commit:

```bash
cd /home/capstone2/zroact-stage2
git init
git add README.md docs .gitignore benchmark2/scripts benchmark2/prompts benchmark2/training/scripts benchmark2/training/configs pipeline pipeline_ver2 serving requirements.txt benchmark2/training/requirements.txt
git status
```

Review `git status` carefully before committing.
