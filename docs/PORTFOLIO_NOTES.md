# Portfolio Notes

This page is written for a reviewer who wants to understand the work without downloading private datasets or large model files.

## Project One-liner

Built a two-stage CCTV intrusion risk analysis pipeline that combines YOWOv3 action detection with Qwen3.5 Vision fine-tuning to classify video moments as `normal`, `unsafe`, or `danger`.

## Problem

Simple object detection is not enough for intrusion monitoring because the risk level depends on temporal context:

- a person near a fence may be normal,
- a person preparing near a fence may be unsafe,
- a person climbing or crossing the fence is dangerous.

The project turns short CCTV sequences into structured risk states that a backend or dashboard can consume.

## My Work

- Designed the Stage 2 request format using three temporal images and Stage 1 action summaries.
- Built scripts to merge image frames, Stage 1 action JSON, and frame-level risk labels into model-ready JSONL.
- Added dataset validation for image paths, split leakage, schema consistency, and class distribution.
- Implemented Qwen3.5 Vision LoRA training with response-only JSON loss.
- Evaluated models using accuracy, macro F1, class-level recall, JSON success, schema success, and confusion matrices.
- Built a FastAPI serving layer for backend video upload jobs.
- Prototyped a daemon-based pipeline that keeps Stage 1 and Stage 2 workers loaded for lower-latency runtime.

## Technical Choices

| Choice | Reason |
|---|---|
| Three-frame VLM input | Gives short temporal context without making each request too heavy. |
| Stage 1 action summaries | Adds motion/action priors that a single image may not reveal. |
| Video-level split | Avoids train/test leakage from adjacent frames of the same video. |
| `unsafe` oversampling | Compensates for the smallest and most ambiguous class. |
| JSON-only output | Makes backend integration simple and testable. |
| Separate serving logs and event logs | Allows the frontend to show only meaningful unsafe/danger events while preserving full traces. |

## Results Worth Highlighting

- Qwen3.5-0.8B zero-shot collapsed mostly to `danger`.
- Qwen3.5-0.8B LoRA learned stable JSON output and reached strong `normal`/`danger` separation.
- Qwen3.5-2B zero-shot was much stronger overall, but still weak on `unsafe`.
- The main remaining challenge is not formatting, but defining and learning the middle-risk boundary.

## Honest Limitations

- `unsafe` recall is still the key weakness.
- Adjacent stride-1 requests are highly correlated, so raw request counts can overstate independent events.
- Some errors are concentrated in a small number of videos, which suggests transition-boundary or label-consistency issues.
- Full reproduction requires private/local data and model weights not stored in git.

## Good Next Steps

1. Evaluate all Qwen3.5-2B LoRA checkpoints on validation.
2. Compare 2B LoRA against 2B zero-shot using macro F1 and `unsafe` recall.
3. Review the top error-concentrated videos frame by frame.
4. Add a small public mock dataset or synthetic example so reviewers can run the pipeline shape without private data.
5. Add screenshots or a short demo GIF from the serving output, without exposing private CCTV data.
