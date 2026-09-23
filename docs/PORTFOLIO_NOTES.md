# Research and contribution scope

## Research motivation

A small VLM can inspect only a limited number of frames. Action detection provides an additional representation of motion and behavior, which can be passed with sampled images and temporal context to the VLM. This project explores that design for CCTV intrusion risk classification.

The connection to my broader research interests is **selecting and representing useful temporal information under a computation budget**, followed by task-specific adaptation with LoRA. It is a video/multimodal capstone, not a claim of a new general-purpose video foundation model.

## Personal role and team scope

**Seongwoo Lim: AI Leader / modeling**, as reported by the project author.

The capstone included an end-to-end CCTV → backend → AI server → frontend dashboard system. This public repository contains AI-side dataset preparation, prompt variants, LoRA training/evaluation, pipeline integration, and a FastAPI serving interface. It does not contain the entire backend or frontend.

The code demonstrates these project components, but its one-commit public snapshot does not establish who authored each component. Exact file-level ownership, other team members' roles, and division of backend/serving work should be added from project records before making narrower personal-contribution claims. YOWOv3, Qwen, Unsloth, and other dependencies are upstream work.

## What a reviewer can inspect

| Question | Public evidence |
|---|---|
| How is temporal context supplied? | Three selected images plus frame/time/action placeholders in [prompts](../benchmark2/prompts/action_timev1.txt) |
| How are controlled input variants constructed? | Action/no-action prompts, [one-frame manifest builder](../benchmark2/scripts/build_one_frame_manifest.py), and [no-bbox image manifest builder](../benchmark2/scripts/build_no_bbox_manifest.py) |
| How is task adaptation implemented? | [LoRA configuration](../benchmark2/training/configs/qwen35_08b_action_v2.json) and [training script](../benchmark2/training/scripts/train_lora.py) |
| How is data leakage checked? | Video-level splits and [dataset validation](../benchmark2/training/scripts/validate_dataset.py) |
| What are the reported findings? | [Experiment summary](EXPERIMENTS_SUMMARY.md), with split and artifact caveats in [result provenance](RESULT_PROVENANCE.md) |
| What can run without private resources? | [Synthetic evaluation example](SETUP_AND_RUN.md#1-cpu-only-evaluation-example) |

## Limitations and next experiment

The intermediate `unsafe` class is difficult, and stride-1 evaluation repeats nearby temporal evidence. Current public reports cannot isolate a LoRA gain because the recorded LoRA and baseline rows use different splits. The specific contribution of action cues also requires matching prompt versions, manifests, image sources, model checkpoints, and evaluation support.

The next useful research step is to recover those run records, select checkpoints on validation, and evaluate the chosen configuration and its controls on the same held-out support. Publish a small de-identified metric artifact and protocol alongside any new claim. Do not substitute new experiments for missing records of earlier results.
