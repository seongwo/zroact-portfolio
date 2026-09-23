# Implementation and result provenance

This page distinguishes implementation evidence, results reported in the public snapshot, and additional results reported by the project author. The audit baseline is commit `a75430d`. No training or model inference was rerun during this documentation review.

## Implementation map

| Portfolio statement | Evidence and boundary |
|---|---|
| Action-guided three-frame VLM input | [Dataset builder](../benchmark2/training/scripts/build_dataset.py) renders frame/time/action fields for `t, t+10, t+20`; [v2 config](../benchmark2/training/configs/qwen35_08b_action_v2.json) selects top-2 names without numeric confidence |
| Temporal evidence across the video | [Sequential pipeline](../pipeline/main.py) orchestrates Stage 1 and Stage 2; [serving config](../serving/config.json) sets `stage1_sample_rate=10`, not every-frame inference |
| Bounding-box component ablation | [No-bbox builder](../benchmark2/scripts/build_no_bbox_manifest.py) changes image paths to unannotated frames. It does not remove a structured text box field; the training prompt contains no box-coordinate placeholders |
| Prompt / component controls | [Prompt directory](../benchmark2/prompts), no-action variants, and [one-frame builder](../benchmark2/scripts/build_one_frame_manifest.py) are present; completed matched result artifacts are not |
| LoRA adaptation | [0.8B config](../benchmark2/training/configs/qwen35_08b_action_v2.json) and [2B config](../benchmark2/training/configs/qwen35_2b_action_v2.json) agree on rank/alpha 16/16, vision/language/attention/MLP targets, 3 epochs, effective batch 32, learning rate `1e-4` |
| Full application integration | AI-side [job interface](../serving/app.py) and [runtime wrapper](../serving/run_job.py) are present. Backend/frontend application code and a full-system demo are not included |

## Results already recorded in the public snapshot

[EXPERIMENTS_SUMMARY.md](EXPERIMENTS_SUMMARY.md) records 0.8B zero-shot test accuracy/Macro F1 **0.4757/0.2149**, 0.8B LoRA checkpoint-2794 validation **0.9297/0.8454**, and 2B zero-shot test **0.8870/0.6970**. The [detailed v2 report](../benchmark2/training/V2_RESULTS_REPORT_KO.md) provides the 0.8B validation confusion matrix and training record.

The existing narrative reports are retained as historical records. Their metric tables were not replaced. Raw `metrics.json`, predictions, split assignments, and checkpoints are not tracked, so these numbers are documented observations rather than independently reproduced measurements. Validation and test rows cannot be subtracted to quantify a matched improvement.

The reports mention completed 2B LoRA training in an earlier local workspace. Those checkpoints and a 2B LoRA evaluation table are not in this checkout; their existence cannot be verified here.

## Author-reported results awaiting run association

The following values were supplied by the project author during portfolio preparation. They were **not found in the tracked experiment reports**. They may describe other experiments; they must not be silently relabeled as the v2 validation/test runs above.

| Author-reported setting | Accuracy | Macro F1 / per-class F1 |
|---|---:|---|
| Prompt v0 | 0.9410 | Macro F1 0.7697 |
| Prompt v1 | 0.9263 | Macro F1 0.8359 |
| v1: Action + BBox + Time | Not supplied | Macro F1 0.8359 |
| v1: without Action | Not supplied | Macro F1 0.6486 |
| v1: without BBox / Action | Not supplied | Macro F1 0.7520 |
| LoRA | 94.96% | Normal 0.9790; Unsafe 0.7391; Danger 0.9686 |

Before promoting these values to the README results table, recover: model/checkpoint identity, prompt hash, data version and split, request count, image/box variant, label/coverage rule, prediction artifacts, and exact evaluator command. The matching LoRA hyperparameters alone do not identify the reported run. The label “without BBox / Action” is preserved as supplied and still needs an exact input-variant definition.

## Reproducibility gaps

- Both v2 configs reference `benchmark2/training/splits/qwen35_08b_action_v0.json`, which is absent. The dataset builder calls `load_fixed_split`; changing it to make a new split would define a new experiment, not reproduce the reported one.
- Frames, Stage 1 action records, risk labels, YOWOv3 code/weights, Qwen model files, and trained adapters are excluded or external.
- The root requirements file is a historical package snapshot. It does not include all imports used by the serving/inference paths, including FastAPI, Uvicorn and Transformers; CUDA wheel versions also need the corresponding package source. No clean-install environment has been validated.
- The committed serving config lacks `jobs_root` and `conda_bin`, which are required by the sequential wrapper. It also carries machine-specific paths. Serving launch commands in historical notes should not be treated as ready-to-run instructions for a fresh checkout.
- `pipeline/main.py` has a separate `--stage1-config` default tied to the original machine. Setting only `--stage1-root` does not update that default, and `serving/run_job.py` does not currently pass a custom Stage 1 config path.

## Publication hygiene

The reviewed baseline tracks 58 small source/config/documentation files, no dataset or checkpoint blobs, and no file larger than about 62 KB. A bounded scan of tracked text found no common GitHub, AWS, Hugging Face, OpenAI token or private-key patterns. This is not a comprehensive secret audit.

Machine-specific paths occur in 17 baseline files, including historical reports and runtime defaults. They are portability issues and can reveal local account names; new quickstart commands avoid these paths. Historical experimental records and runtime defaults were retained in this documentation pass.

The job API checks `X-API-Key` only when `api_key` is set in its config; the current config leaves authentication disabled. `/health` exposes configured paths/model information without that check. The API reads tracked `serving/config.json` directly and does not load a key from environment variables. Do not add credentials to that tracked file or expose the current prototype as a public endpoint. A local-config/environment override and path-free health response remain implementation work.

A release license and data/model redistribution permissions have not been established. No license has been invented for the capstone or upstream assets.
