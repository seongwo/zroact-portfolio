# Experiment Summary

This summary is intentionally small enough to keep in git. Raw datasets, predictions, checkpoints, and generated outputs are excluded.

## Dataset v2

The main dataset version is `qwen35_08b_action_v2`.

| Item | Value |
|---|---:|
| Videos | 180 |
| Images | 53,920 |
| Stage 2 requests | 50,260 |
| Input frames per request | 3 |
| Frame pattern | `t, t+10, t+20` |
| Request stride | 1 frame |
| Image size | `768x432 RGB` |

Groups used:

- `normal_plant_rgb`
- `climb-over-fence_smart_rgb`

The `climb-over-fence_plant_rgb` group was excluded from the v2 setup because its label distribution was not suitable for the target experiment.

## Split Distribution

| Split | Videos | Requests | Normal | Unsafe | Danger |
|---|---:|---:|---:|---:|---:|
| Train | 144 | 40,329 | 16,547 | 4,350 | 19,432 |
| Validation | 18 | 4,924 | 2,101 | 596 | 2,227 |
| Test | 18 | 5,007 | 2,079 | 545 | 2,383 |
| Total | 180 | 50,260 | 20,727 | 5,491 | 24,042 |

Training exposes `unsafe` samples twice, only within the training split.

## Qwen3.5-0.8B LoRA

Training setup:

- model: Qwen3.5-0.8B Vision,
- method: Unsloth BF16 LoRA SFT,
- epochs: 3,
- optimizer steps: 4,191,
- LoRA target: vision and language layers,
- output format: single JSON object.

Best evaluated checkpoint so far:

```text
checkpoint-2794
```

Validation result:

| Metric | Value |
|---|---:|
| Accuracy | 0.9297 |
| Macro F1 | 0.8454 |
| JSON success rate | 1.0000 |
| Schema success rate | 1.0000 |

Per-class result:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.9255 | 0.9881 | 0.9558 | 2,101 |
| Unsafe | 0.9167 | 0.4614 | 0.6138 | 596 |
| Danger | 0.9353 | 1.0000 | 0.9666 | 2,227 |

Main observation:

- `danger` recall reached 100%.
- `normal` was stable.
- `unsafe` recall remained the main bottleneck.

## Qwen3.5-0.8B Zero-shot Baseline

On the test split, the base 0.8B model predicted almost every request as `danger`.

| Metric | Value |
|---|---:|
| Accuracy | 0.4757 |
| Macro F1 | 0.2149 |
| JSON success rate | 0.9998 |

This shows that LoRA was necessary for the 0.8B model to learn the target task and output format.

## Qwen3.5-2B Zero-shot Baseline

The 2B base model was much stronger zero-shot than 0.8B.

| Metric | Value |
|---|---:|
| Samples | 5,007 |
| Accuracy | 0.8870 |
| Macro F1 | 0.6970 |
| JSON success rate | 1.0000 |
| Schema success rate | 1.0000 |

Per-class result:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.9435 | 0.9726 | 0.9578 | 2,079 |
| Unsafe | 0.7396 | 0.1303 | 0.2215 | 545 |
| Danger | 0.8483 | 0.9853 | 0.9117 | 2,383 |

Confusion matrix:

| Ground Truth | Normal | Unsafe | Danger | Invalid |
|---|---:|---:|---:|---:|
| Normal | 2,022 | 0 | 57 | 0 |
| Unsafe | 111 | 71 | 363 | 0 |
| Danger | 10 | 25 | 2,348 | 0 |

Main observation:

- Qwen3.5-2B already separates `normal` and `danger` fairly well.
- The middle `unsafe` state is still difficult.

## Qwen3.5-2B LoRA Status

The local workspace contains completed 2B LoRA training checkpoints:

```text
checkpoint-1397
checkpoint-2794
checkpoint-4191
final_adapter
```

Training completed 3 epochs and 4,191 optimizer steps.

The next useful step is to run validation generation for each checkpoint and compare:

1. Macro F1
2. Unsafe recall
3. Danger recall
4. Invalid output count
5. Accuracy

## Key Research Finding

The project is not mainly limited by JSON formatting or `danger` detection. The core modeling issue is the semantic boundary around `unsafe`, especially during transition periods before or after clear climbing behavior.
