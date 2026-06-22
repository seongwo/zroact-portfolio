#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path


RISK_STATES = ("normal", "unsafe", "danger")


def stage2_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def strip_json_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_prediction(text: str) -> tuple[str, bool, bool]:
    value = strip_json_fence(text)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", value)
        if not match:
            return "invalid", False, False
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return "invalid", False, False

    state = payload.get("risk_state") if isinstance(payload, dict) else None
    return (
        state if state in RISK_STATES else "invalid",
        True,
        state in RISK_STATES,
    )


def class_metrics(
    confusion: Counter,
    state: str,
) -> dict[str, float]:
    tp = confusion[(state, state)]
    fp = sum(confusion[(other, state)] for other in RISK_STATES if other != state)
    fp += confusion[("invalid", state)]
    fn = sum(confusion[(state, other)] for other in (*RISK_STATES, "invalid") if other != state)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": sum(confusion[(state, pred)] for pred in (*RISK_STATES, "invalid")),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    default_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "qwen35_08b_action_v2.json"
    )
    parser = argparse.ArgumentParser(
        description="Generate and evaluate Qwen3.5 CCTV predictions."
    )
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--adapter-path")
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from unsloth import FastVisionModel
    import torch
    from PIL import Image

    root = stage2_root()
    config = load_json(Path(args.config).resolve())
    dataset_dir = resolve_path(root, config["paths"]["dataset_output_dir"])
    rows = load_jsonl(dataset_dir / f"{args.split}.jsonl")
    if args.limit is not None:
        rows = rows[: args.limit]

    model_path = (
        Path(args.adapter_path).resolve()
        if args.adapter_path
        else resolve_path(root, config["paths"]["model_path"])
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else model_path / f"eval_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = config["model"]
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[model_config["dtype"]]
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(model_path),
        load_in_4bit=bool(model_config["load_in_4bit"]),
        dtype=dtype,
    )
    FastVisionModel.for_inference(model)

    image_config = config["image"]
    generation_config = config["generation"]
    system_prompt = config["data"]["system_prompt"]
    predictions = []
    confusion = Counter()
    json_success = 0
    schema_success = 0

    for index, row in enumerate(rows, start=1):
        images = []
        for image_text in row["images"]:
            with Image.open(root / image_text) as image:
                images.append(
                    image.convert("RGB").resize(
                        (
                            int(image_config["width"]),
                            int(image_config["height"]),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": row["prompt_text"]},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=images,
            return_tensors="pt",
        ).to(model.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(generation_config["max_new_tokens"]),
                do_sample=bool(generation_config["do_sample"]),
                use_cache=True,
            )
        latency = time.perf_counter() - start_time
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        raw_response = processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]
        prediction, parsed, valid_schema = parse_prediction(raw_response)
        json_success += int(parsed)
        schema_success += int(valid_schema)
        ground_truth = row["sequence_gt"]
        confusion[(ground_truth, prediction)] += 1
        predictions.append(
            {
                "request_id": row["request_id"],
                "video_id": row["video_id"],
                "group": row["group"],
                "frame_indices": row["frame_indices"],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "match": prediction == ground_truth,
                "json_success": parsed,
                "schema_success": valid_schema,
                "latency_sec": round(latency, 4),
                "raw_response": raw_response,
            }
        )
        print(
            f"[{index}/{len(rows)}] {row['request_id']}: "
            f"gt={ground_truth}, pred={prediction}"
        )

    total = len(rows)
    per_class = {
        state: class_metrics(confusion, state)
        for state in RISK_STATES
    }
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(RISK_STATES)
    correct = sum(confusion[(state, state)] for state in RISK_STATES)
    summary = {
        "model_path": str(model_path),
        "split": args.split,
        "samples": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "json_success_rate": json_success / total if total else 0.0,
        "schema_success_rate": schema_success / total if total else 0.0,
        "per_class": per_class,
        "prediction_distribution": dict(
            Counter(row["prediction"] for row in predictions)
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "predictions.jsonl", predictions)

    with (output_dir / "confusion_matrix.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        predictions_order = (*RISK_STATES, "invalid")
        writer.writerow(["ground_truth", *predictions_order])
        for state in RISK_STATES:
            writer.writerow(
                [state, *[confusion[(state, pred)] for pred in predictions_order]]
            )

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved evaluation:", output_dir)


if __name__ == "__main__":
    main()
