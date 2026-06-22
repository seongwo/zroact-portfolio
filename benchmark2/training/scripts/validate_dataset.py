#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


RISK_STATES = {"normal", "unsafe", "danger"}


def stage2_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL: {path}:{line_number}") from exc
    return rows


def validate_row(row: dict, root: Path) -> list[str]:
    errors = []
    request_id = row.get("request_id", "<missing>")
    images = row.get("images", [])
    if len(images) != 3:
        errors.append(f"{request_id}: expected 3 images, got {len(images)}")
    if len(row.get("frame_indices", [])) != 3:
        errors.append(f"{request_id}: expected 3 frame indices")
    if len(row.get("stage1_actions", [])) != 3:
        errors.append(f"{request_id}: expected 3 action strings")

    for image_text in images:
        image_path = root / image_text
        if not image_path.exists():
            errors.append(f"{request_id}: missing image: {image_path}")
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            errors.append(f"{request_id}: invalid image {image_path}: {exc}")

    state = row.get("sequence_gt")
    if state not in RISK_STATES:
        errors.append(f"{request_id}: invalid sequence_gt: {state}")
    expected = json.dumps(
        {"risk_state": state},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    if row.get("assistant_response") != expected:
        errors.append(f"{request_id}: assistant response mismatch")
    if any("{" in action or "}" in action for action in row.get("stage1_actions", [])):
        errors.append(f"{request_id}: malformed action text")
    if "{frame_" in row.get("prompt_text", ""):
        errors.append(f"{request_id}: unresolved frame placeholder")
    return errors


def parse_args() -> argparse.Namespace:
    default_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "qwen35_08b_action_v2.json"
    )
    parser = argparse.ArgumentParser(description="Validate generated LoRA datasets.")
    parser.add_argument("--config", default=str(default_config))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = stage2_root()
    config = load_json(Path(args.config).resolve())
    output_dir = root / config["paths"]["dataset_output_dir"]
    splits = load_json(output_dir / "splits.json")

    all_errors = []
    seen_requests = set()
    split_videos = {}

    for split in ("train", "validation", "test"):
        rows = load_jsonl(output_dir / f"{split}.jsonl")
        videos = {row["video_id"] for row in rows}
        split_videos[split] = videos
        duplicate_requests = [
            row["request_id"]
            for row in rows
            if row["request_id"] in seen_requests
        ]
        if duplicate_requests:
            all_errors.append(
                f"{split}: duplicate request ids: {duplicate_requests[:5]}"
            )
        seen_requests.update(row["request_id"] for row in rows)

        for row in rows:
            all_errors.extend(validate_row(row, root))

        counts = Counter(row["sequence_gt"] for row in rows)
        print(
            f"{split}: rows={len(rows)}, videos={len(videos)}, "
            f"classes={dict(counts)}"
        )

    if split_videos["train"] & split_videos["validation"]:
        all_errors.append("Train/validation video leakage")
    if split_videos["train"] & split_videos["test"]:
        all_errors.append("Train/test video leakage")
    if split_videos["validation"] & split_videos["test"]:
        all_errors.append("Validation/test video leakage")

    declared = {
        split: set(video_ids)
        for split, video_ids in splits["videos"].items()
    }
    for split in declared:
        if declared[split] != split_videos[split]:
            all_errors.append(f"{split}: splits.json does not match JSONL")

    if all_errors:
        print()
        print(f"Validation failed with {len(all_errors)} error(s):")
        for error in all_errors[:50]:
            print(" -", error)
        raise SystemExit(1)

    print()
    print("Dataset validation passed.")
    print("Unique requests:", len(seen_requests))


if __name__ == "__main__":
    main()
