#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


RISK_STATES = ("normal", "unsafe", "danger")
CLASS_TO_RISK = {0: "unsafe", 1: "danger", 2: "normal"}
RISK_PRIORITY = {"normal": 0, "unsafe": 1, "danger": 2}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def stage2_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_label(path: Path) -> str:
    classes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            classes.add(int(parts[0]))

    if not classes:
        raise ValueError(f"Empty label file: {path}")
    if not classes <= set(CLASS_TO_RISK):
        raise ValueError(f"Unexpected class in label: {path}: {sorted(classes)}")
    if 1 in classes:
        return "danger"
    if 0 in classes:
        return "unsafe"
    return "normal"


def resolve_image(image_dir: Path, video_id: str, frame_idx: int) -> Path:
    candidates = [
        image_dir / f"{video_id}_t{frame_idx:06d}.jpg",
        image_dir / f"{video_id}_t{frame_idx:06d}.jpeg",
        image_dir / f"{video_id}_t{frame_idx:06d}.png",
        image_dir / f"30fps_frame_{frame_idx:03d}.jpg",
        image_dir / f"30fps_frame_{frame_idx:03d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTS:
        matches = sorted(image_dir.glob(f"*t{frame_idx:06d}*{ext}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Image not found: video={video_id}, frame={frame_idx}, dir={image_dir}"
    )


def summarize_actions(frame: dict, top_k: int, include_score: bool) -> str:
    actions = []
    for detection in frame.get("detections", []):
        for action in detection.get("actions", []):
            actions.append(
                {
                    "name": str(action.get("class_name", "unknown")),
                    "score": float(action.get("score", 0.0)),
                }
            )

    actions.sort(key=lambda item: item["score"], reverse=True)
    selected = []
    seen = set()
    for action in actions:
        if action["name"] in seen:
            continue
        seen.add(action["name"])
        selected.append(action)
        if len(selected) == top_k:
            break

    if not selected:
        return "none"
    if include_score:
        return ", ".join(
            f"{item['name']}({item['score']:.2f})" for item in selected
        )
    return ", ".join(item["name"] for item in selected)


def render_prompt(
    template: str,
    frame_indices: list[int],
    frame_times: list[float],
    actions: list[str],
) -> str:
    prompt = template
    for idx, (frame_idx, time_sec, action) in enumerate(
        zip(frame_indices, frame_times, actions),
        start=1,
    ):
        prompt = prompt.replace(f"{{frame_{idx}}}", str(frame_idx))
        prompt = prompt.replace(f"{{time_{idx}}}", f"{time_sec:.3f}")
        prompt = prompt.replace(f"{{action_{idx}}}", action)

    unresolved = [
        token
        for token in ("{frame_", "{time_", "{action_")
        if token in prompt
    ]
    if unresolved:
        raise ValueError(f"Unresolved prompt placeholders: {unresolved}")
    return prompt


def choose_sequence_gt(
    sampled_states: list[str],
    coverage_ratio: dict[str, float],
    danger_threshold: float,
    unsafe_threshold: float,
) -> str:
    if "danger" in sampled_states:
        return "danger"
    if coverage_ratio["danger"] >= danger_threshold:
        return "danger"
    if "unsafe" in sampled_states:
        return "unsafe"
    if coverage_ratio["unsafe"] >= unsafe_threshold:
        return "unsafe"
    return "normal"


def build_rows(config: dict, root: Path) -> tuple[list[dict], dict]:
    paths = config["paths"]
    data_config = config["data"]
    data_root = resolve_path(root, paths["data_root"])
    labeling_root = resolve_path(root, paths["labeling_root"])
    prompt_path = resolve_path(root, paths["prompt_path"])
    prompt_template = prompt_path.read_text(encoding="utf-8")

    fps = float(data_config["fps"])
    stage1_window = int(data_config["stage1_window"])
    gap = int(data_config["within_sequence_gap"])
    stride = int(data_config["request_stride"])
    top_k = int(data_config["action_top_k"])
    include_score = bool(data_config["include_action_confidence"])
    danger_threshold = float(data_config["danger_threshold"])
    unsafe_threshold = float(data_config["unsafe_threshold"])
    exclude_coverage_promoted = bool(
        data_config.get("exclude_coverage_promoted", False)
    )

    if int(data_config["num_frames_per_request"]) != 3:
        raise ValueError("This training pipeline requires exactly 3 images per request.")

    rows = []
    skipped = Counter()
    excluded_examples = defaultdict(list)
    videos_seen = Counter()

    for group, label_subdir in data_config["groups"].items():
        group_root = data_root / group
        image_root = group_root / "images"
        action_root = group_root / "labels"
        risk_root = labeling_root / label_subdir

        if not image_root.exists() or not action_root.exists() or not risk_root.exists():
            raise FileNotFoundError(f"Missing data roots for group: {group}")

        for action_path in sorted(action_root.glob("*.json")):
            video_id = action_path.stem
            image_dir = image_root / video_id
            risk_dir = risk_root / video_id
            if not image_dir.exists() or not risk_dir.exists():
                skipped["missing_sequence_dir"] += 1
                continue

            action_data = load_json(action_path)
            if action_data.get("video", video_id) != video_id:
                raise ValueError(f"Video id mismatch: {action_path}")

            frames = sorted(
                action_data.get("frames", []),
                key=lambda frame: int(frame["frame_idx"]),
            )
            if not frames:
                skipped["empty_action_json"] += 1
                continue

            frame_map = {int(frame["frame_idx"]): frame for frame in frames}
            available = sorted(frame_map)
            start = available[0]
            last = available[-1]
            video_request_count = 0

            while start + gap * 2 <= last:
                selected = [start, start + gap, start + gap * 2]
                if not all(frame_idx in frame_map for frame_idx in selected):
                    skipped["missing_action_frame"] += 1
                    start += stride
                    continue

                label_paths = [
                    risk_dir / f"30fps_frame_{frame_idx:03d}.txt"
                    for frame_idx in selected
                ]
                if not all(path.exists() for path in label_paths):
                    skipped["missing_sampled_gt"] += 1
                    start += stride
                    continue

                try:
                    image_paths = [
                        resolve_image(image_dir, video_id, frame_idx)
                        for frame_idx in selected
                    ]
                except FileNotFoundError:
                    skipped["missing_sampled_image"] += 1
                    start += stride
                    continue

                coverage_start = selected[0] - stage1_window + 1
                coverage_end = selected[-1]
                coverage_paths = [
                    risk_dir / f"30fps_frame_{frame_idx:03d}.txt"
                    for frame_idx in range(coverage_start, coverage_end + 1)
                ]
                if not all(path.exists() for path in coverage_paths):
                    skipped["missing_coverage_gt"] += 1
                    start += stride
                    continue

                sampled_states = [parse_label(path) for path in label_paths]
                sampled_gt = max(
                    sampled_states,
                    key=lambda state: RISK_PRIORITY[state],
                )
                coverage_states = [parse_label(path) for path in coverage_paths]
                coverage_count = Counter(coverage_states)
                coverage_total = len(coverage_states)
                coverage_ratio = {
                    state: coverage_count[state] / coverage_total
                    for state in RISK_STATES
                }
                sequence_gt = choose_sequence_gt(
                    sampled_states=sampled_states,
                    coverage_ratio=coverage_ratio,
                    danger_threshold=danger_threshold,
                    unsafe_threshold=unsafe_threshold,
                )

                selected_frames = [frame_map[frame_idx] for frame_idx in selected]
                actions = [
                    summarize_actions(frame, top_k, include_score)
                    for frame in selected_frames
                ]
                frame_times = [frame_idx / fps for frame_idx in selected]
                prompt_text = render_prompt(
                    template=prompt_template,
                    frame_indices=selected,
                    frame_times=frame_times,
                    actions=actions,
                )
                request_id = (
                    f"{video_id}"
                    f"_f{selected[0]:06d}_{selected[1]:06d}_{selected[2]:06d}"
                )
                coverage_promoted = sampled_gt != sequence_gt
                if exclude_coverage_promoted and coverage_promoted:
                    skipped["coverage_promoted_excluded"] += 1
                    excluded_examples["coverage_promoted"].append(request_id)
                    start += stride
                    continue

                relative_images = [
                    str(path.relative_to(root))
                    for path in image_paths
                ]
                assistant_response = json.dumps(
                    {"risk_state": sequence_gt},
                    ensure_ascii=True,
                    separators=(",", ":"),
                )

                rows.append(
                    {
                        "request_id": request_id,
                        "group": group,
                        "video_id": video_id,
                        "frame_indices": selected,
                        "frame_times_sec": [
                            round(time_sec, 3) for time_sec in frame_times
                        ],
                        "images": relative_images,
                        "stage1_actions": actions,
                        "sampled_frame_gt": sampled_states,
                        "sampled_only_gt": sampled_gt,
                        "coverage_start_frame": coverage_start,
                        "coverage_end_frame": coverage_end,
                        "coverage_frame_count": {
                            state: coverage_count[state] for state in RISK_STATES
                        },
                        "coverage_ratio": {
                            state: round(coverage_ratio[state], 6)
                            for state in RISK_STATES
                        },
                        "sequence_gt": sequence_gt,
                        "coverage_promoted": coverage_promoted,
                        "prompt_text": prompt_text,
                        "assistant_response": assistant_response,
                    }
                )
                video_request_count += 1
                start += stride

            if video_request_count:
                videos_seen[group] += 1
            else:
                skipped["video_without_requests"] += 1

    request_ids = [row["request_id"] for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Duplicate request_id detected.")

    summary = {
        "requests": len(rows),
        "videos_by_group": dict(videos_seen),
        "class_distribution": dict(Counter(row["sequence_gt"] for row in rows)),
        "coverage_promoted": sum(row["coverage_promoted"] for row in rows),
        "skipped": dict(skipped),
        "excluded_examples": dict(excluded_examples),
    }
    return rows, summary


def allocate_counts(size: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: size * ratio for split, ratio in ratios.items()}
    counts = {split: math.floor(value) for split, value in raw.items()}
    remainder = size - sum(counts.values())
    order = sorted(
        ratios,
        key=lambda split: (raw[split] - counts[split], ratios[split]),
        reverse=True,
    )
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def find_balanced_split(
    rows: list[dict],
    ratios: dict[str, float],
    seed_search_count: int,
) -> tuple[int, dict[str, list[str]], dict]:
    by_group_video = defaultdict(lambda: defaultdict(Counter))
    total_classes = Counter()
    for row in rows:
        by_group_video[row["group"]][row["video_id"]][row["sequence_gt"]] += 1
        total_classes[row["sequence_gt"]] += 1

    group_counts = {
        group: allocate_counts(len(video_rows), ratios)
        for group, video_rows in by_group_video.items()
    }
    target = {
        split: {
            state: total_classes[state] * ratios[split]
            for state in RISK_STATES
        }
        for split in ratios
    }

    best = None
    split_order = ("train", "validation", "test")
    for seed in range(seed_search_count):
        rng = random.Random(seed)
        split_videos = {split: [] for split in split_order}
        split_classes = {split: Counter() for split in split_order}

        for group in sorted(by_group_video):
            videos = sorted(by_group_video[group])
            rng.shuffle(videos)
            offset = 0
            for split in split_order:
                count = group_counts[group][split]
                selected = videos[offset : offset + count]
                offset += count
                split_videos[split].extend(selected)
                for video_id in selected:
                    split_classes[split].update(
                        by_group_video[group][video_id]
                    )

        score = 0.0
        for split in split_order:
            for state in RISK_STATES:
                expected = target[split][state]
                score += (
                    (split_classes[split][state] - expected)
                    / max(expected, 1.0)
                ) ** 2

        candidate = (score, seed, split_videos, split_classes)
        if best is None or candidate[0] < best[0]:
            best = candidate

    _, seed, split_videos, split_classes = best
    split_sets = {
        split: set(video_ids)
        for split, video_ids in split_videos.items()
    }
    if split_sets["train"] & split_sets["validation"]:
        raise ValueError("Train/validation sequence leakage detected.")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("Train/test sequence leakage detected.")
    if split_sets["validation"] & split_sets["test"]:
        raise ValueError("Validation/test sequence leakage detected.")

    metadata = {
        "selected_seed": seed,
        "ratios": ratios,
        "sequence_counts": {
            split: len(video_ids)
            for split, video_ids in split_videos.items()
        },
        "request_class_counts": {
            split: dict(split_classes[split])
            for split in split_order
        },
    }
    return seed, split_videos, metadata


def load_fixed_split(
    rows: list[dict],
    split_path: Path,
    ratios: dict[str, float],
) -> tuple[int, dict[str, list[str]], dict]:
    payload = load_json(split_path)
    declared = payload.get("videos", {})
    split_order = ("train", "validation", "test")
    if set(declared) != set(split_order):
        raise ValueError(
            f"Fixed split must contain {split_order}: {split_path}"
        )

    current_videos = {row["video_id"] for row in rows}
    split_videos = {
        split: sorted(current_videos & set(declared[split]))
        for split in split_order
    }
    assigned = [
        video_id
        for split in split_order
        for video_id in split_videos[split]
    ]
    duplicate_videos = [
        video_id
        for video_id, count in Counter(assigned).items()
        if count > 1
    ]
    if duplicate_videos:
        raise ValueError(
            f"Videos assigned to multiple fixed splits: {duplicate_videos[:5]}"
        )

    missing_videos = current_videos - set(assigned)
    if missing_videos:
        raise ValueError(
            f"Videos missing from fixed split: {sorted(missing_videos)[:5]}"
        )

    video_to_split = {
        video_id: split
        for split in split_order
        for video_id in split_videos[split]
    }
    split_classes = {split: Counter() for split in split_order}
    for row in rows:
        split_classes[video_to_split[row["video_id"]]][row["sequence_gt"]] += 1

    seed = int(payload.get("selected_seed", -1))
    metadata = {
        "selected_seed": seed,
        "ratios": ratios,
        "source": str(split_path),
        "sequence_counts": {
            split: len(split_videos[split])
            for split in split_order
        },
        "request_class_counts": {
            split: dict(split_classes[split])
            for split in split_order
        },
    }
    return seed, split_videos, metadata


def split_rows(
    rows: list[dict],
    split_videos: dict[str, list[str]],
) -> dict[str, list[dict]]:
    video_to_split = {}
    for split, video_ids in split_videos.items():
        for video_id in video_ids:
            if video_id in video_to_split:
                raise ValueError(f"Video assigned twice: {video_id}")
            video_to_split[video_id] = split

    output = {split: [] for split in split_videos}
    for row in rows:
        split = video_to_split.get(row["video_id"])
        if split is None:
            raise ValueError(f"Video missing from split: {row['video_id']}")
        output[split].append(row)
    return output


def parse_args() -> argparse.Namespace:
    default_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "qwen35_08b_action_v2.json"
    )
    parser = argparse.ArgumentParser(
        description="Build Qwen3.5 CCTV LoRA JSONL datasets."
    )
    parser.add_argument("--config", default=str(default_config))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = stage2_root()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    output_dir = resolve_path(root, config["paths"]["dataset_output_dir"])

    rows, build_summary = build_rows(config, root)
    ratios = {
        "train": float(config["split"]["train_ratio"]),
        "validation": float(config["split"]["validation_ratio"]),
        "test": float(config["split"]["test_ratio"]),
    }
    if not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0: {ratios}")

    fixed_split_value = config["split"].get("fixed_split_path")
    if fixed_split_value:
        fixed_split_path = resolve_path(root, fixed_split_value)
        seed, split_videos, split_metadata = load_fixed_split(
            rows=rows,
            split_path=fixed_split_path,
            ratios=ratios,
        )
    else:
        seed, split_videos, split_metadata = find_balanced_split(
            rows=rows,
            ratios=ratios,
            seed_search_count=int(config["split"]["seed_search_count"]),
        )
    split_data = split_rows(rows, split_videos)

    write_jsonl(output_dir / "all.jsonl", rows)
    for split, split_rows_data in split_data.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_rows_data)

    split_payload = {
        **split_metadata,
        "videos": {
            split: sorted(video_ids)
            for split, video_ids in split_videos.items()
        },
    }
    write_json(output_dir / "splits.json", split_payload)

    sampling = config["sampling"]
    effective_train = Counter()
    for row in split_data["train"]:
        state = row["sequence_gt"]
        effective_train[state] += int(sampling[f"{state}_factor"])

    summary = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "build": build_summary,
        "split": split_metadata,
        "split_request_counts": {
            split: len(split_rows_data)
            for split, split_rows_data in split_data.items()
        },
        "split_class_counts": {
            split: dict(Counter(row["sequence_gt"] for row in split_rows_data))
            for split, split_rows_data in split_data.items()
        },
        "train_sampling_factors": sampling,
        "effective_train_class_counts": dict(effective_train),
    }
    write_json(output_dir / "summary.json", summary)

    print("Dataset output:", output_dir)
    print("Selected split seed:", seed)
    print("Total requests:", len(rows))
    for split in ("train", "validation", "test"):
        counts = Counter(row["sequence_gt"] for row in split_data[split])
        print(
            f"{split}: requests={len(split_data[split])}, "
            f"videos={len(split_videos[split])}, classes={dict(counts)}"
        )
    print("Effective train classes after oversampling:", dict(effective_train))


if __name__ == "__main__":
    main()
