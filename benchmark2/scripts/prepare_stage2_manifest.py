import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
FRAME_RE = re.compile(r"30fps_frame_(\d+)\.txt$")
RISK_STATES = ["normal", "unsafe", "danger"]

CLASS_TO_RISK = {
    0: "unsafe",
    1: "danger",
    2: "normal",
}

RISK_PRIORITY = {
    "unknown": -1,
    "normal": 0,
    "unsafe": 1,
    "danger": 2,
}

DEFAULT_GT_CONFIG = {
    "danger_threshold": 0.30,
    "unsafe_threshold": 0.30,
    "missing_txt_policy": "normal_if_absent",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def infer_folder_coarse_label(group_name: str) -> str:
    if group_name.startswith("normal"):
        return "normal_candidate"
    if group_name.startswith("climb-over-fence"):
        return "intrusion_candidate"
    return "unknown"


def frame_to_sec(frame_idx: int, fps: float) -> float:
    return round(frame_idx / fps, 3)


def resolve_image_path(video_image_dir: Path, video_id: str, frame: dict) -> Path:
    frame_idx = int(frame["frame_idx"])
    candidates = []

    if frame.get("frame_file"):
        candidates.append(video_image_dir / frame["frame_file"])

    candidates.extend([
        video_image_dir / f"{video_id}_t{frame_idx:06d}.jpg",
        video_image_dir / f"{video_id}_t{frame_idx:06d}.png",
        video_image_dir / f"30fps_frame_{frame_idx:03d}.jpg",
        video_image_dir / f"30fps_frame_{frame_idx:03d}.png",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    patterns = [
        f"*t{frame_idx:06d}*",
        f"*{frame_idx:03d}*",
    ]

    for pattern in patterns:
        matches = []
        for ext in IMAGE_EXTS:
            matches.extend(video_image_dir.glob(pattern + ext))
        if matches:
            return sorted(matches)[0]

    raise FileNotFoundError(
        f"Image not found: video_id={video_id}, frame_idx={frame_idx}, dir={video_image_dir}"
    )


def summarize_actions(frame: dict, top_k: int, include_score: bool) -> str:
    actions = []

    for det in frame.get("detections", []):
        for action in det.get("actions", []):
            actions.append({
                "class_name": action.get("class_name", "unknown"),
                "score": float(action.get("score", 0.0)),
            })

    if not actions:
        return "none"

    actions = sorted(actions, key=lambda x: x["score"], reverse=True)
    unique = []
    seen = set()

    for action in actions:
        name = action["class_name"]
        if name in seen:
            continue
        seen.add(name)
        unique.append(action)
        if len(unique) >= top_k:
            break

    if include_score:
        return ", ".join([
            f"{a['class_name']}({a['score']:.2f})"
            for a in unique
        ])

    return ", ".join([a["class_name"] for a in unique])


def discover_video_samples(data_root: Path, groups: list[str]) -> list[dict]:
    samples = []

    for group in groups:
        group_dir = data_root / group
        image_root = group_dir / "images"
        label_root = group_dir / "labels"

        if not image_root.exists():
            print(f"[SKIP] missing image_root: {image_root}")
            continue
        if not label_root.exists():
            print(f"[SKIP] missing label_root: {label_root}")
            continue

        for label_path in sorted(label_root.glob("*.json")):
            video_id = label_path.stem
            video_image_dir = image_root / video_id

            if not video_image_dir.exists():
                print(f"[SKIP] missing image dir: {video_image_dir}")
                continue

            samples.append({
                "group": group,
                "video_id": video_id,
                "label_path": label_path,
                "image_dir": video_image_dir,
                "folder_coarse_label": infer_folder_coarse_label(group),
            })

    return samples


def build_requests_for_video(sample: dict, config: dict) -> list[dict]:
    label_data = load_json(sample["label_path"])
    frames = sorted(label_data.get("frames", []), key=lambda x: int(x["frame_idx"]))

    if not frames:
        return []

    frame_map = {
        int(frame["frame_idx"]): frame
        for frame in frames
    }
    available_indices = sorted(frame_map.keys())
    first_frame = available_indices[0]
    last_frame = available_indices[-1]

    stage1_window = int(config["stage1_window"])
    fps = float(config.get("fps", 30))
    gap = int(config["within_sequence_gap"])
    stride = int(config["request_stride"])
    num_frames = int(config["num_frames_per_request"])
    top_k = int(config["action_top_k"])
    include_score = bool(config["include_action_score"])

    if num_frames != 3:
        raise ValueError("num_frames_per_request must be 3")

    requests = []
    start = first_frame

    while start + gap * 2 <= last_frame:
        selected_indices = [
            start,
            start + gap,
            start + gap * 2,
        ]

        if not all(idx in frame_map for idx in selected_indices):
            start += stride
            continue

        selected_frames = [frame_map[idx] for idx in selected_indices]

        try:
            image_paths = [
                resolve_image_path(
                    video_image_dir=sample["image_dir"],
                    video_id=sample["video_id"],
                    frame=frame,
                )
                for frame in selected_frames
            ]
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            start += stride
            continue

        action_labels = [
            summarize_actions(
                frame=frame,
                top_k=top_k,
                include_score=include_score,
            )
            for frame in selected_frames
        ]
        coverage_start = selected_indices[0] - stage1_window + 1
        coverage_end = selected_indices[-1]
        frame_times_sec = [frame_to_sec(idx, fps) for idx in selected_indices]
        request_id = (
            f"{sample['video_id']}"
            f"_f{selected_indices[0]:06d}"
            f"_{selected_indices[1]:06d}"
            f"_{selected_indices[2]:06d}"
        )

        requests.append({
            "request_id": request_id,
            "group": sample["group"],
            "video_id": sample["video_id"],
            "folder_coarse_label": sample["folder_coarse_label"],
            "frame_indices": selected_indices,
            "frame_times_sec": frame_times_sec,
            "stage1_window": stage1_window,
            "fps": fps,
            "sequence_coverage": {
                "start_frame": coverage_start,
                "end_frame": coverage_end,
            },
            "images": [str(p) for p in image_paths],
            "stage1_actions": action_labels,
            "gt_risk_state": None,
            "gt_source": "not_available",
        })

        start += stride

    return requests


def strongest_risk(states: list[str], unknown_fallback: str = "normal") -> str:
    known_states = [state for state in states if state in RISK_STATES]
    if not known_states:
        return unknown_fallback
    return max(known_states, key=lambda state: RISK_PRIORITY[state])


def parse_label_file(path: Path) -> str:
    states = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            states.append(CLASS_TO_RISK.get(class_id, "normal"))

    return strongest_risk(states)


def load_frame_gt(labeling_root: Path) -> dict[tuple[str, int], str]:
    frame_gt = {}

    for path in labeling_root.glob("**/obj_train_data/**/30fps_frame_*.txt"):
        match = FRAME_RE.search(path.name)
        if not match:
            continue

        video_id = path.parent.name
        frame_idx = int(match.group(1))
        risk_state = parse_label_file(path)
        key = (video_id, frame_idx)

        if key in frame_gt:
            frame_gt[key] = strongest_risk([frame_gt[key], risk_state])
        else:
            frame_gt[key] = risk_state

    return frame_gt


def absent_frame_state(missing_txt_policy: str) -> str:
    if missing_txt_policy == "normal_if_absent":
        return "normal"
    if missing_txt_policy == "unknown_if_absent":
        return "unknown"
    raise ValueError(
        "missing_txt_policy must be 'normal_if_absent' or 'unknown_if_absent'"
    )


def get_frame_state(
    frame_gt: dict[tuple[str, int], str],
    video_id: str,
    frame_idx: int,
    missing_txt_policy: str,
) -> str:
    return frame_gt.get((video_id, frame_idx), absent_frame_state(missing_txt_policy))


def sampled_only_gt(sampled_frame_gt: list[str]) -> str:
    if "danger" in sampled_frame_gt:
        return "danger"
    if "unsafe" in sampled_frame_gt:
        return "unsafe"
    if "unknown" in sampled_frame_gt:
        return "ignore"
    return "normal"


def validate_sequence_coverage(row: dict) -> tuple[int, int]:
    coverage = row.get("sequence_coverage")
    if not isinstance(coverage, dict):
        raise ValueError(f"Missing sequence_coverage: {row.get('request_id')}")

    try:
        start_frame = int(coverage["start_frame"])
        end_frame = int(coverage["end_frame"])
    except KeyError as exc:
        raise ValueError(
            f"sequence_coverage must contain start_frame/end_frame: {row.get('request_id')}"
        ) from exc

    if end_frame < start_frame:
        raise ValueError(f"Invalid sequence_coverage: {row.get('request_id')}")

    return start_frame, end_frame


def summarize_coverage(
    row: dict,
    frame_gt: dict[tuple[str, int], str],
    missing_txt_policy: str,
) -> tuple[int, dict, dict, int]:
    video_id = row["video_id"]
    start_frame, end_frame = validate_sequence_coverage(row)
    total_frames = end_frame - start_frame + 1
    counts = {"normal": 0, "unsafe": 0, "danger": 0}
    unknown_count = 0

    for frame_idx in range(start_frame, end_frame + 1):
        state = get_frame_state(
            frame_gt=frame_gt,
            video_id=video_id,
            frame_idx=frame_idx,
            missing_txt_policy=missing_txt_policy,
        )
        if state == "unknown":
            unknown_count += 1
        else:
            counts[state] += 1

    ratios = {
        state: round(counts[state] / total_frames, 4)
        for state in RISK_STATES
    }

    return total_frames, counts, ratios, unknown_count


def decide_coverage_gt(counts: dict, ratios: dict, thresholds: dict) -> str:
    if ratios["danger"] >= thresholds["danger_threshold"]:
        return "danger"
    if ratios["unsafe"] >= thresholds["unsafe_threshold"]:
        return "unsafe"
    return "normal"


def decide_sequence_gt(
    sampled_gt: list[str],
    counts: dict,
    ratios: dict,
    thresholds: dict,
) -> str:
    if "danger" in sampled_gt:
        return "danger"
    if ratios["danger"] >= thresholds["danger_threshold"]:
        return "danger"
    if "unsafe" in sampled_gt:
        return "unsafe"
    if ratios["unsafe"] >= thresholds["unsafe_threshold"]:
        return "unsafe"
    if "unknown" in sampled_gt:
        return "ignore"
    return "normal"


def attach_gt(
    rows: list[dict],
    frame_gt: dict[tuple[str, int], str],
    config: dict,
) -> tuple[list[dict], list[dict]]:
    sampled_rows = []
    coverage_rows = []
    missing_txt_policy = config["missing_txt_policy"]
    thresholds = {
        "danger_threshold": float(config["danger_threshold"]),
        "unsafe_threshold": float(config["unsafe_threshold"]),
    }

    for row in rows:
        video_id = row["video_id"]
        frame_indices = [int(idx) for idx in row["frame_indices"]]
        sampled_frame_gt = [
            get_frame_state(
                frame_gt=frame_gt,
                video_id=video_id,
                frame_idx=frame_idx,
                missing_txt_policy=missing_txt_policy,
            )
            for frame_idx in frame_indices
        ]
        sampled_state = sampled_only_gt(sampled_frame_gt)

        sampled_row = dict(row)
        sampled_row["sampled_frame_gt"] = sampled_frame_gt
        sampled_row["sampled_only_gt"] = sampled_state
        sampled_row["sequence_gt"] = sampled_state
        sampled_row["gt_risk_state"] = sampled_state
        sampled_row["use_for_eval"] = sampled_state != "ignore"
        sampled_row["needs_review"] = sampled_state == "ignore"
        sampled_row["gt_source"] = "labeling_yolo_sampled_frames"
        sampled_row["gt_rule"] = "sampled_frames_priority_danger_unsafe_normal"
        sampled_rows.append(sampled_row)

        total_frames, counts, ratios, unknown_count = summarize_coverage(
            row=row,
            frame_gt=frame_gt,
            missing_txt_policy=missing_txt_policy,
        )
        coverage_state = decide_coverage_gt(
            counts=counts,
            ratios=ratios,
            thresholds=thresholds,
        )
        sequence_state = decide_sequence_gt(
            sampled_gt=sampled_frame_gt,
            counts=counts,
            ratios=ratios,
            thresholds=thresholds,
        )
        if sequence_state == "normal" and unknown_count > 0:
            sequence_state = "ignore"

        coverage_row = dict(row)
        coverage_row["sampled_frame_gt"] = sampled_frame_gt
        coverage_row["sampled_only_gt"] = sampled_state
        coverage_row["coverage_total_frames"] = total_frames
        coverage_row["coverage_frame_count"] = counts
        coverage_row["coverage_ratio"] = ratios
        coverage_row["coverage_unknown_frames"] = unknown_count
        coverage_row["coverage_based_gt"] = coverage_state
        coverage_row["sequence_gt"] = sequence_state
        coverage_row["gt_risk_state"] = sequence_state
        coverage_row["use_for_eval"] = sequence_state != "ignore"
        coverage_row["needs_review"] = sequence_state == "ignore"
        coverage_row["gt_source"] = "labeling_yolo_sequence_coverage"
        coverage_row["gt_rule"] = (
            "sampled_danger_then_danger_ratio_"
            "sampled_unsafe_then_unsafe_ratio_else_normal"
        )
        coverage_row["gt_config"] = {
            "danger_threshold": thresholds["danger_threshold"],
            "unsafe_threshold": thresholds["unsafe_threshold"],
            "missing_txt_policy": missing_txt_policy,
        }
        coverage_rows.append(coverage_row)

    return sampled_rows, coverage_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_base_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "request_id",
        "group",
        "video_id",
        "folder_coarse_label",
        "frame_indices",
        "frame_times_sec",
        "sequence_coverage",
        "stage1_actions",
        "image_1",
        "image_2",
        "image_3",
        "gt_risk_state",
        "gt_source",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            images = row["images"]
            writer.writerow({
                "request_id": row["request_id"],
                "group": row["group"],
                "video_id": row["video_id"],
                "folder_coarse_label": row["folder_coarse_label"],
                "frame_indices": json.dumps(row["frame_indices"], ensure_ascii=False),
                "frame_times_sec": json.dumps(row["frame_times_sec"], ensure_ascii=False),
                "sequence_coverage": json.dumps(row["sequence_coverage"], ensure_ascii=False),
                "stage1_actions": json.dumps(row["stage1_actions"], ensure_ascii=False),
                "image_1": images[0],
                "image_2": images[1],
                "image_3": images[2],
                "gt_risk_state": row["gt_risk_state"],
                "gt_source": row["gt_source"],
            })


def write_gt_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "request_id",
        "group",
        "video_id",
        "folder_coarse_label",
        "frame_indices",
        "frame_times_sec",
        "sequence_coverage",
        "stage1_actions",
        "sampled_frame_gt",
        "sampled_only_gt",
        "coverage_total_frames",
        "coverage_frame_count",
        "coverage_ratio",
        "coverage_unknown_frames",
        "coverage_based_gt",
        "sequence_gt",
        "gt_risk_state",
        "use_for_eval",
        "needs_review",
        "gt_source",
        "gt_rule",
        "gt_config",
        "image_1",
        "image_2",
        "image_3",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            images = row.get("images", [])
            writer.writerow({
                "request_id": row.get("request_id", ""),
                "group": row.get("group", ""),
                "video_id": row.get("video_id", ""),
                "folder_coarse_label": row.get("folder_coarse_label", ""),
                "frame_indices": json.dumps(row.get("frame_indices", []), ensure_ascii=False),
                "frame_times_sec": json.dumps(row.get("frame_times_sec", []), ensure_ascii=False),
                "sequence_coverage": json.dumps(row.get("sequence_coverage", {}), ensure_ascii=False),
                "stage1_actions": json.dumps(row.get("stage1_actions", []), ensure_ascii=False),
                "sampled_frame_gt": json.dumps(row.get("sampled_frame_gt", []), ensure_ascii=False),
                "sampled_only_gt": row.get("sampled_only_gt", ""),
                "coverage_total_frames": row.get("coverage_total_frames", ""),
                "coverage_frame_count": json.dumps(row.get("coverage_frame_count", {}), ensure_ascii=False),
                "coverage_ratio": json.dumps(row.get("coverage_ratio", {}), ensure_ascii=False),
                "coverage_unknown_frames": row.get("coverage_unknown_frames", ""),
                "coverage_based_gt": row.get("coverage_based_gt", ""),
                "sequence_gt": row.get("sequence_gt", ""),
                "gt_risk_state": row.get("gt_risk_state", ""),
                "use_for_eval": row.get("use_for_eval", ""),
                "needs_review": row.get("needs_review", ""),
                "gt_source": row.get("gt_source", ""),
                "gt_rule": row.get("gt_rule", ""),
                "gt_config": json.dumps(row.get("gt_config", {}), ensure_ascii=False),
                "image_1": images[0] if len(images) > 0 else "",
                "image_2": images[1] if len(images) > 1 else "",
                "image_3": images[2] if len(images) > 2 else "",
            })


def print_gt_summary(name: str, rows: list[dict]) -> None:
    request_counts = Counter(row["sequence_gt"] for row in rows)
    eval_counts = Counter(str(row["use_for_eval"]) for row in rows)
    group_counts = Counter(
        (row.get("group", "unknown"), row["sequence_gt"])
        for row in rows
    )

    print(f"{name} requests:", len(rows))
    print(f"{name} sequence GT distribution:", dict(sorted(request_counts.items())))
    print(f"{name} use_for_eval distribution:", dict(sorted(eval_counts.items())))
    print()
    print(f"{name} sequence GT by group:")
    for (group, state), count in sorted(group_counts.items()):
        print(f"  {group} / {state}: {count}")


def build_base_manifest(config_path: Path, output_jsonl: Path, output_csv: Path) -> list[dict]:
    project_root = Path.cwd()
    config = load_json(config_path)
    data_root = resolve_path(project_root, config["data_root"])
    groups = config["groups"]

    print("Config:", config_path)
    print("Data root:", data_root)
    print("Groups:", groups)

    video_samples = discover_video_samples(data_root, groups)
    print(f"Videos found: {len(video_samples)}")

    all_requests = []

    for sample in video_samples:
        requests = build_requests_for_video(sample, config)
        all_requests.extend(requests)
        print(
            f"{sample['group']} / {sample['video_id']}: "
            f"{len(requests)} requests"
        )

    write_jsonl(output_jsonl, all_requests)
    write_base_summary_csv(output_csv, all_requests)

    print()
    print("Saved base JSONL:", output_jsonl)
    print("Saved base CSV:", output_csv)
    print("Total Stage 2 requests:", len(all_requests))
    print()

    return all_requests


def attach_and_save_gt(
    rows: list[dict],
    config_path: Path,
    labeling_root: Path,
    sampled_jsonl: Path,
    sampled_csv: Path,
    coverage_jsonl: Path,
    coverage_csv: Path,
) -> None:
    config = {**DEFAULT_GT_CONFIG, **load_json(config_path)}
    frame_gt = load_frame_gt(labeling_root)
    sampled_rows, coverage_rows = attach_gt(
        rows=rows,
        frame_gt=frame_gt,
        config=config,
    )

    write_jsonl(sampled_jsonl, sampled_rows)
    write_gt_summary_csv(sampled_csv, sampled_rows)
    write_jsonl(coverage_jsonl, coverage_rows)
    write_gt_summary_csv(coverage_csv, coverage_rows)

    print("Labeling root:", labeling_root)
    print("Loaded labeled frames:", len(frame_gt))
    print()
    print("Saved sampled-only JSONL:", sampled_jsonl)
    print("Saved sampled-only CSV:", sampled_csv)
    print_gt_summary("Sampled-only", sampled_rows)
    print()
    print("Saved coverage JSONL:", coverage_jsonl)
    print("Saved coverage CSV:", coverage_csv)
    print_gt_summary("Coverage-based", coverage_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="benchmark2/configs/stage2_intrusion_config.json",
    )
    parser.add_argument(
        "--labeling-root",
        default="../labeling",
    )
    parser.add_argument(
        "--base-jsonl",
        default="benchmark2/outputs/manifests/stage2_requests.jsonl",
    )
    parser.add_argument(
        "--base-csv",
        default="benchmark2/outputs/manifests/stage2_requests_summary.csv",
    )
    parser.add_argument(
        "--sampled-jsonl",
        default="benchmark2/outputs/manifests/stage2_requests_sampled_gt.jsonl",
    )
    parser.add_argument(
        "--sampled-csv",
        default="benchmark2/outputs/manifests/stage2_requests_sampled_gt_summary.csv",
    )
    parser.add_argument(
        "--coverage-jsonl",
        default="benchmark2/outputs/manifests/stage2_requests_with_gt.jsonl",
    )
    parser.add_argument(
        "--coverage-csv",
        default="benchmark2/outputs/manifests/stage2_requests_with_gt_summary.csv",
    )

    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    labeling_root = Path(args.labeling_root).resolve()
    base_jsonl = Path(args.base_jsonl).resolve()
    base_csv = Path(args.base_csv).resolve()
    sampled_jsonl = Path(args.sampled_jsonl).resolve()
    sampled_csv = Path(args.sampled_csv).resolve()
    coverage_jsonl = Path(args.coverage_jsonl).resolve()
    coverage_csv = Path(args.coverage_csv).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not labeling_root.exists():
        raise FileNotFoundError(f"Labeling root not found: {labeling_root}")

    rows = build_base_manifest(
        config_path=config_path,
        output_jsonl=base_jsonl,
        output_csv=base_csv,
    )
    attach_and_save_gt(
        rows=rows,
        config_path=config_path,
        labeling_root=labeling_root,
        sampled_jsonl=sampled_jsonl,
        sampled_csv=sampled_csv,
        coverage_jsonl=coverage_jsonl,
        coverage_csv=coverage_csv,
    )


if __name__ == "__main__":
    main()



# python3 benchmark2/scripts/prepare_stage2_manifest.py \
#   --config benchmark2/configs/stage2_intrusion_config.json \
#   --labeling-root /home/capstone2/labeling
