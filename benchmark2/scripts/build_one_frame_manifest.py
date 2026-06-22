import argparse
import csv
import json
from collections import Counter
from pathlib import Path


RISK_STATES = ["normal", "unsafe", "danger"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "request_id",
        "parent_request_id",
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
        "coverage_based_gt",
        "sequence_gt",
        "gt_risk_state",
        "use_for_eval",
        "needs_review",
        "gt_source",
        "gt_rule",
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
                "parent_request_id": row.get("parent_request_id", ""),
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
                "coverage_based_gt": row.get("coverage_based_gt", ""),
                "sequence_gt": row.get("sequence_gt", ""),
                "gt_risk_state": row.get("gt_risk_state", ""),
                "use_for_eval": row.get("use_for_eval", ""),
                "needs_review": row.get("needs_review", ""),
                "gt_source": row.get("gt_source", ""),
                "gt_rule": row.get("gt_rule", ""),
                "image_1": images[0] if len(images) > 0 else "",
                "image_2": images[1] if len(images) > 1 else "",
                "image_3": images[2] if len(images) > 2 else "",
            })


def one_frame_row(row: dict, idx: int) -> dict:
    frame_idx = int(row["frame_indices"][idx])
    frame_time = float(row["frame_times_sec"][idx])
    action = row.get("stage1_actions", ["none", "none", "none"])[idx]
    image = row["images"][idx]
    frame_gt = row.get("sampled_frame_gt", [None, None, None])[idx]

    if frame_gt not in RISK_STATES:
        sequence_gt = "ignore"
    else:
        sequence_gt = frame_gt

    counts = {state: 0 for state in RISK_STATES}
    if sequence_gt in counts:
        counts[sequence_gt] = 1

    ratios = {
        state: round(counts[state] / 1, 4)
        for state in RISK_STATES
    }

    return {
        "request_id": f"{row['video_id']}_f{frame_idx:06d}",
        "parent_request_id": row.get("request_id"),
        "group": row.get("group"),
        "video_id": row.get("video_id"),
        "folder_coarse_label": row.get("folder_coarse_label"),
        "frame_indices": [frame_idx],
        "frame_times_sec": [round(frame_time, 3)],
        "stage1_window": row.get("stage1_window"),
        "fps": row.get("fps"),
        "sequence_coverage": {
            "start_frame": frame_idx,
            "end_frame": frame_idx,
        },
        "original_sequence_coverage": row.get("sequence_coverage"),
        "images": [image],
        "stage1_actions": [action],
        "sampled_frame_gt": [frame_gt],
        "sampled_only_gt": sequence_gt,
        "coverage_total_frames": 1,
        "coverage_frame_count": counts,
        "coverage_ratio": ratios,
        "coverage_based_gt": sequence_gt,
        "sequence_gt": sequence_gt,
        "gt_risk_state": sequence_gt,
        "use_for_eval": sequence_gt != "ignore",
        "needs_review": sequence_gt == "ignore",
        "gt_source": "labeling_yolo_single_frame",
        "gt_rule": "single_frame_gt_from_sampled_frame_label",
    }


def build_one_frame_rows(rows: list[dict]) -> list[dict]:
    output = []
    seen = set()

    for row in rows:
        frame_indices = row.get("frame_indices", [])
        images = row.get("images", [])
        frame_times = row.get("frame_times_sec", [])
        actions = row.get("stage1_actions", [])
        sampled_gt = row.get("sampled_frame_gt", [])

        if not (
            len(frame_indices)
            == len(images)
            == len(frame_times)
            == len(actions)
            == len(sampled_gt)
        ):
            raise ValueError(f"Invalid row lengths: {row.get('request_id')}")

        for idx in range(len(frame_indices)):
            new_row = one_frame_row(row, idx)
            key = new_row["request_id"]
            if key in seen:
                continue
            seen.add(key)
            output.append(new_row)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="benchmark2/outputs/manifests/stage2_requests_with_gt.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default="benchmark2/outputs/manifests/stage2_requests_1frame_with_gt.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark2/outputs/manifests/stage2_requests_1frame_with_gt_summary.csv",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    output_csv = Path(args.output_csv).resolve()

    rows = load_jsonl(manifest_path)
    one_frame_rows = build_one_frame_rows(rows)

    write_jsonl(output_jsonl, one_frame_rows)
    write_csv(output_csv, one_frame_rows)

    print("Source manifest:", manifest_path)
    print("Saved JSONL:", output_jsonl)
    print("Saved CSV:", output_csv)
    print("Total 1-frame requests:", len(one_frame_rows))
    print("GT distribution:", dict(sorted(Counter(row["sequence_gt"] for row in one_frame_rows).items())))


if __name__ == "__main__":
    main()


# cd /home/capstone2/zroact-stage2
# python3 benchmark2/scripts/build_one_frame_manifest.py
