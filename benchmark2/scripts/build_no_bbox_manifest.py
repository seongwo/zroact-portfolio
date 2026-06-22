import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
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
        "group",
        "video_id",
        "frame_indices",
        "image_1",
        "image_2",
        "image_3",
        "sequence_gt",
        "gt_risk_state",
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
                "frame_indices": json.dumps(row.get("frame_indices", []), ensure_ascii=False),
                "image_1": images[0] if len(images) > 0 else "",
                "image_2": images[1] if len(images) > 1 else "",
                "image_3": images[2] if len(images) > 2 else "",
                "sequence_gt": row.get("sequence_gt", ""),
                "gt_risk_state": row.get("gt_risk_state", ""),
            })


def build_no_bbox_rows(rows: list[dict], no_bbox_root: Path) -> list[dict]:
    output_rows = []
    missing = []

    for row in rows:
        group = row["group"]
        video_id = row["video_id"]
        frame_indices = [int(idx) for idx in row["frame_indices"]]

        image_paths = [
            no_bbox_root / group / video_id / f"30fps_frame_{frame_idx:03d}.jpg"
            for frame_idx in frame_indices
        ]

        for path in image_paths:
            if not path.exists():
                missing.append(str(path))

        updated = dict(row)
        updated["images"] = [str(path.resolve()) for path in image_paths]
        updated["image_source"] = "no_bbox_frames"
        output_rows.append(updated)

    if missing:
        shown = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"Missing {len(missing)} no-bbox images. First missing paths:\n{shown}"
        )

    return output_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="benchmark2/outputs/manifests/stage2_requests_with_gt.jsonl",
    )
    parser.add_argument(
        "--no-bbox-root",
        default="benchmark2/data2/no_bbox_frames",
    )
    parser.add_argument(
        "--output-jsonl",
        default="benchmark2/outputs/manifests/stage2_requests_with_gt_no_bbox.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark2/outputs/manifests/stage2_requests_with_gt_no_bbox_summary.csv",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    no_bbox_root = Path(args.no_bbox_root).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    output_csv = Path(args.output_csv).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not no_bbox_root.exists():
        raise FileNotFoundError(f"No-bbox root not found: {no_bbox_root}")

    rows = read_jsonl(manifest_path)
    no_bbox_rows = build_no_bbox_rows(rows, no_bbox_root)
    write_jsonl(output_jsonl, no_bbox_rows)
    write_csv(output_csv, no_bbox_rows)

    print("Input manifest:", manifest_path)
    print("No-bbox root:", no_bbox_root)
    print("Saved JSONL:", output_jsonl)
    print("Saved CSV:", output_csv)
    print("Rows:", len(no_bbox_rows))
    print("Image refs checked:", len(no_bbox_rows) * 3)


if __name__ == "__main__":
    main()
