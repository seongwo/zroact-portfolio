import argparse
import csv
import json
from collections import Counter
from pathlib import Path


RISK_STATES = ["normal", "unsafe", "danger"]
RISK_PRIORITY = {"normal": 0, "unsafe": 1, "danger": 2}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def coverage_majority_gt(counts: dict, tie_break: str) -> str:
    safe_counts = {
        state: int(counts.get(state, 0) or 0)
        for state in RISK_STATES
    }

    max_count = max(safe_counts.values())
    if max_count <= 0:
        return "ignore"

    candidates = [
        state for state, count in safe_counts.items()
        if count == max_count
    ]

    if len(candidates) == 1:
        return candidates[0]

    if tie_break == "higher_risk":
        return max(candidates, key=lambda state: RISK_PRIORITY[state])
    if tie_break == "lower_risk":
        return min(candidates, key=lambda state: RISK_PRIORITY[state])

    raise ValueError("--tie-break must be higher_risk or lower_risk")


def summarize_distribution(name: str, rows: list[dict], key: str) -> None:
    counter = Counter(row.get(key, "") for row in rows)
    print(f"{name}: {dict(sorted(counter.items()))}")


def compare_keys(rows: list[dict], left: str, right: str) -> list[dict]:
    return [
        row for row in rows
        if row.get(left) != row.get(right)
    ]


def write_comparison_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "request_id",
        "group",
        "video_id",
        "frame_indices",
        "sampled_frame_gt",
        "coverage_frame_count",
        "coverage_ratio",
        "sampled_only_gt",
        "coverage_based_gt",
        "sequence_gt",
        "coverage_majority_gt",
        "sequence_vs_majority_match",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "request_id": row.get("request_id", ""),
                "group": row.get("group", ""),
                "video_id": row.get("video_id", ""),
                "frame_indices": json.dumps(row.get("frame_indices", []), ensure_ascii=False),
                "sampled_frame_gt": json.dumps(row.get("sampled_frame_gt", []), ensure_ascii=False),
                "coverage_frame_count": json.dumps(row.get("coverage_frame_count", {}), ensure_ascii=False),
                "coverage_ratio": json.dumps(row.get("coverage_ratio", {}), ensure_ascii=False),
                "sampled_only_gt": row.get("sampled_only_gt", ""),
                "coverage_based_gt": row.get("coverage_based_gt", ""),
                "sequence_gt": row.get("sequence_gt", ""),
                "coverage_majority_gt": row.get("coverage_majority_gt", ""),
                "sequence_vs_majority_match": row.get("sequence_gt") == row.get("coverage_majority_gt"),
            })


def evaluate_results_under_gt(results_csv: Path, gt_rows: list[dict]) -> None:
    result_rows = load_csv(results_csv)
    gt_by_id = {row["request_id"]: row for row in gt_rows}
    joined = []
    for result in result_rows:
        gt_row = gt_by_id.get(result.get("request_id"))
        if not gt_row:
            continue
        pred = result.get("pred_risk_state")
        if pred not in RISK_STATES:
            continue
        joined.append((pred, gt_row))

    print()
    print("Result evaluation:", results_csv)
    print("Evaluated rows:", len(joined))
    for gt_key in ["sequence_gt", "coverage_majority_gt"]:
        if not joined:
            continue
        correct = sum(1 for pred, gt_row in joined if pred == gt_row.get(gt_key))
        print(f"Accuracy vs {gt_key}: {correct}/{len(joined)} ({correct / len(joined):.4f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="benchmark2/outputs/manifests/stage2_requests_with_gt.jsonl",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark2/outputs/manifests/stage2_gt_rule_comparison.csv",
    )
    parser.add_argument(
        "--tie-break",
        choices=["higher_risk", "lower_risk"],
        default="higher_risk",
    )
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--show-diffs", type=int, default=20)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_csv = Path(args.output_csv).resolve()
    rows = load_jsonl(manifest_path)

    for row in rows:
        row["coverage_majority_gt"] = coverage_majority_gt(
            row.get("coverage_frame_count", {}),
            tie_break=args.tie_break,
        )

    print("Manifest:", manifest_path)
    print("Rows:", len(rows))
    print("Tie break:", args.tie_break)
    print()
    summarize_distribution("sampled_only_gt", rows, "sampled_only_gt")
    summarize_distribution("coverage_based_gt", rows, "coverage_based_gt")
    summarize_distribution("sequence_gt", rows, "sequence_gt")
    summarize_distribution("coverage_majority_gt", rows, "coverage_majority_gt")

    print()
    for left, right in [
        ("sampled_only_gt", "coverage_majority_gt"),
        ("coverage_based_gt", "coverage_majority_gt"),
        ("sequence_gt", "coverage_majority_gt"),
        ("sampled_only_gt", "sequence_gt"),
    ]:
        diffs = compare_keys(rows, left, right)
        print(f"{left} != {right}: {len(diffs)}")

    diffs = compare_keys(rows, "sequence_gt", "coverage_majority_gt")
    if diffs and args.show_diffs:
        print()
        print(f"Examples where sequence_gt != coverage_majority_gt (first {args.show_diffs}):")
        for row in diffs[:args.show_diffs]:
            print(
                row.get("request_id"),
                "frames=", row.get("frame_indices"),
                "sampled=", row.get("sampled_frame_gt"),
                "counts=", row.get("coverage_frame_count"),
                "ratios=", row.get("coverage_ratio"),
                "sequence_gt=", row.get("sequence_gt"),
                "majority=", row.get("coverage_majority_gt"),
            )

    write_comparison_csv(output_csv, rows)
    print()
    print("Saved comparison CSV:", output_csv)

    if args.results_csv:
        evaluate_results_under_gt(Path(args.results_csv).resolve(), rows)


if __name__ == "__main__":
    main()
