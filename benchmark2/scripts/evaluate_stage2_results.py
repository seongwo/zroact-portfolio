import argparse
import csv
import json
from collections import Counter
from pathlib import Path


RISK_STATES = ["normal", "unsafe", "danger"]


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).strip().lower() not in {"false", "0", "no"}


def load_gt(path: Path) -> tuple[dict[str, str], set[str]]:
    gt = {}
    ignored = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not as_bool(row.get("use_for_eval", True)):
                ignored.add(row["request_id"])
                continue
            state = row.get("sequence_gt", row.get("gt_risk_state"))
            if state in RISK_STATES:
                gt[row["request_id"]] = state
    return gt, ignored


def load_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_joined(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-manifest",
        default="benchmark2/outputs/manifests/stage2_requests_with_gt.jsonl",
    )
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    gt_manifest = Path(args.gt_manifest).resolve()
    results_csv = Path(args.results_csv).resolve()

    gt, ignored_gt = load_gt(gt_manifest)
    predictions = load_predictions(results_csv)

    joined = []
    confusion = Counter()
    gt_counts = Counter()
    pred_counts = Counter()
    missing_gt = 0
    ignored_predictions = 0

    for row in predictions:
        request_id = row.get("request_id")
        gt_state = gt.get(request_id)
        pred_state = row.get("pred_risk_state") or "unknown"

        if gt_state is None:
            if request_id in ignored_gt:
                ignored_predictions += 1
                continue
            missing_gt += 1
            continue

        joined_row = dict(row)
        joined_row["sequence_gt"] = gt_state
        joined_row["gt_risk_state"] = gt_state
        joined_row["gt_match"] = pred_state == gt_state
        joined.append(joined_row)

        confusion[(gt_state, pred_state)] += 1
        gt_counts[gt_state] += 1
        pred_counts[pred_state] += 1

    correct = sum(
        count
        for (gt_state, pred_state), count in confusion.items()
        if gt_state == pred_state
    )
    total = len(joined)
    accuracy = correct / total if total else 0.0

    print("GT manifest:", gt_manifest)
    print("Results CSV:", results_csv)
    print("Evaluated rows:", total)
    print("Missing GT rows:", missing_gt)
    print("Ignored GT rows:", len(ignored_gt))
    print("Ignored prediction rows:", ignored_predictions)
    print("Accuracy:", f"{correct}/{total}", f"({accuracy:.4f})")
    print("GT distribution:", dict(sorted(gt_counts.items())))
    print("Prediction distribution:", dict(sorted(pred_counts.items())))
    print()
    print("Confusion matrix: GT -> Pred")
    for gt_state in RISK_STATES:
        values = {
            pred_state: confusion[(gt_state, pred_state)]
            for pred_state in [*RISK_STATES, "unknown"]
            if confusion[(gt_state, pred_state)]
        }
        print(f"  {gt_state}: {values}")

    if args.output_csv:
        output_csv = Path(args.output_csv).resolve()
        write_joined(output_csv, joined)
        print()
        print("Saved joined CSV:", output_csv)


if __name__ == "__main__":
    main()


# python3 benchmark2/scripts/evaluate_stage2_results.py \
#   --gt-manifest benchmark2/outputs/manifests/stage2_requests_with_gt.jsonl \
#   --results-csv benchmark2/results/qwen35_2b_v2_gt_all/summary.csv \
#   --output-csv benchmark2/results/qwen35_2b_v2_gt_all/summary_with_gt.csv