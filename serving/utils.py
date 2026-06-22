import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


RISK_PRIORITY = {
    "normal": 0,
    "unsafe": 1,
    "danger": 2,
}


def utc_like_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    stage2_root = resolve_path(config.get("stage2_root"), config_path.parent)
    if stage2_root is None:
        raise ValueError("config.stage2_root is required")

    resolved = dict(config)
    resolved["config_path"] = str(config_path)
    resolved["stage2_root"] = str(stage2_root)

    for key in [
        "stage1_root",
        "stage1_pretrain_path",
        "conda_bin",
        "prompt",
        "vlm_model_path",
        "jobs_root",
    ]:
        if key in resolved and resolved[key] is not None:
            resolved[key] = str(resolve_path(str(resolved[key]), stage2_root))

    return resolved


def make_job_id(prefix: str = "job") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def job_dir(config: dict[str, Any], job_id: str) -> Path:
    return Path(config["jobs_root"]) / job_id


def status_path(job_path: Path) -> Path:
    return job_path / "status.json"


def result_path(job_path: Path) -> Path:
    return job_path / "result.json"


def vlm_logs_path(job_path: Path) -> Path:
    return job_path / "vlm_logs.jsonl"


def event_logs_path(job_path: Path) -> Path:
    return job_path / "event_logs.jsonl"


def update_status(job_path: Path, **updates: Any) -> dict[str, Any]:
    path = status_path(job_path)
    payload = load_json(path) if path.exists() else {}
    payload.update(updates)
    payload["updated_at"] = utc_like_now()
    write_json(path, payload)
    return payload


def parse_cell(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return ""
    if text == "True":
        return True
    if text == "False":
        return False
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({key: parse_cell(value) for key, value in row.items()})
        return rows


def strongest_risk(states: list[str]) -> str:
    if not states:
        return "normal"
    return max(states, key=lambda state: RISK_PRIORITY.get(state, -1))


def copy_video_to_job(video: Path, job_path: Path) -> Path:
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    input_path = job_path / "input" / video.name
    input_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.resolve() != video.resolve():
        shutil.copy2(video, input_path)
    return input_path


def compact_log(row: dict[str, Any]) -> dict[str, Any]:
    frame_times = row.get("frame_times_sec") or []
    time_sec = (
        row.get("event_time_sec")
        or row.get("status_time_sec")
        or (frame_times[-1] if isinstance(frame_times, list) and frame_times else None)
    )
    payload = {
        "video_id": row.get("video_id"),
        "request_id": row.get("request_id"),
        "frame_indices": row.get("frame_indices") or [],
        "time_sec": time_sec,
        "risk_state": row.get("risk_state"),
        "actions": row.get("stage1_actions") or [],
        "images": row.get("snapshot_images") or [],
    }
    if row.get("event_id"):
        payload["event_id"] = row.get("event_id")
    if row.get("overlay_images"):
        payload["overlay_images"] = row.get("overlay_images")
    if row.get("latency_sec") not in {None, ""}:
        payload["latency_sec"] = row.get("latency_sec")
    return payload


def build_result(job_id: str, job_path: Path, run_root: Path) -> dict[str, Any]:
    final_dir = run_root / "final"
    status_rows = read_csv_rows(final_dir / "status_logs.csv")
    event_rows = read_csv_rows(final_dir / "risk_logs.csv")
    timings = load_json(final_dir / "timings.json") if (final_dir / "timings.json").exists() else {}

    logs = [compact_log(row) for row in status_rows]
    events = [compact_log(row) for row in event_rows]

    write_jsonl(vlm_logs_path(job_path), logs)
    write_jsonl(event_logs_path(job_path), events)

    all_states = [
        str(row.get("risk_state"))
        for row in status_rows
        if row.get("risk_state") in RISK_PRIORITY
    ]
    event_states = [
        str(row.get("risk_state"))
        for row in event_rows
        if row.get("risk_state") in RISK_PRIORITY
    ]
    overall = strongest_risk(event_states or all_states)

    summary = {
        "normal_count": sum(1 for state in all_states if state == "normal"),
        "unsafe_count": sum(1 for state in all_states if state == "unsafe"),
        "danger_count": sum(1 for state in all_states if state == "danger"),
        "total_requests": len(status_rows),
        "event_count": len(events),
    }

    return {
        "job_id": job_id,
        "status": "done",
        "overall_risk_state": overall,
        "summary": summary,
        "logs": logs,
        "events": events,
        "timings": timings,
        "paths": {
            "job_dir": str(job_path),
            "pipeline_run_root": str(run_root),
            "vlm_logs_jsonl": str(vlm_logs_path(job_path)),
            "event_logs_jsonl": str(event_logs_path(job_path)),
            "status_logs_csv": str(final_dir / "status_logs.csv"),
            "risk_logs_csv": str(final_dir / "risk_logs.csv"),
            "overlay_dir": str(final_dir / "overlay_images"),
        },
    }
