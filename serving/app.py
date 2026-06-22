import shutil
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile

from serving.run_job import run_job
from serving.utils import (
    event_logs_path,
    job_dir,
    load_config,
    load_json,
    make_job_id,
    read_jsonl,
    result_path,
    status_path,
    utc_like_now,
    vlm_logs_path,
    write_json,
)


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
CONFIG = load_config(CONFIG_PATH)

app = FastAPI(
    title="ZroAct AI Serving API",
    description="Backend-facing API for Stage 1 + Stage 2 intrusion inference jobs.",
    version="0.1.0",
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = CONFIG.get("api_key")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def run_background_job(video_path: str, job_id: str) -> None:
    run_job(
        video=Path(video_path),
        job_id=job_id,
        config_path=CONFIG_PATH,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "zroact-ai-serving",
        "status": "ok",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "stage2_root": CONFIG["stage2_root"],
        "model": CONFIG["vlm_model_path"],
        "prompt": CONFIG["prompt"],
    }


@app.post("/jobs", dependencies=[Depends(require_api_key)])
def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, str]:
    job_id = make_job_id()
    path = job_dir(CONFIG, job_id)
    input_dir = path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(file.filename or "input.mp4").name
    input_path = input_dir / filename

    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    write_json(
        status_path(path),
        {
            "job_id": job_id,
            "status": "queued",
            "created_at": utc_like_now(),
            "input_video": str(input_path),
        },
    )

    background_tasks.add_task(run_background_job, str(input_path), job_id)

    return {
        "job_id": job_id,
        "status": "queued",
    }


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> dict[str, Any]:
    path = job_dir(CONFIG, job_id)
    status_file = status_path(path)
    if not status_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return load_json(status_file)


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_api_key)])
def get_result(job_id: str) -> dict[str, Any]:
    path = job_dir(CONFIG, job_id)
    result_file = result_path(path)
    if not result_file.exists():
        if status_path(path).exists():
            status = load_json(status_path(path))
            raise HTTPException(status_code=202, detail=status)
        raise HTTPException(status_code=404, detail="Job not found")
    return load_json(result_file)


@app.get("/jobs/{job_id}/vlm-logs", dependencies=[Depends(require_api_key)])
def get_vlm_logs(
    job_id: str,
    after: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    path = job_dir(CONFIG, job_id)
    logs_file = vlm_logs_path(path)
    if not logs_file.exists():
        if status_path(path).exists():
            status = load_json(status_path(path))
            raise HTTPException(status_code=202, detail=status)
        raise HTTPException(status_code=404, detail="Job not found")

    logs = read_jsonl(logs_file)
    after = max(after, 0)
    limit = max(min(limit, 1000), 1)
    sliced = logs[after : after + limit]
    return {
        "job_id": job_id,
        "offset": after,
        "count": len(sliced),
        "next_offset": after + len(sliced),
        "total": len(logs),
        "logs": sliced,
    }


@app.get("/jobs/{job_id}/logs", dependencies=[Depends(require_api_key)])
def get_logs(
    job_id: str,
    after: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    return get_vlm_logs(job_id=job_id, after=after, limit=limit)


@app.get("/jobs/{job_id}/events", dependencies=[Depends(require_api_key)])
def get_events(
    job_id: str,
    after: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    path = job_dir(CONFIG, job_id)
    events_file = event_logs_path(path)
    if not events_file.exists():
        if status_path(path).exists():
            status = load_json(status_path(path))
            raise HTTPException(status_code=202, detail=status)
        raise HTTPException(status_code=404, detail="Job not found")

    events = read_jsonl(events_file)
    after = max(after, 0)
    limit = max(min(limit, 1000), 1)
    sliced = events[after : after + limit]

    result_file = result_path(path)
    result = load_json(result_file) if result_file.exists() else {}
    return {
        "job_id": job_id,
        "status": result.get("status"),
        "overall_risk_state": result.get("overall_risk_state"),
        "offset": after,
        "count": len(sliced),
        "next_offset": after + len(sliced),
        "total": len(events),
        "events": sliced,
    }
