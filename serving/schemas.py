from typing import Any, Literal, TypedDict


RiskState = Literal["normal", "unsafe", "danger"]
JobStatus = Literal["queued", "running", "done", "failed"]


class JobStatusPayload(TypedDict, total=False):
    job_id: str
    status: JobStatus
    created_at: str
    started_at: str
    finished_at: str
    updated_at: str
    error: str
    input_video: str
    pipeline_run_root: str


class EventPayload(TypedDict, total=False):
    event_id: str
    video_id: str
    request_id: str
    frame_indices: list[int]
    frame_times_sec: list[float]
    event_time_sec: float
    risk_state: RiskState
    stage1_actions: list[str]
    snapshot_images: list[str]
    overlay_images: list[str]
    latency_sec: float


class ResultPayload(TypedDict, total=False):
    job_id: str
    status: JobStatus
    overall_risk_state: RiskState
    summary: dict[str, int]
    events: list[EventPayload]
    timings: dict[str, Any]
    paths: dict[str, str]


class VlmLogsPayload(TypedDict, total=False):
    job_id: str
    offset: int
    count: int
    next_offset: int
    total: int
    logs: list[dict[str, Any]]
