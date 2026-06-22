# ZroAct AI Serving

This folder is the backend-facing test interface for the ZroAct AI pipeline.
It keeps the existing `benchmark2/` experiment code intact and wraps
`pipeline/main.py` for upload-based backend integration.

## Folder Roles

```text
serving/
├── app.py            # FastAPI server used by the backend
├── run_job.py        # CLI wrapper for one video job
├── config.json       # paths, model, prompt, and runtime options
├── schemas.py        # typed JSON payload shapes
├── utils.py          # job/status/result helpers
├── jobs/             # per-job input, status, result, and pipeline outputs
└── workers/          # reserved for future always-loaded realtime workers
```

## Current MVP Flow

```text
Backend uploads video
-> serving/app.py creates a job
-> serving/run_job.py calls pipeline/main.py
-> pipeline runs Stage 1 + Stage 2
-> serving saves request-level VLM logs
-> backend reads VLM logs through the API
```

This MVP launches the existing pipeline per job. It is good for backend
integration testing. For lower latency, the next phase should replace subprocess
execution with always-loaded Stage 1 / Stage 2 workers.

Stage 1 currently runs with `sample_rate=10` to reduce latency. That means
YOWOv3 processes frames at 10-frame intervals instead of every frame.

## CLI Test

```bash
cd /home/capstone2/zroact-stage2

/home/capstone2/miniconda3/bin/conda run -n qwen35 python serving/run_job.py \
  --video /path/to/input.mp4 \
  --job-id test_001
```

Result:

```text
serving/jobs/test_001/
├── input/
├── status.json
├── result.json
├── vlm_logs.jsonl
├── event_logs.jsonl
└── pipeline_run/
```

## API Server

```bash
cd /home/capstone2/zroact-stage2

/home/capstone2/miniconda3/bin/conda run -n qwen35 \
  uvicorn serving.app:app --host 0.0.0.0 --port 9000
```

## API Test

```bash
curl -X POST http://127.0.0.1:9000/jobs \
  -H "X-API-Key: change-this" \
  -F "file=@/path/to/input.mp4"
```

```bash
curl http://127.0.0.1:9000/jobs/{job_id} \
  -H "X-API-Key: change-this"
```

```bash
curl http://127.0.0.1:9000/jobs/{job_id}/result \
  -H "X-API-Key: change-this"
```

```bash
curl "http://127.0.0.1:9000/jobs/{job_id}/logs?after=0&limit=100" \
  -H "X-API-Key: change-this"
```

```bash
curl http://127.0.0.1:9000/jobs/{job_id}/events \
  -H "X-API-Key: change-this"
```

## Result JSON

`result.json` is the final job summary. It is useful for checking one completed
video job, but it is not the main frontend log stream.

```json
{
  "job_id": "test_001",
  "status": "done",
  "overall_risk_state": "danger",
  "summary": {
    "normal_count": 0,
    "unsafe_count": 0,
    "danger_count": 5,
    "total_requests": 5,
    "event_count": 5
  },
  "timings": {},
  "paths": {}
}
```

## Backend Log Payloads

`vlm_logs.jsonl` contains every Stage 2 VLM request result, including normal,
unsafe, and danger. The backend can read it through `/jobs/{job_id}/logs`.

`event_logs.jsonl` contains only unsafe/danger events. This is what the frontend
would usually show in the event panel.

```text
GET /jobs/{job_id}/logs       # all request-level VLM logs
GET /jobs/{job_id}/vlm-logs   # same as /logs, kept for clarity
GET /jobs/{job_id}/events     # unsafe/danger event logs
GET /jobs/{job_id}/result     # final summary
```

## Realtime Plan

The current MVP is not the final realtime architecture. The next step is:

```text
Stage1Worker: load YOWOv3 once
Stage2Worker: load VLM once
Scheduler: create [t, t+10, t+20] requests from buffers
```

This avoids reloading models for every request and is the path toward
single-GPU near-realtime serving.
