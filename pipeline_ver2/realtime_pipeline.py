import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import requests
import asyncio
import aiohttp
from PIL import Image

# Add project root to python path
project_dir = Path(__file__).resolve().parents[1]
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

from pipeline_ver2.main import (
    project_root,
    default_conda_bin,
    timestamp_run_id,
    resolve_path,
    prepare_input,
    extract_30fps,
    discover_frame_video_dirs,
    resolve_stage1_image,
    summarize_actions,
    write_timings,
    overlay_risk_badge,
    RISK_STATES
)

def default_stage1_root() -> Path:
    return Path("/home/capstone2/zroact-stage1/YOWOv3")

def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def wait_for_server(url: str, timeout: int = 60) -> bool:
    """Wait for FastAPI server to respond with status ok."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200 and r.json().get("status") in ("ok", "loading"):
                # If status is ok, we are ready. If status is loading, wait a bit more.
                if r.json().get("status") == "ok":
                    return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.0)
    return False

def build_paths(args) -> dict[str, Path]:
    root = args.stage2_root
    run_root = args.output_root / args.run_id
    return {
        "root": root,
        "run_root": run_root,
        "input_dir": run_root / "input",
        "frames_dir": args.frames_dir if args.frames_dir else run_root / "frames_30fps",
        "stage1_dir": run_root / "stage1",
        "stage2_dir": run_root / "stage2",
        "final_dir": run_root / "final",
        "overlay_dir": run_root / "final" / "overlay_images",
        "status_logs_jsonl": run_root / "final" / "status_logs.jsonl",
        "status_logs_csv": run_root / "final" / "status_logs.csv",
        "risk_logs_jsonl": run_root / "final" / "risk_logs.jsonl",
        "risk_logs_csv": run_root / "final" / "risk_logs.csv",
        "timings_json": run_root / "final" / "timings.json",
        "timings_csv": run_root / "final" / "timings.csv",
    }

async def detect_clip_batch(session, url, clips, conf_thresh, top_k):
    payload = {
        "clips": clips,
        "conf_threshold": conf_thresh,
        "top_k": top_k,
        "use_blacklist": True
    }
    async with session.post(url, json=payload) as resp:
        if resp.status == 200:
            return await resp.json()
        else:
            txt = await resp.text()
            raise RuntimeError(f"Stage 1 detector server error (status {resp.status}): {txt}")

async def evaluate_vlm(session, url, req_data, semaphore):
    async with semaphore:
        async with session.post(url, json=req_data) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                txt = await resp.text()
                raise RuntimeError(f"Stage 2 VLM server error (status {resp.status}): {txt}")

def frame_to_sec(frame_idx: int, fps: float) -> float:
    return round(frame_idx / fps, 3)

async def run_pipeline_async(args, paths, config, timings):
    # Determine input folders
    video_dirs = discover_frame_video_dirs(paths["frames_dir"])
    if args.frame_video_filter:
        video_dirs = [
            vdir for vdir in video_dirs
            if args.frame_video_filter in str(vdir.relative_to(paths["frames_dir"]))
        ]
    if args.max_frame_videos is not None:
        video_dirs = video_dirs[:args.max_frame_videos]

    if not video_dirs:
        raise RuntimeError("No frame video folders found to process.")

    stage1_url = f"http://{config['stage1_daemon_host']}:{config['stage1_daemon_port']}"
    stage2_url = f"http://{config['stage2_daemon_host']}:{config['stage2_daemon_port']}"

    print(f"Connecting to Stage 1 at: {stage1_url}")
    print(f"Connecting to Stage 2 at: {stage2_url}")

    # Set up VLM concurrency limit (usually 1 or 2 to avoid VRAM OOM)
    vlm_semaphore = asyncio.Semaphore(3)

    all_status_rows = []
    all_events = []

    os.makedirs(paths["stage1_dir"] / "labels", exist_ok=True)
    os.makedirs(paths["stage2_dir"], exist_ok=True)
    os.makedirs(paths["overlay_dir"], exist_ok=True)

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        for video_dir in video_dirs:
            video_id = video_dir.name
            print(f"\nProcessing video stream for: {video_id}")
            
            # Find frames
            frame_paths = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
            num_frames = len(frame_paths)
            if num_frames == 0:
                print(f"  No frames found in {video_dir}, skipping.")
                continue

            # Compute keyframe indices (AVA standard: starting at clip_length, step by sample_rate)
            clip_length = config.get("stage1_window", 16)
            sampling_rate = config.get("stage1_sample_rate", 10)
            keyframe_indices = list(range(clip_length, num_frames + 1, sampling_rate))
            if not keyframe_indices:
                keyframe_indices = [num_frames]

            # ─── 1. Run Stage 1 & Stage 2 Pipelined ───
            print(f"  Running Pipelined Inference (Stage 1 & Stage 2)...")
            
            # Group keyframe indices into batches
            batch_size = args.stage1_batch_size
            keyframe_batches = [keyframe_indices[i:i + batch_size] for i in range(0, len(keyframe_indices), batch_size)]
            
            predictions = {} # key_idx -> detections list
            vlm_tasks = []
            vlm_requests_map = {}
            scheduled_starts = set()
            
            gap = args.stage2_gap
            stride = args.stage2_stride
            first_frame = keyframe_indices[0]
            last_frame = keyframe_indices[-1]
            frame_map = {idx: frame_paths[idx - 1] for idx in range(1, num_frames + 1)}
            
            s1_start = time.perf_counter()
            s2_start = time.perf_counter()
            
            for kf_batch in keyframe_batches:
                clips_payload = []
                for key_idx in kf_batch:
                    clip_frame_paths = []
                    for i in reversed(range(clip_length)):
                        cur_idx = key_idx - i * sampling_rate
                        if cur_idx < 1:
                            cur_idx = 1
                        clip_frame_paths.append(str(frame_paths[cur_idx - 1]))
                    clips_payload.append(clip_frame_paths)
                
                # Send to Stage 1 Daemon
                resp_data = await detect_clip_batch(
                    session=session,
                    url=f"{stage1_url}/detect",
                    clips=clips_payload,
                    conf_thresh=args.stage1_conf_threshold,
                    top_k=args.action_top_k
                )
                
                # Parse results
                for idx, key_idx in enumerate(kf_batch):
                    predictions[key_idx] = resp_data["results"][idx]
                
                # Pipelining: Check if we can schedule any new Stage 2 VLM requests
                start = first_frame
                while start + gap * 2 <= last_frame:
                    selected_indices = [start, start + gap, start + gap * 2]
                    if start not in scheduled_starts and all(idx in predictions for idx in selected_indices):
                        scheduled_starts.add(start)
                        
                        image_paths = [str(frame_map[idx]) for idx in selected_indices]
                        
                        # Format Stage 1 action summary texts
                        actions_summary = []
                        for idx in selected_indices:
                            dets = predictions[idx]
                            if not dets:
                                actions_summary.append("none")
                                continue
                            flat_actions = []
                            for det in dets:
                                for act in det.get("actions", []):
                                    flat_actions.append(act)
                            flat_actions.sort(key=lambda x: x["score"], reverse=True)
                            
                            unique_actions = []
                            seen = set()
                            for act in flat_actions:
                                name = act["class_name"]
                                if name not in seen:
                                    seen.add(name)
                                    unique_actions.append(act)
                                    if len(unique_actions) >= args.action_top_k:
                                        break
                            if args.include_action_score:
                                summary_str = ", ".join(f"{a['class_name']}({a['score']:.2f})" for a in unique_actions)
                            else:
                                summary_str = ", ".join(a['class_name'] for a in unique_actions)
                            
                            actions_summary.append(summary_str if summary_str else "none")

                        coverage_start = selected_indices[0] - clip_length + 1
                        coverage_end = selected_indices[-1]
                        request_id = f"{video_id}_f{selected_indices[0]:06d}_{selected_indices[1]:06d}_{selected_indices[2]:06d}"
                        
                        req = {
                            "request_id": request_id,
                            "video_id": video_id,
                            "images": image_paths,
                            "stage1_actions": actions_summary,
                            "frame_indices": selected_indices,
                            "frame_times_sec": [frame_to_sec(idx, args.fps) for idx in selected_indices],
                            "sequence_coverage": {
                                "start_frame": coverage_start,
                                "end_frame": coverage_end,
                            }
                        }
                        
                        api_payload = {
                            "request_id": req["request_id"],
                            "video_id": req["video_id"],
                            "images": req["images"],
                            "stage1_actions": req["stage1_actions"],
                            "frame_indices": req["frame_indices"],
                            "frame_times_sec": req["frame_times_sec"],
                            "fps": args.fps,
                            "max_new_tokens": args.max_new_tokens
                        }
                        
                        task = asyncio.create_task(
                            evaluate_vlm(session, f"{stage2_url}/evaluate", api_payload, vlm_semaphore)
                        )
                        vlm_tasks.append(task)
                        vlm_requests_map[task] = req
                        
                    start += stride

            s1_elapsed = time.perf_counter() - s1_start
            timings[f"stage1_yowov3_{video_id}"] = s1_elapsed
            print(f"  Stage 1 finished in {s1_elapsed:.4f}s ({s1_elapsed/len(keyframe_indices)*1000:.2f} ms per clip)")

            # Save simulated stage1 labels JSON
            json_records = []
            for key_idx in sorted(predictions.keys()):
                frame_filename = frame_paths[key_idx - 1].name
                dets = predictions[key_idx]
                detections_out = []
                for det in dets:
                    detections_out.append({
                        'box': det['box'],
                        'actions': det['actions']
                    })
                json_records.append({
                    'frame_idx': key_idx,
                    'frame_file': frame_filename,
                    'detections': detections_out
                })
            
            stage1_json_path = paths["stage1_dir"] / "labels" / f"{video_id}.json"
            stage1_json_path.parent.mkdir(parents=True, exist_ok=True)
            with stage1_json_path.open("w", encoding="utf-8") as jf:
                json.dump({'video': video_id, 'frames': json_records}, jf, indent=2)

            print(f"  Generated {len(vlm_tasks)} Stage 2 requests.")
            if not vlm_tasks:
                print("  No VLM requests generated, skipping Stage 2.")
                continue

            # ─── 3. Wait for Stage 2 VLM Evaluation (Async Parallel) ───
            print(f"  Waiting for {len(vlm_tasks)} requests to complete Stage 2 VLM Daemon...")
            done_count = 0
            total_tasks = len(vlm_tasks)

            def _on_done(fut):
                nonlocal done_count
                done_count += 1
                print(f"  [{done_count}/{total_tasks}] Stage 2 request done", flush=True)

            for t in vlm_tasks:
                t.add_done_callback(_on_done)

            vlm_results = await asyncio.gather(*vlm_tasks, return_exceptions=True)
            s2_elapsed = time.perf_counter() - s2_start
            timings[f"stage2_vlm_{video_id}"] = s2_elapsed
            print(f"  Stage 2 finished in {s2_elapsed:.4f}s ({s2_elapsed/len(vlm_tasks):.3f}s per VLM eval)")

            # ─── 4. Process Results & Save Logs ───
            video_summary = []
            for idx, res in enumerate(vlm_results):
                req = vlm_requests_map[vlm_tasks[idx]]
                if isinstance(res, Exception):
                    print(f"  [ERROR] VLM evaluation failed for {req['request_id']}: {res}")
                    continue

                risk_state = res.get("pred_risk_state")
                latency = res.get("latency_sec")
                json_success = res.get("json_success")
                schema_success = res.get("schema_success")
                raw_response = res.get("raw_response")

                status_row = {
                    "video_id": video_id,
                    "request_id": req["request_id"],
                    "frame_indices": req["frame_indices"],
                    "frame_times_sec": req["frame_times_sec"],
                    "status_time_sec": req["frame_times_sec"][-1],
                    "risk_state": risk_state,
                    "stage1_actions": req["stage1_actions"],
                    "snapshot_images": req["images"],
                    "latency_sec": latency,
                    "json_success": json_success,
                    "schema_success": schema_success,
                }
                all_status_rows.append(status_row)
                video_summary.append(status_row)

                if risk_state in {"unsafe", "danger"}:
                    overlay_images = []
                    # Overlay risk badge on images
                    for i_idx, img_path_str in enumerate(req["images"], start=1):
                        img_path = Path(img_path_str)
                        if not img_path.exists():
                            continue
                        out_overlay_path = paths["overlay_dir"] / f"{req['request_id']}_F{i_idx}_{risk_state}.jpg"
                        try:
                            overlay_risk_badge(
                                image_path=img_path,
                                output_path=out_overlay_path,
                                risk_state=risk_state,
                                request_id=req["request_id"]
                            )
                            overlay_images.append(str(out_overlay_path))
                        except Exception as e:
                            print(f"  [WARN] Failed to write overlay: {e}")

                    event = {
                        "event_id": f"evt_{len(all_events) + 1:06d}",
                        "video_id": video_id,
                        "request_id": req["request_id"],
                        "frame_indices": req["frame_indices"],
                        "frame_times_sec": req["frame_times_sec"],
                        "event_time_sec": req["frame_times_sec"][-1],
                        "risk_state": risk_state,
                        "stage1_actions": req["stage1_actions"],
                        "snapshot_images": req["images"],
                        "overlay_images": overlay_images,
                        "latency_sec": latency,
                        "json_success": json_success,
                        "schema_success": schema_success,
                    }
                    all_events.append(event)
            
            # Save raw_results.jsonl and summary.csv for compatibility
            video_results_dir = paths["stage2_dir"]
            video_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Write summary CSV for the video
            summary_csv_path = video_results_dir / "summary.csv"
            with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["request_id", "video_id", "pred_risk_state", "latency_sec", "json_success", "schema_success", "image_1", "image_2", "image_3"])
                for row in video_summary:
                    writer.writerow([
                        row["request_id"],
                        row["video_id"],
                        row["risk_state"],
                        row["latency_sec"],
                        row["json_success"],
                        row["schema_success"],
                        row["snapshot_images"][0],
                        row["snapshot_images"][1],
                        row["snapshot_images"][2]
                    ])

    # ─── 5. Write Final logs ───
    # Write status logs
    with paths["status_logs_jsonl"].open("w", encoding="utf-8") as f:
        for r in all_status_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with paths["status_logs_csv"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "request_id", "frame_indices", "frame_times_sec", "status_time_sec", 
            "risk_state", "stage1_actions", "snapshot_images", "latency_sec", "json_success", "schema_success"
        ])
        writer.writeheader()
        for r in all_status_rows:
            writer.writerow({
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                for k, v in r.items()
            })

    # Write risk logs
    with paths["risk_logs_jsonl"].open("w", encoding="utf-8") as f:
        for e in all_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with paths["risk_logs_csv"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_id", "video_id", "request_id", "frame_indices", "frame_times_sec", "event_time_sec",
            "risk_state", "stage1_actions", "snapshot_images", "overlay_images", "latency_sec", "json_success", "schema_success"
        ])
        writer.writeheader()
        for e in all_events:
            writer.writerow({
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                for k, v in e.items()
            })

    print(f"\nFinal status logs: {paths['status_logs_jsonl']}")
    print(f"Final risk logs: {paths['risk_logs_jsonl']}")
    print(f"Total status rows: {len(all_status_rows)}")
    print(f"Total risk events: {len(all_events)}")


def main():
    parser = argparse.ArgumentParser(description="Run Parallel Real-time Pipeline for ZroAct.")
    parser.add_argument("--video", default=None, help="Single input video path.")
    parser.add_argument("--video-dir", default=None, help="Input video directory.")
    parser.add_argument("--frames-dir", default=None, help="Pre-extracted frames directory.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--stage1-root", default=None)
    parser.add_argument("--stage2-root", default=None)
    parser.add_argument("--conda-bin", default=str(default_conda_bin()))
    parser.add_argument("--stage1-env", default="yowov3")
    parser.add_argument("--stage2-env", default="qwen35")
    
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--stage1-window", type=int, default=16)
    parser.add_argument("--stage1-sample-rate", type=int, default=10)
    parser.add_argument("--stage1-conf-threshold", type=float, default=0.3)
    parser.add_argument("--stage1-batch-size", type=int, default=8)
    parser.add_argument("--action-top-k", type=int, default=2)
    parser.add_argument("--include-action-score", action="store_true")

    parser.add_argument("--stage2-gap", type=int, default=10)
    parser.add_argument("--stage2-stride", type=int, default=30)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--vlm-model-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=640 * 360,
                        help="Max pixels per image for VLM. Default 230400 (640x360).")

    parser.add_argument("--frame-video-filter", default=None)
    parser.add_argument("--max-frame-videos", type=int, default=None)
    parser.add_argument("--config", default="serving/config.json")
    parser.add_argument("--extraction-method", choices=["fps", "interpolate"], default="fps")
    parser.add_argument("--extraction-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-spawn", action="store_true",
                        help="Skip server spawning; connect to already-running servers.")

    args = parser.parse_args()

    # Load configuration
    stage2_dir = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, stage2_dir)
    config = load_config(config_path)

    # Fill defaults from config
    args.stage2_root = Path(args.stage2_root or config.get("stage2_root", str(stage2_dir))).resolve()
    args.stage1_root = Path(args.stage1_root or config.get("stage1_root", "/home/capstone2/zroact-stage1/YOWOv3")).resolve()
    args.conda_bin = Path(args.conda_bin or config.get("conda_bin", "/home/capstone2/miniconda3/bin/conda")).resolve()
    args.prompt = Path(args.prompt or args.stage2_root / config.get("prompt", "benchmark2/prompts/action_timev1.txt")).resolve()
    args.vlm_model_path = Path(args.vlm_model_path or args.stage2_root / config.get("vlm_model_path", "benchmark2/models/Qwen3.5-2B")).resolve()
    args.output_root = Path(args.output_root or args.stage2_root / "benchmark2" / "runs").resolve()
    args.run_id = args.run_id or timestamp_run_id()

    args.video = resolve_path(args.video, Path.cwd())
    args.video_dir = resolve_path(args.video_dir, Path.cwd())
    args.frames_dir = resolve_path(args.frames_dir, Path.cwd())

    paths = build_paths(args)

    print("=== Starting ZroAct Real-Time Parallel Pipeline ===")
    print("Run ID:", args.run_id)
    print("Output root:", paths["run_root"])
    
    timings = {}
    total_start = time.perf_counter()

    s1_process = None
    s2_process = None

    if args.no_spawn:
        # Attach to already-running servers
        s1_url = f"http://{config['stage1_daemon_host']}:{config['stage1_daemon_port']}"
        s2_url = f"http://{config['stage2_daemon_host']}:{config['stage2_daemon_port']}"
        print(f"\n[--no-spawn] Attaching to existing servers at {s1_url} and {s2_url}")
        if not wait_for_server(f"{s1_url}/health", timeout=5):
            raise RuntimeError(f"Stage 1 server not reachable at {s1_url}")
        if not wait_for_server(f"{s2_url}/health", timeout=5):
            raise RuntimeError(f"Stage 2 server not reachable at {s2_url}")
        print("Both servers confirmed ready.")
    else:
        # 1. Spawning Daemons
        print("\n--- Spawning Stage 1 (YOWOv3 ONNX) Server ---")
        s1_spawn_start = time.perf_counter()
        stage1_python = config.get("stage1_python", f"/data2/cache/pipeline/envs/yowov3/bin/python")
        stage1_script = config.get("stage1_server_script", "serving/workers/stage1_launcher.py")
        s1_cmd = [
            stage1_python, stage1_script,
            "--stage1-root", str(args.stage1_root),
            "--host", config["stage1_daemon_host"],
            "--port", str(config["stage1_daemon_port"]),
        ]
        s1_process = subprocess.Popen(s1_cmd, cwd=str(args.stage2_root))
        s1_url = f"http://{config['stage1_daemon_host']}:{config['stage1_daemon_port']}"
        print(f"Waiting for Stage 1 daemon to start at {s1_url}...")
        if not wait_for_server(f"{s1_url}/health"):
            s1_process.terminate()
            raise RuntimeError("Stage 1 server failed to startup in time.")
        s1_spawn_time = time.perf_counter() - s1_spawn_start
        timings["stage1_spawn_time"] = s1_spawn_time
        print(f"Stage 1 server ready. Startup time: {s1_spawn_time:.2f}s")

        print("\n--- Spawning Stage 2 (Qwen3.5 vLLM) Server ---")
        s2_spawn_start = time.perf_counter()
        stage2_python = config.get("stage2_python", f"/data2/cache/pipeline/envs/qwen35/bin/python3")
        stage2_script = config.get("stage2_server_script", "serving/workers/stage2_vllm_server.py")
        s2_cmd = [
            stage2_python, stage2_script,
            "--host", config["stage2_daemon_host"],
            "--port", str(config["stage2_daemon_port"]),
            "--model-path", str(args.vlm_model_path),
            "--prompt-path", str(args.prompt),
            "--max-pixels", str(args.max_pixels),
            "--max-new-tokens", str(config.get("max_new_tokens", 256)),
            "--gpu-memory-utilization", str(config.get("gpu_memory_utilization", 0.5)),
            "--max-model-len", str(config.get("max_model_len", 4096)),
        ]
        # Build env with LD_LIBRARY_PATH for libcudart.so.13 and PATH for ninja
        s2_env = os.environ.copy()
        ld_prepend = config.get("stage2_ld_library_path_prepend", "")
        if ld_prepend:
            existing_ld = s2_env.get("LD_LIBRARY_PATH", "")
            s2_env["LD_LIBRARY_PATH"] = f"{ld_prepend}:{existing_ld}" if existing_ld else ld_prepend
        qwen35_bin = str(Path(stage2_python).parent)
        s2_env["PATH"] = f"{qwen35_bin}:{s2_env.get('PATH', '')}"
        s2_process = subprocess.Popen(s2_cmd, cwd=str(args.stage2_root), env=s2_env)
        s2_url = f"http://{config['stage2_daemon_host']}:{config['stage2_daemon_port']}"
        s2_startup_timeout = config.get("stage2_startup_timeout", 600)
        print(f"Waiting for Stage 2 daemon to start at {s2_url} (timeout={s2_startup_timeout}s, vLLM compiles CUDA graphs)...")
        if not wait_for_server(f"{s2_url}/health", timeout=s2_startup_timeout):
            s1_process.terminate()
            s2_process.terminate()
            raise RuntimeError("Stage 2 server failed to startup in time.")
        s2_spawn_time = time.perf_counter() - s2_spawn_start
        timings["stage2_spawn_time"] = s2_spawn_time
        print(f"Stage 2 server ready. Startup time: {s2_spawn_time:.2f}s")

    try:
        # 2. Extract frames if needed (Simulated Camera stream ingestion)
        if args.frames_dir:
            if not Path(args.frames_dir).exists():
                raise FileNotFoundError(f"Frames dir not found: {args.frames_dir}")
            print(f"[USE] pre-extracted frames: {args.frames_dir}")
        else:
            print("\n--- Extracting Video Frames (Simulating Stream) ---")
            frame_ext_start = time.perf_counter()
            prepare_input(args.video, args.video_dir, paths["input_dir"])
            extract_30fps(args, paths)
            frame_ext_time = time.perf_counter() - frame_ext_start
            timings["frame_extraction"] = frame_ext_time
            print(f"Frame extraction complete: {frame_ext_time:.2f}s")

        # 3. Run streaming logic asynchronously
        print("\n--- Starting Streaming Inference Loop ---")
        loop_start = time.perf_counter()
        asyncio.run(run_pipeline_async(args, paths, config, timings))
        loop_time = time.perf_counter() - loop_start
        timings["streaming_loop_time"] = loop_time
        print(f"Streaming loop finished in: {loop_time:.2f}s")

    finally:
        if not args.no_spawn and s1_process and s2_process:
            # Graceful shutdown of daemons
            print("\n--- Terminating Daemon Servers ---")
            s1_process.terminate()
            s2_process.terminate()
            try:
                s1_process.wait(timeout=5)
                s2_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                s1_process.kill()
                s2_process.kill()
            print("Daemons stopped successfully.")
        elif args.no_spawn:
            print("\n[--no-spawn] Leaving servers running.")

    total_wall = time.perf_counter() - total_start
    timings["total_wall"] = total_wall
    write_timings(paths, timings)
    print(f"\nPipeline finished. Total wall time: {total_wall:.2f}s")

if __name__ == "__main__":
    main()
