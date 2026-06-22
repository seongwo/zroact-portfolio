"""
ZroAct Streaming Pipeline

실시간 슬라이딩 윈도우 추론:
- 프레임이 하나씩 들어올 때, 윈도우 트리거 프레임 도착 즉시 Stage1(3클립) → Stage2 발사
- 배치 대기 없음 → 이벤트 발생 후 ~1.9s 이내 첫 경보 가능
- Stage1·Stage2 윈도우 단위로 파이프라인 병렬 실행

realtime_pipeline.py와의 차이:
- Stage1을 8개 배치로 묶지 않고, 윈도우당 3개 클립 즉시 처리
- 각 윈도우가 독립 async 태스크로 실행 (Stage1 완료 즉시 Stage2 발사)
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import requests

project_dir = Path(__file__).resolve().parents[1]
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

from pipeline_ver2.main import (
    default_conda_bin,
    timestamp_run_id,
    resolve_path,
    write_timings,
    overlay_risk_badge,
    RISK_STATES,
)
from pipeline_ver2.realtime_pipeline import (
    load_config,
    wait_for_server,
    detect_clip_batch,
    evaluate_vlm,
    frame_to_sec,
)


def find_ffmpeg() -> str:
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    candidates = [
        "/home/capstone2/miniconda3/envs/yowov3/bin/ffmpeg",
        "/home/deepfake/miniforge3/envs/yowov3/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError("ffmpeg를 찾을 수 없습니다. PATH에 ffmpeg를 추가하거나 conda env yowov3을 활성화하세요.")


def extract_frames_ffmpeg(video_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        find_ffmpeg(), "-y", "-i", str(video_path),
        "-vf", "fps=30",
        "-q:v", "2",
        str(out_dir / "30fps_frame_%03d.jpg"),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


async def process_window(
    session, s1_url, s2_url,
    clips, selected_indices,
    frame_map, video_id, args, vlm_semaphore,
    overlay_dir, all_results,
):
    """Stage1 → Stage2 를 윈도우 단위로 실행."""
    t0 = time.perf_counter()

    # ── Stage 1 ──────────────────────────────────────────────
    resp = await detect_clip_batch(
        session=session,
        url=f"{s1_url}/detect",
        clips=clips,
        conf_thresh=args.stage1_conf_threshold,
        top_k=args.action_top_k,
    )

    actions_summary = []
    for i in range(len(selected_indices)):
        dets = resp["results"][i]
        if not dets:
            actions_summary.append("none")
            continue
        flat = []
        for det in dets:
            flat.extend(det.get("actions", []))
        flat.sort(key=lambda x: x["score"], reverse=True)
        seen, unique = set(), []
        for a in flat:
            if a["class_name"] not in seen:
                seen.add(a["class_name"])
                unique.append(a)
                if len(unique) >= args.action_top_k:
                    break
        actions_summary.append(", ".join(a["class_name"] for a in unique) or "none")

    # ── Stage 2 ──────────────────────────────────────────────
    image_paths = [str(frame_map[idx]) for idx in selected_indices]
    request_id = (
        f"{video_id}"
        f"_f{selected_indices[0]:06d}"
        f"_{selected_indices[1]:06d}"
        f"_{selected_indices[2]:06d}"
    )

    api_payload = {
        "request_id": request_id,
        "video_id": video_id,
        "images": image_paths,
        "stage1_actions": actions_summary,
        "frame_indices": selected_indices,
        "frame_times_sec": [frame_to_sec(idx, args.fps) for idx in selected_indices],
        "fps": args.fps,
        "max_new_tokens": args.max_new_tokens,
    }

    res = await evaluate_vlm(session, f"{s2_url}/evaluate", api_payload, vlm_semaphore)

    window_latency = time.perf_counter() - t0
    risk_state = res.get("pred_risk_state", "unknown")

    trigger_frame = selected_indices[-1]
    print(
        f"  [f{trigger_frame:04d}] {risk_state:8s} | "
        f"window={selected_indices} | "
        f"actions={actions_summary} | "
        f"{window_latency:.2f}s",
        flush=True,
    )

    result = {
        "request_id": request_id,
        "video_id": video_id,
        "frame_indices": selected_indices,
        "frame_times_sec": [frame_to_sec(idx, args.fps) for idx in selected_indices],
        "trigger_frame": trigger_frame,
        "trigger_time_sec": frame_to_sec(trigger_frame, args.fps),
        "risk_state": risk_state,
        "stage1_actions": actions_summary,
        "snapshot_images": image_paths,
        "vlm_latency_sec": res.get("latency_sec"),
        "window_latency_sec": window_latency,
        "json_success": res.get("json_success"),
        "schema_success": res.get("schema_success"),
    }

    if risk_state in RISK_STATES - {"normal"}:
        overlay_images = []
        for i_idx, img_str in enumerate(image_paths, start=1):
            img_path = Path(img_str)
            if not img_path.exists():
                continue
            out = overlay_dir / f"{request_id}_F{i_idx}_{risk_state}.jpg"
            try:
                overlay_risk_badge(img_path, out, risk_state, request_id)
                overlay_images.append(str(out))
            except Exception as e:
                print(f"  [WARN] overlay failed: {e}")
        result["overlay_images"] = overlay_images

    all_results.append(result)
    return result


async def run_streaming_async(args, paths, config, timings):
    """
    프레임을 순서대로 하나씩 수신(시뮬레이션)하고,
    윈도우 트리거 프레임 도착 즉시 Stage1→Stage2 태스크를 발사.
    """
    clip_length = config.get("stage1_window", 16)
    sampling_rate = config.get("stage1_sample_rate", 10)
    gap = args.stage2_gap    # 프레임 내 간격 (예: 10)
    stride = args.stage2_stride  # 윈도우 간 간격 (예: 30)

    s1_url = f"http://{config['stage1_daemon_host']}:{config['stage1_daemon_port']}"
    s2_url = f"http://{config['stage2_daemon_host']}:{config['stage2_daemon_port']}"

    vlm_semaphore = asyncio.Semaphore(3)
    all_results = []

    frames_root = paths["frames_dir"]
    video_dirs = sorted([
        d for d in frames_root.iterdir()
        if d.is_dir() and any(
            f.suffix.lower() in {".jpg", ".png"} for f in d.iterdir()
        )
    ])
    if not video_dirs:
        raise RuntimeError(f"No frame directories found in {frames_root}")

    os.makedirs(paths["overlay_dir"], exist_ok=True)

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        for video_dir in video_dirs:
            video_id = video_dir.name
            frame_paths = sorted(
                list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
            )
            num_frames = len(frame_paths)
            if num_frames == 0:
                continue

            print(f"\n[Stream] {video_id} — {num_frames} frames @ {args.fps}fps")

            # 1-인덱스 frame_map
            frame_map = {i + 1: frame_paths[i] for i in range(num_frames)}

            # 윈도우 트리거 계산:
            #   선택 프레임: [start, start+gap, start+gap*2]
            #   start = clip_length (= 16), stride=30 씩 증가
            #   트리거 = start + gap*2 (= 36, 66, 96, ...)
            first_start = clip_length
            trigger_base = first_start + gap * 2  # 36

            def build_clip(kf: int) -> list[str]:
                """keyframe kf에 대한 16-frame 클립 경로 리스트."""
                clip = []
                for i in range(clip_length):
                    idx = kf - (clip_length - 1 - i) * sampling_rate
                    idx = max(1, min(idx, num_frames))
                    clip.append(str(frame_map[idx]))
                return clip

            tasks = []
            loop_start = time.perf_counter()

            # 프레임을 순서대로 수신 (실시간 시뮬레이션)
            for frame_idx in range(1, num_frames + 1):
                # 트리거 프레임인지 확인
                if frame_idx < trigger_base:
                    continue
                if (frame_idx - trigger_base) % stride != 0:
                    continue

                window_start = frame_idx - gap * 2
                selected = [window_start, window_start + gap, frame_idx]
                clips = [build_clip(kf) for kf in selected]

                task = asyncio.create_task(
                    process_window(
                        session, s1_url, s2_url,
                        clips, selected,
                        frame_map, video_id, args, vlm_semaphore,
                        paths["overlay_dir"], all_results,
                    )
                )
                tasks.append(task)

            print(f"  {len(tasks)} windows triggered. Waiting...")
            await asyncio.gather(*tasks, return_exceptions=True)

            loop_elapsed = time.perf_counter() - loop_start
            timings[f"streaming_loop_{video_id}"] = loop_elapsed
            print(f"  Done in {loop_elapsed:.2f}s")

    # ── 결과 저장 ──────────────────────────────────────────────
    all_results.sort(key=lambda r: r["frame_indices"][0])
    risk_events = [r for r in all_results if r.get("risk_state") in {"unsafe", "danger"}]

    paths["risk_logs_jsonl"].parent.mkdir(parents=True, exist_ok=True)

    with paths["status_logs_jsonl"].open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with paths["risk_logs_jsonl"].open("w", encoding="utf-8") as f:
        for r in risk_events:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTotal windows : {len(all_results)}")
    print(f"Risk events   : {len(risk_events)}")
    print(f"  danger : {sum(1 for r in risk_events if r['risk_state'] == 'danger')}")
    print(f"  unsafe : {sum(1 for r in risk_events if r['risk_state'] == 'unsafe')}")
    if all_results:
        lats = [r["window_latency_sec"] for r in all_results]
        print(f"Window latency: avg={sum(lats)/len(lats):.2f}s  min={min(lats):.2f}s  max={max(lats):.2f}s")


def _all_descendants(pid: int) -> list:
    """leaf → parent 순서로 자손 PID 목록 반환 (pgrep 사용)."""
    r = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
    children = [int(p) for p in r.stdout.split() if p.strip().isdigit()]
    result = []
    for child in children:
        result.extend(_all_descendants(child))
        result.append(child)
    return result


def kill_port(port: int, timeout: float = 10.0) -> None:
    """해당 포트를 점유한 프로세스와 모든 하위 프로세스를 종료하고
    포트 해제 + GPU 메모리 반환까지 대기."""
    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}"],
        capture_output=True, text=True,
    )
    pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    if not pids:
        return

    # 자손 먼저(leaf→root), 그 다음 포트 소유 프로세스 순으로 SIGTERM
    all_pids = []
    for pid in pids:
        all_pids.extend(_all_descendants(pid))
        all_pids.append(pid)

    for pid in all_pids:
        print(f"  [port {port}] SIGTERM → PID {pid}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        r = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
        if not r.stdout.strip():
            # 포트 해제됨 — GPU 메모리가 OS에 반환될 때까지 잠시 대기
            time.sleep(2.0)
            return
        time.sleep(0.3)

    # 시간 초과 시 잔존 프로세스 트리 강제 종료
    r = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True)
    remaining = [int(p) for p in r.stdout.split() if p.strip().isdigit()]
    for pid in remaining:
        for p in _all_descendants(pid) + [pid]:
            print(f"  [port {port}] SIGKILL → PID {p}")
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(2.0)


def spawn_servers(config, args):
    """Stage1·Stage2 서버 프로세스 시작 후 (process1, process2) 반환."""
    s1_port = config["stage1_daemon_port"]
    s2_port = config["stage2_daemon_port"]

    print(f"--- 포트 정리 (기존 프로세스 종료) ---")
    kill_port(s1_port)
    kill_port(s2_port)

    stage2_root = args.stage2_root

    stage1_python = config.get("stage1_python", "/data2/cache/pipeline/envs/yowov3/bin/python")
    stage1_script = config.get("stage1_server_script", "serving/workers/stage1_launcher.py")
    s1_cmd = [
        stage1_python, stage1_script,
        "--stage1-root", str(args.stage1_root),
        "--host", config["stage1_daemon_host"],
        "--port", str(config["stage1_daemon_port"]),
    ]
    s1_proc = subprocess.Popen(s1_cmd, cwd=str(stage2_root))

    stage2_python = config.get("stage2_python", "/data2/cache/pipeline/envs/qwen35/bin/python3")
    stage2_script = config.get("stage2_server_script", "serving/workers/stage2_vllm_server.py")
    s2_cmd = [
        stage2_python, stage2_script,
        "--host", config["stage2_daemon_host"],
        "--port", str(config["stage2_daemon_port"]),
        "--model-path", str(args.vlm_model_path),
        "--prompt-path", str(args.prompt),
        "--max-pixels", str(args.max_pixels),
        "--max-new-tokens", str(config.get("max_new_tokens", 256)),
        "--gpu-memory-utilization", str(config.get("gpu_memory_utilization", 0.45)),
        "--max-model-len", str(config.get("max_model_len", 4096)),
    ]
    s2_env = os.environ.copy()
    ld = config.get("stage2_ld_library_path_prepend", "")
    if ld:
        s2_env["LD_LIBRARY_PATH"] = f"{ld}:{s2_env.get('LD_LIBRARY_PATH', '')}"
    s2_env["PATH"] = f"{Path(stage2_python).parent}:{s2_env.get('PATH', '')}"
    s2_proc = subprocess.Popen(s2_cmd, cwd=str(stage2_root), env=s2_env)

    return s1_proc, s2_proc


def main():
    parser = argparse.ArgumentParser(description="ZroAct Streaming Pipeline")
    parser.add_argument("--video", default=None, help="입력 영상 경로")
    parser.add_argument("--frames-dir", default=None, help="사전 추출된 프레임 디렉터리")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", default="serving/config.json")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--stage1-conf-threshold", type=float, default=0.3)
    parser.add_argument("--action-top-k", type=int, default=2)
    parser.add_argument("--stage2-gap", type=int, default=10)
    parser.add_argument("--stage2-stride", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=640 * 360)
    parser.add_argument("--no-spawn", action="store_true",
                        help="이미 실행 중인 서버에 연결 (서버 재시작 생략)")
    parser.add_argument("--keep-servers", action="store_true",
                        help="파이프라인 완료 후 서버를 종료하지 않고 유지 (다음 실행에서 --no-spawn 사용 가능)")
    args = parser.parse_args()

    stage2_dir = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, stage2_dir)
    config = load_config(config_path)

    args.stage2_root = Path(config.get("stage2_root", str(stage2_dir))).resolve()
    args.stage1_root = Path(config.get("stage1_root", "/home/capstone2/zroact-stage1/YOWOv3")).resolve()
    args.vlm_model_path = (args.stage2_root / config.get("vlm_model_path", "benchmark2/models/Qwen3.5-2B")).resolve()
    args.prompt = (args.stage2_root / config.get("prompt", "benchmark2/prompts/action_timev1.txt")).resolve()
    args.run_id = args.run_id or timestamp_run_id()

    output_root = (args.stage2_root / "benchmark2" / "runs").resolve()
    run_root = output_root / args.run_id

    # frames_dir 결정
    if args.frames_dir:
        frames_dir = Path(args.frames_dir).resolve()
    elif args.video:
        video_path = Path(args.video).resolve()
        frames_dir = run_root / "frames_30fps" / video_path.stem
        print(f"Extracting frames from {video_path} ...")
        extract_frames_ffmpeg(video_path, frames_dir)
        # wrap in parent so discovery finds it
        frames_dir = run_root / "frames_30fps"
    else:
        raise ValueError("--video 또는 --frames-dir 중 하나는 필요합니다.")

    final_dir = run_root / "final"
    paths = {
        "frames_dir": frames_dir,
        "final_dir": final_dir,
        "overlay_dir": final_dir / "overlay_images",
        "status_logs_jsonl": final_dir / "status_logs.jsonl",
        "risk_logs_jsonl": final_dir / "risk_logs.jsonl",
        "timings_json": final_dir / "timings.json",
        "timings_csv": final_dir / "timings.csv",
    }

    print("=== ZroAct Streaming Pipeline ===")
    print(f"Run ID : {args.run_id}")
    print(f"Output : {run_root}")

    timings = {}
    total_start = time.perf_counter()

    s1_proc = s2_proc = None
    s1_url = f"http://{config['stage1_daemon_host']}:{config['stage1_daemon_port']}"
    s2_url = f"http://{config['stage2_daemon_host']}:{config['stage2_daemon_port']}"

    if args.no_spawn:
        print(f"\n[--no-spawn] 기존 서버에 연결: {s1_url}, {s2_url}")
        if not wait_for_server(f"{s1_url}/health", timeout=5):
            raise RuntimeError(
                f"Stage1 서버 응답 없음: {s1_url}\n"
                "  힌트: 먼저 --keep-servers 없이 실행하거나, 이전 실행에서 --keep-servers를 사용했는지 확인하세요."
            )
        if not wait_for_server(f"{s2_url}/health", timeout=5):
            raise RuntimeError(
                f"Stage2 서버 응답 없음: {s2_url}\n"
                "  힌트: 먼저 --keep-servers 없이 실행하거나, 이전 실행에서 --keep-servers를 사용했는지 확인하세요."
            )
        print("서버 연결 확인.")
    else:
        print("\n--- Stage1 서버 시작 ---")
        t = time.perf_counter()
        s1_proc, s2_proc = spawn_servers(config, args)
        if not wait_for_server(f"{s1_url}/health"):
            s1_proc.terminate(); s2_proc.terminate()
            raise RuntimeError("Stage1 서버 시작 실패")
        timings["stage1_spawn_time"] = time.perf_counter() - t
        print(f"Stage1 준비 ({timings['stage1_spawn_time']:.2f}s)")

        print("--- Stage2 서버 시작 (vLLM 컴파일 포함 최대 600s) ---")
        t = time.perf_counter()
        timeout = config.get("stage2_startup_timeout", 600)
        if not wait_for_server(f"{s2_url}/health", timeout=timeout):
            s1_proc.terminate(); s2_proc.terminate()
            raise RuntimeError("Stage2 서버 시작 실패")
        timings["stage2_spawn_time"] = time.perf_counter() - t
        print(f"Stage2 준비 ({timings['stage2_spawn_time']:.2f}s)")

    try:
        print("\n--- 스트리밍 추론 시작 ---")
        loop_start = time.perf_counter()
        asyncio.run(run_streaming_async(args, paths, config, timings))
        timings["streaming_loop_time"] = time.perf_counter() - loop_start

    finally:
        if s1_proc and s2_proc:
            if args.keep_servers:
                print(f"\n[--keep-servers] 서버 유지 중: {s1_url}, {s2_url}")
                print("  다음 실행: python streaming_pipeline.py --no-spawn --video <파일>")
            else:
                print("\n--- 서버 종료 ---")
                s1_proc.terminate(); s2_proc.terminate()
                try:
                    s1_proc.wait(timeout=5); s2_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    s1_proc.kill(); s2_proc.kill()
        elif args.no_spawn:
            print("\n[--no-spawn] 서버를 그대로 유지합니다.")

    timings["total_wall"] = time.perf_counter() - total_start
    write_timings(paths, timings)
    print(f"\n완료. 총 소요시간: {timings['total_wall']:.2f}s")


if __name__ == "__main__":
    main()
