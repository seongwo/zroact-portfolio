"""
ZroAct Lightweight Streaming Pipeline

streaming_pipeline.py 대비 개선 사항:

[방안 A] Stage1 gather 병렬화
  - 모든 윈도우 Stage1 요청을 동시에 전송 (asyncio.gather)
  - --s1-workers 로 동시 요청 수 제어 (기본 6)

[방안 B] Stage2 동시성 조절
  - --s2-workers 로 VLM Consumer 수 설정 (기본 3)

[방안 C] asyncio Producer-Consumer 큐
  - Stage1 Producer: 완료된 윈도우 즉시 Queue 투입
  - Stage2 Consumer(N): Queue에서 꺼내 VLM 추론
  - Stage1 완료 즉시 Stage2 시작 → 최대 오버랩

[경량화] --skip-stage1
  - Stage1 HTTP 호출 생략, actions = "none"
  - Stage1 서버 불필요 → Stage2(VLM)만 기동
  - 레이턴시 최단, 테스트·데모용

파이프라이닝 타임라인 (이상적):
  Stage1: [w0]──[w1]──[w2]──[w3]──...──[wN]   (s1_workers개 병렬)
                ↓    ↓    ↓    ↓
  Stage2:    [w0] [w1] [w2] [w3] ...           (s2_workers개 Consumer)

streaming_pipeline.py와의 차이:
  현재: process_window(Stage1→Stage2) N개 task gather
         → 모든 윈도우가 개별로 Stage1 완료 후 Stage2
  경량화: Producer(Stage1 all) + Consumer(Stage2 N)
         → Stage1이 하나라도 끝나면 Stage2 즉시 투입
"""

import argparse
import asyncio
import json
import os
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
from pipeline_ver2.streaming_pipeline import (
    find_ffmpeg,
    extract_frames_ffmpeg,
    kill_port,
    spawn_servers,
    _all_descendants,
)


# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────

def _parse_actions(resp: dict, n_clips: int, top_k: int) -> list[str]:
    """Stage1 응답에서 액션 요약 문자열 목록 추출."""
    summaries = []
    for i in range(n_clips):
        dets = resp["results"][i]
        if not dets:
            summaries.append("none")
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
                if len(unique) >= top_k:
                    break
        summaries.append(", ".join(a["class_name"] for a in unique) or "none")
    return summaries


def _spawn_stage2_only(config: dict, args) -> subprocess.Popen:
    """Stage2 서버만 기동 (--skip-stage1 모드용)."""
    s2_port = config["stage2_daemon_port"]
    print(f"--- 포트 정리: Stage2 port {s2_port} ---")
    kill_port(s2_port)

    stage2_root = args.stage2_root
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
    return subprocess.Popen(s2_cmd, cwd=str(stage2_root), env=s2_env)


# ─────────────────────────────────────────────────────────────
# Producer — Stage1 (방안 A + C)
# ─────────────────────────────────────────────────────────────

async def _stage1_producer(
    session: aiohttp.ClientSession,
    s1_url: str,
    window_cfgs: list,        # [(clips, selected_indices, image_paths), ...]
    queue: asyncio.Queue,
    args,
    n_consumers: int,
    s1_semaphore: asyncio.Semaphore,
) -> None:
    """
    [방안 A] 모든 윈도우 Stage1을 동시에 전송.
    [방안 C] 완료된 윈도우를 즉시 Queue에 투입 → Consumer와 오버랩.
    --skip-stage1 시 Stage1 호출 없이 none으로 바로 투입.
    """

    async def _run_one(clips: list, selected: list, image_paths: list) -> None:
        if args.skip_stage1:
            actions = ["none"] * len(selected)
        else:
            try:
                async with s1_semaphore:
                    resp = await detect_clip_batch(
                        session=session,
                        url=f"{s1_url}/detect",
                        clips=clips,
                        conf_thresh=args.stage1_conf_threshold,
                        top_k=args.action_top_k,
                    )
                actions = _parse_actions(resp, len(selected), args.action_top_k)
            except Exception as e:
                print(f"  [WARN] Stage1 실패 window={selected}: {e}")
                actions = ["none"] * len(selected)

        await queue.put((selected, actions, image_paths))

    tasks = [
        asyncio.create_task(_run_one(clips, sel, imgs))
        for clips, sel, imgs in window_cfgs
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Consumer 종료 신호 (poison pill)
    for _ in range(n_consumers):
        await queue.put(None)


# ─────────────────────────────────────────────────────────────
# Consumer — Stage2 VLM (방안 B + C)
# ─────────────────────────────────────────────────────────────

async def _stage2_consumer(
    session: aiohttp.ClientSession,
    s2_url: str,
    queue: asyncio.Queue,
    vlm_semaphore: asyncio.Semaphore,
    args,
    video_id: str,
    all_results: list,
    overlay_dir: Path,
) -> None:
    """[방안 C] Queue에서 꺼내 VLM 추론. 종료 신호(None)를 받으면 종료."""
    while True:
        item = await queue.get()
        if item is None:
            break

        selected_indices, actions_summary, image_paths = item
        t0 = time.perf_counter()
        trigger_frame = selected_indices[-1]

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

        try:
            res = await evaluate_vlm(session, f"{s2_url}/evaluate", api_payload, vlm_semaphore)
        except Exception as e:
            print(f"  [ERROR] Stage2 실패 window={selected_indices}: {e}")
            continue

        window_latency = time.perf_counter() - t0
        risk_state = res.get("pred_risk_state", "unknown")

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
                    print(f"  [WARN] overlay 실패: {e}")
            result["overlay_images"] = overlay_images

        all_results.append(result)


# ─────────────────────────────────────────────────────────────
# 메인 비동기 루프
# ─────────────────────────────────────────────────────────────

async def run_lightweight_async(args, paths: dict, config: dict, timings: dict) -> None:
    """
    Producer(Stage1) + N×Consumer(Stage2) 기반 경량화 스트리밍 파이프라인.

    흐름:
      1. 모든 윈도우 설정 사전 계산
      2. asyncio.Queue 생성
      3. Producer: 모든 Stage1 요청 동시 발사 → 완료 즉시 Queue 투입
      4. Consumer(s2_workers개): Queue에서 꺼내 VLM 추론
      5. asyncio.gather(producer, *consumers) 완료 대기
    """
    clip_length = config.get("stage1_window", 16)
    sampling_rate = config.get("stage1_sample_rate", 10)
    gap = args.stage2_gap
    stride = args.stage2_stride
    n_consumers = args.s2_workers

    s1_url = f"http://{config['stage1_daemon_host']}:{config['stage1_daemon_port']}"
    s2_url = f"http://{config['stage2_daemon_host']}:{config['stage2_daemon_port']}"

    s1_semaphore = asyncio.Semaphore(args.s1_workers)
    vlm_semaphore = asyncio.Semaphore(n_consumers)
    all_results: list = []

    frames_root = paths["frames_dir"]
    video_dirs = sorted([
        d for d in frames_root.iterdir()
        if d.is_dir() and any(
            f.suffix.lower() in {".jpg", ".png"} for f in d.iterdir()
        )
    ])
    if not video_dirs:
        raise RuntimeError(f"프레임 디렉터리를 찾을 수 없습니다: {frames_root}")

    os.makedirs(paths["overlay_dir"], exist_ok=True)

    connector = aiohttp.TCPConnector(limit=max(10, n_consumers + args.s1_workers))
    async with aiohttp.ClientSession(connector=connector) as session:
        for video_dir in video_dirs:
            video_id = video_dir.name
            frame_paths = sorted(
                list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
            )
            num_frames = len(frame_paths)
            if num_frames == 0:
                continue

            print(f"\n[Lightweight] {video_id} — {num_frames} frames @ {args.fps}fps")

            frame_map = {i + 1: frame_paths[i] for i in range(num_frames)}

            def build_clip(kf: int) -> list[str]:
                """keyframe kf 기준 16-frame 클립 경로 목록."""
                clip = []
                for i in range(clip_length):
                    idx = kf - (clip_length - 1 - i) * sampling_rate
                    idx = max(1, min(idx, num_frames))
                    clip.append(str(frame_map[idx]))
                return clip

            # 트리거 프레임: clip_length + gap*2, clip_length + gap*2 + stride, ...
            trigger_base = clip_length + gap * 2

            window_cfgs: list[tuple] = []
            for frame_idx in range(1, num_frames + 1):
                if frame_idx < trigger_base:
                    continue
                if (frame_idx - trigger_base) % stride != 0:
                    continue
                window_start = frame_idx - gap * 2
                selected = [window_start, window_start + gap, frame_idx]
                clips = [build_clip(kf) for kf in selected]
                image_paths = [str(frame_map[idx]) for idx in selected]
                window_cfgs.append((clips, selected, image_paths))

            n_win = len(window_cfgs)
            skip_tag = " (Stage1 생략)" if args.skip_stage1 else ""
            print(
                f"  {n_win} windows | "
                f"s1_workers={args.s1_workers}{skip_tag} | "
                f"s2_workers={n_consumers}"
            )

            # Queue 크기: Consumer 버퍼 (n_consumers×4) — 너무 크면 메모리, 너무 작으면 Producer 블록
            queue: asyncio.Queue = asyncio.Queue(maxsize=n_consumers * 4)
            loop_start = time.perf_counter()

            producer_task = asyncio.create_task(
                _stage1_producer(
                    session, s1_url,
                    window_cfgs, queue, args,
                    n_consumers, s1_semaphore,
                )
            )
            consumer_tasks = [
                asyncio.create_task(
                    _stage2_consumer(
                        session, s2_url,
                        queue, vlm_semaphore,
                        args, video_id, all_results, paths["overlay_dir"],
                    )
                )
                for _ in range(n_consumers)
            ]

            await asyncio.gather(producer_task, *consumer_tasks, return_exceptions=True)

            loop_elapsed = time.perf_counter() - loop_start
            timings[f"lightweight_loop_{video_id}"] = loop_elapsed
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
        print(
            f"Window latency: avg={sum(lats)/len(lats):.2f}s  "
            f"min={min(lats):.2f}s  max={max(lats):.2f}s"
        )


# ─────────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZroAct Lightweight Streaming Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
경량화 옵션:
  --skip-stage1        Stage1 없이 VLM만 실행 (가장 빠른 모드)
  --s1-workers N       Stage1 동시 요청 수 (기본 6)
  --s2-workers N       Stage2 VLM Consumer 수 (기본 3)

예시:
  # 표준 경량화 (Stage1+Stage2 동시, Consumer 3개)
  python lightweight_streaming_pipeline.py --frames-dir /path/frames

  # 초경량 (Stage1 생략, VLM만)
  python lightweight_streaming_pipeline.py --frames-dir /path/frames --skip-stage1

  # 서버 유지 후 재사용
  python lightweight_streaming_pipeline.py --frames-dir /path/frames --keep-servers
  python lightweight_streaming_pipeline.py --frames-dir /path/frames --no-spawn
""",
    )
    # 입력
    parser.add_argument("--video", default=None, help="입력 영상 경로")
    parser.add_argument("--frames-dir", default=None, help="사전 추출된 프레임 디렉터리")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", default="serving/config.json")

    # 추론 파라미터
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--stage1-conf-threshold", type=float, default=0.3)
    parser.add_argument("--action-top-k", type=int, default=2)
    parser.add_argument("--stage2-gap", type=int, default=10)
    parser.add_argument("--stage2-stride", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=640 * 360)

    # 경량화 핵심 옵션
    parser.add_argument("--s1-workers", type=int, default=6,
                        help="Stage1 동시 요청 수 (방안 A, 기본 6)")
    parser.add_argument("--s2-workers", type=int, default=3,
                        help="Stage2 Consumer 수 = VLM 병렬도 (방안 B, 기본 3)")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Stage1 건너뜀 — VLM만 실행 (초경량 모드)")

    # 서버 관리
    parser.add_argument("--no-spawn", action="store_true",
                        help="이미 실행 중인 서버에 연결")
    parser.add_argument("--keep-servers", action="store_true",
                        help="파이프라인 완료 후 서버 유지 (다음 --no-spawn 실행용)")
    args = parser.parse_args()

    stage2_dir = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, stage2_dir)
    config = load_config(config_path)

    args.stage2_root = Path(config.get("stage2_root", str(stage2_dir))).resolve()
    args.stage1_root = Path(config.get("stage1_root", "/home/capstone2/zroact-stage1/YOWOv3")).resolve()
    args.vlm_model_path = (
        args.stage2_root / config.get("vlm_model_path", "benchmark2/models/Qwen3.5-2B")
    ).resolve()
    args.prompt = (
        args.stage2_root / config.get("prompt", "benchmark2/prompts/action_timev1.txt")
    ).resolve()
    args.run_id = args.run_id or timestamp_run_id()

    output_root = (args.stage2_root / "benchmark2" / "runs").resolve()
    run_root = output_root / args.run_id

    # 프레임 디렉터리 결정
    if args.frames_dir:
        frames_dir = Path(args.frames_dir).resolve()
    elif args.video:
        video_path = Path(args.video).resolve()
        frames_dir = run_root / "frames_30fps" / video_path.stem
        print(f"프레임 추출 중: {video_path} ...")
        extract_frames_ffmpeg(video_path, frames_dir)
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

    print("=== ZroAct Lightweight Streaming Pipeline ===")
    print(f"Run ID     : {args.run_id}")
    print(f"Output     : {run_root}")
    print(f"s1_workers : {args.s1_workers}  {'→ SKIP (--skip-stage1)' if args.skip_stage1 else ''}")
    print(f"s2_workers : {args.s2_workers}")

    timings: dict = {}
    total_start = time.perf_counter()

    s1_proc = s2_proc = None
    s1_url = f"http://{config['stage1_daemon_host']}:{config['stage1_daemon_port']}"
    s2_url = f"http://{config['stage2_daemon_host']}:{config['stage2_daemon_port']}"

    # ── 서버 기동 ──────────────────────────────────────────────
    if args.no_spawn:
        print(f"\n[--no-spawn] 기존 서버에 연결: {s1_url}, {s2_url}")
        if not args.skip_stage1:
            if not wait_for_server(f"{s1_url}/health", timeout=5):
                raise RuntimeError(
                    f"Stage1 서버 응답 없음: {s1_url}\n"
                    "  힌트: --keep-servers 로 이전 실행을 유지했는지 확인하세요."
                )
        if not wait_for_server(f"{s2_url}/health", timeout=5):
            raise RuntimeError(
                f"Stage2 서버 응답 없음: {s2_url}\n"
                "  힌트: --keep-servers 로 이전 실행을 유지했는지 확인하세요."
            )
        print("서버 연결 확인.")
    else:
        if args.skip_stage1:
            print("\n--- Stage2 서버만 기동 (Stage1 생략 모드) ---")
            s2_proc = _spawn_stage2_only(config, args)
        else:
            print("\n--- Stage1·Stage2 서버 기동 ---")
            s1_proc, s2_proc = spawn_servers(config, args)

            t = time.perf_counter()
            if not wait_for_server(f"{s1_url}/health"):
                s1_proc.terminate(); s2_proc.terminate()
                raise RuntimeError("Stage1 서버 시작 실패")
            timings["stage1_spawn_time"] = time.perf_counter() - t
            print(f"Stage1 준비 ({timings['stage1_spawn_time']:.2f}s)")

        print("--- Stage2 서버 대기 (vLLM 컴파일 포함 최대 600s) ---")
        t = time.perf_counter()
        timeout = config.get("stage2_startup_timeout", 600)
        if not wait_for_server(f"{s2_url}/health", timeout=timeout):
            if s1_proc:
                s1_proc.terminate()
            s2_proc.terminate()
            raise RuntimeError("Stage2 서버 시작 실패")
        timings["stage2_spawn_time"] = time.perf_counter() - t
        print(f"Stage2 준비 ({timings['stage2_spawn_time']:.2f}s)")

    # ── 추론 실행 ──────────────────────────────────────────────
    try:
        print("\n--- 경량화 스트리밍 추론 시작 ---")
        loop_start = time.perf_counter()
        asyncio.run(run_lightweight_async(args, paths, config, timings))
        timings["lightweight_loop_time"] = time.perf_counter() - loop_start

    finally:
        procs = [p for p in (s1_proc, s2_proc) if p is not None]
        if procs:
            if args.keep_servers:
                print(f"\n[--keep-servers] 서버 유지 중: {s1_url}, {s2_url}")
                print("  다음 실행: python lightweight_streaming_pipeline.py --no-spawn ...")
            else:
                print("\n--- 서버 종료 ---")
                for p in procs:
                    p.terminate()
                for p in procs:
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
        elif args.no_spawn:
            print("\n[--no-spawn] 서버를 그대로 유지합니다.")

    timings["total_wall"] = time.perf_counter() - total_start
    write_timings(paths, timings)
    print(f"\n완료. 총 소요시간: {timings['total_wall']:.2f}s")


if __name__ == "__main__":
    main()
