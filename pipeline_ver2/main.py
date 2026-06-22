from __future__ import annotations
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RISK_STATES = {"normal", "unsafe", "danger"}
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_stage1_root() -> Path:
    return Path("/data2/cache/pipeline/zroact-stage1/YOWOv3")


def default_conda_bin() -> Path:
    return Path("/home/deepfake/miniforge3/bin/conda")


def timestamp_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def resolve_path(path_text: str | None, base: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def run_command(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print()
    print("[CMD]", " ".join(cmd))
    print("[CWD]", cwd)
    if dry_run:
        return

    subprocess.run(cmd, cwd=str(cwd), check=True)


def timed_step(name: str, timings: dict[str, float], fn, *args, **kwargs):
    print()
    print(f"[STEP] {name}")
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    timings[name] = elapsed
    print(f"[DONE] {name}: {elapsed:.2f}s")
    return result


def write_timings(paths: dict[str, Path], timings: dict[str, float]) -> None:
    step_total = sum(
        value
        for name, value in timings.items()
        if name != "total_wall"
    )
    payload = {
        "step_total_sec": round(step_total, 4),
        "total_wall_sec": round(timings.get("total_wall", step_total), 4),
        "steps": {
            name: round(value, 4)
            for name, value in timings.items()
        },
    }
    paths["final_dir"].mkdir(parents=True, exist_ok=True)
    paths["timings_json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with paths["timings_csv"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "elapsed_sec"])
        writer.writeheader()
        for name, value in timings.items():
            writer.writerow({"step": name, "elapsed_sec": round(value, 4)})
        writer.writerow({"step": "step_total", "elapsed_sec": round(step_total, 4)})

    print("Saved timings:", paths["timings_json"])


def conda_python_cmd(conda_bin: Path, env_name: str, python_args: list[str]) -> list[str]:
    return [
        str(conda_bin),
        "run",
        "-n",
        env_name,
        "python",
        "-u",
        *python_args,
    ]


def prepare_input(video: Path | None, video_dir: Path | None, input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)

    if video:
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video}")
        dst = input_dir / video.name
        if dst.resolve() != video.resolve():
            shutil.copy2(video, dst)
        return

    if video_dir:
        if not video_dir.exists():
            raise FileNotFoundError(f"Video dir not found: {video_dir}")
        for item in sorted(video_dir.iterdir()):
            dst = input_dir / item.name
            if item.is_file():
                shutil.copy2(item, dst)
            elif item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
        return

    raise ValueError("Either --video or --video-dir is required.")


def extract_30fps(args, paths: dict[str, Path]) -> None:
    cmd = conda_python_cmd(
        args.conda_bin,
        args.stage1_env,
        [
            "make_30fps.py",
            "--src",
            str(paths["input_dir"]),
            "--dest",
            str(paths["frames_dir"]),
            "--method",
            args.extraction_method,
            "--workers",
            str(args.extraction_workers),
        ],
    )
    run_command(cmd, cwd=args.stage1_root, dry_run=args.dry_run)


def discover_frame_video_dirs(frames_root: Path) -> list[Path]:
    video_dirs = []
    for root, _, files in os.walk(frames_root):
        if any(Path(name).suffix.lower() in IMAGE_EXTS for name in files):
            video_dirs.append(Path(root))
    return sorted(video_dirs)


def link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return

    try:
        os.link(src, dst)
    except OSError:
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def prepare_selected_frames(args, paths: dict[str, Path]) -> None:
    source_root = paths["source_frames_dir"]
    selected_root = paths["frames_dir"]

    video_dirs = discover_frame_video_dirs(source_root)
    if args.frame_video_filter:
        video_dirs = [
            video_dir for video_dir in video_dirs
            if args.frame_video_filter in str(video_dir.relative_to(source_root))
        ]

    if args.max_frame_videos is not None:
        video_dirs = video_dirs[:args.max_frame_videos]

    if not video_dirs:
        raise RuntimeError(
            f"No frame video folders matched under {source_root}. "
            "Check --frame-video-filter."
        )

    if selected_root.exists():
        shutil.rmtree(selected_root)
    selected_root.mkdir(parents=True, exist_ok=True)

    print("[USE] selected frame videos:", len(video_dirs))
    for video_dir in video_dirs:
        rel_dir = video_dir.relative_to(source_root)
        print("  -", rel_dir)
        dst_dir = selected_root / rel_dir
        for image_path in sorted(video_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                link_or_copy_file(image_path, dst_dir / image_path.name)


def run_stage1(args, paths: dict[str, Path]) -> None:
    stage1_config = prepare_runtime_stage1_config(args, paths)
    cmd = conda_python_cmd(
        args.conda_bin,
        args.stage1_env,
        [
            "main.py",
            "-m",
            "custom_frame_infer",
            "-cf",
            str(stage1_config),
            "--frames_dir",
            str(paths["frames_dir"]),
            "--output_dir",
            str(paths["stage1_dir"]),
            "--conf_threshold",
            str(args.stage1_conf_threshold),
            "--top_k",
            str(args.action_top_k),
            "--sample_rate",
            str(args.stage1_sample_rate),
            "--batch_size",
            str(args.stage1_batch_size),
            "--extraction_method",
            args.extraction_method,
            "--extraction_workers",
            str(args.extraction_workers),
        ],
    )

    if args.stage1_make_video:
        cmd.extend(["--make_video", "--video_fps", str(args.stage1_video_fps)])

    if args.use_onnx:
        if not args.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {args.onnx_path}")
        cmd.extend(["--onnx_path", str(args.onnx_path)])

    run_command(cmd, cwd=args.stage1_root, dry_run=args.dry_run)


def parse_pretrain_path(config_path: Path) -> str | None:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^pretrain_path\s*:\s*(.*)$", text)
    if not match:
        return None
    value = match.group(1).strip()
    if value in {"", "null", "None"}:
        return None
    return value


def resolve_stage1_config_path(path_text: str | None, stage1_root: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (stage1_root / path).resolve()


def prepare_runtime_stage1_config(args, paths: dict[str, Path]) -> Path:
    source_config = args.stage1_config
    pretrain_path = args.stage1_pretrain_path

    if pretrain_path:
        if not pretrain_path.exists():
            raise FileNotFoundError(f"Stage 1 pretrain checkpoint not found: {pretrain_path}")

        runtime_config = paths["run_root"] / "config" / "stage1_config.yaml"
        runtime_config.parent.mkdir(parents=True, exist_ok=True)
        text = source_config.read_text(encoding="utf-8")
        text = re.sub(
            r"(?m)^pretrain_path\s*:.*$",
            f"pretrain_path     : {pretrain_path}",
            text,
        )
        runtime_config.write_text(text, encoding="utf-8")
        print("[USE] runtime Stage 1 config:", runtime_config)
        print("[USE] Stage 1 pretrain:", pretrain_path)
        return runtime_config

    config_pretrain = resolve_stage1_config_path(
        parse_pretrain_path(source_config),
        args.stage1_root,
    )

    if config_pretrain and not config_pretrain.exists():
        raise FileNotFoundError(
            "Stage 1 pretrain checkpoint configured in YAML was not found: "
            f"{config_pretrain}\n"
            "Pass --stage1-pretrain-path /path/to/ema_epoch_9.pth, "
            "or restore the checkpoint at the configured path."
        )

    return source_config


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frame_to_sec(frame_idx: int, fps: float) -> float:
    return round(frame_idx / fps, 3)


def summarize_actions(frame: dict, top_k: int, include_score: bool) -> str:
    actions = []
    for det in frame.get("detections", []):
        for action in det.get("actions", []):
            actions.append({
                "class_name": action.get("class_name", "unknown"),
                "score": float(action.get("score", 0.0)),
            })

    if not actions:
        return "none"

    actions = sorted(actions, key=lambda item: item["score"], reverse=True)
    unique = []
    seen = set()
    for action in actions:
        name = action["class_name"]
        if name in seen:
            continue
        seen.add(name)
        unique.append(action)
        if len(unique) >= top_k:
            break

    if include_score:
        return ", ".join(f"{a['class_name']}({a['score']:.2f})" for a in unique)
    return ", ".join(a["class_name"] for a in unique)


def discover_stage1_videos(stage1_dir: Path) -> list[dict]:
    samples = []
    for label_path in sorted(stage1_dir.glob("**/labels/*.json")):
        video_id = label_path.stem
        rel_parent = label_path.parent.parent
        image_dir = rel_parent / "images" / video_id
        if not image_dir.exists():
            print(f"[SKIP] missing Stage 1 image dir: {image_dir}")
            continue

        samples.append({
            "video_id": video_id,
            "label_path": label_path,
            "image_dir": image_dir,
        })

    return samples


def resolve_stage1_image(image_dir: Path, video_id: str, frame: dict) -> Path:
    frame_idx = int(frame["frame_idx"])
    candidates = [
        image_dir / f"{video_id}_t{frame_idx:06d}.jpg",
        image_dir / f"{video_id}_t{frame_idx:06d}.png",
        image_dir / f"30fps_frame_{frame_idx:03d}.jpg",
        image_dir / f"30fps_frame_{frame_idx:03d}.png",
    ]

    if frame.get("frame_file"):
        candidates.append(image_dir / frame["frame_file"])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTS:
        matches = sorted(image_dir.glob(f"*{frame_idx:06d}*{ext}"))
        if matches:
            return matches[0]
        matches = sorted(image_dir.glob(f"*{frame_idx:03d}*{ext}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Stage 1 image not found: video_id={video_id}, frame_idx={frame_idx}, dir={image_dir}"
    )


def build_requests_for_stage1_video(sample: dict, args) -> list[dict]:
    label_data = load_json(sample["label_path"])
    frames = sorted(label_data.get("frames", []), key=lambda item: int(item["frame_idx"]))
    if not frames:
        return []

    frame_map = {int(frame["frame_idx"]): frame for frame in frames}
    available_indices = sorted(frame_map.keys())
    first_frame = available_indices[0]
    last_frame = available_indices[-1]

    requests = []
    start = first_frame
    gap = args.stage2_gap
    stride = args.stage2_stride

    while start + gap * 2 <= last_frame:
        selected_indices = [start, start + gap, start + gap * 2]

        if not all(idx in frame_map for idx in selected_indices):
            start += stride
            continue

        selected_frames = [frame_map[idx] for idx in selected_indices]
        try:
            image_paths = [
                resolve_stage1_image(sample["image_dir"], sample["video_id"], frame)
                for frame in selected_frames
            ]
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            start += stride
            continue

        actions = [
            summarize_actions(
                frame=frame,
                top_k=args.action_top_k,
                include_score=args.include_action_score,
            )
            for frame in selected_frames
        ]

        coverage_start = selected_indices[0] - args.stage1_window + 1
        coverage_end = selected_indices[-1]
        request_id = (
            f"{sample['video_id']}"
            f"_f{selected_indices[0]:06d}"
            f"_{selected_indices[1]:06d}"
            f"_{selected_indices[2]:06d}"
        )

        requests.append({
            "request_id": request_id,
            "group": "runtime",
            "video_id": sample["video_id"],
            "folder_coarse_label": "runtime_input",
            "frame_indices": selected_indices,
            "frame_times_sec": [frame_to_sec(idx, args.fps) for idx in selected_indices],
            "stage1_window": args.stage1_window,
            "fps": args.fps,
            "sequence_coverage": {
                "start_frame": coverage_start,
                "end_frame": coverage_end,
            },
            "images": [str(path) for path in image_paths],
            "stage1_actions": actions,
            "gt_risk_state": None,
            "gt_source": "not_available",
            "use_for_eval": None,
            "needs_review": None,
        })

        start += stride

    return requests


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "request_id",
        "video_id",
        "frame_indices",
        "frame_times_sec",
        "sequence_coverage",
        "stage1_actions",
        "image_1",
        "image_2",
        "image_3",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            images = row["images"]
            writer.writerow({
                "request_id": row["request_id"],
                "video_id": row["video_id"],
                "frame_indices": json.dumps(row["frame_indices"], ensure_ascii=False),
                "frame_times_sec": json.dumps(row["frame_times_sec"], ensure_ascii=False),
                "sequence_coverage": json.dumps(row["sequence_coverage"], ensure_ascii=False),
                "stage1_actions": json.dumps(row["stage1_actions"], ensure_ascii=False),
                "image_1": images[0],
                "image_2": images[1],
                "image_3": images[2],
            })


def build_stage2_manifest(args, paths: dict[str, Path]) -> list[dict]:
    samples = discover_stage1_videos(paths["stage1_dir"])
    if not samples:
        raise RuntimeError(f"No Stage 1 label JSON files found in {paths['stage1_dir']}")

    all_requests = []
    for sample in samples:
        requests = build_requests_for_stage1_video(sample, args)
        print(f"{sample['video_id']}: {len(requests)} Stage 2 requests")
        all_requests.extend(requests)

    if not all_requests:
        raise RuntimeError("No Stage 2 requests were created.")

    write_jsonl(paths["manifest_jsonl"], all_requests)
    write_manifest_csv(paths["manifest_csv"], all_requests)
    print("Saved Stage 2 manifest:", paths["manifest_jsonl"])
    print("Saved Stage 2 manifest summary:", paths["manifest_csv"])
    return all_requests


def run_stage2_vlm(args, paths: dict[str, Path]) -> None:
    cmd = conda_python_cmd(
        args.conda_bin,
        args.stage2_env,
        [
            "benchmark2/scripts/run_stage2_vlm.py",
            "--manifest",
            str(paths["manifest_jsonl"]),
            "--prompt",
            str(args.prompt),
            "--model-path",
            str(args.vlm_model_path),
            "--output-dir",
            str(paths["stage2_dir"]),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--batch-size",
            str(args.stage2_batch_size),
            "--max-pixels",
            str(args.max_pixels),
        ],
    )

    if args.vlm_limit is not None:
        cmd.extend(["--limit", str(args.vlm_limit)])

    run_command(cmd, cwd=args.stage2_root, dry_run=args.dry_run)


def parse_jsonish(value: str | None, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def load_stage2_summary(summary_csv: Path) -> list[dict]:
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest_by_request_id(manifest_jsonl: Path) -> dict[str, dict]:
    rows = {}
    if not manifest_jsonl.exists():
        return rows

    with manifest_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            request_id = row.get("request_id")
            if request_id:
                rows[request_id] = row

    return rows


def badge_color(risk_state: str) -> tuple[int, int, int]:
    if risk_state == "danger":
        return (220, 32, 32)
    if risk_state == "unsafe":
        return (230, 132, 20)
    return (96, 96, 96)


def overlay_risk_badge(image_path: Path, output_path: Path, risk_state: str, request_id: str) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, _ = image.size
    color = badge_color(risk_state)
    label = risk_state.upper()

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    pad_x = 14
    pad_y = 8
    label_box = draw.textbbox((0, 0), label, font=font)
    label_w = label_box[2] - label_box[0]
    label_h = label_box[3] - label_box[1]
    box_w = min(width - 20, label_w + pad_x * 2)
    box_h = label_h + pad_y * 2

    x1, y1 = 10, 10
    x2, y2 = x1 + box_w, y1 + box_h
    draw.rounded_rectangle((x1, y1, x2, y2), radius=6, fill=(*color, 220))
    draw.text((x1 + pad_x, y1 + pad_y - 2), label, fill=(255, 255, 255), font=font)

    footer = request_id[-42:]
    footer_box = draw.textbbox((0, 0), footer, font=small_font)
    footer_w = footer_box[2] - footer_box[0]
    footer_h = footer_box[3] - footer_box[1]
    fx1, fy1 = 10, y2 + 6
    draw.rectangle((fx1, fy1, fx1 + footer_w + 10, fy1 + footer_h + 8), fill=(0, 0, 0, 150))
    draw.text((fx1 + 5, fy1 + 4), footer, fill=(255, 255, 255), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_risk_outputs(paths: dict[str, Path]) -> None:
    summary_path = paths["stage2_dir"] / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Stage 2 summary not found: {summary_path}")

    rows = load_stage2_summary(summary_path)
    manifest_rows = load_manifest_by_request_id(paths["manifest_jsonl"])
    status_rows = []
    events = []
    overlay_dir = paths["overlay_dir"]

    for row in rows:
        manifest_row = manifest_rows.get(row.get("request_id"), {})
        risk_state = row.get("pred_risk_state")
        if risk_state not in RISK_STATES:
            continue

        frame_indices = parse_jsonish(row.get("frame_indices"), manifest_row.get("frame_indices", []))
        frame_times_sec = parse_jsonish(row.get("frame_times_sec"), manifest_row.get("frame_times_sec", []))
        stage1_actions = parse_jsonish(row.get("stage1_actions"), manifest_row.get("stage1_actions", []))
        image_paths = [
            row.get("image_1"),
            row.get("image_2"),
            row.get("image_3"),
        ]
        if not any(image_paths) and manifest_row.get("images"):
            image_paths = list(manifest_row["images"][:3])

        status_row = {
            "video_id": row.get("video_id"),
            "request_id": row.get("request_id"),
            "frame_indices": frame_indices,
            "frame_times_sec": frame_times_sec,
            "status_time_sec": frame_times_sec[-1] if frame_times_sec else None,
            "risk_state": risk_state,
            "stage1_actions": stage1_actions,
            "snapshot_images": image_paths,
            "latency_sec": row.get("latency_sec"),
            "json_success": row.get("json_success"),
            "schema_success": row.get("schema_success"),
        }
        status_rows.append(status_row)

        if risk_state not in {"unsafe", "danger"}:
            continue

        overlay_images = []
        if risk_state in {"unsafe", "danger"}:
            for idx, image_text in enumerate(image_paths, start=1):
                if not image_text:
                    continue
                image_path = Path(image_text)
                if not image_path.exists():
                    continue
                output_path = overlay_dir / f"{row['request_id']}_F{idx}_{risk_state}.jpg"
                overlay_risk_badge(
                    image_path=image_path,
                    output_path=output_path,
                    risk_state=risk_state,
                    request_id=row["request_id"],
                )
                overlay_images.append(str(output_path))

        event = {
            "event_id": f"evt_{len(events) + 1:06d}",
            "video_id": row.get("video_id"),
            "request_id": row.get("request_id"),
            "frame_indices": frame_indices,
            "frame_times_sec": frame_times_sec,
            "event_time_sec": frame_times_sec[-1] if frame_times_sec else None,
            "risk_state": risk_state,
            "stage1_actions": stage1_actions,
            "snapshot_images": image_paths,
            "overlay_images": overlay_images,
            "latency_sec": row.get("latency_sec"),
            "json_success": row.get("json_success"),
            "schema_success": row.get("schema_success"),
        }
        events.append(event)

    paths["final_dir"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["status_logs_jsonl"], status_rows)
    write_jsonl(paths["risk_logs_jsonl"], events)

    fieldnames = [
        "video_id",
        "request_id",
        "frame_indices",
        "frame_times_sec",
        "status_time_sec",
        "risk_state",
        "stage1_actions",
        "snapshot_images",
        "latency_sec",
        "json_success",
        "schema_success",
    ]
    with paths["status_logs_csv"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for status_row in status_rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value
                for key, value in status_row.items()
            })

    event_fieldnames = [
        "event_id",
        "video_id",
        "request_id",
        "frame_indices",
        "frame_times_sec",
        "event_time_sec",
        "risk_state",
        "stage1_actions",
        "snapshot_images",
        "overlay_images",
        "latency_sec",
        "json_success",
        "schema_success",
    ]
    with paths["risk_logs_csv"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=event_fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value
                for key, value in event.items()
            })

    print("Saved status logs:", paths["status_logs_jsonl"])
    print("Saved status log CSV:", paths["status_logs_csv"])
    print("Saved risk logs:", paths["risk_logs_jsonl"])
    print("Saved risk log CSV:", paths["risk_logs_csv"])
    print("Saved overlay images:", overlay_dir)
    print("Total status rows:", len(status_rows))
    print("Unsafe/danger events:", len(events))
    print("Unsafe/danger overlays:", sum(1 for event in events if event["overlay_images"]))


def build_paths(args) -> dict[str, Path]:
    root = args.stage2_root
    run_root = args.output_root / args.run_id
    use_selected_frames = (
        args.frames_dir is not None
        and (args.frame_video_filter is not None or args.max_frame_videos is not None)
    )
    return {
        "root": root,
        "run_root": run_root,
        "input_dir": run_root / "input",
        "source_frames_dir": args.frames_dir,
        "frames_dir": (
            run_root / "frames_selected"
            if use_selected_frames else
            (args.frames_dir if args.frames_dir else run_root / "frames_30fps")
        ),
        "stage1_dir": run_root / "stage1",
        "stage2_dir": run_root / "stage2",
        "final_dir": run_root / "final",
        "manifest_jsonl": run_root / "stage2" / "stage2_requests.jsonl",
        "manifest_csv": run_root / "stage2" / "stage2_requests_summary.csv",
        "overlay_dir": run_root / "final" / "overlay_images",
        "status_logs_jsonl": run_root / "final" / "status_logs.jsonl",
        "status_logs_csv": run_root / "final" / "status_logs.csv",
        "risk_logs_jsonl": run_root / "final" / "risk_logs.jsonl",
        "risk_logs_csv": run_root / "final" / "risk_logs.csv",
        "timings_json": run_root / "final" / "timings.json",
        "timings_csv": run_root / "final" / "timings.csv",
    }


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Run the integrated ZroAct Stage 1 + Stage 2 pipeline."
    )

    parser.add_argument("--video", default=None, help="Single input video path.")
    parser.add_argument("--video-dir", default=None, help="Input video directory.")
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Pre-extracted 30fps frame directory. If set, frame extraction is skipped.",
    )
    parser.add_argument(
        "--frame-video-filter",
        default=None,
        help="Only use frame video folders whose relative path contains this text.",
    )
    parser.add_argument(
        "--max-frame-videos",
        type=int,
        default=None,
        help="Limit the number of frame video folders passed to Stage 1.",
    )
    parser.add_argument("--run-id", default=timestamp_run_id())
    parser.add_argument(
        "--output-root",
        default=str(root / "benchmark2" / "runs"),
        help="Directory where run outputs are saved.",
    )

    parser.add_argument("--stage1-root", default=str(default_stage1_root()))
    parser.add_argument(
        "--stage1-config",
        default=str(default_stage1_root() / "config/cf/custom_shufflenet.yaml"),
    )
    parser.add_argument(
        "--stage1-pretrain-path",
        default="/data2/cache/pipeline/zroact-stage1/ckpt/YOWOv3/checkpoint/M23/ema_epoch_9.pth",
        help="Override pretrain_path in Stage 1 YAML without editing the original config.",
    )
    parser.add_argument("--stage2-root", default=str(root))
    parser.add_argument("--conda-bin", default=str(default_conda_bin()))
    parser.add_argument("--stage1-env", default="yowov3")
    parser.add_argument("--stage2-env", default="qwen35")

    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--stage1-window", type=int, default=16)
    parser.add_argument("--stage1-sample-rate", type=int, default=10)
    parser.add_argument("--stage1-conf-threshold", type=float, default=0.3)
    parser.add_argument("--stage1-batch-size", type=int, default=8)
    parser.add_argument("--stage1-video-fps", type=int, default=2)
    parser.add_argument("--stage1-make-video", action="store_true")
    parser.add_argument("--action-top-k", type=int, default=2)
    parser.add_argument("--include-action-score", action="store_true")

    parser.add_argument("--stage2-gap", type=int, default=10)
    parser.add_argument("--stage2-stride", type=int, default=30)
    parser.add_argument(
        "--prompt",
        default=str(root / "benchmark2/prompts/no_action_timev1.txt"),
    )
    parser.add_argument(
        "--vlm-model-path",
        default=str(root / "benchmark2/models/Qwen3.5-2B"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--stage2-batch-size", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=640 * 360,
                        help="Max pixels per image for VLM. Default 230400 (640x360).")
    parser.add_argument("--vlm-limit", type=int, default=None)

    parser.add_argument("--extraction-method", choices=["fps", "interpolate"], default="fps")
    parser.add_argument("--extraction-workers", type=int, default=4)

    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--use-onnx", action="store_true", default=True, help="Use ONNX Runtime for Stage 1 inference.")
    parser.add_argument("--no-onnx", dest="use_onnx", action="store_false", help="Disable ONNX and use PyTorch for Stage 1.")
    parser.add_argument(
        "--onnx-path",
        default="/data2/cache/pipeline/zroact-stage1/YOWOv3/yowov3.onnx",
        help="Path to YOWOv3 ONNX model.",
    )

    args = parser.parse_args()

    args.stage2_root = resolve_path(args.stage2_root, Path.cwd())
    args.stage1_root = resolve_path(args.stage1_root, Path.cwd())
    args.stage1_config = resolve_path(args.stage1_config, args.stage1_root)
    args.stage1_pretrain_path = resolve_path(args.stage1_pretrain_path, args.stage1_root)
    args.onnx_path = resolve_path(args.onnx_path, args.stage1_root)
    args.output_root = resolve_path(args.output_root, args.stage2_root)
    args.conda_bin = resolve_path(args.conda_bin, Path.cwd())
    args.prompt = resolve_path(args.prompt, args.stage2_root)
    args.vlm_model_path = resolve_path(args.vlm_model_path, args.stage2_root)
    args.video = resolve_path(args.video, Path.cwd())
    args.video_dir = resolve_path(args.video_dir, Path.cwd())
    args.frames_dir = resolve_path(args.frames_dir, Path.cwd())

    return args


def main() -> None:
    start = time.perf_counter()
    timings = {}
    args = parse_args()
    paths = build_paths(args)

    print("Run ID:", args.run_id)
    print("Run root:", paths["run_root"])
    print("Stage 1 root:", args.stage1_root)
    print("Stage 2 root:", args.stage2_root)
    print("Prompt:", args.prompt)
    print("VLM model:", args.vlm_model_path)

    if args.frames_dir:
        if not args.frames_dir.exists():
            raise FileNotFoundError(f"Frames dir not found: {args.frames_dir}")
        print("[USE] pre-extracted frames:", args.frames_dir)
        if paths["frames_dir"] != args.frames_dir:
            timed_step("prepare_selected_frames", timings, prepare_selected_frames, args, paths)
    elif not args.skip_extract:
        timed_step("prepare_input", timings, prepare_input, args.video, args.video_dir, paths["input_dir"])
        timed_step("extract_30fps", timings, extract_30fps, args, paths)
    else:
        print("[SKIP] frame extraction")
        if not paths["frames_dir"].exists():
            raise FileNotFoundError(
                "Frame extraction was skipped, but frames_dir does not exist: "
                f"{paths['frames_dir']}. Use --frames-dir or prepare this directory first."
            )

    if not args.skip_stage1:
        timed_step("stage1_yowov3", timings, run_stage1, args, paths)
    else:
        print("[SKIP] Stage 1")

    if args.dry_run:
        print("[DRY-RUN] Stop before manifest/VLM because Stage 1 outputs were not created.")
        return

    timed_step("build_stage2_manifest", timings, build_stage2_manifest, args, paths)

    if not args.skip_vlm:
        timed_step("stage2_vlm", timings, run_stage2_vlm, args, paths)
    else:
        print("[SKIP] Stage 2 VLM")

    if not args.skip_overlay and not args.skip_vlm and not args.dry_run:
        timed_step("final_logs_overlay", timings, write_risk_outputs, paths)
    else:
        print("[SKIP] final risk log/overlay")

    elapsed = time.perf_counter() - start
    timings["total_wall"] = elapsed
    if not args.dry_run:
        write_timings(paths, timings)
    print()
    print("Pipeline complete.")
    print("Run root:", paths["run_root"])
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
