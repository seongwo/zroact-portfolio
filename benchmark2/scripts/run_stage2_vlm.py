import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


ALLOWED_RISK_STATES = {"normal", "unsafe", "danger"}
DEFAULT_FPS = 30


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return rows


def strip_json_fence(text: str) -> str:
    text = text.strip()

    fence = "`" * 3

    if text.startswith(fence):
        text = re.sub(r"^`{3}(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"`{3}$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text


def parse_model_json(text: str):
    cleaned = strip_json_fence(text)

    try:
        return json.loads(cleaned), None
    except Exception as exc:
        return None, str(exc)


def resolve_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return (project_root / path).resolve()


def load_images(image_paths: list[str], project_root: Path) -> list[Image.Image]:
    images = []

    for image_path in image_paths:
        resolved = resolve_path(image_path, project_root)

        if not resolved.exists():
            raise FileNotFoundError(f"Image not found: {resolved}")

        images.append(Image.open(resolved).convert("RGB"))

    return images


def compute_frame_times_sec(frame_indices: list[int], fps: float = DEFAULT_FPS) -> list[float]:
    return [
        round(int(frame_idx) / fps, 3)
        for frame_idx in frame_indices
    ]


def get_request_frame_times(request: dict) -> list[float]:
    frame_indices = request.get("frame_indices", [])

    if not frame_indices:
        raise ValueError("frame_indices must contain at least 1 value")

    frame_times_sec = request.get("frame_times_sec")

    if frame_times_sec is None:
        return compute_frame_times_sec(
            frame_indices=frame_indices,
            fps=float(request.get("fps", DEFAULT_FPS)),
        )

    if len(frame_times_sec) != len(frame_indices):
        raise ValueError("frame_times_sec must match frame_indices length")

    return [
        round(float(time_sec), 3)
        for time_sec in frame_times_sec
    ]


def fill_prompt_template(
    prompt_template: str,
    stage1_actions: list[str],
    frame_indices: list[int],
    frame_times_sec: list[float],
) -> str:
    if not frame_indices:
        raise ValueError("frame_indices must contain at least 1 value")

    if len(stage1_actions) != len(frame_indices):
        raise ValueError("stage1_actions must match frame_indices length")

    if len(frame_times_sec) != len(frame_indices):
        raise ValueError("frame_times_sec must match frame_indices length")

    prompt = prompt_template

    for idx, (action, frame_idx, time_sec) in enumerate(
        zip(stage1_actions, frame_indices, frame_times_sec),
        start=1,
    ):
        prompt = prompt.replace(f"{{action_{idx}}}", action)
        prompt = prompt.replace(f"{{frame_{idx}}}", str(frame_idx))
        prompt = prompt.replace(f"{{time_{idx}}}", f"{time_sec:.3f}")

    return prompt


def risk_state_to_binary(risk_state: str | None) -> str | None:
    if risk_state == "normal":
        return "normal"

    if risk_state in {"unsafe", "danger"}:
        return "intrusion"

    return None


def folder_label_to_binary(folder_coarse_label: str | None) -> str | None:
    if folder_coarse_label == "normal_candidate":
        return "normal"

    if folder_coarse_label == "intrusion_candidate":
        return "intrusion"

    return None


def run_one_request(
    processor,
    model,
    request: dict,
    prompt_template: str,
    project_root: Path,
    max_new_tokens: int,
) -> dict:
    images = load_images(request["images"], project_root)

    stage1_actions = request.get("stage1_actions")
    frame_indices = request.get("frame_indices", [])
    frame_times_sec = get_request_frame_times(request)

    if stage1_actions is None:
        stage1_actions = ["none"] * len(frame_indices)

    prompt_text = fill_prompt_template(
        prompt_template=prompt_template,
        stage1_actions=stage1_actions,
        frame_indices=frame_indices,
        frame_times_sec=frame_times_sec,
    )

    system_prompt = (
        "You are an industrial CCTV intrusion classifier. "
        "Return only one JSON object with risk_state."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                *[
                    {"type": "image", "image": image}
                    for image in images
                ],
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
    ).to(model.device)

    start_time = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    latency_sec = time.perf_counter() - start_time

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    parsed, parse_error = parse_model_json(raw_response)

    pred_risk_state = None
    json_success = parsed is not None
    schema_success = False

    if parsed is not None:
        pred_risk_state = parsed.get("risk_state")
        schema_success = pred_risk_state in ALLOWED_RISK_STATES

    pred_binary = risk_state_to_binary(pred_risk_state)
    folder_binary_label = folder_label_to_binary(request.get("folder_coarse_label"))

    folder_binary_match = None
    if pred_binary is not None and folder_binary_label is not None:
        folder_binary_match = pred_binary == folder_binary_label

    sequence_gt = request.get("sequence_gt", request.get("gt_risk_state"))

    return {
        "request_id": request.get("request_id"),
        "group": request.get("group"),
        "video_id": request.get("video_id"),
        "folder_coarse_label": request.get("folder_coarse_label"),
        "folder_binary_label": folder_binary_label,
        "frame_indices": json.dumps(request.get("frame_indices"), ensure_ascii=False),
        "frame_times_sec": json.dumps(frame_times_sec, ensure_ascii=False),
        "sequence_coverage": json.dumps(request.get("sequence_coverage"), ensure_ascii=False),
        "stage1_actions": json.dumps(stage1_actions, ensure_ascii=False),
        "image_1": request["images"][0] if len(request["images"]) > 0 else "",
        "image_2": request["images"][1] if len(request["images"]) > 1 else "",
        "image_3": request["images"][2] if len(request["images"]) > 2 else "",
        "sampled_frame_gt": json.dumps(request.get("sampled_frame_gt"), ensure_ascii=False),
        "sampled_only_gt": request.get("sampled_only_gt"),
        "coverage_frame_count": json.dumps(request.get("coverage_frame_count"), ensure_ascii=False),
        "coverage_ratio": json.dumps(request.get("coverage_ratio"), ensure_ascii=False),
        "coverage_based_gt": request.get("coverage_based_gt"),
        "sequence_gt": sequence_gt,
        "gt_risk_state": request.get("gt_risk_state"),
        "use_for_eval": request.get("use_for_eval"),
        "needs_review": request.get("needs_review"),
        "gt_source": request.get("gt_source"),
        "gt_rule": request.get("gt_rule"),
        "pred_risk_state": pred_risk_state,
        "pred_binary": pred_binary,
        "folder_binary_match": folder_binary_match,
        "json_success": json_success,
        "schema_success": schema_success,
        "latency_sec": round(latency_sec, 4),
        "raw_response": raw_response,
        "parse_error": parse_error,
        "prompt_text": prompt_text,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "request_id",
        "group",
        "video_id",
        "folder_coarse_label",
        "folder_binary_label",
        "frame_indices",
        "frame_times_sec",
        "sequence_coverage",
        "stage1_actions",
        "sampled_frame_gt",
        "sampled_only_gt",
        "coverage_frame_count",
        "coverage_ratio",
        "coverage_based_gt",
        "sequence_gt",
        "gt_risk_state",
        "use_for_eval",
        "needs_review",
        "gt_source",
        "gt_rule",
        "pred_risk_state",
        "pred_binary",
        "folder_binary_match",
        "json_success",
        "schema_success",
        "latency_sec",
        "raw_response",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        default="benchmark2/outputs/manifests/stage2_requests.jsonl",
    )
    parser.add_argument(
        "--prompt",
        default="benchmark2/prompts/intrusion_short.txt",
    )
    parser.add_argument(
        "--model-path",
        default="benchmark2/models/Qwen3.5-0.8B",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark2/results/qwen35_08b_intrusion_short",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)

    args = parser.parse_args()

    project_root = Path.cwd()

    manifest_path = resolve_path(args.manifest, project_root)
    prompt_path = resolve_path(args.prompt, project_root)
    model_path = resolve_path(args.model_path, project_root)
    output_dir = resolve_path(args.output_dir, project_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", project_root)
    print("Manifest:", manifest_path)
    print("Prompt:", prompt_path)
    print("Model:", model_path)
    print("Output dir:", output_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    requests = load_jsonl(manifest_path)

    if args.limit is not None:
        requests = requests[:args.limit]

    if not requests:
        raise RuntimeError("No requests found in manifest.")

    prompt_template = prompt_path.read_text(encoding="utf-8")

    print("Requests:", len(requests))

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("Model device:", next(model.parameters()).device)

    rows = []

    for idx, request in enumerate(requests, start=1):
        print()
        print(f"[{idx}/{len(requests)}] {request.get('request_id')}")

        try:
            row = run_one_request(
                processor=processor,
                model=model,
                request=request,
                prompt_template=prompt_template,
                project_root=project_root,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            frame_indices = request.get("frame_indices")
            try:
                frame_times_sec = get_request_frame_times(request)
            except Exception:
                frame_times_sec = request.get("frame_times_sec")

            row = {
                "request_id": request.get("request_id"),
                "group": request.get("group"),
                "video_id": request.get("video_id"),
                "folder_coarse_label": request.get("folder_coarse_label"),
                "frame_indices": json.dumps(frame_indices, ensure_ascii=False),
                "frame_times_sec": json.dumps(frame_times_sec, ensure_ascii=False),
                "sequence_coverage": json.dumps(request.get("sequence_coverage"), ensure_ascii=False),
                "stage1_actions": json.dumps(request.get("stage1_actions"), ensure_ascii=False),
                "pred_risk_state": None,
                "pred_binary": None,
                "folder_binary_match": None,
                "json_success": False,
                "schema_success": False,
                "latency_sec": None,
                "raw_response": "",
                "parse_error": str(exc),
                "prompt_text": "",
            }

        rows.append(row)

        print("folder label:", row.get("folder_coarse_label"))
        print("pred:", row.get("pred_risk_state"))
        print("binary:", row.get("pred_binary"))
        print("folder binary match:", row.get("folder_binary_match"))
        print("latency:", row.get("latency_sec"))
        print("raw:", row.get("raw_response"))

    raw_path = output_dir / "raw_results.jsonl"
    summary_path = output_dir / "summary.csv"

    write_jsonl(raw_path, rows)
    write_summary_csv(summary_path, rows)

    print()
    print("Saved raw results:", raw_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()


# 실행 예시 (Stage 1이 끝나고 Stage 2에 들어갈 request 목록만 만드는 코드 참고)


# cd ~/zroact-stage2

# python3 benchmark2/scripts/run_stage2_vlm.py \
#   --manifest benchmark2/outputs/manifests/stage2_requests_with_gt.jsonl \
#   --prompt benchmark2/prompts/intrusion_action_timev2.txt \
#   --model-path benchmark2/models/Qwen3.5-2B \
#   --output-dir benchmark2/results/qwen35_2b_v2_gt_all \
#   --max-new-tokens 32
