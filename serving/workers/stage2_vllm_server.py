"""
Stage 2 VLM server using vLLM backend (Qwen3.5-2B).
Same /evaluate and /health API as stage2_server.py.
"""
import argparse
import asyncio
import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

class EvaluateRequest(BaseModel):
    request_id: str
    video_id: str
    images: List[str]          # file paths
    stage1_actions: List[str]
    frame_indices: List[int]
    frame_times_sec: Optional[List[float]] = None
    fps: float = 30.0
    max_new_tokens: int = 256  # larger than transformers default; vLLM handles it

app = FastAPI(title="Stage 2 Qwen3.5 vLLM Daemon")

# Globals set at startup
llm_engine = None
sampling_params = None
prompt_template = ""
max_pixels_cfg = 230400

import threading
_inference_lock = threading.Lock()  # vLLM LLM class is not thread-safe; serialize calls

ALLOWED_RISK_STATES = {"normal", "unsafe", "danger"}


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


def fill_prompt_template(
    template: str,
    stage1_actions: List[str],
    frame_indices: List[int],
    frame_times_sec: List[float],
) -> str:
    prompt = template
    for idx, (action, frame_idx, time_sec) in enumerate(
        zip(stage1_actions, frame_indices, frame_times_sec), start=1
    ):
        prompt = prompt.replace(f"{{action_{idx}}}", action)
        prompt = prompt.replace(f"{{frame_{idx}}}", str(frame_idx))
        prompt = prompt.replace(f"{{time_{idx}}}", f"{time_sec:.3f}")
    return prompt


def resize_image(img: Image.Image, max_pixels: int) -> Image.Image:
    """Resize image so total pixels <= max_pixels."""
    w, h = img.size
    total = w * h
    if total <= max_pixels:
        return img
    import math
    scale = math.sqrt(max_pixels / total)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # round to multiples of 16 for vision encoder patch alignment
    new_w = max(16, (new_w // 16) * 16)
    new_h = max(16, (new_h // 16) * 16)
    return img.resize((new_w, new_h), Image.BICUBIC)


def image_to_base64(path: str, max_pixels: int) -> str:
    img = Image.open(path).convert("RGB")
    img = resize_image(img, max_pixels)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
def health():
    if llm_engine is None:
        return {"status": "loading"}
    return {"status": "ok", "backend": "vllm"}


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    # Validate image paths
    for p in req.images:
        if not Path(p).exists():
            raise HTTPException(status_code=400, detail=f"Image not found: {p}")

    frame_times_sec = req.frame_times_sec
    if frame_times_sec is None:
        frame_times_sec = [round(idx / req.fps, 3) for idx in req.frame_indices]

    try:
        prompt_text = fill_prompt_template(
            template=prompt_template,
            stage1_actions=req.stage1_actions,
            frame_indices=req.frame_indices,
            frame_times_sec=frame_times_sec,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prompt template error: {exc}")

    system_prompt = (
        "You are an industrial CCTV intrusion classifier. "
        "Return only one JSON object with risk_state."
    )

    # Build vLLM chat messages with base64 images
    image_contents = []
    for img_path in req.images:
        b64 = image_to_base64(img_path, max_pixels_cfg)
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                *image_contents,
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    # Run inference (serialized — vLLM LLM class is not thread-safe)
    t0 = time.perf_counter()
    try:
        loop = asyncio.get_event_loop()
        def _infer():
            with _inference_lock:
                return llm_engine.chat(messages, sampling_params=sampling_params)
        outputs = await loop.run_in_executor(None, _infer)
        raw_response = outputs[0].outputs[0].text
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"vLLM inference error: {exc}")
    latency_sec = time.perf_counter() - t0

    parsed, parse_error = parse_model_json(raw_response)
    pred_risk_state = None
    json_success = parsed is not None
    schema_success = False
    if parsed is not None:
        pred_risk_state = parsed.get("risk_state")
        schema_success = pred_risk_state in ALLOWED_RISK_STATES

    return {
        "request_id": req.request_id,
        "pred_risk_state": pred_risk_state,
        "json_success": json_success,
        "schema_success": schema_success,
        "latency_sec": round(latency_sec, 4),
        "raw_response": raw_response,
        "parse_error": parse_error,
    }


if __name__ == "__main__":
    import uvicorn
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--max-pixels", type=int, default=640 * 360)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    max_pixels_cfg = args.max_pixels

    print(f"Loading vLLM engine from {args.model_path}...")
    print(f"max_pixels: {args.max_pixels}, max_new_tokens: {args.max_new_tokens}")

    llm_engine = LLM(
        model=args.model_path,
        dtype="half",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        repetition_penalty=1.05,
    )

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    print(f"vLLM engine ready. Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
