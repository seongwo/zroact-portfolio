import argparse
import sys
import os
import json
import time
import re
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForImageTextToText

# Define API request model
class EvaluateRequest(BaseModel):
    request_id: str
    video_id: str
    images: List[str]
    stage1_actions: List[str]
    frame_indices: List[int]
    frame_times_sec: Optional[List[float]] = None
    fps: float = 30.0
    max_new_tokens: int = 32

app = FastAPI(title="Stage 2 Qwen VLM Daemon")

# Global variables for model and processor
model = None
processor = None
prompt_template = ""

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

def load_images(image_paths: List[str]) -> List[Image.Image]:
    images = []
    for image_path in image_paths:
        resolved = Path(image_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Image not found: {resolved}")
        images.append(Image.open(resolved).convert("RGB"))
    return images

def fill_prompt_template(
    template: str,
    stage1_actions: List[str],
    frame_indices: List[int],
    frame_times_sec: List[float],
) -> str:
    if not frame_indices:
        raise ValueError("frame_indices must contain at least 1 value")
    if len(stage1_actions) != len(frame_indices):
        raise ValueError("stage1_actions must match frame_indices length")
    if len(frame_times_sec) != len(frame_indices):
        raise ValueError("frame_times_sec must match frame_indices length")

    prompt = template
    for idx, (action, frame_idx, time_sec) in enumerate(
        zip(stage1_actions, frame_indices, frame_times_sec),
        start=1,
    ):
        prompt = prompt.replace(f"{{action_{idx}}}", action)
        prompt = prompt.replace(f"{{frame_{idx}}}", str(frame_idx))
        prompt = prompt.replace(f"{{time_{idx}}}", f"{time_sec:.3f}")
    return prompt

@app.get("/health")
def health():
    if model is None or processor is None:
        return {"status": "loading"}
    return {
        "status": "ok",
        "device": str(next(model.parameters()).device)
    }

@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load")
    
    try:
        images = load_images(req.images)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    
    frame_indices = req.frame_indices
    frame_times_sec = req.frame_times_sec
    if frame_times_sec is None:
        frame_times_sec = [round(idx / req.fps, 3) for idx in frame_indices]
        
    try:
        prompt_text = fill_prompt_template(
            template=prompt_template,
            stage1_actions=req.stage1_actions,
            frame_indices=frame_indices,
            frame_times_sec=frame_times_sec
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prompt template filling failed: {str(exc)}")

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
            max_new_tokens=req.max_new_tokens,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-path", required=True)
    args = parser.parse_args()

    print(f"Loading processor and model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
        
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
