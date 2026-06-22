import sys
import os
import time
from pathlib import Path
from typing import List
from collections import OrderedDict
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

frame_cache = LRUCache(capacity=300)


# Add yowov3 root to python path to import its utils
stage1_root = os.environ.get("STAGE1_ROOT", "/data2/cache/pipeline/zroact-stage1/YOWOv3")
if stage1_root not in sys.path:
    sys.path.insert(0, stage1_root)
# Note: sys.path is set by stage1_launcher.py before this module is imported.

import onnxruntime
from utils.build_config import build_config
from utils.box import non_max_suppression
import torchvision.transforms.functional as FT

app = FastAPI(title="Stage 1 YOWOv3 ONNX Daemon")

# Load config
config_path = os.path.join(stage1_root, "config/cf/custom_shufflenet.yaml")
config = build_config(config_path)

mapping = config['idx2name']
clip_length = config['clip_length']  # 16
sampling_rate = config['sampling_rate'] # 10
img_size = config['img_size'] # 224

# Default blacklist
DEFAULT_BLACK_LIST = [2, 4, 7, 9, 13, 16, 18, 19, 21, 22, 23, 24, 25, 29, 31, 32, 33, 35, 36, 39, 40, 41, 42, 44, 45, 46, 49, 50, 52, 53, 55, 56, 57, 58, 59, 60, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]

# Initialize ONNX runtime session
onnx_path = os.environ.get("STAGE1_ONNX_PATH", os.path.join(stage1_root, "yowov3.onnx"))
_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
ort_session = onnxruntime.InferenceSession(onnx_path, providers=_providers)
input_name = ort_session.get_inputs()[0].name
_active = ort_session.get_providers()
device = "cuda" if "CUDAExecutionProvider" in _active else "cpu"
print(f"[Stage 1 Server] Loaded ONNX session | device: {device} | providers: {_active}")

class live_transform():
    def __init__(self, img_size):
        self.img_size = img_size

    def to_tensor(self, image):
        return FT.to_tensor(image)
    
    def normalize(self, clip):
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std  = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        clip -= mean
        clip /= std
        return clip
    
    def __call__(self, img):
        img = img.resize([self.img_size, self.img_size])
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img

transform = live_transform(img_size)

class ClipRequest(BaseModel):
    # A list of image paths of length 16 representing a single clip.
    # To support dynamic batching, we accept a list of clips.
    clips: List[List[str]]
    conf_threshold: float = 0.3
    top_k: int = 2
    use_blacklist: bool = True

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "active_providers": ort_session.get_providers(),
        "input_name": input_name,
        "clip_length": clip_length,
        "img_size": img_size
    }

@app.post("/detect")
async def detect(req: ClipRequest):
    if not req.clips:
        raise HTTPException(status_code=400, detail="No clips provided")
    
    # Preprocess each clip
    batch_clips = []
    for clip_paths in req.clips:
        if len(clip_paths) != clip_length:
            raise HTTPException(
                status_code=400, 
                detail=f"Each clip must have exactly {clip_length} frames. Got {len(clip_paths)}"
            )
        
        clip_tensors = []
        for path in clip_paths:
            cached_tensor = frame_cache.get(path)
            if cached_tensor is not None:
                clip_tensors.append(cached_tensor)
                continue
            
            if not os.path.exists(path):
                raise HTTPException(status_code=400, detail=f"Image path does not exist: {path}")
            try:
                img = Image.open(path).convert('RGB')
                tensor = transform(img)
                frame_cache.put(path, tensor)
                clip_tensors.append(tensor)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load image {path}: {str(e)}")
        
        # Stack to shape [C, T, H, W]
        clip_tensor = torch.stack(clip_tensors, dim=0).permute(1, 0, 2, 3).contiguous()
        batch_clips.append(clip_tensor)
    
    # Stack to shape [B, C, T, H, W]
    batch_tensor = torch.stack(batch_clips, dim=0)
    clips_np = batch_tensor.numpy()
    
    # Inference
    start_time = time.perf_counter()
    ort_inputs = {input_name: clips_np}
    ort_outputs = ort_session.run(None, ort_inputs)
    inference_time = time.perf_counter() - start_time
    
    outputs = torch.from_numpy(ort_outputs[0]).to(device)
    
    # Post-processing (NMS)
    # non_max_suppression expects float tensors on device
    nms_outputs = non_max_suppression(outputs.float(), conf_threshold=req.conf_threshold, iou_threshold=0.5)
    
    black_list = DEFAULT_BLACK_LIST if req.use_blacklist else []
    
    batch_results = []
    for idx, dets in enumerate(nms_outputs):
        grouped_dets = {}
        if dets is not None and dets.size(0) > 0:
            for det in dets:
                x1, y1, x2, y2, score, label = det.tolist()
                label_1indexed = int(label) + 1
                
                if req.use_blacklist and (label_1indexed in black_list):
                    continue

                if score < req.conf_threshold:
                    continue

                box_key = (round(x1 / img_size, 4), round(y1 / img_size, 4), 
                           round(x2 / img_size, 4), round(y2 / img_size, 4))
                
                if box_key not in grouped_dets:
                    grouped_dets[box_key] = []
                grouped_dets[box_key].append((label_1indexed, score))
        
        detections_out = []
        for box_key, actions in grouped_dets.items():
            actions.sort(key=lambda x: x[1], reverse=True)
            kept = actions[:req.top_k]
            if not kept:
                continue
            
            detections_out.append({
                'box': {'x1': box_key[0], 'y1': box_key[1], 'x2': box_key[2], 'y2': box_key[3]},
                'actions': [
                    {'class_id': aid, 'class_name': mapping.get(aid - 1, f'id_{aid}'), 'score': round(sc, 4)}
                    for aid, sc in kept
                ]
            })
            
        batch_results.append(detections_out)
        
    return {
        "results": batch_results,
        "inference_time_sec": inference_time
    }
