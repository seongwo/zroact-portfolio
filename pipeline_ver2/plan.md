# ZroAct 실시간 병렬 처리 파이프라인 설계 계획

## 현재 구조 파악

### 디렉터리 구성
```
zroact-stage2/
├── pipeline/
│   └── main.py                  # 구버전 순차 처리 파이프라인
├── pipeline_ver2/
│   ├── main.py                  # 공통 유틸리티 (extract_30fps, write_timings 등)
│   └── realtime_pipeline.py     # 현재 버전 (asyncio + aiohttp HTTP 기반)
└── serving/
    ├── app.py                   # FastAPI 잡 제출 API
    ├── config.json              # 포트, 경로 설정
    ├── run_job.py               # 단일 잡 실행기
    └── workers/
        ├── stage1_server.py     # YOWOv3 ONNX HTTP 데몬 (Port 8001)
        ├── stage2_server.py     # Qwen3.5 VLM HTTP 데몬 (Port 8002)
        └── scheduler.py        # RealtimeScheduler (미구현 stub)
```

### 현재 파이프라인 흐름 (realtime_pipeline.py)
```
main()
 ├── Stage 1 서버 spawn (uvicorn, Port 8001)
 ├── Stage 2 서버 spawn (uvicorn, Port 8002)
 └── asyncio.run(run_pipeline_async())
      ├── [배치 루프] keyframe_batches 순서대로 await detect_clip_batch()  ← Stage 1 HTTP POST
      ├── [인라인 체크] predictions 준비된 3-frame 슬롯마다 asyncio.create_task(evaluate_vlm())  ← Stage 2 비동기 태스크 생성
      └── await asyncio.gather(*vlm_tasks)  ← Stage 2 전부 완료 대기
```

### 현재 성능 (419 프레임, 14초 영상 기준)
| 구분 | 시간 |
|------|------|
| Stage 1 (41 클립) | 6.71초 (클립당 163ms) |
| Stage 2 (13 VLM 요청, semaphore=1 직렬) | 26.93초 (요청당 2.07초) |
| 스트리밍 루프 전체 | **27.91초** |

### 하드웨어
- GPU: **NVIDIA RTX A6000 (47.5 GB VRAM)**
- Stage 1 ONNX: ~1 GB VRAM
- Stage 2 Qwen3.5-2B: ~5 GB VRAM
- 여유 VRAM: 약 15~16 GB

---

## 현재 구현의 잔존 병목

### 병목 1: Stage 1 배치 루프가 동기적 (await per batch)
```python
for kf_batch in keyframe_batches:         # ← 순서대로 기다림
    resp_data = await detect_clip_batch(...)  # ← 이전 배치가 끝날 때까지 블로킹
    # → Stage 1 자체 내부는 병렬화되지 않음
```
배치가 1개씩 직렬로 전송되므로, 배치 사이즈 1일 때 ONNX 추론 사이의 idle 시간이 큼.

### 병목 2: VLM semaphore=1 (Stage 2 직렬 처리)
```python
vlm_semaphore = asyncio.Semaphore(1)
```
VLM이 1개 요청씩만 처리하므로, 13개 요청이 2.07초 × 13 = 26.9초 직렬 소요.
RTX A6000의 여유 VRAM(~16GB)을 활용하지 못하고 있음.

### 병목 3: Stage 1 → Stage 2 전환 시점 지연
Stage 2 태스크가 `asyncio.create_task`로 즉시 큐에 들어가지만,  
asyncio 이벤트 루프가 Stage 1 await 처리 중에는 실제 HTTP 요청 시작이 지연됨.

---

## 병렬처리 개선 방안

---

### 방안 A: Stage 1 배치 완전 비동기화 (asyncio.gather for Stage 1)

#### 개념
현재 `for` 루프로 순서대로 Stage 1 배치를 보내는 대신,  
**모든 배치를 동시에 asyncio.gather로 Stage 1에 전송**하여 Stage 1 자체 처리를 병렬화.

#### 구현 방식
```python
# 현재 (순차)
for kf_batch in keyframe_batches:
    resp = await detect_clip_batch(...)

# 방안 A (병렬)
tasks = [detect_clip_batch(..., clips=build_payload(kf_batch)) for kf_batch in keyframe_batches]
results = await asyncio.gather(*tasks)
```

#### 장점
- 구현 단순, 코드 변경 최소
- Stage 1 HTTP 왕복 레이턴시 오버헤드 제거
- Stage 1 ONNX 서버가 동시 요청 처리 가능 (uvicorn은 async 처리 지원)

#### 단점
- ONNX 서버가 동시 요청을 받으면 실제로 GPU에서 직렬 처리됨 (ONNX Session은 스레드 비안전)
- 큰 배치를 한꺼번에 Stage 1에 보내면 첫 Stage 2 태스크 시작이 오히려 지연될 수 있음

#### 예상 효과
- Stage 1 HTTP 오버헤드 감소: 약 20~30% 단축 가능

---

### 방안 B: Stage 2 VLM 병렬 처리 (semaphore 해제 또는 완화)

#### 개념
현재 `vlm_semaphore = asyncio.Semaphore(1)`이 Stage 2 호출을 직렬화하고 있음.  
RTX A6000의 여유 VRAM(약 15GB)을 활용해 VLM 요청을 동시에 처리하도록 semaphore 값을 올림.

#### 구현 방식
```python
# 현재
vlm_semaphore = asyncio.Semaphore(1)   # 직렬

# 방안 B-1: semaphore=2 (두 요청 동시)
vlm_semaphore = asyncio.Semaphore(2)

# 방안 B-2: semaphore 완전 제거 (모두 동시)
# evaluate_vlm() 에서 semaphore 사용 제거
```

#### VLM 동시 처리 시 VRAM 추정
- Qwen3.5-2B 모델 가중치: ~5 GB
- 입력 (3장 이미지 + 프롬프트) 활성화 메모리: 요청당 ~1~2 GB 추가
- semaphore=2: ~7~9 GB (안전)
- semaphore=4: ~11~13 GB (안전권 내)
- semaphore=무제한: 13개 동시 → 약 26+ GB (OOM 위험)

#### 주의사항
- Qwen VLM 내부 `model.generate()`는 `torch.no_grad()` 아래 단일 스레드 GPU 연산
- 동시 HTTP 요청이 와도 stage2_server의 uvicorn이 async endpoint를 받아 처리하지만,  
  `model.generate()`는 GIL + CUDA 직렬 연산이므로 실제 throughput 향상은 제한적
- **asyncio 단계에서의 이미지 로딩(I/O)** 은 병렬화됨

#### 예상 효과
- semaphore=2~3: Stage 2 이미지 로딩 병렬화로 5~15% 단축
- 실제 GPU 추론 가속은 CUDA stream 활용 없이는 어려움

---

### 방안 C: asyncio Producer-Consumer 큐 기반 파이프라이닝

#### 개념
Stage 1과 Stage 2를 **완전히 독립적인 비동기 태스크로 분리**하여,  
Stage 1이 한 클립씩 결과를 생산하는 동시에 Stage 2가 즉시 소비하는 스트림 구조.

#### 구현 방식
```python
stage2_queue = asyncio.Queue()

async def stage1_producer(session, keyframe_indices, stage2_queue):
    """Stage 1 결과를 하나씩 생산하여 큐에 넣음"""
    for kf_batch in keyframe_batches:
        resp = await detect_clip_batch(...)
        for key_idx, result in zip(kf_batch, resp["results"]):
            predictions[key_idx] = result
            # 3-frame 슬롯이 완성되면 즉시 Stage 2 요청을 큐에 넣음
            if check_slot_ready(predictions, key_idx):
                await stage2_queue.put(build_vlm_request(predictions, key_idx))
    await stage2_queue.put(None)  # sentinel

async def stage2_consumer(session, stage2_queue, results_list):
    """Stage 2 큐에서 꺼내 VLM 추론 수행"""
    async with asyncio.Semaphore(1):
        while True:
            req = await stage2_queue.get()
            if req is None:
                break
            result = await evaluate_vlm(session, stage2_url, req)
            results_list.append(result)

# 두 태스크를 동시에 실행
await asyncio.gather(
    stage1_producer(session, keyframe_indices, stage2_queue),
    stage2_consumer(session, stage2_queue, vlm_results)
)
```

#### 장점
- Stage 1이 한 클립 처리하자마자 Stage 2가 즉시 시작 → 최대 오버랩
- 메모리 효율적 (results 전체를 한꺼번에 hold하지 않음)
- 실시간 CCTV 스트림 연동 시 자연스러운 구조

#### 단점
- 코드 복잡도 증가 (Producer/Consumer 패턴 + sentinel 처리)
- `check_slot_ready` 로직이 기존 `scheduled_starts` 집합 관리와 중복될 수 있음
- 순서 보장 필요 시 추가 구현 필요

#### 예상 효과
- 현재 구현 대비 Stage 2 시작 시점을 최대 6~8초 앞당길 수 있음
- 전체 루프 시간 27.91초 → **20~22초** 수준 목표

---

### 방안 D: multiprocessing + 공유 메모리 기반 Zero-Copy 파이프라인

#### 개념
Stage 1과 Stage 2를 **별도 OS 프로세스**로 분리하고,  
`torch.multiprocessing`의 shared memory 또는 `mp.Queue`로 텐서를 직접 전달하여  
HTTP 직렬화/역직렬화 오버헤드를 제거.

#### 구현 방식
```
프로세스 구성:
  Main Process
    ├── Stage1Process  (yowov3 conda env)
    │     ├── ONNX 추론
    │     └── mp.Queue → predictions_queue에 결과 넣음
    └── Stage2Process  (qwen35 conda env)
          ├── predictions_queue에서 꺼냄
          └── VLM 추론 → results_queue에 결과 넣음
```

#### 구현 예시
```python
from multiprocessing import Process, Queue

predictions_queue = Queue(maxsize=20)
results_queue = Queue()

def stage1_worker(frame_dir, predictions_queue):
    # ONNX 모델 로드
    # 프레임 읽어서 추론
    # predictions_queue.put(result)
    pass

def stage2_worker(predictions_queue, results_queue):
    # VLM 모델 로드
    # predictions_queue.get() → 추론 → results_queue.put()
    pass

p1 = Process(target=stage1_worker, ...)
p2 = Process(target=stage2_worker, ...)
p1.start(); p2.start()
p1.join(); p2.join()
```

#### 장점
- HTTP 오버헤드 완전 제거 (JSON 직렬화 없음)
- 두 conda 환경을 하나의 파이썬 스크립트로 통합 불가하므로 현재 구조 유지 필요
- GIL 우회로 CPU 전처리 병렬화 가능

#### 단점
- **현재 conda 환경 분리**(yowov3, qwen35)와 충돌 → subprocess로 실행해야 함
- 프로세스 간 GPU 텐서 공유는 CUDA IPC 필요 → 복잡도 매우 높음
- 디버깅 어려움, 에러 전파 복잡

#### 예상 효과
- 이상적: HTTP 오버헤드(요청당 ~10~50ms) 제거
- 현실적: conda 환경 분리 문제로 직접 적용 어렵고, HTTP 방식 유지가 더 실용적

---

### 방안 E: Stage 1 ONNX 서버 내부 배치 처리 최적화

#### 개념
현재 Stage 1 서버는 요청이 오면 클립을 스택하여 배치 추론을 수행하지만,  
`await` 사이에 여러 요청이 쌓일 경우 **동적 배치(Dynamic Batching)** 를 자동으로 묶어 처리.

#### 구현 방식 (stage1_server.py 내부)
```python
# 기존: 요청마다 바로 추론
# 개선: 짧은 시간(예: 10ms) 기다린 후 쌓인 요청을 묶어서 한 번에 추론

pending_clips = []
pending_futures = []
batch_lock = asyncio.Lock()

async def flush_batch():
    async with batch_lock:
        if pending_clips:
            # 한 번에 묶어서 ONNX 추론
            batch = stack_clips(pending_clips)
            results = ort_session.run(None, {input_name: batch})
            for fut, res in zip(pending_futures, results):
                fut.set_result(res)
            pending_clips.clear()
            pending_futures.clear()
```

#### 장점
- Stage 1 GPU 활용도 향상 (배치 처리 효율)
- 요청량 많을수록 이점 증가 (멀티 CCTV 상황)

#### 단점
- 단일 CCTV 환경에서는 이점 제한적
- batching 대기 시간(latency) 추가

---

### 방안 F: asyncio + threading 혼합 (run_in_executor)

#### 개념
`asyncio`의 이벤트 루프를 Block하는 이미지 로딩, 파일 I/O 등을  
`loop.run_in_executor(ThreadPoolExecutor)`로 스레드 풀에서 실행하여  
Stage 1 HTTP 대기 중에 다음 배치 프리페치(prefetch) 수행.

#### 구현 방식
```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def prefetch_clip_frames(loop, frame_paths, key_idx, clip_length, sampling_rate):
    """다음 배치 프레임을 미리 디스크에서 읽어 놓음 (블로킹 I/O를 스레드로 분리)"""
    def _load():
        return [str(frame_paths[max(0, key_idx - i * sampling_rate - 1)])
                for i in reversed(range(clip_length))]
    return await loop.run_in_executor(executor, _load)
```

#### 장점
- 현재 LRU 캐시와 시너지 효과
- 구현 비교적 간단

#### 단점
- 이미 LRU 캐시가 적용되어 디스크 I/O가 거의 없어 추가 효과 제한적

---

## 방안별 비교 요약

| 방안 | 구현 난이도 | 예상 효과 | 적용 추천 여부 |
|------|-----------|----------|--------------|
| A. Stage 1 gather 병렬화 | ⭐ 쉬움 | Stage 1 20~30% 단축 | ✅ 즉시 적용 가능 |
| B. VLM semaphore 완화 | ⭐ 쉬움 | Stage 2 5~15% 단축 (VRAM 확인 필요) | ✅ 즉시 적용 가능 |
| C. Producer-Consumer 큐 | ⭐⭐ 보통 | 전체 20~30% 단축 | ✅ 중기 적용 추천 |
| D. multiprocessing IPC | ⭐⭐⭐⭐ 어려움 | HTTP 오버헤드 제거 (현실적으로 어려움) | ❌ 현재 구조와 충돌 |
| E. 동적 배치 처리 | ⭐⭐⭐ 보통 | 멀티CCTV 환경에서 효과적 | 🔶 멀티CCTV 상황 시 |
| F. run_in_executor 프리페치 | ⭐ 쉬움 | LRU 이미 있어 제한적 | 🔶 필요 시 |

---

## 추천 적용 순서

### Phase 1 (즉시, 코드 변경 최소)
1. **방안 B**: `vlm_semaphore` 값을 2~3으로 올려 VLM 이미지 로딩 병렬화  
   → VRAM 여유분(~15GB)으로 안전하게 처리 가능
2. **방안 A**: Stage 1 배치 전송을 `asyncio.gather`로 변경

### Phase 2 (중기, 구조 개선)
3. **방안 C**: asyncio Queue 기반 Producer-Consumer로 전면 리팩토링  
   → Stage 1의 첫 클립 완료 즉시 Stage 2 시작으로 진정한 실시간 파이프라이닝 구현  
   → `scheduler.py`의 `RealtimeScheduler` stub을 이 구조로 채워넣기

### Phase 3 (장기, 멀티카메라)
4. **방안 E**: Stage 1 동적 배치로 멀티 CCTV 동시 처리 지원

---

## 참고: 현재 파이프라인 타임라인 (배치 사이즈=1)

```
시간축 (초):
0     2     4     6     8    10    12    14    16    18    20    22    24    26    28
|     |     |     |     |     |     |     |     |     |     |     |     |     |     |
├─────────────── Stage 1 (6.7초) ───────────────┤
                                    ├─ VLM1 ─┤
                                          ├─ VLM2 ─┤
                                                 ├─ VLM3 ─┤
                                                       ...
                                                              ├─ VLM13─┤  (27.9초)

이상적인 완전 파이프라이닝 목표:
0     2     4     6     8    10    12    14
|     |     |     |     |     |     |     |
├─────────────── Stage 1 ───────────────┤
              ├─ VLM1 ─┤
                        ├─ VLM2 ─┤
                              ├─ VLM3 ─┤ ← Stage 2 VLM 직렬이 병목
                                              ...끝
→ 목표 Wall Time: ~15~18초 (현재 27.9초 대비 ~40% 단축)
```
