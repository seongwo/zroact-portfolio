# Qwen3.5-0.8B CCTV LoRA Training

현재 `viz_shufflenet_all_frame` 데이터로 Qwen3.5-0.8B Vision LoRA를 학습하기 위한 구성입니다.
학습 스크립트는 실수로 실행되지 않도록 `--run` 플래그가 있어야만 시작됩니다.

데이터 구성, GT 산정, 분할, LoRA, loss 및 평가 구현의 상세 설명은
[`IMPLEMENTATION_REPORT_KO.md`](IMPLEMENTATION_REPORT_KO.md)를 참고합니다.

## 확정 구성

- 입력: `t`, `t+10`, `t+20`의 CCTV 이미지 3장
- 이미지 간격: 요청당 10프레임 (`t`, `t+10`, `t+20`)
- 요청 시작 stride: 1프레임 (`t=16`, `17`, `18`, ...)
- 행동 정보: Stage 1 행동 이름 confidence 상위 2개
- 행동 confidence 숫자: 모델 입력에서 제외
- GT: sampled frame 우선 + 36프레임 coverage 30% 규칙
- Coverage만으로 승격되는 입력 모호 샘플 60건: 학습에서 제외
- 데이터 그룹: Normal Plant + Smart Climb
- Split: 영상 단위 Train 80% / Validation 10% / Test 10%
- Train unsafe: 정확히 2배 노출
- 이미지: 메모리에서 RGB `768x432`로 변환
- 모델: Qwen3.5-0.8B, BF16 16-bit LoRA
- LoRA: Vision + Language, `r=16`, `alpha=16`, dropout 0
- Loss: 빈 think 블록 뒤의 assistant JSON에만 적용

## 폴더

```text
benchmark2/training/
├── configs/qwen35_08b_action_v2.json
├── datasets/qwen35_08b_action_v2/
├── outputs/qwen35_08b_action_v2/
├── splits/qwen35_08b_action_v0.json
├── scripts/
│   ├── build_dataset.py
│   ├── validate_dataset.py
│   ├── train_lora.py
│   └── evaluate_lora.py
└── requirements.txt
```

## 1. 학습 환경

기존 `qwen35` 추론 환경과 패키지 충돌이 생기지 않도록 별도의 독립 venv에
Unsloth 학습 환경을 설치합니다.

```bash
mkdir -p /home/capstone2/.venvs
/home/capstone2/miniconda3/envs/qwen35/bin/python \
  -m venv /home/capstone2/.venvs/qwen35-lora
source /home/capstone2/.venvs/qwen35-lora/bin/activate
cd /home/capstone2/zroact-stage2
python -m pip install --upgrade pip
python -m pip install --upgrade --no-cache-dir \
  -r benchmark2/training/requirements.txt
```

GPU 확인:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"
```

설치 후 실제 버전을 보존합니다.

```bash
mkdir -p benchmark2/training/outputs
python -m pip freeze > benchmark2/training/outputs/installed_packages.txt
```

Unsloth import 중 `flash-linear-attention` 또는 `causal_conv1d`가 없다는 오류가
나올 때만 공식 Qwen3.5 노트북과 같은 추가 커널을 설치합니다.

```bash
python -m pip install --no-cache-dir flash-linear-attention
python -m pip install --no-build-isolation --no-cache-dir causal_conv1d==1.6.0
python -m pip install --no-deps --upgrade --no-cache-dir "torchao>=0.16.0"
```

## 2. 데이터 생성

```bash
cd /home/capstone2/zroact-stage2
python benchmark2/training/scripts/build_dataset.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json
python benchmark2/training/scripts/validate_dataset.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json
```

생성 파일:

```text
benchmark2/training/datasets/qwen35_08b_action_v2/
├── all.jsonl
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── splits.json
└── summary.json
```

`splits.json`은 영상 단위 분할을 고정합니다. 실험 도중 다시 생성하지 않습니다.

## 3. Smoke Test

먼저 학습 없이 모델, LoRA, 이미지 collator와 JSON loss mask만 검사합니다.

```bash
python benchmark2/training/scripts/train_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --preflight \
  --limit-train 1 \
  --limit-validation 1
```

다음 메시지가 출력되어야 합니다.

```text
Preflight passed. Trainer was not created and training was not started.
```

32개 Train 샘플로 20 step만 실행하여 환경과 loss mask를 확인합니다.
제한 샘플은 데이터 파일 앞부분이 아니라 그룹과 클래스별로 순환 선택됩니다.

```bash
python benchmark2/training/scripts/train_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --run \
  --limit-train 32 \
  --limit-validation 16 \
  --max-steps 20 \
  --output-dir benchmark2/training/outputs/qwen35_08b_action_v2_smoke
```

시작할 때 다음이 출력되어야 합니다.

```text
Active loss tokens decode: ...{"risk_state":"..."}...
```

이 출력에 사용자 프롬프트가 포함되면 본 학습을 시작하지 않습니다.

## 4. Zero-shot 기준선

LoRA 없이 Validation을 평가합니다.

```bash
python benchmark2/training/scripts/evaluate_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --split validation \
  --output-dir benchmark2/training/outputs/zero_shot_validation
```

## 5. 본 학습

다음 명령을 실행해야 실제 학습이 시작됩니다.

```bash
python benchmark2/training/scripts/train_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --run
```

기본 출력:

```text
benchmark2/training/outputs/qwen35_08b_action_v2/
├── checkpoint-*/
└── final_adapter/
```

중단된 checkpoint에서 재개:

```bash
python benchmark2/training/scripts/train_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --run \
  --resume-from-checkpoint benchmark2/training/outputs/qwen35_08b_action_v2/checkpoint-N
```

## 6. Validation 평가

각 epoch checkpoint를 평가해 `macro_f1`이 가장 높은 것을 선택합니다.

```bash
python benchmark2/training/scripts/evaluate_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --adapter-path benchmark2/training/outputs/qwen35_08b_action_v2/checkpoint-N \
  --split validation
```

평가 결과:

```text
checkpoint-N/eval_validation/
├── metrics.json
├── predictions.jsonl
└── confusion_matrix.csv
```

## 7. 최종 Test

Validation으로 checkpoint를 선택한 뒤 Test는 한 번만 실행합니다.

```bash
python benchmark2/training/scripts/evaluate_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --adapter-path benchmark2/training/outputs/qwen35_08b_action_v2/checkpoint-N \
  --split test
```

주요 지표:

- `macro_f1`
- danger recall
- unsafe recall
- normal precision/recall
- JSON parse 성공률
- JSON schema 성공률

JSON이 아닌 출력은 `invalid` 오답으로 계산됩니다.

## 주의

- Test 결과를 보고 하이퍼파라미터를 변경하지 않습니다.
- 원본 이미지는 변경하지 않고 학습 시 메모리에서만 `768x432`로 변환합니다.
- Qwen 이미지 processor가 patch 크기에 맞추는 내부 정렬 외에는 collator 추가 축소를 하지 않습니다.
- 디스크 여유가 작으므로 checkpoint는 최대 3개만 유지합니다.
- 전체 프레임이 겹치는 요청 중 적어도 하나에는 입력 이미지로 포함됩니다.

## 참고

- [Unsloth Qwen3.5 fine-tuning](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Unsloth Vision fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning)
