# Qwen3.5-0.8B CCTV 위험 상태 분류 LoRA 학습 구현 보고서

## 1. 문서 개요

| 항목 | 내용 |
|---|---|
| 작성 기준일 | 2026-06-09 |
| 프로젝트 경로 | `/home/capstone2/zroact-stage2` |
| 학습 구성 경로 | `benchmark2/training` |
| 대상 모델 | Qwen3.5-0.8B Vision |
| 학습 방식 | Unsloth 기반 BF16 LoRA SFT |
| 입력 | 시간순 CCTV 이미지 3장 + Stage 1 행동 정보 |
| 출력 | `{"risk_state":"normal|unsafe|danger"}` |
| 데이터 분할 | 영상 단위 Train 80% / Validation 10% / Test 10% |
| 현재 상태 | v2 학습, checkpoint-2794 Validation 및 base zero-shot Test 완료 |

이 문서는 현재 구현된 데이터 생성, GT 산정, 데이터 분할, 이미지 전처리,
프롬프트 구성, Qwen3.5-0.8B LoRA 학습, loss masking, 평가 및 실행 절차를
하나의 보고서로 정리한 것이다.

최신 v2 학습 및 평가 결과는 다음 별도 보고서에 요약했다.

```text
benchmark2/training/V2_RESULTS_REPORT_KO.md
```

1차 데이터로 LoRA 학습과 Validation/Test 평가까지 수행했으나 이후 라벨
문제가 확인되어 해당 checkpoint, adapter, 평가 결과와 JSONL 데이터셋을
폐기했다. 현재는 Plant Climb을 제외한 전체 프레임 데이터로 v2 JSONL 생성,
검증 및 학습 전 preflight까지 완료한 상태다.

---

## 2. 구현 목적

Stage 1에서 추출한 행동 정보와 CCTV 이미지 시퀀스를 함께 입력하여, Stage 2
멀티모달 모델이 침입 위험 상태를 다음 세 등급 중 하나로 분류하도록
미세 조정하는 것이 목적이다.

| 위험 상태 | 의미 |
|---|---|
| `normal` | 침입과 관련된 행동이 확인되지 않음 |
| `unsafe` | 펜스나 경계에 접근, 확인, 접촉, 대기 또는 준비 중이지만 명확한 등반은 아님 |
| `danger` | 펜스 등반, 발판 진입, 신체의 경계 통과 등 명확한 침입 증거가 있음 |

최종 모델은 설명문이나 confidence 값을 출력하지 않고 다음처럼 JSON 하나만
출력하도록 학습한다.

```json
{"risk_state":"danger"}
```

---

## 3. 구현 결과 요약

현재 생성된 데이터셋은 총 258개 영상과 2,342개 학습 요청으로 구성된다.
각 요청은 서로 다른 시간의 이미지 3장을 사용하므로 실제 참조 이미지 수는
7,026장이다.

| 구분 | 영상 수 | 요청 수 | Normal | Unsafe | Danger |
|---|---:|---:|---:|---:|---:|
| 전체 | 258 | 2,342 | 746 | 220 | 1,376 |
| Train 원본 | 206 | 1,862 | 596 | 176 | 1,090 |
| Validation | 26 | 242 | 77 | 22 | 143 |
| Test | 26 | 238 | 73 | 22 | 143 |

전체 클래스 비율은 다음과 같다.

| 클래스 | 개수 | 비율 |
|---|---:|---:|
| Normal | 746 | 31.85% |
| Unsafe | 220 | 9.39% |
| Danger | 1,376 | 58.75% |

Train에서만 `unsafe` 요청을 2배로 노출한다. JSONL 파일 자체를 복제하지 않고
학습 시작 시 메모리의 데이터 목록에서 해당 행을 두 번 넣는다.

| Train 유효 노출 | 개수 | 비율 |
|---|---:|---:|
| Normal | 596 | 29.24% |
| Unsafe | 352 | 17.27% |
| Danger | 1,090 | 53.48% |
| 합계 | 2,038 | 100% |

`unsafe`를 2배로 늘려도 가장 많은 클래스는 `danger`다. 따라서 이 설정은
완전한 클래스 균형이 아니라, 적은 실험 횟수에서 `unsafe` 학습 신호를
보강하는 보수적인 oversampling이다.

---

## 4. 생성된 폴더와 파일

```text
benchmark2/training/
├── IMPLEMENTATION_REPORT_KO.md
├── README.md
├── requirements.txt
├── configs/
│   └── qwen35_08b_action_v0.json
├── datasets/
│   └── qwen35_08b_action_v0/
│       ├── all.jsonl
│       ├── train.jsonl
│       ├── validation.jsonl
│       ├── test.jsonl
│       ├── splits.json
│       └── summary.json
└── scripts/
    ├── build_dataset.py
    ├── validate_dataset.py
    ├── train_lora.py
    └── evaluate_lora.py
```

각 파일의 역할은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `qwen35_08b_action_v0.json` | 데이터, 이미지, 분할, LoRA, Trainer, loss 및 생성 설정 |
| `build_dataset.py` | 원본 이미지, 행동 JSON, GT txt를 결합해 JSONL 생성 |
| `validate_dataset.py` | 이미지 존재 여부, JSON 형식, split 누수 및 요청 구조 검사 |
| `train_lora.py` | Unsloth Vision LoRA 학습 |
| `evaluate_lora.py` | Validation/Test 생성 평가와 분류 지표 계산 |
| `all.jsonl` | 분할 전 전체 요청 |
| `train.jsonl` | Train 요청 |
| `validation.jsonl` | Validation 요청 |
| `test.jsonl` | Test 요청 |
| `splits.json` | 각 split에 속한 영상 ID와 분할 seed |
| `summary.json` | 클래스 수, split 수, oversampling 후 유효 개수 |
| `README.md` | 환경 설치와 실행 명령 |

---

## 5. 데이터 원천

### 5.1 이미지와 행동 JSON

이미지와 Stage 1 행동 JSON은 다음 루트에서 읽는다.

```text
benchmark2/data/viz_shufflenet_full/
├── normal_plant_rgb/
├── climb-over-fence_plant_rgb/
└── climb-over-fence_smart_rgb/
```

각 그룹은 다음 구조를 사용한다.

```text
<group>/
├── images/
│   └── <video_id>/
│       ├── <video_id>_t000016.jpg
│       ├── <video_id>_t000026.jpg
│       └── ...
└── labels/
    └── <video_id>.json
```

여기서 `labels/<video_id>.json`은 위험 상태 GT가 아니라 Stage 1 행동 탐지
결과다.

### 5.2 위험 상태 GT

위험 상태 GT는 다음 외부 labeling 경로에서 읽는다.

```text
/home/capstone2/labeling/
├── normal_plant/obj_train_data/normal_plant_rgb/
├── plant_climb/obj_train_data/climb-over-fence_plant_rgb/
└── smart_climb/obj_train_data/climb-over-fence_smart_rgb/
```

영상별 GT 폴더에는 다음처럼 각 30fps 프레임의 txt 라벨이 있다.

```text
<video_id>/
├── 30fps_frame_001.txt
├── 30fps_frame_002.txt
└── ...
```

### 5.3 GT 클래스 매핑

GT txt 각 줄의 첫 번째 숫자만 위험 상태 판정에 사용한다. bbox 좌표는
이번 Stage 2 분류 학습에서 사용하지 않는다.

| 원본 클래스 ID | Stage 2 상태 |
|---:|---|
| 0 | `unsafe` |
| 1 | `danger` |
| 2 | `normal` |

하나의 txt에 여러 객체나 클래스가 있으면 다음 위험 우선순위로 한 프레임의
상태를 정한다.

```text
danger > unsafe > normal
```

즉, 클래스 1이 하나라도 있으면 해당 프레임은 `danger`, 클래스 1은 없지만
클래스 0이 있으면 `unsafe`, 나머지는 `normal`이다.

빈 GT 파일이나 0, 1, 2 이외 클래스가 있는 파일은 오류로 처리한다.

---

## 6. 이미지 시퀀스 생성 방식

### 6.1 기본 프레임 구성

각 학습 요청은 다음 세 프레임으로 구성한다.

```text
F1 = t
F2 = t + 10
F3 = t + 20
```

원본이 30fps이므로 시간 차이는 다음과 같다.

| 구간 | 프레임 차이 | 시간 차이 |
|---|---:|---:|
| F1 -> F2 | 10 | 약 0.333초 |
| F2 -> F3 | 10 | 약 0.333초 |
| F1 -> F3 | 20 | 약 0.667초 |

다음 요청의 시작점은 30프레임 뒤다.

```text
다음 요청 시작 = t + 30
```

따라서 약 1초마다 하나의 요청을 생성한다. 첫 요청의 `t`는 0이나 1로
강제하지 않고 해당 영상의 Stage 1 행동 JSON에서 가장 처음 사용할 수 있는
프레임으로 정한다. 예를 들어 Stage 1 window 때문에 첫 행동 결과가
16프레임부터 존재하면 첫 요청은 `[16, 26, 36]`이 된다.

### 6.2 생성 조건

요청 하나를 생성하려면 다음 조건을 모두 만족해야 한다.

1. `t`, `t+10`, `t+20` 세 프레임이 행동 JSON에 존재해야 한다.
2. 세 프레임의 이미지 파일이 모두 존재해야 한다.
3. 세 프레임의 GT txt가 모두 존재해야 한다.
4. coverage 구간의 모든 GT txt가 존재해야 한다.

현재 생성 결과에서 `skipped` 항목은 비어 있다. 즉, 현재 남아 있는 행동
JSON을 기준으로 위 조건을 만족하지 못해 건너뛴 요청은 없다.

### 6.3 이미지 경로 해석

이미지 파일명은 우선 다음 규칙으로 찾는다.

```text
<video_id>_t<6자리 frame>.jpg
<video_id>_t<6자리 frame>.jpeg
<video_id>_t<6자리 frame>.png
30fps_frame_<3자리 frame>.jpg
30fps_frame_<3자리 frame>.png
```

정확한 이름이 없으면 지원 확장자 내에서 같은 frame index가 포함된 파일을
검색한다.

---

## 7. Stage 1 행동 정보 처리

각 선택 프레임의 Stage 1 JSON에서 모든 detection의 action 후보를 모은 뒤,
confidence가 높은 순서로 정렬한다.

처리 규칙은 다음과 같다.

1. 모든 detection의 `actions`를 하나의 후보 목록으로 모은다.
2. confidence 내림차순으로 정렬한다.
3. 같은 행동 이름은 중복 제거한다.
4. 상위 2개의 행동 이름만 선택한다.
5. confidence 숫자는 프롬프트에서 제외한다.
6. 행동이 없으면 `none`을 사용한다.

예시는 다음과 같다.

```text
walk, carry/hold (an object)
```

confidence 숫자를 출력하지 않지만 상위 행동을 고를 때는 confidence 순위를
사용한다.

전체 7,026개 입력 프레임 중 행동이 없어 `none`으로 들어간 프레임은 42개다.

### 7.1 행동 정보 사용상의 한계

현재 구현은 사람별 track과 위험 대상 인물을 연결하지 않는다. 한 프레임에
여러 사람이 탐지되면 모든 detection의 행동 후보 중 confidence가 높은
2개를 선택한다.

따라서 사람이 여러 명인 장면에서는 화면의 다른 사람 행동이 프롬프트에
포함될 가능성이 있다. 이번 1차 학습에서는 단순하고 재현 가능한 방식으로
구현했지만, 향후에는 bbox 또는 track ID를 이용해 위험 대상 사람의 행동만
연결하는 개선을 고려할 수 있다.

---

## 8. Sequence GT 산정

### 8.1 sampled frame GT

먼저 모델 입력으로 사용하는 세 프레임의 GT를 읽는다.

```text
sampled_states = [GT(t), GT(t+10), GT(t+20)]
```

세 상태 중 가장 위험한 상태를 `sampled_only_gt`로 저장한다.

```text
danger > unsafe > normal
```

### 8.2 coverage 구간

Stage 1 window가 16프레임이므로 coverage 구간은 다음과 같이 정의했다.

```text
coverage_start = t - 16 + 1 = t - 15
coverage_end   = t + 20
```

양 끝을 포함하므로 coverage 길이는 항상 36프레임이다.

```text
[t-15, ..., t, ..., t+10, ..., t+20]
```

실제 생성된 2,342개 요청 모두 coverage 길이가 36프레임임을 확인했다.

각 coverage에서 `normal`, `unsafe`, `danger` 프레임 수와 비율을 계산해
JSONL에 함께 기록한다.

### 8.3 최종 sequence_gt 우선순위

현재 최종 정답은 다음 순서로 결정한다.

```text
1. 선택된 이미지 3장 중 danger가 하나라도 있으면 danger
2. coverage의 danger 비율이 30% 이상이면 danger
3. 선택된 이미지 3장 중 unsafe가 하나라도 있으면 unsafe
4. coverage의 unsafe 비율이 30% 이상이면 unsafe
5. 위 조건이 모두 아니면 normal
```

이를 의사 코드로 표현하면 다음과 같다.

```python
if "danger" in sampled_states:
    sequence_gt = "danger"
elif coverage_danger_ratio >= 0.30:
    sequence_gt = "danger"
elif "unsafe" in sampled_states:
    sequence_gt = "unsafe"
elif coverage_unsafe_ratio >= 0.30:
    sequence_gt = "unsafe"
else:
    sequence_gt = "normal"
```

### 8.4 coverage로 승격된 사례

coverage 규칙 때문에 sampled-only GT보다 최종 GT가 높아진 요청은 전체
2,342개 중 2개다.

#### 사례 1: Normal에서 Unsafe로 승격

```text
video: intrusion_normal_rgb_0585_cctv2
selected: [346, 356, 366]
sampled GT: [normal, normal, normal]
coverage: normal 25 / unsafe 11 / danger 0
unsafe ratio: 11 / 36 = 30.56%
final sequence_gt: unsafe
```

#### 사례 2: Unsafe에서 Danger로 승격

```text
video: intrusion_climb-over-fence_rgb_0410_cctv2
selected: [166, 176, 186]
sampled GT: [unsafe, unsafe, unsafe]
coverage: normal 0 / unsafe 24 / danger 12
danger ratio: 12 / 36 = 33.33%
final sequence_gt: danger
```

### 8.5 coverage GT의 의미와 위험

coverage GT는 모델이 직접 보지 않는 중간 프레임의 라벨을 이용한다. 따라서
중간 프레임에서만 등반이 나타나고 선택된 이미지 3장에는 증거가 없다면,
모델 입장에서는 입력만으로 정답을 맞히기 어려운 label ambiguity가 생길 수
있다.

현재는 이러한 승격이 2개뿐이어서 전체 데이터에 미치는 영향은 작다.
다만 30% 임계값이나 프레임 간격을 변경하면 승격 수가 크게 달라질 수
있으므로, 향후 데이터 설정 변경 시 반드시 다시 집계해야 한다.

---

## 9. 프롬프트 구성

프롬프트 템플릿은 다음 파일을 사용한다.

```text
benchmark2/prompts/action_timev0.txt
```

Qwen3.5-0.8B의 작은 모델 크기와 고정된 3-class 분류 작업을 고려해, 초기
`action_timev1.txt`보다 짧은 v0 템플릿으로 최종 전환했다. V0는 107단어,
843바이트이며 v1의 193단어, 1,374바이트보다 약 45% 짧다.

프롬프트에는 다음 정보가 포함된다.

1. 이미지 3장이 시간순이라는 설명
2. 세 위험 상태의 의미
3. 펜스 등반 및 경계 침입 판정 규칙
4. 각 프레임 번호와 초 단위 시각
5. 각 프레임의 Stage 1 행동 상위 2개
6. 이미지를 우선 사용하고 행동은 시간적 보조 정보로 사용하라는 지시
7. JSON 하나만 반환하라는 출력 규칙

프레임별 실제 삽입 형태는 다음과 같다.

```text
F1: frame=16, time=0.533s, action=walk, carry/hold (an object)
F2: frame=26, time=0.867s, action=walk, carry/hold (an object)
F3: frame=36, time=1.200s, action=stand, carry/hold (an object)
```

시각은 다음 식으로 계산한다.

```text
time_sec = frame_index / 30.0
```

프롬프트 템플릿의 `{frame_n}`, `{time_n}`, `{action_n}`이 치환되지 않고
남아 있으면 데이터 생성 단계에서 오류를 발생시킨다.

---

## 10. JSONL 레코드 구조

각 요청은 한 줄의 JSON 객체로 저장된다. 주요 필드는 다음과 같다.

| 필드 | 의미 |
|---|---|
| `request_id` | 영상 ID와 세 frame index를 결합한 고유 ID |
| `group` | 데이터 그룹 |
| `video_id` | 원본 영상 ID |
| `frame_indices` | `[t, t+10, t+20]` |
| `frame_times_sec` | 각 프레임의 초 단위 시각 |
| `images` | 프로젝트 루트 기준 상대 이미지 경로 3개 |
| `stage1_actions` | 각 프레임의 행동 문자열 |
| `sampled_frame_gt` | 세 선택 프레임 각각의 GT |
| `sampled_only_gt` | 선택 프레임만으로 정한 우선순위 GT |
| `coverage_start_frame` | coverage 시작 프레임 |
| `coverage_end_frame` | coverage 종료 프레임 |
| `coverage_frame_count` | coverage의 클래스별 프레임 수 |
| `coverage_ratio` | coverage의 클래스별 비율 |
| `sequence_gt` | 최종 학습 정답 |
| `coverage_promoted` | coverage 때문에 정답이 승격되었는지 여부 |
| `prompt_text` | 실제 user 텍스트 |
| `assistant_response` | 정답 JSON |

정답 문자열에는 불필요한 공백을 넣지 않는다.

```json
{"risk_state":"normal"}
```

---

## 11. 데이터 분할

### 11.1 분할 단위

분할은 개별 요청이나 프레임 단위가 아니라 `video_id` 단위로 수행한다.

같은 영상에서 생성된 여러 요청은 장면, 배경, 카메라 위치와 인물이 매우
유사할 수 있다. 이를 요청 단위로 무작위 분할하면 같은 영상의 다른 프레임이
Train과 Test에 동시에 들어가 성능이 과대평가될 수 있다.

현재 구현은 한 영상의 모든 요청을 반드시 하나의 split에만 배치한다.

### 11.2 비율

현재 비율은 다음과 같다.

```text
Train      0.8
Validation 0.1
Test       0.1
```

총 258개 영상의 실제 분할은 다음과 같다.

| Split | 영상 수 | 비율 |
|---|---:|---:|
| Train | 206 | 79.84% |
| Validation | 26 | 10.08% |
| Test | 26 | 10.08% |

### 11.3 그룹별 분할

각 데이터 그룹에서도 80/10/10에 가깝도록 영상을 나눈다.

| Split | Normal Plant | Plant Climb | Smart Climb |
|---|---:|---:|---:|
| Train | 80 | 62 | 64 |
| Validation | 10 | 8 | 8 |
| Test | 10 | 8 | 8 |

### 11.4 균형 seed 탐색

영상 단위 분할은 클래스 요청 수가 영상마다 다르기 때문에 단순 random seed
하나로 나누면 Validation 또는 Test에서 `unsafe`가 지나치게 적어질 수 있다.

이를 줄이기 위해 seed 0부터 9,999까지 총 10,000개 후보를 검사했다.
각 후보에 대해 다음 값을 계산했다.

```text
각 split의 실제 클래스 요청 수
vs.
전체 클래스 요청 수 x 목표 split 비율
```

정규화된 제곱 오차 합이 가장 작은 seed를 선택했고, 최종 seed는 `1045`다.

분할 결과는 `splits.json`에 고정 저장된다. 실험 중 Test 결과에 맞춰
분할을 다시 생성하면 안 된다.

---

## 12. 이미지 전처리

### 12.1 디스크 원본

원본 이미지는 디스크에서 수정하거나 별도 resize 이미지로 저장하지 않는다.
학습 또는 평가 시 PIL로 읽어 메모리에서만 변환한다.

### 12.2 처리 순서

```text
원본 이미지 읽기
-> RGB 변환
-> 768 x 432 LANCZOS resize
-> Qwen image processor
-> vision tensor
```

원본 이미지가 1920x1080의 16:9이고 목표 크기 768x432도 16:9이므로
PIL resize 단계에서는 종횡비 왜곡이 없다.

Qwen processor는 vision patch 규격에 맞춰 내부 정렬을 수행한다. 실제
processor 검사에서 각 이미지의 grid는 `[1, 28, 48]`로 나타났다. 즉,
사용자가 지정한 768x432 이미지는 모델 내부에서 patch 배수에 맞게 처리된다.

Unsloth collator에는 다음 옵션을 지정했다.

```text
resize = "max"
```

이는 데이터셋에서 이미 맞춘 이미지에 대해 collator가 별도의 고정 크기로
강제 축소하는 것을 피하기 위한 설정이다.

### 12.3 입력 길이 검사

로컬 Qwen processor로 이미지 3장과 프롬프트를 실제 tokenization했다.

| 검사 대상 | 토큰 수 |
|---|---:|
| 일반 샘플 | 1,294 |
| 가장 긴 프롬프트 샘플 | 1,318 |
| 설정된 최대 길이 | 2,048 |

현재 데이터는 이미지 토큰을 포함해 `max_length=2048` 안에 들어가며, 가장
긴 샘플도 730 token의 여유가 있다.

---

## 13. 모델과 LoRA 구성

### 13.1 기본 모델

```text
benchmark2/models/Qwen3.5-0.8B
```

로컬 모델 크기는 약 1.7GB이며, 모델 파일과 tokenizer, chat template,
image processor 설정이 존재한다.

### 13.2 BF16 16-bit LoRA의 의미

현재 설정은 다음과 같다.

```json
{
  "load_in_4bit": false,
  "dtype": "bfloat16"
}
```

즉, QLoRA처럼 base model을 4-bit로 양자화해 불러오지 않는다. base model을
BF16 경로로 로드하고, 원본 전체 가중치를 직접 업데이트하는 대신 LoRA
adapter 파라미터를 학습한다.

`adamw_8bit`는 optimizer state의 메모리를 줄이는 설정이다. 이는 base
model을 8-bit로 로드한다는 뜻이 아니다.

### 13.3 LoRA 대상

현재 다음 영역을 모두 LoRA 학습 대상으로 지정했다.

| 설정 | 값 |
|---|---|
| Vision layers | 활성화 |
| Language layers | 활성화 |
| Attention modules | 활성화 |
| MLP modules | 활성화 |

1차 학습에서 이미지 특징과 언어 출력 형식을 함께 적응시키기 위한 설정이다.

### 13.4 LoRA 하이퍼파라미터

| 하이퍼파라미터 | 값 | 의미 |
|---|---:|---|
| `r` | 16 | LoRA 저랭크 차원 |
| `lora_alpha` | 16 | LoRA update scale |
| `lora_dropout` | 0.0 | LoRA 경로 dropout |
| `bias` | `none` | bias 추가 학습 안 함 |
| `random_state` | 42 | LoRA 초기화 재현 seed |
| `use_rslora` | false | Rank-stabilized LoRA 미사용 |
| `loftq_config` | null | LoftQ 미사용 |

`r=16`, `alpha=16`, dropout 0은 Unsloth Qwen3.5 Vision 예제에 가까운
보수적인 1차 설정이다.

---

## 14. 학습 데이터 로더

`CCTVConversationDataset`은 JSONL을 읽은 뒤 샘플을 요청받을 때 이미지 3장을
그때그때 로드한다. 전체 이미지를 한꺼번에 RAM에 적재하지 않는다.

모델에 전달되는 대화 구조는 다음과 같다.

```text
system:
  industrial CCTV intrusion classifier 역할과 JSON 출력 지시

user:
  image 1
  image 2
  image 3
  prompt text

assistant:
  {"risk_state":"..."}
```

Train 데이터만 oversampling 후 seed 42로 섞는다. Validation은 복제하지
않으며 원본 분포를 유지한다.

---

## 15. 학습 하이퍼파라미터

| 항목 | 설정 |
|---|---:|
| Epoch | 3 |
| Train batch size per GPU | 16 |
| Eval batch size per GPU | 16 |
| Gradient accumulation | 2 |
| 단일 GPU 기준 유효 batch | 32 |
| Learning rate | `1e-4` |
| Optimizer | `adamw_8bit` |
| Weight decay | `0.001` |
| LR scheduler | `linear` |
| Warmup ratio | `0.05` |
| Max grad norm | `1.0` |
| Logging interval | 10 step |
| Evaluation | 매 epoch |
| Checkpoint 저장 | 매 epoch |
| 최대 checkpoint 수 | 3 |
| BF16 | 활성화 |
| FP16 | 비활성화 |
| TF32 | 활성화 |
| Gradient checkpointing | 활성화 |
| Data loader workers | 4 |
| Seed | 42 |
| Data seed | 42 |
| Max sequence length | 2048 |
| Logging backend | TensorBoard |

Train의 oversampling 후 샘플 수는 2,038개다. 단일 GPU에서 micro batch 16,
gradient accumulation 2를 사용해 유효 batch 32를 유지한다. Epoch당
optimizer update는 약 64회이며, 3 epoch 전체는 약 192회다.

실험 횟수가 많지 않을 가능성을 고려해 별도 grid search는 구성하지 않았다.
기본 계획은 다음과 같다.

```text
1. Zero-shot Validation 기준선
2. 소규모 smoke test
3. LoRA 본 학습 1회
4. 각 epoch checkpoint를 Validation macro F1으로 비교
5. 선택된 checkpoint로 Test 1회
```

---

## 16. Assistant JSON 전용 Loss

### 16.1 목적

학습 입력에는 system prompt, 이미지 토큰, user prompt, 행동 정보와 정답 JSON이
모두 포함된다. 그러나 모델이 맞혀야 하는 것은 assistant의 최종 JSON이다.

따라서 다음 부분은 loss 계산에서 제외한다.

```text
system prompt
user prompt
이미지 토큰
행동 설명
Qwen의 빈 thinking prefix
```

정답 JSON과 응답 종료에 해당하는 token만 loss 대상으로 사용하도록
Unsloth Vision collator의 response-only masking을 설정했다.

### 16.2 Qwen3.5 chat template

로컬 Qwen3.5 chat template로 정답 샘플을 렌더링하면 assistant 부분은 다음
형태가 된다.

```text
<|im_start|>assistant
<think>

</think>

{"risk_state":"normal"}<|im_end|>
```

현재 loss 시작 구분자는 다음과 같다.

```text
</think>\n\n
```

즉, 빈 thinking 블록 뒤에서부터 정답을 학습한다.

### 16.3 시작 전 안전 검사

실제 학습 시작 직전에 collator로 첫 샘플을 처리하고 `labels != -100`인
token만 다시 decode한다.

다음 두 조건을 확인한다.

1. 활성 loss token에 정답 JSON이 포함되어야 한다.
2. 활성 loss token에 user prompt 문장인 `Classify the risk state`가
   포함되면 안 된다.

조건을 만족하지 않으면 학습을 시작하지 않고 오류를 발생시킨다.

현재 로컬 Qwen processor로 chat template 문자열이 예상 형태인지 확인했고,
Unsloth preflight에서 실제 `UnslothVisionDataCollator`의 runtime mask도
검사했다. 활성 loss token을 decode한 결과는 다음과 같았다.

```text
{"risk_state":"normal"}<|im_end|>
```

User prompt와 이미지 token은 활성 loss 대상에 포함되지 않았고 preflight는
정상 종료했다.

### 16.4 JSON이 아닌 출력과 학습 가능 여부

SFT 학습 자체는 모델이 추론 시 항상 JSON을 생성해야만 진행되는 방식이
아니다. 학습 데이터의 정답 token이 JSON 형태로 주어지면 teacher forcing으로
그 정답 token에 loss를 계산한다.

따라서 학습 도중 모델이 아직 JSON을 잘 생성하지 못해도 학습은 가능하다.
다만 최종 추론 출력이 JSON 형식을 지키지 않으면 평가에서는 실패 또는
형식 오류로 처리해야 한다.

---

## 17. 평가 구현

### 17.1 평가 입력

평가도 학습과 동일하게 다음을 사용한다.

```text
RGB 768x432 이미지 3장
동일한 system prompt
동일한 user prompt와 행동 정보
```

### 17.2 생성 설정

| 항목 | 값 |
|---|---:|
| `max_new_tokens` | 32 |
| `do_sample` | false |
| decoding | greedy |

세 클래스 JSON 중 하나만 출력하는 작업이므로 긴 생성이나 sampling을 사용하지
않는다. 같은 입력에서 결과가 재현되도록 greedy decoding을 사용한다.

### 17.3 산출 지표

평가 결과에는 다음 지표가 포함된다.

| 지표 | 의미 |
|---|---|
| Accuracy | 전체 정답률 |
| Macro F1 | 세 클래스 F1의 동일 가중 평균 |
| Class precision | 해당 클래스로 예측한 것 중 정답 비율 |
| Class recall | 실제 해당 클래스를 찾아낸 비율 |
| Class F1 | precision과 recall의 조화 평균 |
| JSON success rate | parse 가능한 JSON 객체를 찾은 비율 |
| Schema success rate | `risk_state`가 세 클래스 중 하나인 비율 |
| Confusion matrix | GT별 예측 분포 |
| Latency | 요청별 생성 시간 |

최종 checkpoint 선택의 기본 지표는 Validation `macro_f1`이다.

안전 관점에서는 다음 지표도 반드시 같이 확인해야 한다.

1. `danger recall`
2. `unsafe recall`
3. `normal precision`
4. `invalid` 출력 수

### 17.4 현재 JSON parser의 허용 범위

현재 평가기는 다음 출력도 parse할 수 있도록 다소 관대하게 구현되어 있다.

```text
코드 펜스 시작: ```json
JSON 본문: {"risk_state":"danger"}
코드 펜스 종료: ```
```

또는 출력 전체가 JSON이 아니더라도 내부에 단일 JSON 객체가 있으면 이를
추출해 평가한다.

따라서 현재 `json_success_rate`와 `schema_success_rate`는 "오직 JSON만
출력했는가"를 엄격하게 측정하는 지표는 아니다. 최종 운영 요구가 완전한
strict JSON이면 다음 지표를 추가하는 것이 좋다.

```text
strict_json_rate =
전체 응답을 trim한 문자열이 JSON 객체 하나로 바로 parse되는 비율
```

---

## 18. 데이터 검증 구현

`validate_dataset.py`는 다음 사항을 검사한다.

1. Train, Validation, Test JSONL이 정상 JSON인지 확인
2. 각 요청에 이미지가 정확히 3장인지 확인
3. frame index가 정확히 3개인지 확인
4. 행동 문자열이 정확히 3개인지 확인
5. 모든 이미지 파일이 존재하는지 확인
6. PIL로 이미지가 손상 없이 열리는지 확인
7. `sequence_gt`가 세 클래스 중 하나인지 확인
8. `assistant_response`가 GT와 일치하는 JSON인지 확인
9. 행동 문자열에 잘못된 `{` 또는 `}`가 없는지 확인
10. 프롬프트 placeholder가 남아 있지 않은지 확인
11. 요청 ID가 split 사이에서 중복되지 않는지 확인
12. 영상 ID가 split 사이에서 누수되지 않는지 확인
13. `splits.json`과 실제 JSONL 영상 목록이 일치하는지 확인

최종 검증 결과는 다음과 같다.

```text
Dataset validation passed.
Unique requests: 2342
```

---

## 19. 현재 데이터 상세 통계

### 19.1 그룹별 전체 영상

| 그룹 | 영상 수 |
|---|---:|
| `normal_plant_rgb` | 100 |
| `climb-over-fence_plant_rgb` | 78 |
| `climb-over-fence_smart_rgb` | 80 |
| 합계 | 258 |

### 19.2 Train 그룹별 구성

| 그룹 | 영상 | 요청 | Normal | Unsafe | Danger |
|---|---:|---:|---:|---:|---:|
| Normal Plant | 80 | 621 | 562 | 59 | 0 |
| Plant Climb | 62 | 448 | 0 | 16 | 432 |
| Smart Climb | 64 | 793 | 34 | 101 | 658 |

### 19.3 Validation 그룹별 구성

| 그룹 | 영상 | 요청 | Normal | Unsafe | Danger |
|---|---:|---:|---:|---:|---:|
| Normal Plant | 10 | 74 | 74 | 0 | 0 |
| Plant Climb | 8 | 69 | 0 | 0 | 69 |
| Smart Climb | 8 | 99 | 3 | 22 | 74 |

### 19.4 Test 그룹별 구성

| 그룹 | 영상 | 요청 | Normal | Unsafe | Danger |
|---|---:|---:|---:|---:|---:|
| Normal Plant | 10 | 70 | 70 | 0 | 0 |
| Plant Climb | 8 | 64 | 0 | 2 | 62 |
| Smart Climb | 8 | 104 | 3 | 20 | 81 |

한 영상에서 생성되는 요청 수는 전체적으로 최소 4개, 최대 16개 수준이며
Train 중앙값은 영상당 9개 요청이다.

---

## 20. 제외된 데이터

### 20.1 행동 JSON이 없는 추가 이미지

추가 이미지 1,113장은 이미지와 위험 상태 GT는 있지만, 현재 학습 입력에
필요한 프레임별 Stage 1 행동 JSON이 없다.

현재 builder는 `labels/*.json`을 기준으로 영상을 순회하므로 행동 JSON에
참조되지 않는 추가 이미지는 자동으로 학습 요청에서 제외된다.

이 이미지들은 손상되거나 잘못된 데이터라서 제외한 것이 아니다. 현재
학습 입력 형식을 "이미지 3장 + 행동 라벨"로 고정했기 때문에 제외된 것이다.

향후 사용할 수 있는 방법은 다음과 같다.

1. 해당 프레임에 Stage 1 행동 추론을 다시 실행해 행동 JSON 생성
2. 행동이 없는 샘플을 `action=none`으로 사용하는 별도 데이터 정책 수립
3. 행동 없는 이미지 전용 보조 학습 단계 구성

현재 1차 학습에는 포함하지 않는다.

### 20.2 제거된 불완전 영상

이전 데이터 정리에서 다음 영상은 학습 요청을 만들 수 없는 상태라 제거됐다.

| 영상 | 사유 |
|---|---|
| `1098_cctv4` | 이미지 0장, GT 없음, JSON만 존재 |
| `1097_cctv4` | 남은 GT 구간이 짧아 3프레임 요청 생성 불가 |

현재 builder 결과의 `skipped`가 비어 있으므로, 정리 후 남은 행동 JSON
기준으로는 추가 누락 없이 요청이 생성됐다.

---

## 21. 실행 환경 상태

확인된 GPU 환경은 다음과 같다.

| 항목 | 확인값 |
|---|---|
| GPU | NVIDIA RTX A6000 |
| VRAM | 약 48GB |
| BF16 지원 | 지원 |
| 학습 venv PyTorch | 2.10.0+cu128 |
| PyTorch CUDA | 12.8 |
| 로컬 모델 | 존재 |

독립 학습 환경은 다음 경로에 설치됐다.

```text
/home/capstone2/.venvs/qwen35-lora
```

주요 설치 버전은 `unsloth 2026.6.1`, `unsloth_zoo 2026.6.1`,
`transformers 5.5.0`, `trl 0.24.0`, `torch 2.10.0+cu128`,
`torchvision 0.25.0`, `bitsandbytes 0.49.2`다. `pip check` 결과는
`No broken requirements found`로 통과했다.

Unsloth는 Flash Attention 2 대신 Xformers 0.0.35를 사용한다. 또한
flash-linear-attention과 causal-conv1d가 없어 Qwen의 일부 fast path는
Torch 구현으로 fallback한다. 이는 preflight와 학습 시작을 막는 오류는
아니지만 학습 속도에는 영향을 줄 수 있다.

또한 `nvidia-smi`에서 다음 문제가 확인됐다.

```text
Failed to initialize NVML: Driver/library version mismatch
```

PyTorch에서는 CUDA와 BF16 사용이 가능하고 GPU 이름 및 메모리를 읽을 수
있었지만, 장시간 학습 전에 재부팅 또는 NVIDIA driver/library 정합성 확인을
권장한다.

학습 환경 설치 후 디스크 가용 공간은 약 18GB이며 venv 크기는 약 7.9GB다.
checkpoint 저장 수를 3개로 제한한 이유 중 하나다.

본 학습은 oversampling 후 2,038개 요청, 유효 batch 32, 3 epoch이므로 약
192 optimizer update가 예상된다. Smoke에서 측정한 checkpoint 크기는 개당
약 100.2MB, final adapter 디렉터리는 약 72.8MB였다. 세 checkpoint와 final
adapter의 예상 핵심 저장량은 약 356MiB이며 TensorBoard와 평가 파일을
포함해도 보수적으로 약 0.5GB 수준이다. 현재 가용 공간으로 학습 결과 저장은
충분하지만, 파일 시스템 전체 사용률이 97%이므로 동시에 다른 대용량 파일을
생성하지 않는 것이 좋다.

---

## 22. 설치 및 실행 절차

### 22.1 학습 환경 생성

기존 qwen35 추론 환경의 vLLM, Torch 및 Transformers 버전과 충돌하지 않도록
완전히 독립된 venv를 사용한다.

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

### 22.2 GPU와 BF16 확인

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"
```

### 22.3 데이터 재생성

현재 데이터는 이미 생성돼 있다. 원본 데이터나 설정을 변경했을 때만 다시
실행한다.

```bash
python benchmark2/training/scripts/build_dataset.py
python benchmark2/training/scripts/validate_dataset.py
```

데이터를 재생성하면 split seed 검색도 다시 수행한다. 동일한 데이터와
동일한 코드에서는 다시 seed 1045가 선택되지만, 원본을 변경한 뒤에는
분할 결과가 달라질 수 있다.

### 22.4 Zero-shot 기준선

```bash
python benchmark2/training/scripts/evaluate_lora.py \
  --split validation \
  --output-dir benchmark2/training/outputs/zero_shot_validation
```

### 22.5 Smoke test

먼저 optimizer나 Trainer를 생성하지 않는 preflight 검사를 실행한다.

```bash
python benchmark2/training/scripts/train_lora.py \
  --preflight \
  --limit-train 1 \
  --limit-validation 1
```

Preflight는 모델 로드, LoRA 부착, 이미지 collator 구성과 JSON 전용 loss
mask까지만 검사하고 종료한다. forward/backward 학습은 수행하지 않는다.

본 학습 전에 32개 Train 샘플, 16개 Validation 샘플, 20 step으로 실행한다.

```bash
python benchmark2/training/scripts/train_lora.py \
  --run \
  --limit-train 32 \
  --limit-validation 16 \
  --max-steps 20 \
  --output-dir benchmark2/training/outputs/qwen35_08b_action_v0_smoke
```

시작 로그에서 다음을 확인해야 한다.

```text
Active loss tokens decode: ...{"risk_state":"..."}...
```

이 decode 결과에 user prompt가 포함되거나 정답 JSON이 없으면 본 학습을
실행하면 안 된다.

2026-06-06에 위 smoke test를 실제 실행했으며 정상 완료됐다.

| 항목 | 결과 |
|---|---:|
| 제한 전 Train 요청 | 32 |
| Unsafe 2배 적용 후 Train 요청 | 41 |
| Validation 요청 | 16 |
| 실행 step | 20 |
| 실제 반복 epoch | 약 6.762 |
| 학습 시간 | 약 5분 46초 |
| Train loss | 약 0.0917 |
| 마지막 Eval loss | 약 0.000277 |
| 학습된 파라미터 | 13,181,952 / 866,167,872 (1.52%) |

`max_steps=20`이 `num_train_epochs=3`보다 우선 적용되며 데이터가 매우 작아서
같은 41개 요청을 약 6.762 epoch 반복했다. 따라서 낮은 Eval loss는 smoke
샘플에 빠르게 맞춰진 결과일 뿐, 일반화 성능을 의미하지 않는다.

첫 smoke 실행 당시 제한 방식은 JSONL 앞에서부터 자르는 방식이었다. 그
결과 Train 32개는 `normal` 23개와 `unsafe` 9개였으며 모두
`normal_plant_rgb` 그룹이었다. Validation 16개는 모두 `normal`이었다.
따라서 첫 smoke의 낮은 Eval loss는 클래스 성능을 나타내지 않는다.

이 확인 이후 `--limit-train`과 `--limit-validation`은 그룹과 클래스 조합별
bucket을 순환하며 선택하는 deterministic stratified limit로 수정했다. 이
변경은 제한 옵션을 사용하는 preflight와 smoke에만 적용되며, 제한 옵션이
없는 본 학습 데이터에는 영향을 주지 않는다.

최근 checkpoint 3개인 `checkpoint-15`, `checkpoint-18`, `checkpoint-20`과
`final_adapter`가 정상 저장됐다. Final adapter 가중치 크기는 약 52.8MB다.

저장된 final adapter를 다시 불러와 Validation 1건을 생성하는 end-to-end
검사도 수행했다. 출력은 정상 JSON이었고 GT `normal`과 예측 `normal`이
일치했다. 1건 결과는 저장, 재로딩, 이미지 입력과 JSON 생성 경로 확인용이며
성능 지표로 해석하지 않는다.

검증이 끝난 smoke 출력 폴더는 본 학습 전 2026-06-06에 삭제했다. Smoke
실행 결과와 확인 수치는 이 보고서에 보존했으며, `outputs`에는 환경 재현용
`installed_packages.txt`만 남겨 두었다.

### 22.6 본 학습

```bash
python benchmark2/training/scripts/train_lora.py --run
```

`--run`은 실수로 학습이 시작되지 않도록 만든 필수 안전 플래그다.
이 플래그가 없으면 스크립트는 다음 메시지와 함께 종료된다.

```text
Training was not started. Re-run with --run after checking the config.
```

2026-06-06에 본 학습을 완료했다. 실제 실행 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| 유효 Train 요청 | 2,038 |
| Epoch | 3 |
| Optimizer step | 192 |
| Device batch | 16 |
| Gradient accumulation | 2 |
| 유효 batch | 32 |
| 학습 파라미터 | 13,181,952 / 866,167,872 (1.52%) |
| 학습 시간 | 약 1시간 54초 |
| 최종 train loss | 0.02644 |

`checkpoint-64`, `checkpoint-128`, `checkpoint-192`와 `final_adapter`가
정상 저장됐다. SHA-256 확인 결과 `final_adapter`와 `checkpoint-192`의
adapter 가중치는 동일하다. 따라서 `final_adapter`는 Validation 성능이 가장
좋은 checkpoint를 의미하지 않는다.

학습 시작 시 tokenizer의 EOS token ID를 model 및 generation config와
맞췄다는 안내가 출력됐다. 이는 설정 동기화 안내이며 학습 중단이나 데이터
오류가 아니다.

### 22.7 중단된 학습 재개

```bash
python benchmark2/training/scripts/train_lora.py \
  --run \
  --resume-from-checkpoint \
  benchmark2/training/outputs/qwen35_08b_action_v0/checkpoint-N
```

### 22.8 Validation 평가

```bash
python benchmark2/training/scripts/evaluate_lora.py \
  --adapter-path \
  benchmark2/training/outputs/qwen35_08b_action_v0/checkpoint-N \
  --split validation
```

세 epoch checkpoint 중 Validation macro F1이 가장 높은 checkpoint를
선택한다. macro F1이 비슷하면 `danger recall`, `unsafe recall`, invalid 수를
함께 비교한다.

세 checkpoint 모두 242개 Validation 요청에 대한 생성 평가를 완료했다.
최종 선택 결과는 `checkpoint-128`이다. 상세 수치는 28절에 정리했다.

### 22.9 최종 Test

```bash
python benchmark2/training/scripts/evaluate_lora.py \
  --adapter-path \
  benchmark2/training/outputs/qwen35_08b_action_v0/checkpoint-N \
  --split test
```

Test는 checkpoint와 하이퍼파라미터를 모두 확정한 뒤 한 번만 실행한다.

---

## 23. 학습 및 평가 출력

본 학습 출력은 다음 위치에 생성된다.

```text
benchmark2/training/outputs/qwen35_08b_action_v0/
├── checkpoint-*/
└── final_adapter/
    ├── adapter 설정과 가중치
    ├── processor 설정
    └── training_config.json
```

평가 결과는 checkpoint 아래에 다음처럼 저장된다.

```text
eval_validation/
├── metrics.json
├── predictions.jsonl
└── confusion_matrix.csv
```

`predictions.jsonl`에는 요청별로 다음 정보가 기록된다.

```text
request_id
video_id
group
frame_indices
ground_truth
prediction
match
json_success
schema_success
latency_sec
raw_response
```

---

## 24. 재현성 관리

현재 재현성 관련 설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| Split 탐색 범위 | seed 0~9999 |
| 선택된 split seed | 1045 |
| Trainer seed | 42 |
| Data seed | 42 |
| LoRA random state | 42 |
| Decoding sampling | 사용 안 함 |

학습 환경 설치 후 실제 패키지 버전을 저장한다.

```bash
mkdir -p benchmark2/training/outputs
python -m pip freeze > benchmark2/training/outputs/installed_packages.txt
```

재현을 위해 함께 보존해야 할 파일은 다음과 같다.

1. 학습 config JSON
2. `splits.json`
3. `summary.json`
4. 설치 패키지 목록
5. 사용 checkpoint
6. 평가 `metrics.json`
7. 평가 `predictions.jsonl`

---

## 25. 구현 중 확정한 가정

다음은 코드에 현재 반영돼 있지만 실험 결과에 따라 변경될 수 있는 정책이다.

| 항목 | 현재 결정 |
|---|---|
| Coverage 구간 | `t-15`부터 `t+20`까지 36프레임 |
| Danger coverage 임계값 | 30% |
| Unsafe coverage 임계값 | 30% |
| 분할 | 영상 단위 80/10/10 |
| 행동 정보 | confidence 상위 2개 이름 |
| 행동 confidence 숫자 | 입력에서 제외 |
| 이미지 입력 크기 | 768x432 |
| Unsafe oversampling | Train에서만 정확히 2배 |
| LoRA 대상 | Vision + Language + Attention + MLP |
| LoRA rank/alpha | 16/16 |
| Epoch | 3 |
| Learning rate | `1e-4` |
| Checkpoint 선택 | Validation macro F1 |
| Test 사용 | 최종 1회 |

---

## 26. 사용자 확인이 필요한 핵심 항목

구현을 막는 미확정 사항은 없지만, 다음 항목은 연구 정책에 해당하므로
사용자가 최종적으로 승인하거나 실험 후 재검토하는 것이 좋다.

### 26.1 Coverage GT 30% 정책

현재 2개 요청이 이 규칙으로 승격됐다. 세 이미지에 명확한 증거가 없는데
중간 프레임 때문에 정답이 바뀌는 것을 허용할지 확인이 필요하다.

선택지는 다음과 같다.

1. 현재처럼 coverage 30% 승격 유지
2. coverage는 metadata로만 저장하고 정답은 세 sampled frame으로 결정
3. coverage에서 승격된 경우 해당 위험 프레임을 세 이미지 중 하나로 교체
4. 임계값을 30%보다 높임

현재 구현은 1번이다.

### 26.2 행동이 없는 추가 1,113장

현재는 제외한다. 향후 이 이미지까지 활용하려면 행동 재추론 또는
`action=none` 정책 중 하나를 선택해야 한다.

### 26.3 행동 top-2 집계 단위

현재는 모든 detection을 합쳐 top-2를 고른다. 다중 인물 장면에서 특정
위험 인물의 행동만 사용할지 확인이 필요하다.

### 26.4 엄격한 JSON 평가

현재 평가기는 응답에 JSON 객체가 포함되어 있으면 주변 텍스트가 있어도
정답으로 인정할 수 있다. 운영에서 완전한 JSON-only 출력이 필수라면
strict JSON 지표를 추가해야 한다.

### 26.5 1차 하이퍼파라미터

3 epoch와 learning rate `1e-4`는 제한된 실험 횟수를 고려한 1차 설정이다.
Smoke test와 epoch별 Validation 결과를 보기 전에는 최적값이라고 확정할 수
없다.

---

## 27. 알려진 위험과 대응

### 27.1 클래스 불균형

`danger`가 전체의 58.75%, `unsafe`가 9.39%다. Unsafe 2배 적용 후에도
Train에서 danger가 53.48%로 가장 많다.

대응:

1. Accuracy만 보지 않고 macro F1과 클래스별 recall 사용
2. Unsafe oversampling 2배 유지
3. Validation unsafe 22개를 개별 오류 분석

### 27.2 영상별 강한 시각적 유사성

같은 영상에서 여러 요청이 생성된다. 영상 단위 split로 직접 누수는
차단했지만, 같은 장소나 비슷한 CCTV 구도의 다른 영상 간 유사성은 남을 수
있다.

대응:

1. 그룹별 지표 확인
2. 필요하면 카메라 또는 장소 단위 split 검토

### 27.3 행동 라벨 오류 전파

Stage 1 행동이 틀리면 Stage 2 프롬프트에 잘못된 힌트가 들어간다.

대응:

1. 프롬프트에 이미지를 우선 사용하도록 명시
2. 행동 confidence 숫자를 제거해 과도한 신뢰 방지
3. 향후 image-only와 image+action 성능 비교

### 27.4 Coverage와 관측 이미지 불일치

모델이 보지 않은 프레임으로 정답이 승격될 수 있다.

대응:

1. `coverage_promoted=true` 사례 별도 추적
2. 현재 두 사례를 수동 확인
3. 향후 승격 수가 늘면 샘플 프레임 선택 정책 변경

### 27.5 환경 패키지 호환성

Unsloth, TRL, Transformers API는 버전에 따라 인자명이 바뀔 수 있다.

대응:

1. 별도 venv 사용
2. 패키지 버전 저장
3. 20-step smoke test 선행
4. 응답 loss mask decode 확인

### 27.6 디스크 부족

현재 디스크 사용률이 약 95%다.

대응:

1. `save_total_limit=3`
2. Smoke test 출력은 본 학습과 다른 폴더 사용
3. 불필요한 checkpoint와 pip cache 점검
4. 최종 adapter와 선택 checkpoint를 제외한 중간 산출물 정리

---

## 28. 폐기된 1차 학습 및 Validation 결과

> 이 절의 수치는 잘못된 라벨이 포함된 v0 데이터로 얻은 과거 기록이다.
> 모델 산출물과 평가 파일은 삭제했으며, 현재 모델 성능으로 사용하지 않는다.

### 28.1 완료된 항목

- 데이터셋 생성
- 총 2,342개 요청 생성 확인
- 참조 이미지 7,026장 확인
- 이미지 존재 및 PIL decode 확인
- 요청별 이미지 3장 확인
- 요청별 행동 문자열 3개 확인
- GT 클래스와 assistant JSON 일치 확인
- Train/Validation/Test 요청 ID 중복 없음
- 영상 단위 split 누수 없음
- `splits.json`과 JSONL 일치
- Coverage 길이 36프레임 확인
- Coverage 승격 사례 2개 확인
- 로컬 Qwen chat template의 assistant 형식 확인
- 이미지 포함 최대 입력 길이 1,318 token 확인
- `max_length=2048` 이내 확인
- 학습 스크립트 Python 문법 검사
- `--run` 없는 경우 학습 차단 확인
- RTX A6000 및 BF16 지원 확인
- 독립 `qwen35-lora` venv 설치
- `pip check` 의존성 검사 통과
- 로컬 Qwen3.5-0.8B 16-bit 모델 로드
- Vision 및 Language LoRA 부착
- Unsloth Vision collator 생성
- JSON 전용 runtime loss mask 확인
- `--preflight` 정상 종료 및 Trainer 미생성 확인
- 20-step smoke training 정상 완료
- Checkpoint 3개와 final adapter 저장 확인
- Smoke adapter 재로딩과 Validation 1건 JSON 생성 확인
- 2,038개 유효 Train 요청으로 3 epoch 본 학습 완료
- 192 optimizer step 정상 완료
- Epoch별 checkpoint 3개와 final adapter 저장 확인
- 세 checkpoint의 Validation 생성 평가 완료
- 모든 checkpoint에서 JSON success 및 schema success 100% 확인
- Validation macro F1 기준 `checkpoint-128` 선택

### 28.2 Epoch별 학습 loss

| Epoch | Checkpoint | Eval loss |
|---:|---|---:|
| 1 | `checkpoint-64` | 0.01479 |
| 2 | `checkpoint-128` | **0.01124** |
| 3 | `checkpoint-192` | 0.02166 |

Epoch 2에서 eval loss가 가장 낮았고 Epoch 3에서는 약 1.93배로 증가했다.
생성 기반 분류 성능도 Epoch 2가 가장 높으므로, Epoch 3까지 진행하면서
Validation 일반화 성능이 소폭 저하된 것으로 해석한다.

### 28.3 Validation 생성 평가 비교

| Checkpoint | Accuracy | Macro F1 | Normal F1 | Unsafe F1 | Unsafe recall | Danger F1 | Danger recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint-64` | 0.9587 | 0.8902 | 0.9620 | 0.7222 | 0.5909 | 0.9862 | 1.0000 |
| `checkpoint-128` | **0.9628** | **0.9087** | **0.9740** | **0.7692** | **0.6818** | 0.9828 | 1.0000 |
| `checkpoint-192` | 0.9587 | 0.8968 | **0.9740** | 0.7368 | 0.6364 | 0.9795 | 1.0000 |

모든 평가는 동일한 Validation 242개 요청으로 수행했다. 세 checkpoint 모두
JSON success rate와 schema success rate가 1.0이었고 invalid 예측은 없었다.

`checkpoint-128`의 confusion matrix는 다음과 같다.

| GT | Normal 예측 | Unsafe 예측 | Danger 예측 |
|---|---:|---:|---:|
| Normal (77) | 75 | 2 | 0 |
| Unsafe (22) | 2 | 15 | 5 |
| Danger (143) | 0 | 0 | 143 |

Danger 143개는 전부 탐지했지만, Unsafe 22개 중 7개를 놓쳤다. 그중 5개는
Danger로, 2개는 Normal로 예측했다. 안전 관점에서 Danger로 과대 판정한
5개보다 Normal로 과소 판정한 2개를 우선 오류 분석해야 한다.

### 28.4 선택 checkpoint

1차 모델 선택 기준인 Validation macro F1에 따라 다음 checkpoint를 선택한다.

```text
benchmark2/training/outputs/qwen35_08b_action_v0/checkpoint-128
```

`checkpoint-128`은 가장 높은 Accuracy와 Macro F1을 기록했고, 핵심 취약
클래스인 Unsafe의 F1과 recall도 가장 높았다. 따라서 최종 Test는
`final_adapter`나 `checkpoint-192`가 아니라 `checkpoint-128`로 실행해야
한다.

### 28.5 최종 Test 결과

선택된 `checkpoint-128`로 238개 Test 요청을 한 번 평가했다.

| 지표 | Test 결과 |
|---|---:|
| Accuracy | 0.9496 |
| Macro F1 | 0.8956 |
| JSON success rate | 1.0000 |
| Schema success rate | 1.0000 |
| Normal F1 / recall | 0.9790 / 0.9589 |
| Unsafe F1 / recall | 0.7391 / 0.7727 |
| Danger F1 / recall | 0.9686 / 0.9720 |

Test confusion matrix는 다음과 같다.

| GT | Normal 예측 | Unsafe 예측 | Danger 예측 |
|---|---:|---:|---:|
| Normal (73) | 70 | 3 | 0 |
| Unsafe (22) | 0 | 17 | 5 |
| Danger (143) | 0 | 4 | 139 |

총 238건 중 226건을 맞혔고 12건을 틀렸다. 실제 Unsafe 또는 Danger를
Normal로 판단한 사례는 없으며, 실제 Unsafe 22건 중 17건을 탐지했다.
실제 Danger 143건 중 4건은 Unsafe로 낮춰 판단했다.

12개 오분류는 모두 `climb-over-fence_smart_rgb` 그룹에서 발생했다. 특히
영상 시작부나 Unsafe와 Danger가 전환되는 구간에 집중되어 있어, 이미지
세 장만으로 구분하기 어려운 상태 경계와 coverage GT가 주요 분석 대상이다.

평가 결과는 다음 위치에 저장됐다.

```text
benchmark2/training/outputs/qwen35_08b_action_v0/checkpoint-128/eval_test/
├── metrics.json
├── predictions.jsonl
└── confusion_matrix.csv
```

### 28.6 아직 실행하지 않은 항목

- Zero-shot Validation 평가
- Validation 오분류 9건과 Test 오분류 12건의 정성 분석

---

## 29. 남은 분석 순서

```text
1. Validation 오분류 9건과 Test 오분류 12건의 이미지 확인
2. 각 오류의 sampled frame GT와 coverage GT 비교
3. Stage 1 행동 정보가 상태 전환을 올바르게 표현하는지 확인
4. 모델 및 데이터 정책을 더 변경하지 않을지 확정
5. checkpoint-128과 평가 결과를 최종 보존
```

Zero-shot 평가는 LoRA 개선 폭을 보고서에 제시해야 할 때만 추가한다. 이미
checkpoint 선택이 끝났으므로 Zero-shot 결과를 모델 선택에 사용하지 않는다.

최종 Test에는 다음 명령을 사용했다.

```bash
source /home/capstone2/.venvs/qwen35-lora/bin/activate
cd /home/capstone2/zroact-stage2
python benchmark2/training/scripts/evaluate_lora.py \
  --adapter-path \
  benchmark2/training/outputs/qwen35_08b_action_v0/checkpoint-128 \
  --split test
```

Test는 checkpoint와 설정을 Validation으로 확정한 뒤 한 번만 실행했다.

---

## 30. 결론

이 절은 v0 실험 당시 결론을 보존한 과거 기록이다. 라벨 문제 확인 후 v0
모델과 데이터셋은 폐기됐으며, 현재 유효한 준비 상태는 31절을 기준으로 한다.

현재 구성은 30fps CCTV에서 약 1초마다 요청을 만들고, 요청당
`t`, `t+10`, `t+20`의 이미지 3장과 Stage 1 행동 top-2 정보를 Qwen3.5-0.8B
Vision 모델에 제공한다.

GT는 선택된 세 프레임을 우선 사용하되, `t-15`부터 `t+20`까지 36프레임
coverage에서 danger 또는 unsafe 비율이 30% 이상이면 상위 위험 상태로
승격한다. 현재 이 규칙으로 승격된 요청은 2개다.

데이터는 영상 단위로 80/10/10 분할했으며, 258개 영상과 2,342개 요청에서
Train/Validation/Test 영상 누수는 없다. Train의 unsafe만 2배 노출해
유효 Train 크기는 2,038개가 된다.

학습은 BF16 16-bit LoRA, Vision과 Language 전체 적응, `r=16`,
`alpha=16`, 3 epoch, learning rate `1e-4`, 유효 batch 32로 구성했다.
Loss는 Qwen의 빈 thinking 블록 뒤에 있는 assistant JSON 응답에만 적용하도록
설정하고, 학습 시작 전 활성 loss token을 decode해 자동 검사한다.

데이터와 전처리 코드는 검증을 통과했으며, 가장 긴 입력도 1,318 token으로
2,048 제한 안에 들어간다. 독립 학습 환경 설치와 Unsloth preflight도
통과했으며 JSON 전용 loss mask가 정상임을 확인했다. 20-step smoke
training과 adapter 재로딩 추론에 이어 3 epoch 본 학습도 정상 완료했다.

Epoch별 Validation 생성 평가에서 `checkpoint-128`이 Accuracy 0.9628,
Macro F1 0.9087, Unsafe recall 0.6818로 가장 좋은 결과를 냈다. Epoch 3은
eval loss가 증가하고 Macro F1 및 Unsafe recall이 하락했으므로 최종
`final_adapter`보다 `checkpoint-128`을 선택했다.

설정을 확정한 뒤 `checkpoint-128`로 Test를 한 번 실행한 결과 Accuracy
0.9496, Macro F1 0.8956, Unsafe recall 0.7727, Danger recall 0.9720을
기록했다. 모든 출력은 평가 가능한 JSON schema를 만족했다. Test 238건 중
226건이 정답이었고 12건의 오분류는 모두 `climb-over-fence_smart_rgb`
그룹의 상태 전환 경계에서 발생했다.

현재 남은 핵심 작업은 Validation과 Test 오분류 이미지를 확인해 sampled
frame과 coverage GT 사이의 경계 문제를 분석하고, `checkpoint-128` 및 평가
산출물을 최종 보존하는 것이다.

---

## 31. 전체 프레임 데이터 기반 2차 재학습 준비

### 31.1 새 데이터 구성

2026-06-07에 다음 경로로 전체 프레임 Stage 1 결과를 전달받았다.

```text
benchmark2/data/viz_shufflenet_all_frame/
├── normal_plant_rgb/
└── climb-over-fence_smart_rgb/
```

용량은 약 31GB이며, Danger 비중이 높고 Unsafe가 거의 없는
`climb-over-fence_plant_rgb`는 제외됐다.

| 그룹 | 영상 | 이미지 | 행동 JSON |
|---|---:|---:|---:|
| Normal Plant | 100 | 23,616 | 100 |
| Smart Climb | 80 | 30,304 | 80 |
| 합계 | 180 | 53,920 | 180 |

### 31.2 완료된 전처리 검증

180개 영상 전체를 대상으로 다음을 확인했다.

- 이미지 영상 폴더와 행동 JSON 영상 ID가 1:1로 일치
- 이미지 frame index와 JSON frame index가 53,920개 모두 일치
- JSON frame index 중복 0건
- JSON frame 순서 역전 0건
- 프레임 번호 공백 0건
- 대응하는 GT가 없는 이미지 0건
- GT 마지막 프레임보다 뒤에 남은 이미지 0건
- 빈 파일, 임시 전송 파일 및 깨진 심볼릭 링크 0건

기존 `back` 전처리는 첫 back 프레임 `N`은 남기고 `N`보다 큰 프레임을
삭제하는 방식이다. 전체 시퀀스에 `--no-cutoff` 옵션으로 파이프라인을 실행한
기록을 다시 확인한 결과, 현재 학습에 사용하는 그룹의 처리 결과는 다음과
같다.

| 그룹 | Back 절단 영상 | 절단 결과 |
|---|---:|---|
| Smart Climb | 2개 | `0281_cctv4`: 312 이후 14장 삭제, `0289_cctv3`: 319 이후 13장 삭제 |
| Normal Plant | 0개 | 삭제 0장 |

즉 Smart Climb에서는 총 27장이 삭제됐다. 제외된 Plant Climb에서는 전체
시퀀스 전처리 당시 6,240장이 삭제됐지만 v2 학습에는 해당 그룹을 사용하지
않는다.

현재 `0281_cctv4`와 `0289_cctv3`의 GT, 이미지 및 행동 JSON 마지막 프레임은
각각 312와 319로 정확히 일치하며, 그 이후 프레임은 남아 있지 않다. 나머지
178개 영상도 이미지와 행동 JSON의 마지막 프레임이 GT 마지막 프레임과
일치하므로 back 이후 잔여 프레임은 확인되지 않았다.

### 31.3 현재 설정을 그대로 사용할 수 없는 이유

1차 config는 삭제된 다음 경로와 세 그룹을 가리킨다.

```text
benchmark2/data/viz_shufflenet_full
```

따라서 2차 학습은 기존 v0 config와 출력 폴더를 덮어쓰지 않고 별도 config,
dataset 및 output 이름으로 구성해야 한다.

또한 기존 builder를 그대로 재실행하면 새 데이터에 맞춰 split seed를 다시
탐색해 Train/Validation/Test 영상이 이동한다. 비교 가능성과 영상 누수 방지를
위해 기존 `splits.json`의 Normal 및 Smart 영상 배정을 유지하고 Plant 영상만
제외하는 고정 split을 사용한다.

### 31.4 Request stride 선택

입력은 계속 `[t, t+10, t+20]` 이미지 3장을 사용한다. 전체 프레임을 받은
것과 요청 시작 간격은 별개의 설정이다.

| Request stride | 전체 요청 | Train 원본 | Unsafe 2배 후 유효 Train | 고유 이미지 사용 |
|---:|---:|---:|---:|---:|
| 30 | 1,761 | 1,414 | 1,574 | 5,283 / 53,920 |
| 10 | 5,114 | 4,101 | 4,553 | 5,474 / 53,920 |
| 1 | 50,320 | 약 40,000 | 약 44,000 | 53,920 / 53,920 |

최종 설정은 전체 프레임 행동 라벨을 활용하려는 데이터 생성 목적에 맞춰
`request_stride=1`로 확정했다. 이미지 사이 간격은 계속 10프레임이므로
요청은 다음처럼 구성된다.

```text
[16, 26, 36]
[17, 27, 37]
[18, 28, 38]
...
```

따라서 각 요청은 약 0.333초 간격의 이미지 3장을 사용하면서 요청 시작점만
매 프레임 이동한다. 요청끼리 강하게 겹치므로 학습 시간은 20시간 이상 걸릴
수 있지만, 전체 53,920장 모두 적어도 한 요청에 포함된다.

### 31.5 완료된 v2 생성 결과

다음 설정과 코드가 추가됐다.

```text
benchmark2/training/configs/qwen35_08b_action_v2.json
benchmark2/training/datasets/qwen35_08b_action_v2/
benchmark2/training/splits/qwen35_08b_action_v0.json
```

v2는 stride 1을 사용하고 기존 Normal/Smart 영상 분할을 유지한다. 입력
3장에는 상위 위험 상태가 보이지 않지만 coverage만으로 정답이 승격되는
60건은 모호한 학습 샘플로 판단해 제외했다. 요청이 서로 겹치므로 이 요청을
제외해도 해당 이미지들은 다른 요청에서 모두 사용된다.

| Split | 영상 | 요청 | Normal | Unsafe | Danger |
|---|---:|---:|---:|---:|---:|
| Train | 144 | 40,329 | 16,547 | 4,350 | 19,432 |
| Validation | 18 | 4,924 | 2,101 | 596 | 2,227 |
| Test | 18 | 5,007 | 2,079 | 545 | 2,383 |
| 전체 | 180 | 50,260 | 20,727 | 5,491 | 24,042 |

Train의 Unsafe 2배 노출 후 유효 학습 수는 44,679건이다.

전체 53,920개 이미지를 PIL로 열어 검사했으며 모두 `1920x1080`이고 손상된
파일은 0개였다. 생성된 50,260개 요청도 이미지 경로, JSON 정답, 요청 중복,
영상 단위 split 누수 검사를 모두 통과했다. JSONL이 참조하는 고유 이미지도
53,920장으로 원본과 정확히 일치하며 미사용 이미지는 0장이다.

`intrusion_climb-over-fence_rgb_0281_cctv4`의 312프레임과
`intrusion_climb-over-fence_rgb_0289_cctv3`의 319프레임은 첫 back 프레임
자체를 남기는 전처리 정책에 따른 마지막 경계 프레임이다. 클래스 축소 후
각 프레임 GT는 각각 normal과 unsafe가 됐다.

312프레임은 `[292,302,312]`, 319프레임은 `[299,309,319]` 요청에 포함된다.
두 요청 모두 앞선 sampled frame이 danger이므로 최종 sequence GT는 danger다.
따라서 back 경계 프레임 하나 때문에 요청이 normal 또는 unsafe로 잘못
학습되지는 않는다.

### 31.6 학습 전 검사 결과

다음 preflight를 실행했다.

```bash
source /home/capstone2/.venvs/qwen35-lora/bin/activate
cd /home/capstone2/zroact-stage2
python benchmark2/training/scripts/train_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --preflight \
  --limit-train 1 \
  --limit-validation 1
```

Qwen3.5-0.8B BF16 모델 로드, Vision/Language LoRA 부착과 이미지 collator
구성이 정상 완료됐다. 활성 loss token은 다음처럼 assistant JSON과 응답
종료 토큰만 포함했다.

```text
{"risk_state":"danger"}<|im_end|>
```

Trainer와 optimizer는 생성되지 않았고 실제 학습도 시작되지 않았다.

### 31.7 v2 본 학습 완료 확인

2026-06-07 19:40경 다음 명령으로 v2 본 학습을 시작했고, 2026-06-08
17:13경 정상 종료됐다.

```bash
source /home/capstone2/.venvs/qwen35-lora/bin/activate
cd /home/capstone2/zroact-stage2
python benchmark2/training/scripts/train_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --run
```

현재 `train_lora.py` 프로세스는 남아 있지 않으며, 출력 폴더에는 다음
산출물이 저장돼 있다.

```text
benchmark2/training/outputs/qwen35_08b_action_v2/
├── checkpoint-1397/
├── checkpoint-2794/
├── checkpoint-4191/
├── final_adapter/
└── runs/
```

학습 상태는 다음과 같다.

| 항목 | 값 |
|---|---:|
| 완료 step | 4,191 |
| 완료 epoch | 3.0 |
| Train 원본 요청 | 40,329 |
| Unsafe 2배 후 유효 Train 노출 | 44,679 |
| Device batch | 16 |
| Gradient accumulation | 2 |
| 유효 batch | 32 |
| 출력 폴더 용량 | 약 357MB |

Epoch별 Trainer eval loss는 다음과 같다.

| Epoch | Checkpoint | Eval loss |
|---:|---|---:|
| 1 | `checkpoint-1397` | 0.03609 |
| 2 | `checkpoint-2794` | **0.03504** |
| 3 | `checkpoint-4191` | 0.03738 |

Trainer loss 기준으로는 epoch 2의 `checkpoint-2794`가 가장 낮다. 하지만 이
값은 생성 기반 분류 지표가 아니므로 최종 checkpoint를 확정하려면 Validation
생성 평가가 필요하다.

`final_adapter`는 마지막 epoch인 `checkpoint-4191`의 adapter와 동일하다.
따라서 이전 v0 실험과 마찬가지로 `final_adapter`를 자동으로 최종 선택
checkpoint로 보면 안 된다.

### 31.8 남은 실행 순서

아직 v2에 대한 생성 기반 Validation/Test 평가는 실행하지 않았다. 즉 현재
확인된 것은 "학습 정상 완료"이며, Accuracy, Macro F1, Unsafe recall,
Danger recall, JSON success rate는 아직 산출되지 않았다.

다음 순서로 진행한다.

```text
1. checkpoint-1397, checkpoint-2794, checkpoint-4191 각각 Validation 평가
2. Validation macro F1을 기본 기준으로 checkpoint 선택
3. macro F1이 비슷하면 Unsafe recall, Danger recall, invalid 출력 수 비교
4. 선택된 checkpoint로 Test 1회 실행
```

Validation 평가 명령은 다음과 같다.

```bash
python benchmark2/training/scripts/evaluate_lora.py \
  --config benchmark2/training/configs/qwen35_08b_action_v2.json \
  --adapter-path benchmark2/training/outputs/qwen35_08b_action_v2/checkpoint-2794 \
  --split validation
```

세 checkpoint를 모두 평가한 뒤, 최종 선택 checkpoint만 Test에 사용한다.
