# Qwen3.5-0.8B CCTV 위험 상태 분류 LoRA v2 중간 결과 보고서

## 1. 문서 개요

| 항목 | 내용 |
|---|---|
| 작성 기준일 | 2026-06-09 |
| 프로젝트 | `/home/capstone2/zroact-stage2` |
| 모델 | Qwen3.5-0.8B Vision |
| 학습 방식 | Unsloth 기반 BF16 LoRA SFT |
| 입력 | 시간순 CCTV 이미지 3장 + Stage 1 행동 정보 |
| 출력 | `{"risk_state":"normal|unsafe|danger"}` |
| 학습 상태 | v2 3 epoch 학습 완료 |
| 완료된 평가 | `checkpoint-2794` Validation, base model zero-shot Test |
| 미완료 평가 | 다른 epoch checkpoint Validation, 선택 모델 Test |

이 문서는 v2 데이터 전처리부터 LoRA 학습, Validation 생성 평가와
학습하지 않은 base model의 zero-shot Test 평가까지 현재 완료된 결과를
정리한다.

세부 구현 과정은 다음 문서를 참고한다.

```text
benchmark2/training/IMPLEMENTATION_REPORT_KO.md
```

---

## 2. 핵심 결과 요약

1. v2 데이터는 총 180개 영상, 53,920개 이미지, 50,260개 요청으로 구성됐다.
2. 요청 시작 간격을 1프레임으로 설정해 전체 53,920개 이미지를 모두 사용했다.
3. Train의 `unsafe`만 2배 노출하여 유효 학습 요청 수는 44,679개가 됐다.
4. 3 epoch, 총 4,191 optimizer step의 LoRA 학습이 정상 완료됐다.
5. Trainer eval loss는 epoch 2의 `checkpoint-2794`가 가장 낮았다.
6. `checkpoint-2794`의 Validation Accuracy는 92.97%, Macro F1은 84.54%다.
7. Validation에서 Danger recall은 100%지만 Unsafe recall은 46.14%다.
8. Validation 오분류 346건은 모두 Smart Climb 그룹의 7개 영상에 집중됐다.
9. 학습하지 않은 base model은 Test 5,007건 중 5,006건을 `danger`로 예측했다.
10. LoRA는 JSON 출력 형식과 작업별 분류를 학습하는 데 명확한 효과가 있었다.

현재 가장 중요한 문제는 `unsafe`와 `normal` 또는 `danger` 사이의 상태
경계를 안정적으로 구분하는 것이다.

---

## 3. v2 데이터 구성

### 3.1 데이터 원천

```text
benchmark2/data/viz_shufflenet_all_frame/
├── normal_plant_rgb/
└── climb-over-fence_smart_rgb/
```

Danger 비중이 높고 Unsafe가 거의 없는 Plant Climb 그룹은 v2에서 제외했다.

| 그룹 | 영상 수 | 이미지 수 |
|---|---:|---:|
| Normal Plant | 100 | 23,616 |
| Smart Climb | 80 | 30,304 |
| 합계 | 180 | 53,920 |

### 3.2 요청 생성 규칙

각 요청은 다음 3장의 이미지를 사용한다.

```text
[t, t+10, t+20]
```

30fps 기준 이미지 간 시간 차이는 약 0.333초이고, 첫 이미지부터 마지막
이미지까지의 구간은 약 0.667초다.

v2에서는 요청 시작점을 매 프레임 이동한다.

```text
[16, 26, 36]
[17, 27, 37]
[18, 28, 38]
...
```

설정값은 다음과 같다.

| 설정 | 값 |
|---|---:|
| 요청당 이미지 | 3장 |
| 이미지 간 프레임 차이 | 10 |
| 요청 시작 stride | 1 |
| 이미지 크기 | 768x432 |
| 행동 정보 | 프레임별 confidence 상위 2개 이름 |
| 행동 confidence 숫자 | 입력에서 제외 |

요청끼리 많은 이미지를 공유하지만 전체 53,920개 이미지는 모두 적어도 한
요청에 포함된다.

### 3.3 Back 전처리 확인

Back 이후 프레임을 남기지 않는 기존 전처리 정책을 확인했다.

| 영상 | 마지막 유지 프레임 | 삭제된 이후 이미지 |
|---|---:|---:|
| `intrusion_climb-over-fence_rgb_0281_cctv4` | 312 | 14장 |
| `intrusion_climb-over-fence_rgb_0289_cctv3` | 319 | 13장 |

두 영상 모두 GT, 이미지 및 행동 JSON의 마지막 프레임이 정확히 일치한다.
나머지 178개 영상에서도 back 이후 잔여 프레임은 확인되지 않았다.

### 3.4 행동 정보 완전성

| 항목 | 수 |
|---|---:|
| 전체 고유 이미지 | 53,920 |
| 실제 행동 후보가 있는 이미지 | 53,635 |
| 행동이 없어 `none`인 이미지 | 285 |
| `none` 프레임을 하나 이상 포함한 요청 | 468 |
| 이미지 3장이 모두 `none`인 요청 | 92 |

모든 이미지 프레임에는 대응하는 행동 JSON frame 항목이 존재한다. 위 285개는
JSON 누락이 아니라 해당 프레임에서 사용할 행동 후보가 없어 `none`으로
입력된 경우다.

### 3.5 모호한 Coverage 승격 제거

선택 이미지 3장에는 상위 위험 상태가 보이지 않지만 coverage 구간만으로
정답이 승격되는 60건은 제외했다.

이 요청들은 stride 1로 서로 겹치기 때문에, 60건을 제외해도 해당 이미지
자체는 다른 요청에서 계속 사용된다.

---

## 4. 데이터 분할과 클래스 분포

분할은 요청 단위가 아니라 영상 단위로 수행했으며, 기존 고정 split을
유지했다.

| Split | 영상 수 | 요청 수 | Normal | Unsafe | Danger |
|---|---:|---:|---:|---:|---:|
| Train | 144 | 40,329 | 16,547 | 4,350 | 19,432 |
| Validation | 18 | 4,924 | 2,101 | 596 | 2,227 |
| Test | 18 | 5,007 | 2,079 | 545 | 2,383 |
| 전체 | 180 | 50,260 | 20,727 | 5,491 | 24,042 |

전체 클래스 비율은 다음과 같다.

| 클래스 | 요청 수 | 비율 |
|---|---:|---:|
| Normal | 20,727 | 41.24% |
| Unsafe | 5,491 | 10.93% |
| Danger | 24,042 | 47.84% |

Train에서만 Unsafe를 2배로 노출한다.

| 클래스 | 원본 Train | 유효 Train 노출 |
|---|---:|---:|
| Normal | 16,547 | 16,547 |
| Unsafe | 4,350 | 8,700 |
| Danger | 19,432 | 19,432 |
| 합계 | 40,329 | 44,679 |

Oversampling 이후에도 Danger가 가장 많지만, Unsafe 비중은 원본 약 10.79%에서
유효 노출 약 19.47%로 증가한다.

---

## 5. LoRA 학습 설정

### 5.1 모델과 LoRA

| 항목 | 값 |
|---|---|
| Base model | `benchmark2/models/Qwen3.5-0.8B` |
| Base dtype | BF16 |
| 4-bit load | 사용 안 함 |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Vision layers | 학습 |
| Language layers | 학습 |
| Attention modules | 학습 |
| MLP modules | 학습 |

### 5.2 Trainer 설정

| 항목 | 값 |
|---|---:|
| Epoch | 3 |
| Device batch | 16 |
| Gradient accumulation | 2 |
| 유효 batch | 32 |
| Learning rate | `1e-4` |
| Optimizer | `adamw_8bit` |
| Scheduler | Linear |
| Warmup ratio | 0.05 |
| Max sequence length | 2048 |
| Seed | 42 |

Loss는 system/user prompt와 이미지 토큰에는 적용하지 않고 assistant의 정답
JSON 및 응답 종료 토큰에만 적용했다.

---

## 6. 학습 완료 결과

학습은 2026-06-07 19:40경 시작해 2026-06-08 17:13경 종료됐다.

| 항목 | 결과 |
|---|---:|
| 완료 epoch | 3.0 |
| Optimizer step | 4,191 |
| 유효 Train 노출 | 44,679 |
| 소요 시간 | 약 21시간 33분 |
| 출력 폴더 용량 | 약 362MB |

생성된 산출물은 다음과 같다.

```text
benchmark2/training/outputs/qwen35_08b_action_v2/
├── checkpoint-1397/
├── checkpoint-2794/
├── checkpoint-4191/
├── final_adapter/
└── runs/
```

### 6.1 Epoch별 Eval loss

| Epoch | Checkpoint | Eval loss |
|---:|---|---:|
| 1 | `checkpoint-1397` | 0.03609 |
| 2 | `checkpoint-2794` | **0.03504** |
| 3 | `checkpoint-4191` | 0.03738 |

Epoch 2에서 eval loss가 가장 낮고 epoch 3에서 다시 증가했다. 따라서 현재
생성 평가를 먼저 수행한 대상은 `checkpoint-2794`다.

`final_adapter`는 마지막 epoch인 `checkpoint-4191`과 동일하므로 자동으로
최종 모델로 선택하면 안 된다.

---

## 7. Checkpoint-2794 Validation 결과

평가 대상은 Validation 전체 4,924건이다.

| 지표 | 결과 |
|---|---:|
| Accuracy | 0.9297 |
| Macro F1 | 0.8454 |
| JSON success rate | 1.0000 |
| Schema success rate | 1.0000 |
| 정답 요청 | 4,578 |
| 오분류 요청 | 346 |

### 7.1 클래스별 결과

| 클래스 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.9255 | 0.9881 | 0.9558 | 2,101 |
| Unsafe | 0.9167 | 0.4614 | 0.6138 | 596 |
| Danger | 0.9353 | 1.0000 | 0.9666 | 2,227 |

### 7.2 Confusion matrix

| 실제 GT | Normal 예측 | Unsafe 예측 | Danger 예측 |
|---|---:|---:|---:|
| Normal 2,101 | 2,076 | 25 | 0 |
| Unsafe 596 | 167 | 275 | 154 |
| Danger 2,227 | 0 | 0 | 2,227 |

### 7.3 결과 해석

강점은 다음과 같다.

1. Danger 2,227건을 모두 탐지해 Danger recall 100%를 기록했다.
2. 실제 Danger를 Normal 또는 Unsafe로 낮춰 판단한 경우가 없다.
3. Normal recall도 98.81%로 높다.
4. 4,924개 응답이 모두 올바른 JSON schema를 만족했다.

가장 큰 약점은 Unsafe다.

| Unsafe 결과 | 요청 수 | Unsafe GT 대비 비율 |
|---|---:|---:|
| 정확히 Unsafe로 예측 | 275 | 46.14% |
| Normal로 과소 판정 | 167 | 28.02% |
| Danger로 과대 판정 | 154 | 25.84% |

Unsafe 596건 중 321건을 놓쳤다. 특히 Unsafe를 Normal로 판단한 167건은
안전 관점에서 우선 분석해야 하는 오류다.

---

## 8. Validation 오분류 집중 구간

346개 오분류는 모두 `climb-over-fence_smart_rgb` 그룹의 7개 영상에서
발생했다. Normal Plant 그룹에서는 오분류가 없었다.

| 영상 | 전체 요청 | 오분류 | 오류율 | 주요 오류 |
|---|---:|---:|---:|---|
| `0361_cctv2` | 440 | 148 | 33.64% | Unsafe -> Normal 146 |
| `0081_cctv4` | 390 | 81 | 20.77% | Unsafe -> Danger 81 |
| `0962_cctv2` | 309 | 36 | 11.65% | Unsafe -> Normal/Danger |
| `1109_cctv3` | 294 | 32 | 10.88% | Unsafe -> Danger 27 |
| `0401_cctv4` | 409 | 29 | 7.09% | Normal -> Unsafe 25 |
| `0110_cctv2` | 423 | 15 | 3.55% | Unsafe -> Danger 15 |
| `1177_cctv1` | 291 | 5 | 1.72% | Unsafe -> Danger 5 |

상위 두 영상인 `0361_cctv2`와 `0081_cctv4`가 전체 오분류의 229건,
약 66.18%를 차지한다.

주의할 점은 request stride가 1이라는 것이다. 인접 요청은 이미지 3장 중
대부분을 공유하므로 148개 오류가 서로 독립적인 148개 사건을 의미하지
않는다. 하나의 상태 전환 경계나 장면 해석 오류가 연속 요청으로 반복 집계될
수 있다.

따라서 다음 분석에서는 요청 단위 오류 수와 함께 영상별 연속 오류 구간,
상태 전환 시점 및 대표 프레임을 확인해야 한다.

---

## 9. 학습 전 Base Model Zero-shot Test 결과

LoRA adapter를 적용하지 않은 Qwen3.5-0.8B base model로 Test 전체
5,007건을 평가했다.

| 지표 | 결과 |
|---|---:|
| Accuracy | 0.4757 |
| Macro F1 | 0.2149 |
| JSON success rate | 0.9998 |
| Schema success rate | 0.9998 |

### 9.1 클래스별 결과

| 클래스 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.0000 | 0.0000 | 0.0000 | 2,079 |
| Unsafe | 0.0000 | 0.0000 | 0.0000 | 545 |
| Danger | 0.4758 | 0.9996 | 0.6447 | 2,383 |

### 9.2 Confusion matrix

| 실제 GT | Normal 예측 | Unsafe 예측 | Danger 예측 | Invalid |
|---|---:|---:|---:|---:|
| Normal 2,079 | 0 | 0 | 2,079 | 0 |
| Unsafe 545 | 0 | 0 | 545 | 0 |
| Danger 2,383 | 0 | 0 | 2,382 | 1 |

Base model은 5,007건 중 5,006건을 `danger`로 예측했다. 나머지 1건은 JSON
대신 설명문을 생성해 invalid로 처리됐다.

즉 zero-shot Accuracy 47.57%는 실제 분류 능력이라기보다 Test의 Danger
비율과 거의 동일한 값이다. Base model은 현재 프롬프트만으로 세 위험 상태를
구분하지 못한다.

Validation의 LoRA 결과와 zero-shot Test는 서로 다른 split에서 계산됐으므로
두 수치를 정확한 성능 향상 폭으로 직접 빼면 안 된다. 다만 다음 사실은
분명하다.

1. Base model은 사실상 Danger 한 클래스만 출력한다.
2. LoRA 모델은 Normal, Unsafe, Danger 세 클래스를 모두 출력한다.
3. LoRA 모델은 JSON schema를 100% 준수한다.
4. LoRA 학습이 작업 형식과 클래스 구분을 실질적으로 학습시켰다.

---

## 10. 현재 결론

### 10.1 확인된 성과

v2 LoRA 학습은 정상 완료됐으며, 모델은 학습 전 base model의 단일 클래스
편향에서 벗어났다.

`checkpoint-2794`는 Validation에서 다음 특성을 보인다.

- Normal과 Danger를 매우 안정적으로 구분
- Danger 누락 0건
- 올바른 JSON 출력 100%
- 전체 Accuracy 약 93%
- Macro F1 약 84.5%

### 10.2 현재 병목

Unsafe recall 46.14%가 가장 큰 병목이다. Unsafe는 Normal과 Danger 사이의
중간 상태이므로 다음 문제가 함께 작용할 가능성이 있다.

1. Unsafe 자체의 학습 샘플이 상대적으로 적음
2. 상태 전환 경계에서 프레임별 시각 증거가 모호함
3. stride 1 요청이 동일한 경계 오류를 반복 증폭함
4. Stage 1 행동 정보가 위험 대상 인물과 정확히 연결되지 않을 수 있음
5. Unsafe 라벨 기준이 영상 또는 구간별로 일관되지 않을 가능성

현재 수치만으로 특정 원인 하나를 확정할 수는 없다. 우선 오류가 집중된
7개 영상, 특히 `0361_cctv2`와 `0081_cctv4`를 확인해야 한다.

### 10.3 아직 최종 모델이 아닌 이유

현재 생성 평가를 완료한 학습 checkpoint는 `checkpoint-2794` 하나뿐이다.

Trainer eval loss가 가장 낮다는 이유만으로 최종 checkpoint를 확정하면
안 된다. 다음 두 checkpoint도 동일한 Validation 생성 평가가 필요하다.

```text
checkpoint-1397
checkpoint-4191
```

세 checkpoint의 Macro F1, Unsafe recall, Danger recall 및 invalid 출력 수를
비교한 뒤 최종 모델을 선택해야 한다.

---

## 11. 다음 실행 순서

### 11.1 남은 Validation 평가

1. `checkpoint-1397` Validation 평가
2. `checkpoint-4191` Validation 평가
3. 세 checkpoint 결과 비교
4. Validation 기준으로 최종 checkpoint 확정

최종 선택 기준의 우선순위는 다음과 같다.

```text
1. Macro F1
2. Unsafe recall
3. Danger recall
4. Invalid 출력 수
5. Accuracy
```

### 11.2 정성 오류 분석

다음 영상을 우선 확인한다.

```text
intrusion_climb-over-fence_rgb_0361_cctv2
intrusion_climb-over-fence_rgb_0081_cctv4
intrusion_climb-over-fence_rgb_0962_cctv2
intrusion_climb-over-fence_rgb_1109_cctv3
intrusion_climb-over-fence_rgb_0401_cctv4
intrusion_climb-over-fence_rgb_0110_cctv2
intrusion_climb-over-fence_rgb_1177_cctv1
```

각 영상에서 다음 항목을 함께 확인한다.

1. 이미지 3장의 실제 시각적 상태
2. 각 sampled frame GT
3. Stage 1 행동 문자열
4. Unsafe 시작 및 종료 시점
5. 연속 요청에서 예측이 전환되는 프레임
6. 라벨 기준의 일관성

### 11.3 최종 Test

최종 Test는 checkpoint를 Validation으로 확정한 뒤 선택된 LoRA 모델에 대해
한 번 실행한다.

이번에 수행한 base model zero-shot Test 결과는 baseline으로만 사용하며,
LoRA checkpoint 선택이나 하이퍼파라미터 변경 근거로 Test 정답을 사용하면
안 된다.

---

## 12. 저장된 결과 경로

### 학습 설정

```text
benchmark2/training/configs/qwen35_08b_action_v2.json
```

### 데이터셋

```text
benchmark2/training/datasets/qwen35_08b_action_v2/
```

### Checkpoint-2794 Validation

```text
benchmark2/training/outputs/qwen35_08b_action_v2/checkpoint-2794/eval_validation/
├── metrics.json
├── predictions.jsonl
└── confusion_matrix.csv
```

### Base model zero-shot Test

```text
benchmark2/training/outputs/qwen35_08b_action_v2/zero_shot_test/
├── metrics.json
├── predictions.jsonl
└── confusion_matrix.csv
```

---

## 13. 운영상 주의사항

2026-06-09 확인 시 루트 파일 시스템의 남은 공간은 약 1.3GB로 사용률이
100%에 가깝다.

현재 v2 학습 및 평가 폴더 자체는 약 362MB이고, zero-shot 및 Validation
예측 파일은 각각 약 2MB다. 디스크 부족의 주원인은 현재 평가 결과가 아닌
다른 대용량 모델 및 프로젝트 파일을 포함한 전체 파일 시스템 사용량이다.

추가 checkpoint 평가나 최종 Test 전에 여유 공간을 확보하는 것이 안전하다.
특히 학습 재실행, 모델 복사, adapter merge 또는 새로운 checkpoint 저장은
현재 공간에서 진행하지 않는 것이 좋다.

---

## 14. 최종 요약

Qwen3.5-0.8B base model은 zero-shot Test에서 거의 모든 요청을 Danger로
판단해 실제 3-class 분류 능력을 보이지 못했다.

반면 v2 LoRA의 `checkpoint-2794`는 Validation에서 Accuracy 92.97%,
Macro F1 84.54%, Danger recall 100%, JSON schema success 100%를 기록했다.
따라서 LoRA 학습이 CCTV 위험 상태 분류 작업과 출력 형식을 모델에
성공적으로 적응시킨 것으로 판단한다.

다만 Unsafe recall은 46.14%로 낮으며, 346개 오분류가 Smart Climb의 7개
영상에 집중돼 있다. 특히 두 영상이 전체 오류의 약 66%를 차지하므로
데이터 전체를 다시 수정하기 전에 해당 영상의 상태 경계와 라벨 일관성을
먼저 확인하는 것이 효율적이다.

현재 단계의 올바른 다음 작업은 나머지 두 epoch checkpoint를 같은 Validation
조건으로 평가하고, 최종 checkpoint를 선택한 뒤 LoRA Test를 한 번 수행하는
것이다.
