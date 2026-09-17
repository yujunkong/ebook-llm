# 제64강. Mixed Precision과 Gradient Accumulation

> **학습 목표**
> - FP32 / FP16 / BF16이 무엇(비트·역할)인지 개념적으로 구분한다.
> - Mixed Precision 학습의 아이디어와 loss scaling이 필요한 이유를 설명한다.
> - Gradient Accumulation으로 실효 배치를 키우는 식을 쓴다.
> - 메모리·속도·수치 안정성 사이의 트레이드오프를 표로 정리한다.
> - 사실(하드웨어/형식 정의)과 설명(관행·직관)을 섞지 않고 서술한다.

---
## 1. 왜 이것을 배우는가

LLM 학습의 병목은 종종 “계산이 느리다”보다 **메모리에 안 들어간다**에 가깝다. Activations, 파라미터, Adam 상태($m,v$), 기울기가 동시에 산다.

```text
같은 모델
  ├─ 정밀도를 낮추면 → 텐서당 바이트↓ (대개)
  └─ micro-batch를 줄이고 누적하면 → 한 번에 올리는 activation↓
```

이 두 기법은 “모델을 작게 만들기” 위한 것이 아니라, **같은 모델을 학습 가능하게 만들기** 위한에 가깝다.

## 2. 먼저 알아야 할 개념

1. **Tensor dtype / GPU 메모리 감각** — 제9, 19, 52강
2. **Autograd와 `.grad` 누적** — 제20, 23강
3. **Train step** — 제62강
4. **AdamW 상태 메모리** — 제63강

벤치마크 TFLOPS·속도up % 같은 **구체 성능 수치를 이 강의에서 단정하지 않는다**. 하드웨어·커널·모델에 따라 달라지기 때문이다.

## 3. 숫자 형식 — FP32, FP16, BF16

### 3.1 용어 정의

| 형식 | 대략적 의미 | 비고 |
|---|---|---|
| **FP32** | 32비트 부동소수점 (단정밀도) | 전통적 기본 학습 정밀도 |
| **FP16** | 16비트 부동소수점 (반정밀도) | 지수 비트가 FP32보다 적어 **범위가 좁음** |
| **BF16 (bfloat16)** | 16비트, 지수는 FP32에 가깝게 유지하고 가수를 줄인 형식 | 범위는 상대적으로 넓고, 정밀도(가수)는 희생 |

사실:

- 비트 배치(부호·지수·가수)는 IEEE/벤더 정의에 따른 **형식 사실**이다.
- “어떤 GPU가 BF16 텐서 코어를 지원하는가”는 **하드웨어·소프트웨어 스택의 사실**이며, 세대마다 다르다. 특정 칩 성능을 여기서 숫자로 단언하지 않는다.

설명:

- 낮은 정밀도는 메모리·대역폭·(지원 시) 연산 처리에 유리할 **수 있다**.
- 대신 반올림 오차·오버플로/언더플로 위험이 커질 **수 있다**.

### 3.2 왜 FP16에서 범위가 문제인가

기울기나 loss 관련 값이 매우 작으면 FP16에서 **0으로 밑으로 꺼지거나(underflow)**, 크면 **Inf로 터질(overflow)** 수 있다. BF16은 지수 비트 설계상 범위 문제가 FP16보다 덜한 경우가 많다(설명: 과제·텐서 분포에 의존).

## 4. Mixed Precision이란

**Mixed Precision Training**은 연산·텐서의 일부는 낮은 정밀도(FP16/BF16)로, 일부는 FP32로 유지하는 학습 방식이다.

전형적 아이디어:

```text
forward / backward의 행렬곱 등: low precision
가중치 마스터 카피(또는 중요 축소 연산): FP32 유지
optimizer가 갱신하는 파라미터: 대개 FP32 마스터
```

PyTorch에서는 `torch.autocast`가 “이 영역 안에서는 허용된 연산을 자동으로 낮은 정밀도로” 수행하게 돕는다.

```python
from torch import amp

# device_type은 'cuda' 등. dtype은 float16 또는 bfloat16
with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(input_ids)
    loss = loss_fn(logits, labels)
```

설명: autocast는 마법이 아니라 **연산 화이트리스트에 따라 dtype을 선택**하는 컨텍스트다. 모든 연산이 항상 16비트로 가는 것은 아니다.

## 5. Loss Scaling 개념 (특히 FP16)

### 5.1 문제

FP16 경로에서는 backward 중 **작은 gradient가 반정밀도에서 사라지는** 경우가 있다.

### 5.2 아이디어

Loss에 큰 배수 $S$를 곱해 backward하면 기울기도 대략 $S$배 커진다. 최적화 직전(또는 unscale 후)에 $1/S$로 되돌려 실효 업데이트는 같게 유지한다.

```text
loss' = loss * S
backward(loss')
grads ≈ S * true_grads
unscale: grads /= S
optimizer.step()
```

**동적 loss scaling**: overflow(Inf/NaN)가 보이면 $S$를 줄이고, 안정이면 점진적으로 키우는 식의 적응 전략이 흔하다.

```python
scaler = amp.GradScaler()  # 주로 FP16 경로에서 사용

optimizer.zero_grad(set_to_none=True)
with amp.autocast(device_type="cuda", dtype=torch.float16):
    loss = compute_loss(model, batch)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # clip 전에 unscale하는 패턴이 흔함
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

### 5.3 BF16과 loss scaling

설명: BF16은 범위가 상대적으로 넓어 **loss scaler 없이** autocast만 쓰는 레시피가 많다. 그러나 “BF16이면 절대 스케일러 불필요”를 만유인력처럼 단정할 수는 없다. 구현·라이브러리 기본값을 확인한다.

사실/설명 구분 요약:

| 진술 | 종류 |
|---|---|
| FP16은 FP32보다 비트 수가 적다 | 사실 |
| FP16 혼합정밀도에서 loss scaling이 자주 쓰인다 | 관행에 가까운 사실·설명 |
| BF16이 항상 FP16보다 빠르다 | **단정 금지** (하드웨어 의존) |
| AMP를 쓰면 항상 N% 빠르다 | **단정 금지** |

## 6. Gradient Accumulation이란

**Gradient Accumulation**은 micro-batch를 여러 번 forward/backward하여 `.grad`를 쌓은 뒤, **한 번만** `optimizer.step()` 하는 기법이다.

목적: GPU에 한 번에 올릴 수 있는 배치($B_{\text{micro}}$)는 작아도, 실효 배치

$$

B_{\text{eff}} = B_{\text{micro}} \times N_{\text{accum}}

$$

(분산 학습이면 여기에 GPU 수 등이 더 곱해진다)를 키운다.

### 6.1 왜 Loss를 나누는가

PyTorch의 `backward`는 기울기를 **더한다**. 평균 CE loss를 micro-batch마다 그대로 backward하면, $N$번 누적 시 기울기 스케일이 약 $N$배가 된다. 실효적으로 “합”이 되므로, 평균을 유지하려면 보통:

```python
loss = loss / N_accum
loss.backward()
```

을 micro-step마다 수행한다.

### 6.2 루프 골격

```python
accum = 4  # 예시 값일 뿐
optimizer.zero_grad(set_to_none=True)

for micro_idx, batch in enumerate(micro_batches):
    loss = compute_loss(model, batch) / accum
    loss.backward()  # grads 누적

    if (micro_idx + 1) % accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

AMP와 결합:

```python
accum = 4
optimizer.zero_grad(set_to_none=True)
scaler = amp.GradScaler()

for micro_idx, batch in enumerate(micro_batches):
    with amp.autocast(device_type="cuda", dtype=torch.float16):
        loss = compute_loss(model, batch) / accum
    scaler.scale(loss).backward()

    if (micro_idx + 1) % accum == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

## 7. 실효 배치·토큰·LR의 연결

실효 배치가 바뀌면 **노이즈 있는 기울기의 분산**이 바뀐다. 그래서 대형 레시피에서는 배치와 lr을 함께 스케일하는 논의가 있다.

이 강의에서 고정 법칙(예: “배치 2배면 lr 2배”)을 선포하지 않는다. 다만 실험 시:

- $B_{\text{eff}}$ 정의를 로그에 명시할 것
- accumulation만 바꾸고 lr·스케줄을 잊지 말 것
- token/sec와 step 정의( micro vs optimizer step )를 혼동하지 말 것

제62강의 `global_step`은 보통 **optimizer.step 횟수**다. micro-batch 횟수와 이름을 분리하라.

## 8. 메모리 트레이드오프

| 선택 | 대체로 줄이는 것 | 대체로 대가 |
|---|---|---|
| micro-batch ↓ | activation 메모리 | 같은 $B_{\text{eff}}$를 위해 accum ↑, 동기·오버헤드 |
| accum ↑ | ピーク activation(배치 축) | step당 벽시계 시간, 구현 복잡도 |
| FP16/BF16 가중·활성 | 텐서 바이트 | 수치 이슈, (FP16) scaler 관리 |
| AdamW 상태 | (줄이기 어려움; 별도 기법 필요) | 저정밀 상태 등 고급 주제는 범위 밖 |

설명: Mixed Precision이 줄이는 쪽과 Accumulation이 줄이는 쪽은 완전히 같지 않다. 전자는 dtype, 후자는 **동시 상주하는 배치 크기**에 가깝다. 둘을 묶어 “메모리 마법”으로 부르지 말 것.

Activation checkpointing(재계산) 등은 제52강 복잡도·메모리 맥락의 연장이나, 이번 강의 범위를 넘어가므로 이름만 언급한다.

## 9. 수치 안정성 체크리스트

학습 중 다음을 보면 AMP/accum 쪽을 의심한다.

1. Loss가 갑자기 NaN/Inf
2. `grad_norm`이 폭주하거나 전부 0
3. scaler scale이 비정상적으로 내려가 step이 스킵됨(FP16)
4. BF16/FP16 전환만으로 수렴이 무너짐 → lr·warmup·데이터 버그와 분리 실험

디버깅 순서 제안:

```text
1) FP32에서 같은 배치가 학습되는지
2) autocast만 (BF16) 추가
3) FP16 + scaler
4) accumulation 배수 추가
```

한 번에 모두 켜면 원인 분리가 안 된다.

## 10. 제62강 루프에 통합한 예시

```python
from torch import amp

def train_optimizer_step(
    model,
    micro_batches,
    optimizer,
    scheduler,
    scaler,          # FP16이면 GradScaler, BF16/FP32면 None 가능
    use_amp=True,
    amp_dtype=torch.bfloat16,
    max_norm=1.0,
):
    """micro_batches 길이가 N_accum. 한 번의 optimizer step을 수행."""
    optimizer.zero_grad(set_to_none=True)
    n = len(micro_batches)
    last_loss = None

    for batch in micro_batches:
        if use_amp:
            with amp.autocast(device_type="cuda", dtype=amp_dtype):
                loss = compute_loss(model, batch) / n
        else:
            loss = compute_loss(model, batch) / n

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        last_loss = loss.detach() * n  # 로그용으로 스케일 복원

    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    scheduler.step()
    return float(last_loss.item())
```

주의: `compute_loss`가 이미 평균인지 합인지에 따라 `/ n` 위치가 달라질 수 있다. **평균 loss를 N번 더해 평균을 유지**하려는 의도인지 팀 규약을 고정하라.

## 11. 메모리 그림으로 보기

한 번의 optimizer step에 대략 동시에 존재하는 것:

```text
┌─────────────────────────────────────────┐
│ Parameters (종종 FP32 마스터)            │
│ Gradients                               │
│ Adam m, v  (파라미터당 추가 상태)         │
│ Activations (micro-batch, 정밀도 의존)   │
│ Autocast workspace / 임시 버퍼          │
└─────────────────────────────────────────┘
```

Accumulation은 “Activations를 한 micro만큼만 올려두고, Gradients만 디스크가 아니라 **파라미터 `.grad`에** 쌓는다”에 가깝다.  
Mixed Precision은 Activations·일부 연산의 **바이트/폭**을 줄이려 한다.

둘을 켜도 Adam 상태가 사라지지는 않는다. “AMP만 켜면 Optimizer 메모리도 반으로”는 사실이 아니다.

## 12. FP32 기준선에서 올리는 순서 (복습 강화)

권장 실험 사다리:

```text
A. FP32, accum=1
B. BF16 autocast, accum=1          (지원 환경)
C. FP16 autocast + GradScaler
D. 선택 dtype + accum=N
```

각 단계에서 고정할 것: 데이터 샘플, seed, lr, step 수(짧게).  
바꾸는 축이 하나여야 “AMP 때문에 깨졌는지” 알 수 있다.

## 13. Dynamic Loss Scale의 동작 스케치

```text
초기 scale S가 큼
  → overflow 관측 시: step 스킵(또는 무효), S ← S / 2
  → 안정이 계속되면: 가끔 S ← 2S (상한까지)
```

로그에 `scaler.get_scale()`을 남기면, “학습이 안 되는 줄” vs “스케일이 무너져 step이 스킵되는 중”을 분리하기 쉽다.

사실: GradScaler API·기본값은 PyTorch 버전에 따라 다를 수 있다. 문서의 현재 시그니처를 따른다.  
설명: scale이 계속 내려가기만 하면 원인(폭발 기울기·버그)이 따로 있을 수 있다. scale만 탓하지 말 것.

## 14. Accumulation과 Dataloader

micro-batch를 어떻게 공급할까?

패턴 A — 같은 loader에서 N번 next:

```python
micros = [next(data_iter) for _ in range(accum)]
loss = train_optimizer_step(model, micros, ...)
```

패턴 B — 한 루프에서 카운터:

```python
for batch in loader:
    micro_idx += 1
    (loss/accum).backward()
    if micro_idx % accum == 0:
        optimizer.step(); scheduler.step(); zero_grad()
```

패턴 B에서 loader 길이가 `accum`의 배수가 아니면 **마지막 잔여 micro**를 어떻게 처리할지(버림/패딩 step) 정책을 정해야 한다. 잔여를 무시하면 실효 토큰이 약간 줄고, 잔여만으로 step하면 `/accum` 스케일이 어긋날 수 있다.

## 15. 처리량 관점 — 측정은 하되 숫자를 지어내지 말 것

기록할 메트릭:

- optimizer steps / sec
- tokens / sec (PAD 제외 여부 명시)
- peak allocated memory

비교 시 조건(모델, $T$, $B_{\text{micro}}$, dtype, accum)을 표로 고정한다.  
이 책 본문에 “AMP = +X%” 같은 **가상 벤치마크를 넣지 않는다**.

## 16. 흔한 버그

1. **accum 시 `zero_grad`를 micro마다 호출** — 누적이 리셋되어 실효 배치가 작아짐.
2. **`loss / n` 누락** — 실효 lr이 의도보다 커짐.
3. **clip을 scale된 grad에 적용(FP16)** — unscale 전에 clip하면 임계값 의미가 왜곡.
4. **scaler.step 실패(overflow) 후에도 scheduler만 진행** — step 스킵과 스케줄 불일치 가능. 성공 시에만 scheduler를 진각할지 정책을 정한다.
5. **`global_step`을 micro-step으로 증가** — 로그·코사인·ckpt 간격이 전부 어긋남.
6. **BF16 미지원 장치에서 묵시 실패** — 환경 확인은 사실 문제다. 추측으로 덮지 말 것.
7. **로그 loss에 `/n`된 값을 그대로 기록** — train curve가 가짜로 낮아 보임. 로그 시 스케일 복원.

## 17. 핵심 정리

- Mixed Precision은 연산·텐서 dtype을 섞어 메모리·효율을 노린다. FP16은 범위 문제로 loss scaling이 자주 동반된다.
- BF16은 형식상 지수 범위가 넓어 스케일러 없이 쓰는 경우가 많지만, 만능 선언은 피한다.
- Gradient Accumulation은 micro-batch 기울기를 쌓아 실효 배치를 키운다.
- `/ N_accum`, `zero_grad` 위치, optimizer step vs micro step 정의가 구현의 핵심이다.
- 성능 %를 지어내지 말고, 트레이드오프와 디버깅 순서를 손에 익힌다.

## 18. 핵심 용어

| 용어 | 의미 |
|---|---|
| FP32 / FP16 / BF16 | 부동소수점 저장·연산 형식 |
| Mixed Precision / AMP | 혼합 정밀도 학습 |
| Autocast | 영역 내 연산 dtype 자동 선택 |
| Loss scaling | FP16 등에서 기울기 underflow 완화 |
| GradScaler | 동적 loss scale 유틸 |
| Micro-batch | 한 번에 forward하는 작은 배치 |
| Gradient accumulation | 여러 backward 후 1 step |
| Effective batch size | micro × accum (× 분산 크기) |
| Master weights | 갱신에 쓰는 FP32 파라미터 카피(개념) |
| Overflow | Inf/NaN으로 숫자가 범위를 벗어남 |

## 19. 연습 문제
### 문제 1 (구분)

다음 문장이 사실(형식/정의)에 가까운지, 설명·관행에 가까운지 분류하시오.  
(a) FP16은 16비트다.  
(b) AMP를 켜면 학습이 30% 빨라진다.  
(c) FP16 혼합정밀도에서 loss scaling이 흔히 쓰인다.

### 문제 2 (개념)

Loss scaling이 실효 업데이트 크기를 바꾸지 않으려면 backward 이후 무엇을 해야 하는가?

### 문제 3 (계산)

$B_{\text{micro}}=2$, $N_{\text{accum}}=8$, GPU 1장일 때 $B_{\text{eff}}$는?

### 문제 4 (코드)

Accumulation 루프에서 micro마다 `optimizer.zero_grad()`를 호출하면 실효적으로 어떤 설정이 되는가?

### 문제 5 (연결)

제63강 cosine scheduler의 `step`을 micro-batch마다 호출하면 어떤 문제가 생기는가?

### 문제 6 (운영)

FP16 학습 로그에서 `scale`이 계속 반만 되고 `step`이 자주 스킵된다. 점검 순서 세 가지를 쓰시오.

---

## 정답 및 해설

### 문제 1

(a) 사실(형식 정의).  
(b) 근거 없는 성능 단정 — 이 책에서는 채택하지 않음(관측 없이는 설명이 아니라 추측).  
(c) 널리 보이는 관행/설명(항상법칙은 아님).

### 문제 2

스케일된 기울기를 $1/S$로 되돌리거나(unscale), Optimizer가 그 스케일을 보정한 뒤 갱신하도록 해야 한다. 핵심은 “키웠으면 업데이트 전에 보정”.

### 문제 3

$2 \times 8 = 16$.

### 문제 4

누적이 리셋되어 $N_{\text{accum}}=1$과 같아진다(각 micro가 독립 step에 가까운 형태). 단 `step` 호출 빈도에 따라 세부는 달라지나, 의도한 실효 배치는 달성되지 않는다.

### 문제 5

스케줄이 설계보다 $N_{\text{accum}}$배 빨리 진행되어 warmup/cosine이 조기에 끝난다. Scheduler는 optimizer.step 단위에 맞춘다.

### 문제 6

예: (1) FP32로 동일 배치 재현 (2) lr·clip·데이터 NaN 확인 (3) autocast 범위/모델 수치 버그 확인. scale만 올리며 강행하지 말 것.

## 20. 다음 강의와 연결

긴 학습은 중간에 끊긴다. 지금 만든 AMP·accum 루프의 **가중치·Optimizer·Scaler·step**을 디스크에 남겨야 한다.

다음 **제65강. Checkpoint 관리**에서는 무엇을 저장하고 어떻게 재개하는지, best vs last, safetensors를 형식 옵션으로 언급한다.

이전 강의: **제63강. Optimizer, Learning Rate, Scheduler**  
다음 강의: **제65강. Checkpoint 관리**

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [63강. Optimizer, Learning Rate, Scheduler](63강_Optimizer_Learning_Rate_Scheduler.md)
- **다음 강:** [65강. Checkpoint 관리](65강_Checkpoint_관리.md)

<!-- /LECTURE_NAV -->
