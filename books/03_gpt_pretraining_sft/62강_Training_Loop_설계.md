# 3권. GPT Pretraining과 SFT

## 제62강. Training Loop 설계

### 1. 이번 강의에서 배울 것

제61강에서 packing된 `input_ids` / `labels` 배치를 만들었다. 이번 강의는 그 배치를 모델에 넣어 파라미터를 갱신하는 **Training Loop(학습 루프)** 의 골격을 설계한다.

이 강의를 마치면 다음을 할 수 있어야 한다.

- 한 train step의 순서를 `batch → forward → loss → backward → clip → step`으로 설명한다.
- Epoch 기반 종료와 Token budget 기반 종료를 구분한다.
- Gradient clipping이 왜 들어가는지 직관적으로 말한다.
- 무엇을 로그할지(최소 세트) 설계한다.
- 제23강의 작은 루프가 LLM Pretraining에서 어떻게 확장되는지 연결한다.

### 2. 왜 이것을 배우는가

모델 코드가 있어도 루프가 흐리면 실험이 재현되지 않는다. Pretraining은 수천만~수조 토큰을 도는 긴 작업이다. 루프의 “한 바퀴”가 무엇을 의미하는지 합의해야 Optimizer(제63강), AMP(제64강), Checkpoint(제65강), Validation(제66강)을 꽂을 구멍이 생긴다.

```text
DataLoader 배치
    → model.train()
    → forward
    → loss
    → backward
    → (grad clip)
    → optimizer.step()
    → scheduler.step()   # 위치는 설계에 따라
    → logging / checkpoint / eval 훅
```

이번 강의는 **훅이 꽂히기 전 뼈대**에 집중한다.

### 3. 먼저 알아야 할 개념

1. **학습 루프·Optimizer 기초** — 제23강
2. **Autograd / `zero_grad`** — 제20강
3. **Causal LM Loss** — 제34, 57강
4. **Packed 배치** — 제61강
5. **GPT forward 골격** — 제48, 56강

AdamW·cosine·AMP 세부 수식은 다음 강의들이다. 여기서는 루프에 **자리가 있다**는 점만 확보한다.

### 4. 핵심 개념 — Train Step

**Train step(학습 스텝)**은 “파라미터를 한 번 갱신하기까지”의 최소 단위다.  
(Gradient Accumulation을 쓰면 micro-batch 여러 번 뒤에야 한 step이 된다. 제64강.)

표준 순서:

```text
1. 배치를 장치로 이동
2. optimizer.zero_grad()
3. logits = model(input_ids)          # forward
4. loss = loss_fn(logits, labels)     # 보통 token-level CE
5. loss.backward()                    # 각 파라미터 .grad 채움
6. clip_grad_norm_(...)               # 선택이나 LLM에서 흔함
7. optimizer.step()                   # 파라미터 갱신
8. lr_scheduler.step()                # 스케줄러를 step 단위로 둘 때
9. 로그 기록 (loss, lr, tokens, ...)
```

제23강과의 차이:

| 항목 | 소규모 예제(제23강) | LLM Pretraining 루프 |
|---|---|---|
| 배치 내용 | 표/이미지 등 | 토큰 윈도우 `[B, T]` |
| Loss | MSE/CE | next-token CE (ignore PAD) |
| 종료 조건 | epoch 수 | 종종 **token budget** |
| 부가 장치 | 거의 없음 | clip, AMP, accum, ckpt, eval |
| 로깅 | print | step/토큰/시간 정규화 지표 |

### 5. Forward와 Loss를 루프에 끼우기

#### 5.1 모델이 로짓만 반환하는 경우

```python
logits = model(input_ids)          # [B, T, V]
# 다음 토큰 예측: 위치 t → 타깃 t+1
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()
loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
```

#### 5.2 모델이 loss까지 계산하는 경우

많은 Causal LM 구현은 `labels`를 받아 내부에서 shift+CE를 한다.

```python
out = model(input_ids=input_ids, labels=labels)
loss = out.loss  # 스칼라
```

루프 설계 관점에서는 **스칼라 loss 하나**만 있으면 backward가 가능하다. 내부 구현은 제57강 목표와 일치해야 한다.

#### 5.3 `model.train()`의 의미

`train()` 모드는 Dropout·일부 Norm 동작 등을 **학습용**으로 켠다. Validation(제66강)에서는 `eval()` + `torch.no_grad()`로 바꾼다. 루프 시작 시 한 번, eval 후 복귀 시 다시 `train()`을 호출하는 패턴이 안전하다.

### 6. Backward, Clip, Step

#### 6.1 `zero_grad` 위치

PyTorch는 기본적으로 gradient를 **누적**한다. 매 step 시작(또는 accumulation 사이클 시작)에 비운다.

```python
optimizer.zero_grad(set_to_none=True)  # True면 .grad=None 할당(메모리 이점 설명)
```

#### 6.2 Gradient Clipping

**Gradient clipping**은 기울기 벡터의 노름이 임계값을 넘으면 스케일을 줄여, 한 스텝의 업데이트 폭주를 완화하는 기법이다.

자주 쓰는 형태 — **global norm clip**:

$$
g \leftarrow g \cdot \min\left(1,\ \frac{\texttt{max\_norm}}{\|g\|_2}\right)
$$

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

설명(직관): Loss 곡면이 가파른 구간이나 불안정한 배치에서 $\nabla$가 커지면 파라미터가 멀리 뛴다. clip은 “방향은 유지하고 길이만 자른다”.

사실: clip 임계값·사용 여부는 레시피 하이퍼파라미터다. “1.0이 항상 최적”은 사실이 아니다.

clip은 **`backward` 이후, `optimizer.step` 이전**에 둔다.

#### 6.3 `optimizer.step()`

Optimizer가 `.grad`를 읽어 파라미터를 갱신한다. AdamW 세부·weight decay는 제63강.

### 7. Epoch vs Token Budget

#### 7.1 Epoch이란

**Epoch**은 학습 데이터 전체를 한 번 순회한 단위다. 작은 표 데이터에서는 “10 epoch”이 자연스럽다.

LLM Pretraining에서는:

- 데이터가 너무 커 **진짜 1 epoch를 안 도는** 설정도 있다.
- 반대로 작은 말뭉치를 **여러 epoch** 반복하는 설정도 있다.
- packing·셔플·샤드 때문에 “전체 1회”의 정의가 모호해질 수 있다.

#### 7.2 Token Budget이란

**Token budget(토큰 예산)**은 “학습 중 모델이 본(또는 갱신에 사용된) 토큰 수”를 종료·스케줄의 축으로 쓰는 방식이다.

```text
예: 100B tokens 학습
    = micro-batch 토큰 수 × step 수 × (accumulation 등)
```

집계 시 주의:

- **PAD 제외**할지, raw `B*T`로 할지 정책을 정한다.
- Gradient Accumulation·분산 학습에서는 “글로벌 토큰” 정의를 문서화한다.
- labels ignore 위치는 “예측에 쓰인 토큰”과 “입력 토큰”을 가른다.

#### 7.3 루프 골격 비교

Epoch 스타일:

```python
global_step = 0
for epoch in range(num_epochs):
    for batch in train_loader:
        train_one_step(batch)
        global_step += 1
```

Token budget 스타일:

```python
tokens_seen = 0
global_step = 0
while tokens_seen < token_budget:
    batch = next(train_iter)  # 무한 셔플 이터레이터일 수 있음
    n = count_tokens(batch)   # 정책에 따라
    train_one_step(batch)
    tokens_seen += n
    global_step += 1
```

Learning rate scheduler(제63강)도 epoch 단위보다 **step 또는 토큰 단위**로 거는 경우가 많다.

### 8. 최소 Training Loop 코드

교육용 단일 GPU 루프(AMP·accum 없음):

```python
import time
import torch
import torch.nn.functional as F


def count_real_tokens(attention_mask: torch.Tensor) -> int:
    """PAD를 제외한 토큰 수. attention_mask: [B, T]"""
    return int(attention_mask.sum().item())


def train_one_step(model, batch, optimizer, max_norm=1.0):
    """한 번의 파라미터 갱신. loss 스칼라(float)를 반환."""
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)  # [B, T, V]
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    return float(loss.detach().item())


def training_loop(
    model,
    loader,
    optimizer,
    device,
    token_budget: int,
    log_every: int = 10,
    max_norm: float = 1.0,
):
    model.train()
    model.to(device)

    tokens_seen = 0
    global_step = 0
    running_loss = 0.0
    t0 = time.time()

    data_iter = iter(loader)
    while tokens_seen < token_budget:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        loss = train_one_step(model, batch, optimizer, max_norm=max_norm)

        ntok = count_real_tokens(batch["attention_mask"])
        tokens_seen += ntok
        global_step += 1
        running_loss += loss

        if global_step % log_every == 0:
            elapsed = time.time() - t0
            avg = running_loss / log_every
            tps = tokens_seen / max(elapsed, 1e-6)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={global_step} loss={avg:.4f} lr={lr:.2e} "
                f"tokens={tokens_seen} tok/s={tps:.1f}"
            )
            running_loss = 0.0

    return {"steps": global_step, "tokens_seen": tokens_seen}
```

이 코드는 **의도적으로** scheduler·AMP·eval·ckpt를 빼 두었다. 자리가 보일 것이다.

```text
train_one_step 안에 들어갈 확장:
  - scaler / autocast          → 제64강
  - loss = loss / accum_steps  → 제64강
  - scheduler.step()           → 제63강

training_loop 안에 들어갈 확장:
  - if step % eval_every: validate()     → 제66강
  - if step % ckpt_every: save_ckpt()    → 제65강
```

### 9. 로깅 설계

로깅은 “예쁜 대시보드”가 아니라 **재현·디버깅 계약**이다.

#### 9.1 최소 세트

| 항목 | 이유 |
|---|---|
| `global_step` | 재개·비교의 기본 축 |
| `loss` (window 평균) | 학습 신호. 단일 배치는 시끄러움 |
| `lr` | scheduler·warmup 오작동 탐지 |
| `tokens_seen` | budget·비교 실험의 공통 축 |
| `tok/s` 또는 `time/step` | 성능 회귀 탐지 |
| `grad_norm` (가능하면) | 폭주·clip 직전 상태 |

#### 9.2 있으면 좋은 항목

- PAD 비율 / ignore label 비율 (데이터 파이프라인 회귀)
- 최대 logit / NaN·Inf 플래그
- GPU 메모리 사용량(대략)
- 호스트 시간 vs 디바이스 시간(간단 프로파일)

#### 9.3 로깅 함정

1. **매 스텝 raw loss만** 보면 분산이 커서 추세가 안 보인다. 이동 평균을 둔다.
2. **토큰 수 정의를 바꿔** 이전 run과 비교하면 안 된다.
3. Loss만 보고 lr이 0에 붙어 있는 줄을 놓친다.
4. 로그 간격이 너무 희귀하면 발산을 늦게 발견한다.

설명: 로그는 학습을 빠르게 만들지 않는다. 그러나 실패한 실험을 **실패한 이유와 함께** 남긴다.

### 10. 배치·디바이스·시드

#### 10.1 장치 이동

```python
batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
```

`non_blocking`은 pinned memory + async copy 환경에서 의미가 있다. 없어도 동작한다.

#### 10.2 시드

재현을 위해:

- Python `random`, `numpy`, `torch` 시드
- DataLoader `generator` / worker 시드
- CUDA deterministic 옵션(완전 재현은 비용·제약이 큼)

사실: 동일 시드여도 비결정적 커널·분산 통신 때문에 비트 단위 일치를 보장하지 못하는 경우가 있다. “재현”의 목표 수준을 팀에서 정의하라.

#### 10.3 `global_step`의 권위

Checkpoint 재개(제65강)의 기준은 보통 `global_step`(그리고 토큰 수)이다. epoch 카운터만으로 재개하면 packing 이터레이터 위치와 어긋나기 쉽다.

### 11. 루프에 훅 지점 배치하기

권장 순서(단일 프로세스 기준):

```text
for each step:
  train_one_step(...)
  scheduler.step()          # step 기반일 때
  log if due
  eval if due               # 제66강: no_grad, model.eval
  checkpoint if due         # 제65강: model+optim+sched+step
```

Eval/Checkpoint를 **너무 자주** 하면 처리량(tok/s)이 떨어진다. 너무 드물면 발산·최고 성능 지점을 놓친다. 간격은 예산·안정성에 맞게 고른다. 구체 숫자는 레시피마다 다르므로 여기서 고정값처럼 제시하지 않는다.

### 12. 디버깅용 “한 배치 과적합” 테스트

본학습 전에 루프가 맞는지 확인하는 고전적 방법:

1. 배치 1~몇 개만 골라 반복 학습한다.
2. Loss가 거의 0 근처로 내려가는지 본다.
3. 내려가지 않으면 모델·label shift·ignore_index·lr·train 모드를 의심한다.

이 테스트는 **일반화**를 보장하지 않는다. “파이프와 그래프가 연결되었는가”만 본다. 과적합·검증은 제24강·제66강 맥락이다.

### 13. 분산 학습을 아직 안 다루더라도

이 책의 루프는 단일 장치 중심이다. 그래도 이름을 미리 분리해 두면 이후가 쉽다.

- `micro_batch_size`: 이 GPU가 한 forward에 넣는 배치
- `global_batch_size`: 모든 장치에 걸친 실효 배치
- `global_step`: **동기화된 optimizer step**

제64강의 accumulation은 “장치 하나에서 global에 가까워지는 법”이다. 로깅 키를 지금부터 통일하라.

### 14. 흔한 버그

1. **`zero_grad` 생략** — 기울기 누적으로 step이 폭주.
2. **`step`과 `backward` 순서 반대** — 제23강 퀴즈와 동일.
3. **eval 후 `train()` 복귀 누락** — Dropout 꺼진 채 학습.
4. **label shift 이중 적용** — 모델 내부와 루프에서 둘 다 shift.
5. **token budget에 PAD 포함/제외 혼선** — run 간 비교 불가.
6. **clip을 `step` 뒤에 적용** — 이미 갱신된 뒤라 의미 없음.
7. **손실을 Python float로 너무 일찍 변환** — 그래프 끊김은 `backward` 전 `loss`에 하면 안 됨. 로그용 `detach`는 가능.

### 15. 핵심 정리

- Train step은 배치 → forward → loss → backward → (clip) → optimizer step이다.
- LLM Pretraining은 epoch보다 **token budget / global_step** 축이 자주 쓰인다.
- Gradient clipping은 step 직전에서 업데이트 폭주를 완화한다.
- 로깅 최소 세트는 step, loss, lr, tokens, throughput이다.
- Eval·Checkpoint·AMP·Scheduler는 이 루프의 훅으로 연결된다.

### 16. 핵심 용어

| 용어 | 의미 |
|---|---|
| Train step | 파라미터 1회 갱신 단위 |
| Global step | 누적 갱신 횟수 |
| Epoch | 데이터 전체 1회 순회 |
| Token budget | 학습에 사용할 토큰 수 목표 |
| Gradient clipping | 기울기 노름 상한으로 스케일 조정 |
| `model.train()` | 학습 모드 |
| Throughput (`tok/s`) | 초당 처리 토큰 수 |
| Hook | eval/ckpt/log가 삽입되는 지점 |

### 17. 복습 문제

#### 문제 1 (순서)

다음 단계를 올바른 순서로 배열하시오: `optimizer.step`, `loss.backward`, `zero_grad`, `forward`, `clip_grad_norm_`, `loss 계산`.

#### 문제 2 (개념)

Epoch 종료와 Token budget 종료의 차이를 Pretraining 맥락에서 설명하시오.

#### 문제 3 (디버깅)

Loss는 나오는데 파라미터가 변하지 않는다. 루프에서 의심할 항목 세 가지를 쓰시오.

#### 문제 4 (로깅)

`loss`만 로그하고 `lr`을 안 남기면 놓치기 쉬운 실패 모드를 하나 쓰시오.

#### 문제 5 (연결)

제61강 packing 배치의 `ignore_index`가 이번 루프의 Loss 계산에 어떻게 반영되어야 하는지 쓰시오.

---

### 정답 및 해설

#### 문제 1

`zero_grad` → `forward` → `loss 계산` → `loss.backward` → `clip_grad_norm_` → `optimizer.step`.

#### 문제 2

Epoch는 데이터 순회 횟수 기준이고, Token budget은 실제로 본(또는 학습에 사용한) 토큰 수 기준이다. 말뭉치가 크거나 스트리밍·다중 에폭이 섞이면 token/step 축이 비교와 스케줄에 더 직접적이다.

#### 문제 3

예: `optimizer.step()` 호출 누락, `lr=0`, `requires_grad=False`/동결, loss가 detach된 텐서, 잘못된 파라미터 그룹.

#### 문제 4

Scheduler 버그로 lr이 0 또는 극소에 붙어 Loss가 정체되는 상황을, “모델이 안 배우는 중”으로 오인하기 쉽다.

#### 문제 5

PAD·무시 위치 label을 `-100` 등으로 두었다면 `cross_entropy(..., ignore_index=-100)`로 그 위치가 Loss/기울기에 기여하지 않게 해야 한다. shift를 쓸 때도 무시 위치가 어긋나지 않게 유지한다.

### 18. 다음 강의와 연결

루프의 `optimizer.step()` 자리가 비어 있다.

다음 **제63강. Optimizer, Learning Rate, Scheduler**에서는 AdamW의 직관, weight decay, warmup+cosine schedule을 제1권 Optimizer 지식 위에 쌓는다. 그 다음 제64강에서 AMP와 Gradient Accumulation으로 메모리·실효 배치를 다룬다.

이전 강의: **제61강. Tokenization Pipeline과 Dataset Packing**  
다음 강의: **제63강. Optimizer, Learning Rate, Scheduler**
