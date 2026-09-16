# 3권. GPT Pretraining과 SFT

## 제66강. Validation과 Evaluation

### 1. 이번 강의에서 배울 것

제65강에서 `best.pt`를 저장하는 훅을 만들었다. 이번 강의는 그 훅에 넣을 **신호** — held-out validation loss와 평가 절차 — 를 다룬다.

이 강의를 마치면 다음을 할 수 있어야 한다.

- Train loss와 Validation loss의 역할을 구분한다.
- 언제 eval을 돌릴지(주기·비용) 설계한다.
- `model.eval()` / `torch.no_grad()`가 하는 일을 설명한다.
- 발산·과적합·데이터 버그의 조기 신호를 로그에서 읽는다.
- 제67강 Perplexity로 넘어가기 전, “loss 평가”의 한계를 말한다.

### 2. 왜 이것을 배우는가

Train loss만 보면 두 가지를 놓친다.

1. **일반화**: 모델이 학습 토큰에만 맞춰지고 있지는 않은가.
2. **운영 사고**: 학습 루프는 도는데 평가 분포에서는 이미 무너지지 않았는가.

```text
Train stream ──► 파라미터 갱신 (제62~64강)
Held-out val ──► 갱신 없이 Loss 측정 (이번 강)
                 └► best ckpt / 조기 경보 (제65강)
```

Pretraining의 val은 분류 정확도처럼 “틀/맞”이 한 방에 보이지 않는다. 그래도 **동일 프로토콜의 held-out CE loss**는 가장 값싼 나침반이다.

### 3. 먼저 알아야 할 개념

1. **Train/Val 분리, 과적합** — 제24강
2. **Cross Entropy / next-token loss** — 제34, 57강
3. **Training loop 훅** — 제62강
4. **Checkpoint best/last** — 제65강
5. **데이터 누수·샤드** — 제60강

생성 품질·사람 평가·Perplexity 해석의 깊게는 제67강이다. 여기서는 **loss 기반 평가 운영**이 중심이다.

### 4. 핵심 개념 — Validation이란

**Validation(검증)**은 학습 파라미터 갱신에 쓰지 않는 held-out 데이터로 모델 상태를 측정하는 과정이다.

용어 정리:

| 용어 | 의미 |
|---|---|
| **Train set** | 기울기를 만드는 데이터 |
| **Validation set** | 하이퍼파라미터·조기 선택·모니터링용 held-out |
| **Test set** | (가능하면) 최종 보고용으로 더 잠가 둔 held-out |
| **Evaluation** | 넓게는 모든 측정. 이 강의에서는 val 루프 중심 |

설명: 실무에서 val/test 이름이 섞여 쓰이기도 한다. 중요한 것은 **갱신에 쓰지 않는 데이터인가**와 **반복 열람으로 선택 편향이 생기지는 않는가**다.

### 5. Held-out Validation Loss

Causal LM에서 기본 지표는 학습과 같은 **토큰 평균 Cross Entropy**다.

$$
L_{\text{val}} = -\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(x_i\mid x_{<i})
$$

$N$은 ignore되지 않은 타깃 토큰 수다. PAD·특수 무시 위치는 제61강 파이프라인과 동일한 `ignore_index` 규칙을 써야 한다.

#### 5.1 Train loss와의 차이

| | Train loss | Val loss |
|---|---|---|
| 데이터 | 학습 스트림 | held-out |
| Dropout 등 | 켜짐(`train`) | 꺼짐(`eval`)이 일반적 |
| 목적 | 최적화 신호 | 모니터링·선택 |
| 노이즈 | 배치 단위로 큼 | 충분히 많은 토큰을 보면 상대적으로 안정 |

한 배치 train loss가 낮아져도 val이 오르면 **과적합·분포 불일치·버그** 후보가 된다.

#### 5.2 데이터 누수

Val 문서가 packing·필터 과정에서 train에 섞이면 val loss는 낙관적으로 왜곡된다. 제60강 샤드·해시 단위 분리 정책을 eval에도 그대로 적용한다.

### 6. 언제 Evaluation을 돌릴 것인가

Eval은 공짜가 아니다. forward를 여러 배치 돌리고, 심하면 생성을 한다.

설계 축:

1. **Step 간격** — 예: 매 $K$ optimizer step마다 (구체 $K$는 예산 의존)
2. **토큰 간격** — token budget 축과 맞출 때
3. **시간 간격** — 벽시계 기준 운영
4. **이벤트** — loss spike, scaler 이상, 수동 트리거

트레이드오프:

```text
자주 ──► 발산·최고점 포착 ↑, tok/s ↓, val 과다 열람 편향 위험 ↑
드물게 ──► 비용 ↓, 사고 발견 지연 ↑
```

설명: Pretraining 초반은 불안정하므로 상대적으로 자주 보고, 안정 구간에서는 간격을 늘리는 전략도 있다. 고정 공식은 없다.

### 7. Eval 루프 구현

```python
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=None):
    """
    held-out CE loss (ignore_index=-100).
    max_batches: 비용 제한용. None이면 로더 전체.
    반환: {"loss": float, "tokens": int, "batches": int}
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in val_loader:
        if max_batches is not None and n_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
        # token-level CE, sum으로 모아 토큰 수로 나눔
        loss_sum = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tok = (labels[:, 1:] != -100).sum().item()
        if n_tok == 0:
            continue

        total_loss += float(loss_sum.item())
        total_tokens += int(n_tok)
        n_batches += 1

    model.train()  # 호출측에서 되돌려도 됨. 안전하게 여기서 복귀.

    if total_tokens == 0:
        return {"loss": float("nan"), "tokens": 0, "batches": n_batches}

    return {
        "loss": total_loss / total_tokens,
        "tokens": total_tokens,
        "batches": n_batches,
    }
```

핵심 포인트:

- **`torch.no_grad()`**: 그래프 미구축 → 메모리·속도.
- **`model.eval()`**: Dropout 등을 평가 모드로.
- **토큰 가중 평균**: 배치 평균을 단순 산술평균하면 배치 길이·PAD 비율에 왜곡된다. `reduction="sum"` 후 토큰 수로 나누는 편이 공정하다.
- 평가 후 **`model.train()` 복귀**를 잊으면 이후 학습이 eval 모드로 진행된다.

### 8. Train 모드 vs Eval 모드

| | `model.train()` | `model.eval()` |
|---|---|---|
| Dropout | 활성(확률적) | 비활성 |
| 일부 Norm 통계 | 학습 중이라면 업데이트 | 보통 고정 사용 |
| 목적 | 학습 | 평가·추론 |

사실: `eval()`은 “기울기를 끈다”가 아니다. 기울기를 끄는 것은 `torch.no_grad()` / `inference_mode()`다. 둘을 함께 쓰는 것이 평가의 정석에 가깝다.

LLM GPT 스타일(LayerNorm+Dropout)에서는 Dropout on/off가 val loss에 영향을 준다. train 모드로 val을 재면 노이즈가 섞인다.

### 9. Training Loop에 훅 연결

```python
best_val = float("inf")

for global_step in range(start_step + 1, total_steps + 1):
    train_one_optimizer_step(...)  # 제62~64강
    log_train(...)

    if global_step % eval_every == 0:
        metrics = evaluate(model, val_loader, device, max_batches=val_max_batches)
        print(
            f"step={global_step} val_loss={metrics['loss']:.4f} "
            f"val_tokens={metrics['tokens']}"
        )

        # 제65강 best 저장
        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            save_checkpoint(..., meta={"best_val_loss": best_val, ...})

        model.train()
```

로그에는 train과 val을 **같은 step 축**에 남겨 곡선을 겹쳐 보게 한다.

### 10. 조기 신호 — 발산과 이상

숫자 벤치마크를 남발하지 말고, **패턴**을 본다.

#### 10.1 발산(divergence) 후보

- Train loss가 갑자기 NaN/Inf
- Train loss가 가파르게 폭주
- `grad_norm`이 clip 상한에 반복적으로 박힘 + loss 악화
- Val loss가 train과 함께 급등

의심 순서: 데이터 깨짐 → lr/스케줄 → AMP overflow → 버그(label shift) → 모델 초기화.

#### 10.2 과적합 후보

- Train loss ↓, Val loss ↑ (또는 개선 정체 후 악화)
- 작은 말뭉치 다에폭에서 특히 흔함

대응 방향(제24강 연결): 더 많은 데이터/정규화/조기 중단(best ckpt)·학습 길이 조정. “항상 Dropout을 키우면 해결”은 단정 금지.

#### 10.3 파이프라인 버그 후보

- Train·Val 모두 비정상적으로 낮음 → label 누수(미래 토큰), ignore 미적용
- Val만 이상 → 전처리 불일치(다른 토크나이저, 다른 EOS 정책)
- 토큰 수 0 배치 다수 → mask 버그

제40강 causal mask 버그는 “train loss만 좋게” 만들 수 있다. Val에서도 미래가 보이면 같이 좋아 보이므로, **생성 스모크 테스트**를 간헐적으로 넣는 것이 안전하다(제58~59, 67강).

### 11. Eval 프로토콜을 고정하라

비교 가능한 실험의 조건:

1. 같은 val 샤드·같은 토크나이저
2. 같은 `block_size` / packing 규칙 (또는 명시적 차이 기록)
3. 같은 `max_batches` 또는 전체 패스
4. 같은 dtype 경로(가능하면). AMP eval 여부를 기록
5. 시드가 개입하는 샘플링 eval은 별도 프로토콜

설명: `max_batches`로 자른 val loss는 **근사 모니터**다. 논문/최종 보고에는 더 긴 평가를 따로 두는 편이 좋다.

### 12. Loss 평가의 한계 (제67강 예고)

Validation CE loss가 내려가도:

- 답이 짧게 끊기거나
- 반복 루프에 빠지거나
- 지시 따르기(SFT 이후 목표)가 나아지지 않을 수 있다.

Loss는 **확률 모델로서의 압축/예측 품질**에 가깝다. 사람 기준 유용성과 1:1은 아니다. 제67강에서 Perplexity($e^{L}$)와 생성 품질 이야기로 확장한다.

그럼에도 Pretraining 중반의 자동 나침반으로는 held-out loss가 여전히 1순위다. 비용 대비 정보가 크다.

### 13. 작은 실습 — 곡선 읽기

다음 가상 로그를 해석해 보자. (숫자는 **예시일 뿐**, 실측 벤치마크가 아니다.)

```text
step 1000  train=3.50  val=3.55
step 2000  train=3.10  val=3.20
step 3000  train=2.80  val=3.40
step 4000  train=2.40  val=3.90
```

읽기:

- 초반 train/val 동반 하락 → 학습 정상 후보
- 이후 train ↓ val ↑ → 과적합 또는 val 분포 이슈 후보
- 운영: step 2000 근처 best를 남겨 둔 정책이 유리할 수 있음

또 다른 패턴:

```text
step 500  train=3.2  val=3.2
step 600  train=nan  val=nan
```

→ 즉시 중단, 직전 last/best로 롤백, AMP/lr/데이터 점검.

### 14. Running train loss와 Val를 같이 보기

Train은 매 step(또는 window 평균), Val은 희소 샘플이다. 비교할 때:

- 같은 **smoothing 창** 개념으로 train을 보고
- val 토큰 수가 충분한지 확인하고
- 축을 `global_step` 또는 `tokens_seen`으로 통일한다

```python
# 개념: 지수이동평균으로 train 노이즈 완화
ema = None
beta = 0.9
ema = loss if ema is None else (beta * ema + (1 - beta) * loss)
```

EMA train과 val을 같은 그래프에 올리면 “진짜 일반화 악화”와 “배치 노이즈”를 덜 혼동한다.

### 15. 온라인 Val 부분집합

전체 val이 비싸면:

1. **monitor set**: 매 `eval_every`마다 짧게
2. **full val**: 드물게 전체 패스
3. best 갱신은 정책을 명시 (monitor 기준인지 full 기준인지)

설명: monitor로 best를 갱신하면 노이즈 우승 확률이 올라간다. 타협안은 “monitor가 개선되면 full val을 한 번 더 돌려 확인 후 best 확정”이다.

### 16. 생성 스모크 테스트 (짧게)

Loss 평가와 별도로, 고정 프롬프트 몇 개에 대해 greedy/sampling 생성을 저장한다.

```python
@torch.no_grad()
def smoke_generate(model, tokenizer, prompts, max_new_tokens=40):
    model.eval()
    outs = []
    for p in prompts:
        # 제58~59강 생성 루틴 호출 (여기선 자리만)
        outs.append(generate_fn(model, tokenizer, p, max_new_tokens))
    model.train()
    return outs
```

주기적으로 텍스트를 읽어보면 causal mask 누수·반복·붕괴를 loss보다 일찍 발견하는 경우가 있다. 정량 점수는 제67강.

### 17. Early stopping 개념

**Early stopping**은 val 지표가 오래 개선되지 않으면 학습을 멈추거나 best로 되돌리는 전략이다.

Pretraining에서는:

- 예산이 토큰으로 고정되어 있으면 “중지”보다 **best 선택**이 더 흔하다.
- 작은 말뭉치·다에폭에서는 early stopping이 더 직접적이다.

사실처럼 “인내 스텝=N이 표준”이라고 쓰지 않는다. 인내 구간은 과제·노이즈에 의존한다.

### 18. 흔한 버그

1. **eval 후 `train()` 미복귀**
2. **val에서 `backward` 또는 optim step** — held-out 오염
3. **배치 평균 loss를 배치 수로만 평균** — 토큰 수 불균형
4. **train과 다른 tokenize/EOS 정책**
5. **너무 짧은 val로 best를 잦게 갱신** — 노이즈 우승
6. **test를 매일 봐서 사실상 val로 사용** — 최종 보고 신뢰 하락
7. **AMP train / FP32 val을 아무 기록 없이 비교** — 곡선 해석이 어긋남

### 19. 핵심 정리

- Held-out validation loss는 Pretraining의 기본 자동 평가 신호다.
- Eval은 `eval()` + `no_grad()` + 토큰 가중 CE로 돌리고, 학습 모드로 복귀한다.
- 주기·비용·노이즈 사이에서 간격을 설계한다.
- Train/Val 곡선 패턴으로 발산·과적합·파이프라인 버그를 조기에 읽는다.
- Loss만으로 생성 품질을 단정하지 말고, 제67강으로 평가 축을 넓힌다.

### 20. 핵심 용어

| 용어 | 의미 |
|---|---|
| Held-out | 학습 갱신에 쓰지 않는 데이터 |
| Validation loss | held-out CE 등 모니터링 지표 |
| `model.eval()` | 평가 모드 스위치 |
| `torch.no_grad()` | Autograd 비활성 |
| Early signal | 발산·과적합의 조기 로그 패턴 |
| Eval protocol | 비교 가능한 평가 절차 고정 |
| Best checkpoint | val 지표 기준 저장본 |
| Leakage | train/val 혼입 |
| Monitor set | 저비용 부분 val |
| Early stopping | 개선 정체 시 중단·되돌리기 |

### 21. 복습 문제

#### 문제 1 (개념)

Train loss가 계속 내려가는데 Val loss가 오른다. 가능한 해석 두 가지를 쓰시오.

#### 문제 2 (코드)

`evaluate` 함수에서 `model.eval()`만 하고 `no_grad`를 안 쓰면 어떤 비용이 남는가? 반대로 `no_grad`만 하고 `eval()`을 안 쓰면?

#### 문제 3 (설계)

Val loss를 배치 평균들의 단순 평균으로 집계할 때 생기는 왜곡을 설명하시오.

#### 문제 4 (운영)

Eval을 매 step 돌리면 생기는 문제 두 가지를 쓰시오.

#### 문제 5 (연결)

제65강 `best.pt` 갱신 조건을 이번 강의 용어로 한 줄로 쓰시오. 제67강으로 넘길 “남는 질문”도 한 줄 쓰시오.

#### 문제 6 (신호)

Train/Val이 동시에 NaN이 되었다. 첫으로 확인할 것 세 가지를 쓰시오.

---

### 정답 및 해설

#### 문제 1

(1) 과적합 — 학습 분포에만 맞춰짐. (2) train/val 전처리·분포 불일치, 또는 val 쪽 데이터/필터 문제. (버그·누수도 후보.)

#### 문제 2

`no_grad` 없으면 그래프 구축으로 메모리·시간이 낭비된다(갱신하지 않아도). `eval()` 없으면 Dropout 등이 켜진 채라 val 추정이 노이즈·편향될 수 있다.

#### 문제 3

배치마다 유효 토큰 수가 다르면, 작은 배치(또는 PAD 많은 배치)가 평균에 과도하게 영향을 준다. 토큰 수 가중(합/총토큰)이 더 공정하다.

#### 문제 4

(1) 학습 처리량 저하. (2) val을 너무 자주 봐 모델 선택 편향·과다 적합 위험. (운영 복잡도 증가도 가능.)

#### 문제 5

예: “held-out val loss가 최저일 때 best checkpoint를 저장한다.”  
남는 질문: “그 best가 생성 품질·Perplexity 해석에서도 좋은가?” → 제67강.

#### 문제 6

예: 직전 배치 데이터/라벨, lr·AMP scale, grad_norm/클립, 체크포인트 롤백 가능성.

### 22. 다음 강의와 연결

Validation loss라는 나침반을 달았다.

다음 **제67강. Perplexity와 생성 품질**에서는 $e^{L}$로서의 Perplexity, loss와 체감 품질의 관계, 생성 샘플을 보는 법을 다룬다. 제68강 Mini GPT Pretraining에서 제61~67강의 파이프라인을 한 프로젝트로 묶는다.

이전 강의: **제65강. Checkpoint 관리**  
다음 강의: **제67강. Perplexity와 생성 품질**
