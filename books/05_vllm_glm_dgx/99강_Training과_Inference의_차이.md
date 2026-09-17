# 제99강. Training과 Inference의 차이

> **학습 목표**
> - Training과 Inference의 목표·그래프·메모리·성공 지표 차이
> - 같은 체크포인트라도 training graph와 inference graph가 다른 이유
> - 왜 서빙 병목이 “loss를 더 줄이는 것”이 아니라 지연·처리량·메모리인지
> - Prefill / Decode / KV Cache / Continuous Batching이 왜 다음 강의에 필요한지（예고）
> - 사실 / 설명 / 해석을 섞지 않고 Training–Inference 경계를 서술하는 습관

---
## 1. 왜 이것을 배우는가

정렬이 끝난 모델은 파일이고, 사용자가 만나는 것은 **매 요청마다 토큰이 나오는 시스템**이다.

```text
4권까지:  θ 를 갱신한다  (학습)
5권부터:  고정 θ 로 y 를 생성·배포한다  (추론·서빙)
```

이 경계를 흐리면 다음 오해가 생긴다.

| 오해 | 실제로 |
|---|---|
| “학습이 빨랐으니 서빙도 빠르다” | 학습 배치와 온라인 요청 배치는 다른 문제 |
| “GPU가 크면 끝” | 대역폭·스케줄·KV·양자화가 병목이 될 수 있다 |
| “batch size만 키우면 된다” | 요청 길이·도착 시각이 제각각이면 static batch는 낭비된다 |
| “정밀도만 낮추면 항상 이득” | 품질·커널 지원·오버헤드에 의존한다 |

**해석:** 커리큘럼이 5권으로 꺾이는 이유는 “할 학습 이야기가 없어서”가 아니라, **다음 병목이 하드웨어·스케줄러·메모리 관리 쪽**으로 이동하기 때문이다（제98강에서 예고한 표의 본문）.

## 2. 먼저 알아야 할 개념

1. **Causal LM / Autoregressive Generation** — 제48, 58강. 프롬프트 뒤 토큰을 하나씩（또는 스펙큘러티브로 여러 개）이어 붙인다.
2. **Forward vs Backward** — 제16~20강. Inference는 기본적으로 forward만, Training은 forward+backward+optimizer.
3. **Mixed Precision** — 제64강. 학습에서도 정밀도를 섞지만, 서빙의 Quantization（제103강）과는 목적·기법이 겹치면서도 다르다.
4. **계산 복잡도·메모리 감각** — 제52강. Attention의 $T^2$와 KV cache 미리보기.
5. **정렬 체크포인트** — 제98강. $\pi_\theta$는 파일이고, 서빙은 그 파일에 대한 반복 생성 시스템이다다.

아직 Prefill/Decode 세부를 깊게 다루지 않는다. 오늘은 **두 세계가 무엇이 다른지**다.

## 3. 핵심 개념 설명

### 3.1 Training(학습)

**Training(트레이닝, 학습)**은 데이터（또는 보상·선호 신호）에 맞춰 파라미터 $\theta$를 갱신하는 과정이다.

전형적인 한 스텝:

```text
배치 샘플 → forward → loss/objective → backward → optimizer.step
```

메모리는 대개 다음을 동시에 든다.

- 가중치（parameters）
- 기울기（gradients）
- 옵티마이저 상태（예: Adam의 $m,v$）
- 역전파용 활성화（activations）
- （분산 학습이면）통신 버퍼

성공 지표의 예: loss, reward, preference win-rate, verifiable accuracy.  
**사실:** 지표의 정의는 과제마다 다르다. **설명:** 학습의 “좋다”는 대개 **분포·정책이 목표에 가까워졌는가**다.

### 3.2 Inference(추론)

**Inference(인퍼런스, 추론)**는 학습이 끝난（또는 freeze된） $\theta$로 입력에 대한 출력을 계산하는 과정이다. LLM에서는 보통 **토큰 생성**을 뜻한다.

전형적인 한 요청:

```text
prompt tokens → (prefill) → 반복: logits → decode 규칙 → append → …
```

메모리는 대개 다음을 든다.

- 가중치（종종 양자화됨）
- **KV Cache**（컨텍스트가 늘수록 커짐 — 제101강）
- 일시적 activation / workspace
- （멀티테넌시라면）여러 요청의 스케줄 상태

성공 지표의 예: TTFT, TPOT, throughput, 오류율, 품질 모니터링.  
**사실:** 서빙 지표 이름은 제품·논문마다 표기가 조금씩 다를 수 있다. **해석:** 학습 성공 ≠ 서빙 성공이다.

### 3.3 Serving(서빙)

**Serving(서빙, 서비스 제공)**은 Inference를 **동시 다수 요청**, SLA, API, 장애 대응까지 포함한 **시스템**으로 올리는 층이다.

```text
Inference = “한 번(또는 소수) 생성 계산”
Serving   = “그 계산을 안정적으로 많이·빠르게 돌리는 운영 시스템”
```

vLLM 같은 **Inference Engine(추론 엔진)**은 Serving의 핵심 부품이다（제104강）. 오늘은 엔진 내부를 열지 않고, Training과 대비되는 **문제 설정**만 고정한다.

### 3.4 한눈에 보는 대조표

| 축 | Training（~4권） | Inference / Serving（5권~） |
|---|---|---|
| 목표 | $\theta$ 갱신 | 고정 $\theta$로 $y\sim\pi_\theta$ 생성 |
| 그래프 | forward + backward + opt | forward（생성 루프） |
| 배치 | 학습 배치·롤아웃 | 요청 배치·연속 배치 |
| 메모리 주인공 | grad·optimizer·activation | 가중치·**KV cache** |
| 성공 | loss·reward·선호 승률 | 지연·처리량·안정성·품질 |
| 전형 병목 | 데이터·보상·분산 학습 | 대역폭·스케줄·양자화·KV |
| 시간 축 | epoch / step | request / token / iteration |

**사실:** 동일 가중치 파일이라도 training 코드 경로와 inference 코드 경로는 다르다.  
**설명:** `model.train()`과 `model.eval()`, `torch.no_grad()`, KV cache, 커널 선택이 갈라진다.

## 4. 직관적으로 이해하기

식당에 비유한다（비유는 **설명**이지 증명이 아니다）.

| | Training | Serving |
|---|---|---|
| 하는 일 | 레시피를 계속 고친다 | 같은 레시피로 주문을 낸다 |
| 병목 | 재료 품질·조리법 실험 | 동시에 온 주문·불 용량·접시 공간 |
| 실패 | 맛이 안 맞음 | 대기 시간·품절·접시 부족 |

정렬（4권）은 “레시피를 더 좋게” 만드는 일이다.  
서빙（5권）은 “점심시간에 주문을 막지 않게” 만드는 일이다.

긴 chain-of-thought를 강화하면（제94강） **평균 출력 토큰 수**가 늘 수 있고, 그러면 decode 비용이 커진다.  
**해석:** 정렬 성공이 곧 서빙 부하 증가로 이어질 수 있다. 숫자를 여기서 단정하지 않는다 — 제107강에서 지표로 다룬다.

## 5. 수학적으로 이해하기

### 5.1 학습 목표（복습 압축）

지도 학습 CE（3권）:

$$

\min_\theta\; \mathbb{E}_{(x,y)}\big[-\log\pi_\theta(y\mid x)\big]

$$

RL 계열（4권, 기호만）:

$$

\max_\theta\; \mathbb{E}\big[r\big] - \beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})

$$

공통점: $\theta$가 **변수**다.

### 5.2 추론 목표

$\theta$를 고정하고, 프롬프트 $x$에 대해 토큰 서열을 샘플（또는 greedy）한다.

$$

y_{1:T} \sim \pi_\theta(\cdot\mid x)

$$

자기회귀:

$$

y_t \sim \pi_\theta(\cdot\mid x, y_{<t})

$$

최적화 변수가 없다. 대신 **계산 그래프를 몇 번·어떤 순서로·어떤 메모리로** 돌릴지가 문제다.

### 5.3 순진한 재계산 vs 캐시

길이 $L$ 프롬프트 뒤 $N$개 토큰을 생성할 때, 매 스텝 전체 컨텍스트를 처음부터 다시 Attention하면 대략

$$

\sum_{t=1}^{L+N} O(t^2) = O\big((L+N)^3\big)

$$

감각의 비용이 된다（제52강）.  
KV를 재사용하면 decode 한 스텝은 대략 **새 토큰 1개**에 대한 계산으로 줄어든다（세부는 제100~101강）.

**사실:** 위는 복잡도 **감각**이지, 실제 FLOPs·벽시계 시간의 측정값이 아니다.

## 6. 작은 숫자로 직접 생각해 보기（예시·가정）

> 아래 숫자는 **교육용 가정**이다. 특정 모델·GPU 벤치마크가 아니다.

가정:

- 프롬프트 토큰 $L=100$
- 생성 토큰 $N=50$
- “순진한 재계산”은 매 스텝 전체 길이만큼 Attention을 다시 한다고 본다

순진한 방식에서 Attention이 보는 길이의 합（대략）:

$$

101+102+\cdots+150 = \frac{50}{2}\,(101+150) = 6275

$$

KV cache를 쓰면:

- Prefill 한 번: 길이 100을 처리
- Decode 50번: 매번 **새 토큰 1개**만 추가（과거 K/V는 재사용）

**설명:** “6275 vs 150”처럼 단순 합으로 속도를 예측하면 안 된다. 실제 병목은 compute인지 memory bandwidth인지에 따라 다르다（제100강）.  
**해석:** 그래도 “왜 캐시가 필수인가”의 직관은 충분하다.

메모리 쪽 가정 예:

- 가중치가 GPU의 큰 몫을 차지한다
- 컨텍스트가 길면 KV가 가중치에 필적하거나 넘을 수 있다

정확한 GB는 모델 크기·층 수·헤드·dtype·동시 요청 수에 달려 있으므로 **여기서 GB를 단정하지 않는다**. 제101강에서 기호로 식을 세운다.

## 7. 코드로 구현하기 — 두 루프의 골격

### 7.1 Training step 골격（이미 익숙한 것）

```python
model.train()
optimizer.zero_grad(set_to_none=True)

logits = model(input_ids)               # forward
loss = loss_fn(logits, labels)          # CE 등
loss.backward()                         # 그래프 유지·역전파
optimizer.step()                        # θ 갱신
```

특징:

- `requires_grad=True` 경로가 살아 있다
- activation을 backward용으로 들고 있을 수 있다
- 옵티마이저 상태가 메모리를 차지한다

### 7.2 Inference step 골격

```python
model.eval()
generated = prompt_ids.clone()

with torch.no_grad():
    for _ in range(max_new_tokens):
        logits = model(generated)           # 순진: 전체 재forward
        next_id = sample_or_argmax(logits[:, -1, :])
        generated = torch.cat([generated, next_id], dim=-1)
        if is_eos(next_id):
            break
```

특징:

- $\theta$ 갱신 없음
- `no_grad()`로 그래프를 안 남긴다
- 위 코드는 **KV cache 없음** — 매 스텝 전체 재계산（교육용）

### 7.3 KV를 쓰는 추론 골격（미리보기）

```python
model.eval()
with torch.no_grad():
    # prefill: 프롬프트 전체를 한 번
    logits, past_kv = model(prompt_ids, use_cache=True)
    next_id = sample_or_argmax(logits[:, -1, :])

    for _ in range(max_new_tokens - 1):
        # decode: 새 토큰만 + past_kv
        logits, past_kv = model(next_id, past_key_values=past_kv, use_cache=True)
        next_id = sample_or_argmax(logits[:, -1, :])
        if is_eos(next_id):
            break
```

API 이름（`past_key_values`, `use_cache`）은 라이브러리마다 다를 수 있다.  
**사실:** Hugging Face Transformers 등 많은 스택이 cache 인자를 제공한다.  
**설명:** 서빙 엔진은 이 아이디어를 **연속 배치·페이지 메모리**로 확장한다（제102, 105강）.

## 8. PyTorch로 확인하는 최소 차이

작은 모델에서 “학습 모드”와 “추론 모드”가 텐서에 미치는 영향만 확인한다.

```python
import torch
import torch.nn as nn

class TinyLM(nn.Module):
    def __init__(self, vocab=32, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.fc = nn.Linear(d, vocab)

    def forward(self, x):
        return self.fc(self.emb(x))

model = TinyLM()
x = torch.randint(0, 32, (2, 8))
y = torch.randint(0, 32, (2, 8))

# --- training-like ---
model.train()
logits = model(x)
loss = nn.functional.cross_entropy(logits.reshape(-1, 32), y.reshape(-1))
loss.backward()
print("grad on emb:", model.emb.weight.grad is not None)

# --- inference-like ---
model.zero_grad(set_to_none=True)
model.eval()
with torch.no_grad():
    logits = model(x)
print("requires_grad (logits):", logits.requires_grad)
```

관찰할 것:

- training 경로: `.grad`가 생긴다
- inference 경로: `no_grad()` 안에서 `requires_grad`가 False

이 차이가 “서빙 최적화 전부”는 아니다. 다만 **그래프를 안 남기는 것**이 추론의 출발점임을 보여 준다.

## 9. 실제 LLM Serving에서는 어떻게 달라지는가

단일 스크립트 `generate()`와 프로덕션 서빙의 차이:

| 단일 추론 스크립트 | LLM Serving |
|---|---|
| 요청 1~소수 | 동시 다수 요청 |
| 배치를 사람이 고정 | 도착 시각·길이가 동적 |
| OOM이면 재시작 | 요청 거절·스왑·우선순위 |
| 품질만 보면 됨 | TTFT/TPOT/처리량·비용 |
| 모델 로드 한 번 | 버전·A/B·롤백 |

Training에서 쓰던 습관이 서빙을 해치는 예:

- **큰 micro-batch 감각**을 온라인에 그대로 적용 → 긴 요청 하나에 GPU가 오래 점유
- **activation checkpointing** 사고방식을 KV에 잘못 투영
- **학습용 정밀도 설정**을 서빙 양자화와 동일시

**설명:** 도구（AMP, grad accum）와 도구（INT4 weight, continuous batching）는 이름만 비슷하고 목적이 다르다.

제98강에서 남긴 매핑을 다시 적는다.

| 4권에서 만든 것 | 5권에서 다시 열리는 문제 |
|---|---|
| 긴 CoT | Decode 길이↑ |
| Chat template | 토크나이저·템플릿 일치 |
| KL로 묶인 문체 | temperature/top-p와 상호작용 |
| 샘플 다수@train | 온라인은 보통 1경로 |

## 10. 실습

### 실습 A — 대조표 채우기

빈칸을 자신의 말로 채운다.

```text
목표: Training = (    ), Inference = (    )
메모리 주인공: Training = (    ), Inference = (    )
성공 지표: Training = (    ), Serving = (    )
```

### 실습 B — 순진 generate 비용 감각

가정 $L=200$, $N=100$일 때, “매 스텝 전체 재forward”라면 Attention이 보는 길이의 합을

$$

\sum_{t=1}^{N} (L+t)

$$

로 계산해 본다（예시·가정）. KV를 쓰면 decode 단계의 “새 토큰 수”는 $N$임을 한 줄로 대조한다.

### 실습 C — 코드 모드 전환

위 TinyLM 예제에서:

1. `model.train()` + `loss.backward()` 후 어떤 파라미터에 `.grad`가 있는지 확인
2. `model.eval()` + `torch.no_grad()`에서 `logits.requires_grad` 확인
3. （선택）`torch.cuda.memory_allocated()`가 있는 환경이면, 학습 한 스텝과 추론 루프의 할당 차이를 **환경 의존**으로만 관찰（숫자를 책에 일반화하지 말 것）

### 실습 D — 제품 문장 고치기

다음 문장을 “사실/해석이 섞였다”고 보고 고친다.

> “이 모델은 RLHF까지 끝났으니 서비스 지연도 자동으로 짧다.”

예시 교정:

> “RLHF로 정책 분포는 정렬 목표에 맞게 바뀌었을 수 있다. 서비스 지연은 Prefill/Decode·배치·하드웨어에 달려 있으며 별도로 측정해야 한다.”

## 11. 자주 하는 실수

1. **Training throughput과 Serving throughput을 같은 단위로 비교**  
   tokens/sec라도 “학습 토큰”과 “생성 토큰”의 의미가 다르다.

2. **`eval()`만 하면 서빙 최적화 완료**라고 생각  
   `eval()`은 dropout 등을 끌 뿐, KV·배치·양자화를 해결하지 않는다.

3. **배치 크기만 키우면 GPU 효율이 항상 오른다**  
   요청 길이 불일치·패딩 낭비·KV 메모리 상한이 있다（제102강）.

4. **학습 Mixed Precision = 서빙 Quantization**  
   둘 다 “비트 수” 이야기지만, 목적·교정·커널이 다르다（제64 vs 제103강）.

5. **벤치마크 숫자를 맥락 없이 암기**  
   모델·GPU·동시성·입출력 길이에 따라 달라진다. 이 시리즈는 **날조·단정 금지**.

6. **정렬 품질만 보고 배포**  
   템플릿 불일치·EOS·stop string·최대 길이 설정이 학습과 다르면 “다른 모델”처럼 동작한다.

## 12. 핵심 정리

- Training은 $\theta$를 바꾸고, Inference는 고정 $\theta$로 토큰을 만든다.
- 메모리 주인공이 grad/optimizer에서 **가중치+KV**로 이동한다.
- Serving은 Inference를 동시 요청·SLA 관점으로 확장한 시스템이다다.
- 순진한 재계산 생성은 비용 감각상 불리하며, Prefill/Decode·KV가 필수 입구다.
- 정렬 성공과 서빙 성공은 **다른 축**이다. 둘 다 필요하다.

## 13. 핵심 용어

| 용어 | 한 줄 의미 |
|---|---|
| Training | 파라미터 갱신 과정 |
| Inference | 고정 파라미터로 출력 계산（LLM에선 주로 생성） |
| Serving | 다수 요청·운영 제약을 포함한 추론 제공 |
| Checkpoint | 저장·로드되는 가중치 스냅샷 |
| Autoregressive generation | 이전 토큰을 조건으로 다음 토큰을 잇는 생성 |
| Prefill / Decode | 프롬프트 일괄 처리 / 토큰 단위 이어붙이기（다음 강） |
| KV Cache | Attention의 K,V 재사용 저장소（제101강） |
| Continuous Batching | 요청을 동적으로 끼워 넣는 배치（제102강） |
| TTFT / TPOT | 첫 토큰·토큰당 지연 지표（제107강） |
| Inference Engine | vLLM 등 고성능 추론 런타임（제104강） |

## 14. 연습 문제
### 문제 1（목표）

Training과 Inference의 최적화/생성 목표를 수식 또는 한 줄로 대조하시오.

### 문제 2（메모리）

학습 시와 추론 시 각각 “메모리를  Dominate하기 쉬운” 항목을 두 개씩 쓰시오.

### 문제 3（그래프）

`loss.backward()`가 있는 루프와 `torch.no_grad()` 생성 루프의 결정적 차이를 한 문장으로.

### 문제 4（오해）

“RLHF가 끝났으니 TTFT가 짧다”가 잘못된 이유를 사실/해석으로 구분해 쓰시오.

### 문제 5（복잡도 감각）

순진한 재계산 생성이 KV cache 대비 불리한 이유를 $T^2$ 감각으로 설명하시오（수치 단정 없이）.

### 문제 6（용어）

Inference와 Serving을 한 줄씩 구분하시오.

### 문제 7（다리）

제100강 제목을 쓰고, Prefill과 Decode가 Training 배치와 다른 이유를 한 줄로 쓰시오.

---

## 정답 및 해설

### 문제 1

Training: $\theta$에 대해 loss/objective 최소화·보상 최대화 등.  
Inference: 고정 $\theta$로 $y\sim\pi_\theta(\cdot\mid x)$ 생성.

### 문제 2

학습: gradients, optimizer states（+ activations）.  
추론: weights, KV cache（+ workspace）.

### 문제 3

전자는 미분 그래프를 구성·역전파하고, 후자는 가중치를 갱신하지 않으며 보통 그래프를 남기지 않는다.

### 문제 4

사실: RLHF는 정책 분포를 바꾸는 학습 단계다.  
해석/연결: TTFT는 Prefill 등 서빙 경로의 지연 지표라 자동으로 보장되지 않는다.

### 문제 5

매 스텝 전체 컨텍스트 Attention을 다시 하면 길이 증가에 따라 제곱 항이 누적되는 감각이 되고, KV 재사용 시 decode는 새 토큰 중심으로 줄어든다.

### 문제 6

Inference: 생성 계산 자체. Serving: 그 계산을 다수 요청·운영 제약 하에서 제공하는 시스템.

### 문제 7

제목: Prefill과 Decode.  
이유: 학습 배치는 보통 고정 길이 텐서를 한 번에 forward/backward하지만, 서빙은 프롬프트 일괄（prefill）과 토큰 단위 연장（decode）이 시간·병목 특성이 다른 두 국면으로 나뉜다.

## 15. 다음 강의와 연결

좌표계가 고정되었다.  
다음 **제100강. Prefill과 Decode**에서는 한 요청의 수명을 **프롬프트를 밀어 넣는 구간**과 **토큰을 잇는 구간**으로 쪼개고, 왜 한쪽은 compute에 가깝고 다른 쪽은 memory bandwidth에 가까운 이야기가 나오는지（설명 수준）를 다룬다.

> 가중치는 같아도, 시간이 흐르는 방식이 다르다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [98강. 4권 총정리 — Inference와 Serving으로](../04_rlhf_ppo_grpo/98강_4권_총정리_Inference와_Serving으로.md)
- **다음 강:** [100강. Prefill과 Decode](100강_Prefill과_Decode.md)

<!-- /LECTURE_NAV -->
