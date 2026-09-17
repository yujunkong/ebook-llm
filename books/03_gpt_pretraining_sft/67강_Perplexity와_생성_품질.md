# 제67강. Perplexity와 생성 품질

> **학습 목표**
> - PPL이 평균 NLL(Negative Log Likelihood)의 지수임을 식으로 쓰기
> - PPL을 “얼마나 놀라는가”로 직관적으로 해석하되, 남용하지 않기
> - 같은 PPL이라도 토크나이저·시퀀스 길이·패딩 처리에 따라 비교가 깨질 수 있음
> - 생성 품질(유창성, 지시 준수, 사실성 등)은 PPL만으로 대체되지 않음
> - 미니 GPT에서 `val_loss → PPL`을 찍고, 샘플 생성과 나란히 보는 습관

---
## 1. 왜 이것을 배우는가

Pretraining 중 화면에 자주 뜨는 숫자는 대략 두 종류다.

```text
train/val loss  (또는 NLL)
perplexity      (= exp(mean NLL))
```

Loss는 최적화 목표에 가깝고, PPL은 그 Loss를 **사람이 읽기 쉬운 척도**로 바꾼 것이다. 문제는 PPL이 “모델이 좋아졌다”는 **만능 성적표**처럼 오해되기 쉽다는 점이다.

실제로는:

- PPL이 내려가도 생성이 지루하거나 반복적일 수 있다.
- PPL이 비슷해도 Temperature / Top-p에 따라 체감 품질이 크게 달라진다.
- 데이터셋·토크나이저가 다르면 PPL 숫자 자체를 비교하면 안 된다.

3권 후반 Instruction Tuning / SFT에서는 “다음 토큰을 잘 맞추는 것”과 “지시를 잘 따르는 것”이 더 갈라진다. 지금 PPL의 한계를 고정해 두지 않으면, 이후 평가 설계가 흔들린다.

## 2. 먼저 알아야 할 개념

- Next-Token Prediction과 Causal LM Loss (제57·62강, 2권 제32·34강)
- Cross Entropy ≈ one-hot 정답에 대한 NLL (2권 제34강)
- Validation과 Evaluation의 역할 분리 (제66강)
- Softmax 확률 $p(x_t \mid x_{<t})$ (2권 제33강)
- 생성: Greedy / Sampling / Temperature (제58·59강)

아직 깊게 들어가지 않는 것:

- BLEU, ROUGE, BERTScore 등 생성 자동 지표의 전면 비교
- Human preference / RLHF 보상 모델 (4권)
- 벤치마크 리더보드 수치 암기 (이 책은 **임의 벤치마크 숫자를 만들지 않는다**)

## 3. 핵심 개념 설명

### 3.1 Perplexity란?

**Perplexity(퍼플렉시티)**는 언어 모델이 평가 시퀀스에 대해 **평균적으로 얼마나 “헷갈리는지”**를 나타내는 척도다. 값이 **작을수록** 모델이 해당 데이터 분포의 다음 토큰을 더 높은 확률로 맞춘다는 뜻이다.

표준 정의(토큰 단위, 길이 $T$인 시퀀스 $x_{1:T}$):

$$

\mathrm{PPL}(x_{1:T})
=
\exp\!\left(
-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})
\right)
=
\exp\!\big(\overline{\mathrm{NLL}}\big)

$$

여기서:

- $p_\theta(x_t \mid x_{<t})$: 모델이 문맥 $x_{<t}$에서 정답 토큰 $x_t$에 부여한 확률
- $-\log p_\theta(\cdot)$: 위치별 NLL (= Cross Entropy, one-hot 정답일 때)
- $\overline{\mathrm{NLL}}$: 유효 토큰에 대한 **평균 NLL**
- $\exp(\cdot)$: 자연상수 $e$를 밑으로 하는 지수

한 줄로 외운다.

> **PPL = exp(평균 NLL)**

학습 로그의 `val_loss`가 이미 **평균 CE/NLL**이라면:

```text
ppl = math.exp(val_loss)
```

이다. (단, `val_loss`가 합인지 평균인지, pad를 제외했는지 확인해야 한다.)

### 3.2 “헷갈림” 직관

이상적인 경우, 매 스텝에서 모델이 마치 **균등한 $k$개 후보** 사이에서 고르는 것처럼 행동한다고 상상하자. 그때의 유효 후보 수 $k$가 PPL에 가깝다는 고전적 비유가 있다.

- PPL ≈ 1: 거의 항상 정답에 확률 1을 줌 (극단적으로 확신)
- PPL ≈ $V$: vocabulary 전체에 거의 균등 (거의 무작위)
- 그 사이: “평균적으로 몇 갈래로 갈라지는가”

비유는 직관에 도움이 되지만, **실제 분포는 균등이 아니다**. 따라서 “PPL=30이면 후보가 정확히 30개”라고 말하면 과하다. “낮을수록 해당 코퍼스에서 다음 토큰을 잘 압축·예측한다” 정도가 안전하다.

### 3.3 토큰 단위 vs 단어 단위

문헌에 따라 Perplexity를 **단어(word)** 단위로 보고하는 경우도 있다. BPE/서브워드 토크나이저를 쓰면 토큰 수 $T_{\text{tok}}$와 단어 수 $T_{\text{word}}$가 다르다.

같은 텍스트라도:

$$

\overline{\mathrm{NLL}}_{\text{tok}}
\quad\text{와}\quad
\overline{\mathrm{NLL}}_{\text{word}}

$$

의 평균 분모가 달라지므로 PPL 숫자도 달라진다. **비교할 때는 단위·토크나이저·전처리를 반드시 명시**한다.

이 책의 미니 실험에서는 기본적으로 **토큰 단위 PPL**을 쓴다. 문자(char) tokenizer면 “문자 단위”가 곧 토큰 단위다.

### 3.4 패딩·마스크와 평균

배치 학습에서는 pad 토큰이 섞인다. Loss에서 `ignore_index`로 pad를 빼면, PPL도 **같은 유효 토큰 집합**으로 계산해야 한다.

잘못된 예:

```text
평균 NLL을 pad 포함으로 계산 → exp
```

pad 위치가 쉬우면(또는 이상하게 쉬우면) PPL이 왜곡된다. 제61강 Dataset Packing을 쓰면 pad가 줄어 PPL이 달라 보일 수 있다. **패킹 전후를 같은 평가 스크립트로** 비교하라.

### 3.5 생성 품질이란? (PPL과 다른 축)

**생성 품질(Generation Quality)**은 모델이 실제로 뽑아낸 텍스트가 **사람에게·과제에** 얼마나 쓸모 있는지를 가리킨다. 단일 숫자로 정의되지 않는다. 대표 축만 나열하면:

| 축 | 질문 |
|---|---|
| 유창성 | 문법·표현이 자연스러운가? |
| 관련성 | 프롬프트/지시에 맞는가? |
| 다양성 | 반복·붕괴 없이 적절한가? |
| 사실성 | 근거 없는 주장을 하지 않는가? |
| 안전·정중 | 유해·무례한 출력을 피하는가? |
| 과제 성공 | 요약·코드·번역 등 목표를 달성하는가? |

Pretraining 단계의 PPL은 주로 **“홀드아웃 텍스트의 다음 토큰 예측”**을 본다. 위 표의 여러 축을 한꺼번에 대체하지 못한다.

## 4. 직관적으로 이해하기

같은 val loss라도 체감이 다를 수 있다.

```text
모델 A: val_loss = 2.3  →  PPL ≈ exp(2.3) ≈ 10.0
모델 B: val_loss = 2.3  →  PPL ≈ 10.0
```

숫자는 같다. 그런데:

- A는 Temperature 0.7 + Top-p 0.9로 읽기 좋은 글을 냄
- B는 Greedy만 쓰면 같은 구절을 반복함

PPL은 **평가 코퍼스에 대한 가능도**이지, **디코딩 전략의 체감**이 아니다. 반대로:

```text
모델 C: PPL이 더 낮음
생성: 학습 말투를 과도하게 외워 창의성이 떨어짐 (과적합 느낌)
```

도 가능하다. 특히 데이터가 작을수록 “낮은 PPL = 좋은 생성” 등식이 깨지기 쉽다.

## 5. 수학적으로 이해하기

### 5.1 시퀀스 가능도와 PPL

결합 확률의 인과 분해:

$$

p_\theta(x_{1:T})=\prod_{t=1}^{T} p_\theta(x_t\mid x_{<t})

$$

양변에 $-\frac{1}{T}\log$를 취하면 평균 NLL:

$$

\overline{\mathrm{NLL}}
=
-\frac{1}{T}\log p_\theta(x_{1:T})
=
-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})

$$

PPL은 그 지수:

$$

\mathrm{PPL}=\exp(\overline{\mathrm{NLL}})

$$

따라서 **Loss를 최소화하는 것**과 **PPL을 최소화하는 것**은 단조 변환($\exp$) 관계라 같은 방향이다. Optimizer가 보는 것은 보통 Loss이고, 사람이 로그에 PPL을 옆에 적어 두는 것이다.

### 5.2 Bits-per-character / Bits-per-byte와의 관계

정보량 단위로 보고할 때는 $\log_2$를 쓴다.

$$

\mathrm{bpc}
=
-\frac{1}{T_{\text{char}}}\sum \log_2 p(\cdot)
=
\frac{\overline{\mathrm{NLL}}_{\ln}}{\ln 2}
\quad\text{(문자 단위일 때)}

$$

PPL과 bpc는 같은 NLL 가족이다. 단위만 다르다. 이 책에서는 미니 실험에 PPL을 우선 쓰고, 필요하면 “$\mathrm{PPL}=e^{\ell}$, $\mathrm{bpc}=\ell/\ln 2$”로 환산한다.

### 5.3 배치·말뭉치 평균

여러 시퀀스 $\mathcal{D}=\{x^{(i)}\}$에 대해 흔히 쓰는 방식은 **토큰 가중 평균**이다.

$$

\overline{\mathrm{NLL}}_{\mathcal{D}}
=
\frac{\sum_i\sum_t -\log p(x^{(i)}_t\mid x^{(i)}_{<t})}{\sum_i T_i}

$$

시퀀스마다 길이로 먼저 평균한 뒤 산술 평균하면, 짧은 문서의 비중이 커져 **다른 숫자**가 된다. 평가 스크립트에서 **어느 평균인지**를 고정하라.

## 6. 작은 숫자로 직접 계산하기

길이가 3인 장난감 시퀀스. 각 위치에서 정답 확률이 다음과 같다고 하자.

| $t$ | $p_{\text{correct}}$ | NLL $=-\ln p$ |
|---:|---:|---:|
| 1 | 0.5 | $-\ln 0.5 \approx 0.693$ |
| 2 | $e^{-1}\approx 0.368$ | $1.000$ |
| 3 | 0.25 | $-\ln 0.25 \approx 1.386$ |

평균 NLL:

$$

\overline{\mathrm{NLL}}=\frac{0.693+1.000+1.386}{3}\approx 1.026

$$

Perplexity:

$$

\mathrm{PPL}=\exp(1.026)\approx 2.79

$$

관찰:

- 한 위치라도 확률이 매우 작으면 NLL이 커져 PPL을 끌어올린다.
- “대체로 잘하다가 한 번 크게 틀림”이 평균을 망가뜨린다.

코드로 같은 값:

```python
import math

ps = [0.5, math.exp(-1), 0.25]
nlls = [-math.log(p) for p in ps]
mean_nll = sum(nlls) / len(nlls)
ppl = math.exp(mean_nll)
print(f"mean_nll={mean_nll:.4f}, ppl={ppl:.4f}")
```

예상 출력(반올림 오차 허용):

```text
mean_nll=1.0265, ppl=2.7912
```

## 7. 코드로 구현하기 — Loss에서 PPL

학습 루프의 validation이 이미 평균 CE를 반환한다고 가정한다.

```python
# ppl_from_loss.py
"""val_loss(평균 NLL) → perplexity."""

from __future__ import annotations

import math

def perplexity_from_mean_nll(mean_nll: float) -> float:
    """PPL = exp(mean NLL). mean_nll이 너무 크면 overflow에 주의."""
    # 예: mean_nll > 20 이면 exp가 매우 커짐 → 로그 비교가 나을 수 있음
    return math.exp(mean_nll)

def safe_perplexity(mean_nll: float, cap: float = 100.0) -> float:
    """폭발 방지용 상한 (로깅용). 수학적 정의의 일부가 아님."""
    return math.exp(min(mean_nll, cap))

if __name__ == "__main__":
    for loss in [0.0, 1.0, 2.3, 5.0]:
        print(f"loss={loss:.1f} → ppl={perplexity_from_mean_nll(loss):.3f}")
```

예상 출력:

```text
loss=0.0 → ppl=1.000
loss=1.0 → ppl=2.718
loss=2.3 → ppl=9.974
loss=5.0 → ppl=148.413
```

학습 초기 char-LM에서 loss가 $\ln V$ 근처면, PPL도 $V$ 근처여야 **랜덤 초기화와 일치**한다. (제50·68강 overfit/초기화 점검과 연결)

## 8. PyTorch로 구현하기 — 평가 루프에 붙이기

```python
# eval_ppl.py
"""평균 CE → PPL. pad는 ignore_index로 제외."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

@torch.no_grad()
def mean_nll_and_ppl(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ignore_index: int = -100,
    device: str = "cpu",
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in batches:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)  # [B, T, V]
        B, T, V = logits.shape
        loss_sum = F.cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
            ignore_index=ignore_index,
            reduction="sum",
        )
        n_valid = int((y != ignore_index).sum().item())
        total_loss += float(loss_sum.item())
        total_tokens += n_valid

    mean_nll = total_loss / max(total_tokens, 1)
    ppl = math.exp(mean_nll)
    return mean_nll, ppl
```

포인트:

1. `reduction="sum"` + 유효 토큰 수로 나누면 **토큰 가중 평균**이 된다.
2. `math.exp(mean_nll)` 전에 mean이 정의와 맞는지 확인한다.
3. 생성 샘플은 **별도 함수**로 뽑아 나란히 로그한다.

## 9. 실제 LLM에서는 어떻게 사용하는가

대규모 Causal LM Pretraining에서 PPL(또는 val loss)는 여전히 **학습이 살아 있는지**, **체크포인트 간 상대 비교**에 쓰인다.

다만 제품·연구 보고에서는 보통 다음을 병행한다.

| 단계 | 자주 보는 것 |
|---|---|
| Pretraining | val loss / PPL, 간헐적 샘플 생성 |
| SFT 이후 | 지시 준수 평가, 과제별 성공률, 사람 평가 |
| RLHF 이후 | preference win rate, 안전성 평가 등 (4권) |

공개 리더보드 숫자는 **데이터 오염·프롬프트 형식·디코딩**에 민감하다. 이 책에서는 특정 모델의 “PPL=XX로 GPT-4를 이겼다” 같은 **검증되지 않은 수치를 만들지 않는다**. 원리면 원리, 실험이면 **재현 가능한 미니 실험**만 적는다.

Hugging Face 등 생태계에서는 `evaluate` 라이브러리의 perplexity 계산기, `lm-eval-harness` 같은 **표준 평가 하네스**가 쓰인다. 핵심은 도구 이름이 아니라 **동일한 토크나이저·동일 전처리·동일 평균 정의**다.

## 10. 실습

### 실습 1 — 손계산과 코드 일치

§7 표의 세 확률로 `mean_nll`과 `ppl`을 손계산한 뒤, §8 스크립트와 소수점 3자리까지 맞춰 보라.

### 실습 2 — $\ln V$ 점검

vocabulary 크기 $V=65$(소문자+기호 가정)일 때, 완전 랜덤 모델의 이상적 평균 NLL과 PPL을 구하라.

$$

\overline{\mathrm{NLL}}\approx \ln 65,\quad \mathrm{PPL}\approx 65

$$

미니 모델 첫 step loss가 이 근처인지 확인하는 버릇을 들인다.

### 실습 3 — 생성과 함께 보기

제58강 greedy와 제59강 temperature 샘플링으로 같은 체크포인트에서 문장을 몇 개 뽑고, val PPL 숫자 **옆에** 붙인다.

```text
step=500  val_loss=2.10  ppl=8.17
sample[0]: ...
sample[1]: ...
```

“숫자만 보고 좋다/나쁘다”를 금지하는 연습이다.

### 실습 4 — (선택) pad 포함 vs 제외

고의로 pad를 loss에 포함한 PPL과 `ignore_index`로 제외한 PPL을 비교하라. 차이가 크면 평가 버그 후보다.

## 11. 자주 하는 실수

1. **다른 토크나이저로 찍은 PPL을 비교한다**  
   서브워드가 잘게 쪼개지면 토큰당 예측이 쉬워져 PPL이 낮아 보일 수 있다.

2. **train PPL만 보고 일반화를 주장한다**  
   암기면 train PPL은 내려가고 val/생성은 망가질 수 있다.

3. **`exp(sum_nll)`처럼 합에 지수를 씌운다**  
   정의는 **평균** NLL의 지수다.

4. **PPL이 낮으면 생성 지표도 자동으로 좋다고 본다**  
   디코딩·과제·안전성은 별 축이다.

5. **overflow를 무시한다**  
   학습 초기에 loss가 크면 `exp`가 `inf`가 된다. 로깅 시 cap 또는 loss만 표시.

6. **평가 시 `model.train()` 상태**  
   Dropout이 켜져 val PPL이 흔들린다. `model.eval()` + `torch.no_grad()`.

## 12. 핵심 정리

- Perplexity는 $\exp(\text{평균 NLL})$이다. val loss가 평균 NLL이면 `ppl = exp(val_loss)`.
- PPL은 홀드아웃 텍스트에 대한 **다음 토큰 예측력**의 요약이다.
- 토크나이저·패딩·평균 단위가 다르면 숫자 비교가 무의미하다.
- 생성 품질은 유창성·지시 준수·다양성·사실성 등 **다축**이며 PPL로 대체되지 않는다.
- Pretraining 로그에는 **PPL + 샘플 생성**을 나란히 두는 것이 실무적이다.

## 13. 핵심 용어

| 용어 | 의미 |
|---|---|
| Perplexity (PPL) | $\exp(\overline{\mathrm{NLL}})$, LM 예측 난이도의 요약 척도 |
| NLL | $-\log p(\text{정답})$, Cross Entropy와 (one-hot 시) 동일 |
| Mean NLL | 유효 토큰에 대한 NLL 평균; 보통 val_loss |
| Bits-per-character (bpc) | $\overline{\mathrm{NLL}}/\ln 2$ 형태의 정보량 보고 |
| Generation quality | 생성 텍스트의 과제·인간 기준 품질 (단일 숫자 아님) |
| Holdout / Validation set | 학습에 쓰지 않은 평가 분할 |
| Decoding | Greedy·Sampling 등 확률→토큰 변환 (PPL과 별개) |

## 14. 연습 문제
### 문제 1 (정의)

평균 NLL이 $2.0$일 때 PPL은? (식으로)

### 문제 2 (계산)

네 위치의 정답 확률이 $1, 1/e, 1/e^2, 1/e$일 때 평균 NLL과 PPL을 구하시오.

### 문제 3 (개념)

왜 train PPL이 매우 낮은데도 생성이 빈약할 수 있는가? 원인 후보를 두 가지 쓰시오.

### 문제 4 (평가 설계)

모델 A는 char tokenizer, 모델 B는 BPE다. 둘의 PPL 숫자를 직접 비교하면 안 되는 이유를 한 문장으로.

### 문제 5 (연결)

제66강 Validation과 이번 PPL의 관계, 그리고 제68강 프로젝트에서 로그에 무엇을 나란히 찍어야 하는지 쓰시오.

---

## 정답 및 해설

### 문제 1

$\mathrm{PPL}=\exp(2.0)=e^2$.

### 문제 2

NLL = $0 + 1 + 2 + 1 = 4$, 평균 $1$, PPL $=e^1=e$.

### 문제 3

예: (1) 학습 집합 암기·과적합으로 val/생성 분포와 괴리 (2) 디코딩이 반복·붕괴 (3) 프롬프트가 학습 분포 밖 (4) eval 모드/토크나이저 불일치.

### 문제 4

토큰 단위가 달라 같은 텍스트라도 평균 NLL의 분모·난이도 정의가 달라지기 때문이다.

### 문제 5

Validation loss(평균 NLL)를 고정된 홀드아웃으로 재고, 그 지수인 PPL을 함께 보고한다. 제68강에서는 `val_loss`/`ppl`과 **샘플 생성문**을 같은 로그에 남겨, 숫자와 체감을 동시에 추적한다.

## 15. 다음 강의와 연결

제66강에서 평가의 자리를 잡았고, 이번 강의에서 PPL의 정의와 한계를 고정했다.

이제 부품을 한데 모아 **작은 코퍼스에서 Pretraining을 끝까지** 돌려 볼 차례다. 다음 **제68강. 프로젝트 — Mini GPT Pretraining**에서는 `config / data / model / train / generate` 구조로 미니 GPT를 학습하고, loss 곡선·PPL·샘플 생성을 한 번에 확인한다.

> 숫자를 읽을 줄 알면, 이제 그 숫자가 움직이는 학습을 직접 돌려 보자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제66강. Validation과 Evaluation](66강_Validation과_Evaluation.md)
- **다음 강:** [제68강. 프로젝트 — Mini GPT Pretraining](68강_프로젝트_Mini_GPT_Pretraining.md)

<!-- /LECTURE_NAV -->
