# 제59강. Temperature, Top-K, Top-P

> **학습 목표**
> - Temperature(온도) 가 logit 스케일을 바꿔 엔트로피를 조절함을 식으로 쓴다
> - Top-k 가 상위 $k$ 토큰만 남기는 절단임을 구현한다
> - Top-p (Nucleus sampling) 가 누적 확률 $p$로 동적 집합을 고름을 손계산한다
> - 작은 logit 벡터로 Temperature / Top-k / Top-p를 숫자로 추적한다
> - 세 기법을 조합할 때의 순서와 함정을 안다
> - “창의성 슬라이더”로만 이해하는 오해를 교정한다

---
## 1. 왜 이것을 배우는가

원분포 Sampling은 이론적으로 모델 그대로지만, 실사용에서는 문제가 자주 생긴다.

```text
꼬리 토큰이 가끔 뽑힘 → 문장이 갑자기 붕괴
greedy는 안전하지만 → 반복·단조
과제마다 원하는 엔트로피가 다름 → 코드 vs 시 vs 대화
```

Temperature · Top-k · Top-p는 **모델을 다시 학습하지 않고** 추론 분포만 조절한다.  
Pretraining Loss(제57강)와 분리된 **디코딩 하이퍼파라미터**다.

제67강에서 PPL이 같아도 Temperature에 따라 체감 품질이 달라진다고 한다. 그 메커니즘이 오늘 내용이다.

## 2. 먼저 알아야 할 개념

- Softmax · logit (제33강)
- CE 학습은 Softmax 온도 1을 전제로 함 (제57강)
- Greedy / multinomial 루프 (제58강)
- 누적합(CDF) 직관
- 정렬(sorting) 후 마스크

빔 서치·대조 디코딩은 다루지 않는다. 필요하면 이후 확장으로 둔다.

## 3. 핵심 개념 설명

### 3.1 Temperature

**Temperature** $T>0$는 Softmax 전 logit을 스케일링한다.

$$

p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}

$$

| $T$ | 효과 |
|---|---|
| $T\to 0^+$ | 최대 logit에 확률 집중 → greedy에 수렴 |
| $T=1$ | 학습 때와 같은 Softmax |
| $T>1$ | 분포가 평평해짐 → 다양성↑, 이상 토큰↑ |
| $T<1$ | 분포가 날카로워짐 |

$T$는 “창의성 마법 숫자”가 아니라 **엔트로피 손잡이**다.

주의: 학습 Loss에 Temperature를 넣는 기법(knowledge distillation 등)과, **추론 디코딩 Temperature**를 혼동하지 않는다. 이 강의는 후자.

### 3.2 Top-k Sampling

**Top-k sampling**은 확률이 높은 $k$개 토큰만 남기고 나머지 확률을 0으로 만든 뒤, 남긴 질량을 다시 정규화하여 샘플한다.

절차:

1. $\mathbf{z}$에서 상위 $k$ logit(또는 확률) 선택
2. 나머지 위치를 $-\infty$(또는 확률 0)로
3. Softmax(또는 재정규화)
4. multinomial

$k=1$이면 greedy와 같다. $k=V$이면 절단 없음.

### 3.3 Top-p (Nucleus) Sampling

**Top-p sampling** 또는 **Nucleus sampling(핵 샘플링)**은 확률 내림차순으로 토큰을 누적해, 누적 합이 $p$ 이상이 되는 **최소 집합**만 남긴다. 집합 크기가 문맥에 따라 변한다.

절차:

1. $p_i=\mathrm{softmax}(z_i)$ (또는 temperature 적용 후)
2. 내림차순 정렬 $p_{(1)}\ge p_{(2)}\ge\cdots$
3. 최소 $m$ s.t. $\sum_{i=1}^{m}p_{(i)}\ge p$
4. 그 $m$개만 남기고 재정규화 후 샘플

직관: “확률 질량의 상위 $p$ 핵(nucleus)만 남긴다.”

Top-k는 개수 고정, Top-p는 **질량** 고정이라 분포가 뾰족할 때는 후보가 줄고, 평평할 때는 후보가 늘어난다.

### 3.4 조합 순서

실무에서 흔한 순서:

```text
logits
  → / temperature
    → (optional) top-k truncate
      → softmax
        → (optional) top-p truncate on probs
          → renormalize
            → sample
```

구현마다 Top-k를 확률 대신 logit 단계에서 하기도 한다. **한 구현의 순서를 문서화**하고 일관되게 쓴다. 이 책은 위 순서를 기본으로 한다.

Greedy는 `temperature→0` 또는 `argmax`로 따로 두는 편이 수치적으로 안전하다.

## 4. 직관적으로 이해하기

식당 메뉴 비유:

- **Temperature↑**: 평소 안 시키던 메뉴도 고를 마음 ↑
- **Top-k**: “상위 k개 추천만 보고 고르기”
- **Top-p**: “추천 확률을 더해 80% 될 때까지만 메뉴판에 남기기”

메뉴판(모델 분포)이 이미 잘못되면 손잡이만으로 명요리가 나오지 않는다.  
손잡이는 **이미 학습된 분포의 사용법**이다.

## 5. 수학적으로 이해하기

### 5.1 Temperature와 엔트로피

이산 분포 엔트로피:

$$

H(p)=-\sum_i p_i\log p_i

$$

$T$가 커지면 일반적으로 $H$가 커지고, $T$가 작아지면 $H$가 줄어든다(동률·퇴화 케이스 제외).

logit 차이가 $\Delta$일 때 $T$로 나누면 유효 차이가 $\Delta/T$로 줄어든다.

### 5.2 Top-k 재정규화

허용 집합 $\mathcal{K}$:

$$

\tilde{p}_i=\frac{p_i\cdot\mathbf{1}[i\in\mathcal{K}]}{\sum_{j\in\mathcal{K}}p_j}

$$

logit 마스크로 구현하면 Softmax 한 번에 끝난다.

### 5.3 Top-p 집합

정렬된 확률에 대해

$$

m=\min\Big\{m:\sum_{i=1}^{m}p_{(i)}\ge p\Big\},\quad
\mathcal{N}=\{(1),\ldots,(m)\}

$$

동률·부동소수점 때문에 “$\ge p$” 경계를 구현할 때 토큰이 하나 더/덜 들어갈 수 있다. 테스트로 고정한다.

## 6. 작은 숫자로 직접 계산하기

공통 logit ($V=5$):

$$

\mathbf{z}=(3.0,\ 1.5,\ 1.2,\ 0.5,\ -1.0)

$$

인덱스는 `0..4`.

### 6.1 Temperature $T=1$ Softmax

$$

e^{z}\approx(20.09,\ 4.48,\ 3.32,\ 1.65,\ 0.37),\ \sum\approx 29.91

$$

$$

\mathbf{p}\approx(0.672,\ 0.150,\ 0.111,\ 0.055,\ 0.012)

$$

### 6.2 Temperature $T=0.5$

$z/T=(6.0,\ 3.0,\ 2.4,\ 1.0,\ -2.0)$

$$

e^{z/T}\approx(403.4,\ 20.1,\ 11.0,\ 2.72,\ 0.14),\ \sum\approx 437.4

$$

$$

\mathbf{p}\approx(0.922,\ 0.046,\ 0.025,\ 0.006,\ 0.0003)

$$

거의 인덱스 0에 집중 → greedy에 가까움.

### 6.3 Temperature $T=2.0$

$z/T=(1.5,\ 0.75,\ 0.6,\ 0.25,\ -0.5)$

$$

e^{z/T}\approx(4.48,\ 2.12,\ 1.82,\ 1.28,\ 0.61),\ \sum\approx 10.31

$$

$$

\mathbf{p}\approx(0.435,\ 0.206,\ 0.177,\ 0.124,\ 0.059)

$$

분포가 평평해져 꼬리(인덱스 4)도  Ignorable하지 않다.

### 6.4 Top-k, $k=2$, $T=1$

상위 2개: 인덱스 0,1 확률 $0.672,\ 0.150$, 합 $0.822$

재정규화:

$$

\tilde{p}_0=0.672/0.822\approx0.817,\quad
\tilde{p}_1=0.150/0.822\approx0.183

$$

나머지 0.

### 6.5 Top-p, $p=0.9$, $T=1$

내림차순 누적:

```text
0: 0.672          cum=0.672
1: 0.150          cum=0.822
2: 0.111          cum=0.933  ← 여기서 ≥0.9
```

핵 = `{0,1,2}`. 재정규화:

합 $0.933$

$$

\tilde{p}\approx(0.720,\ 0.161,\ 0.119,\ 0,\ 0)

$$

### 6.6 Top-p가 Top-k와 달라지는 경우

분포가 매우 뾰족하면 $p=0.9$여도 $m=1$일 수 있다.  
평평하면 $m$이 커진다. 같은 $p$라도 문맥마다 후보 수가 변한다 — Nucleus의 요점이다.

## 7. 코드로 구현하기

### 7.1 Temperature만

```python
import torch
import torch.nn.functional as F

def sample_with_temperature(logits, temperature=1.0):
    # logits: [V] or [B,V]
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use greedy for hard argmax")
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

$T\to0$을 코드에서 `1e-8`로 근사하기보다, greedy 분기를 분리하는 편이 안전하다.

### 7.2 Top-k

```python
def apply_top_k(logits, k):
    # logits: [B, V]
    if k <= 0:
        return logits
    V = logits.size(-1)
    k = min(k, V)
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    # k번째 값(가장 작은 top-k)보다 작은 위치 제거
    cutoff = topk_vals[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < cutoff, float("-inf"))
```

동률이 있으면 $k$개보다 많이 남을 수 있다. 개수를 엄격히 $k$로 고정하려면 인덱스 마스크 방식을 쓴다.

### 7.3 Top-p

```python
def apply_top_p(probs, p):
    """probs: [B,V] 이미 Softmax된 확률. 반환도 확률(재정규화)."""
    if p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # 누적이 p를 넘긴 이후의 토큰 제거. 단, 최소한 하나는 남김
    mask = cum > p
    # 첫 토큰은 항상 유지: 오른쪽으로 한 칸 shift해 True 시작을 미룸
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    # 원래 인덱스로 scatter
    out = torch.zeros_like(probs)
    out.scatter_(1, sorted_idx, sorted_probs)
    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out
```

이 패턴은 공개 구현들에서 흔히 보는 **정렬→누적→마스크→scatter** 방식이다. 경계 off-by-one을 단위 테스트로 고정한다.

### 7.4 통합 decode_rule

```python
def sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """logits: [B,V] → next_id [B,1]"""
    if temperature != 1.0:
        logits = logits / temperature
    if top_k and top_k > 0:
        logits = apply_top_k(logits, top_k)
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        probs = apply_top_p(probs, top_p)
    return torch.multinomial(probs, num_samples=1)
```

제58강 루프의 `next_id = ...`만 이 함수로 교체한다.

```python
next_id = sample_logits(
    logits[:, -1, :],
    temperature=0.8,
    top_k=40,
    top_p=0.9,
)
```

## 8. PyTorch로 손계산 검증

섹션 7 숫자를 코드로 재현한다.

```python
z = torch.tensor([[3.0, 1.5, 1.2, 0.5, -1.0]])
p = F.softmax(z, dim=-1)
print("T=1", p)

p_half = F.softmax(z / 0.5, dim=-1)
print("T=0.5", p_half)

logits_k = apply_top_k(z.clone(), k=2)
print("top-k logits", logits_k)
print("top-k probs", F.softmax(logits_k, dim=-1))

print("top-p", apply_top_p(p, p=0.9))
```

손으로 구한 값과 소수 둘째·셋째 자리까지 맞는지 확인한다.

## 9. 함정과 실패 모드

### 9.1 Temperature 0 또는 음수

$T\le0$는 정의상 Softmax를 망가뜨린다. UI에서 0을 greedy로 매핑할지 명시한다.

### 9.2 Top-k = 0의 의미

라이브러리마다 `0`이 “비활성”이거나 “버그”다. 이 책은 `k<=0` → 비활성으로 둔다.

### 9.3 Top-p = 1.0

절단 없음. `p=0`은 위험(전부 제거)하므로 `p`는 $(0,1]$로 제한하고 최소 1토큰 보장.

### 9.4 이중 Softmax

Top-p를 logit에 잘못 적용하거나 Softmax를 두 번 하면 분포가 왜곡된다.

### 9.5 학습 분포와 큰 괴리

$T$를 극단으로 올리면 모델이 거의 보지 않은 영역에서 샘플한다. “창의적”이 아니라 **붕괴**로 이어지기 쉽다.

### 9.6 Top-k와 Top-p 동시 과절단

둘 다 공격적으로 켜면 후보가 1~2개만 남아 greedy와 다를 바 없다. 로그에 평균 후보 수($m$)를 찍으면 디버깅에 도움이 된다.

### 9.7 배치 내 다른 길이

이미 EOS인 행도 샘플하면 쓰레기 토큰이 append된다. 제58강의 `finished` 마스크와 함께 쓴다.

## 10. 실제 LLM에서는 어떻게 사용하는가

챗/API에서 사용자가 만지는 슬라이더가 대체로 이 셋이다.

| 상황 | 흔한 경향(경험칙, 만능 수치 아님) |
|---|---|
| 짧은 사실형 답 | 낮은 $T$, 작은 nucleus |
| 코드 | 낮은 $T$, 때때로 greedy |
| 브레인스토밍 | 높은 $T$, 넓은 $p$ |
| 번역 | 낮은 $T$ |

정확한 최적값은 과제·모델·토크나이저에 의존한다. **이 책은 특정 상용 모델의 기본값을 사실처럼 적지 않는다.**

서버 측에서는 logit processor 파이프라인으로 금지어·스키마 제약을 같은 마스크 자리에 넣는다.

## 11. 실습

### 실습 1 — 온도 스윕

같은 프롬프트·시드로 $T\in\{0.2,0.7,1.0,1.5\}$ 샘플을 비교하고, 반복률·이상 토큰 빈도를 주관적으로 기록한다.

### 실습 2 — Top-k vs Top-p

섹션 7 벡터로 $k=3$과 $p=0.9$의 허용 집합을 구한 뒤, 코드 출력과 대조한다.

### 실습 3 — 후보 수 로깅

생성 루프에서 Top-p 적용 후 `m=(probs>0).sum(-1).float().mean()`을 프린트한다. $T$를 바꿔 $m$이 어떻게 변하는지 본다.

### 실습 4 — greedy 등가

`top_k=1` 또는 매우 낮은 $T$가 `argmax`와 같은 토큰을 내는지 여러 스텝 비교한다.

### 실습 5 — 버그 주입

`apply_top_p`에서 “최소 1토큰 보장” 줄을 제거하고 $p=0.01`처럼 작은 값으로 호출해 무엇이 깨지는지 관찰한 뒤 복구한다.

## 12. 자주 하는 실수

1. **Temperature를 Softmax 뒤에 곱함**  
   확률에 $T$를 곱하는 것은 다른 연산이다. 정의는 $z/T$.

2. **Top-p를 정렬 없이 누적**  
   인덱스 순서 누적은 핵이 아니다.

3. **재정규화 누락**  
   절단만 하고 합이 1이 아니면 `multinomial` 입력이 깨진다.

4. **`-inf` Softmax 전 배치 NaN**  
   한 행이 전부 `-inf`면 Softmax NaN. 최소 1토큰 보장 필수.

5. **학습 재개와 디코딩 파라미터 혼동**  
   체크포인트에 $T$가 저장되지 않는 것이 정상. 추론 설정이다.

6. **“Top-p=0.9면 항상 90% 창의적”**  
   모델·문맥에 따라 후보 집합이 달라진다.

## 13. LLM 연결 — Pretraining 이후의 체감

Pretraining이 끝난 base 모델(제68강)에 같은 프롬프트를 넣고:

```text
greedy / T=0.2 / T=0.8+top_p=0.9
```

를 비교하면, **가중치는 동일한데 사용자 체감이 달라진다.**  
이후 SFT는 분포 자체를 지시 따르기 쪽으로 옮기고, 디코딩 손잡이는 그 위를 미세 조정한다.

## 14. 핵심 정리

- Temperature는 $z/T$로 Softmax 엔트로피를 조절한다
- Top-k는 고정 개수 절단, Top-p는 누적 확률 핵 절단이다
- 적용 순서를 고정하고 재정규화·최소 1토큰을 지킨다
- 손잡이는 학습 대체가 아니며, 극단값은 붕괴를 부른다
- PPL(제67강)과 체감 품질을 잇는 다리가 디코딩 정책이다

## 15. 핵심 용어

| 용어 | 의미 |
|---|---|
| Temperature | Softmax 전 logit 스케일 $T$ |
| Top-k | 상위 $k$ 토큰만 남겨 샘플 |
| Top-p / Nucleus | 누적확률 $p$ 핵만 남겨 샘플 |
| Renormalize | 절단 후 확률 합을 1로 |
| Entropy | 분포의 불확실성 척도 |
| Logit masking | 불가 토큰에 `-inf` |
| Decoding hyperparameter | 추론 때만 쓰는 생성 파라미터 |

## 16. 연습 문제
### 문제 1 (식)

Temperature Softmax 식을 쓰고, $T\to0$일 때 거동을 설명하시오.

### 문제 2 (손계산)

섹션 7의 $\mathbf{p}$에서 $p=0.8$ Top-p 핵에 들어가는 인덱스를 구하시오.

### 문제 3 (비교)

Top-k와 Top-p의 핵심 차이 한 줄을 쓰시오.

### 문제 4 (코드)

왜 Top-p 마스크에서 “첫 토큰은 항상 유지”가 필요한가?

### 문제 5 (함정)

$T=2.0$, $top\_k=1$을 동시에 쓰면 체감은 어떤가?

### 문제 6 (연결)

이 파라미터들을 바꿔도 Validation CE(제57·67강)가 즉시 바뀌지 않는 이유는?

---

## 정답 및 해설

### 문제 1

$p_i=\exp(z_i/T)/\sum_j\exp(z_j/T)$. $T\to0$이면 최대 성분에 질량이 모여 greedy에 수렴.

### 문제 2

cum: 0.672 → 0.822 ≥ 0.8 이므로 핵 `{0,1}`.

### 문제 3

Top-k는 개수 고정 절단, Top-p는 확률 질량 기준의 가변 집합 절단.

### 문제 4

누적 조건만 쓰면 첫 구간에서도 마스크가 켜져 후보가 0개가 될 수 있다. 최소 1토큰을 남겨 NaN을 방지한다.

### 문제 5

Top-k=1이 이미 단일 토큰만 남기므로 Temperature는 거의 효과 없고 greedy와 동일.

### 문제 6

CE/PPL은 보통 teacher-forcing 정답 분포(온도 1, 절단 없음)로 측정한다. 디코딩 손잡이는 샘플 경로만 바꾼다.

## 17. 다음 강의와 연결

생성 규칙을 갖췄다. 이제 **그 분포를 맞출 연료**가 남았다.

**제60강. Pretraining Dataset 구성**에서는 코퍼스 유형, 정제·중복 제거 개념, 문서 경계를 다룬다. 제61강 Tokenization Pipeline이 그 문서를 토큰 창으로 패킹한다.

이전 강의: [제58강. Text Generation — Greedy와 Sampling](./58강_Text_Generation_Greedy와_Sampling.md)  
다음 강의: [제60강. Pretraining Dataset 구성](./60강_Pretraining_Dataset_구성.md)

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [58강. Text Generation — Greedy와 Sampling](58강_Text_Generation_Greedy와_Sampling.md)
- **다음 강:** [60강. Pretraining Dataset 구성](60강_Pretraining_Dataset_구성.md)

<!-- /LECTURE_NAV -->
