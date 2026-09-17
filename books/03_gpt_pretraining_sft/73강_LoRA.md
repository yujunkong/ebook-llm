# 73강. LoRA
## 이번 강에서 배우는 내용

- $\Delta W \approx BA$ 형태의 저랭크 갱신을 작은 행렬로 손으로 계산한다.
- rank $r$이 학습 파라미터 수에 어떻게 들어가는지 센다.
- 보통 어디에 LoRA를 붙이는지(예: $W_q, W_v$) 이유를 설명한다.
- 학습 후 어댑터를 병합(merge) 해 추론 오버헤드를 없애는 절차를 말한다.
- 전체 fine-tune 대비 LoRA의 트레이드오프를 한 문장으로 정리한다.

## 왜 중요한가?
실무에서 “모델을 도메인에 맞춘다”는 말이 곧 “모든 가중치를 다시 학습한다”는 뜻은 아니다.

| 방식 | 학습 대상 | 전형적인 느낌 |
|---|---|---|
| Full fine-tune | 거의 모든 $\theta$ | 표현력↑, 비용·저장↑ |
| LoRA | 작은 $A,B$만 | 비용↓, 실험 병렬↑ |
| Prompt/Prefix tuning 등 | 입력 쪽 파라미터 | 또 다른 PEFT 계열 |

LoRA를 이해하면 제74강 QLoRA(양자화된 베이스 + LoRA)와, 이후 4권에서 “정책 모델 여러 버전을 돌리는” 실험 설계가 자연스러워진다.

## 선수 개념
1. Linear 층의 $y = xW$（또는 $Wx$ — 구현 관례에 주의）
2. SFT의 next-token CE + instruction mask（제71강）
3. 행렬 곱·rank의 직관（1권 제10강）
4. Optimizer가 “학습 가능한 파라미터”에만 그라디언트를 쌓는다는 사실

## 핵심 아이디어 — 갱신을 저랭크로 둔다
사전학습된 가중치 $W_0 \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$가 있다고 하자. Full fine-tune은

$$

W = W_0 + \Delta W

$$

에서 $\Delta W$ 전체를 자유 행렬로 둔다. 파라미터 수는 $d_{\mathrm{out}} \cdot d_{\mathrm{in}}$이다.

LoRA의 가설은 다음과 같다.

> 미세조정에 필요한 $\Delta W$는 **낮은 랭크**로도 충분히 근사할 수 있다.

즉

$$

\Delta W \approx BA

$$

여기서

- $B \in \mathbb{R}^{d_{\mathrm{out}} \times r}$
- $A \in \mathbb{R}^{r \times d_{\mathrm{in}}}$
- $r \ll \min(d_{\mathrm{in}}, d_{\mathrm{out}})$

학습 시 $W_0$는 **동결(freeze)** 하고 $A,B$만 학습한다. Forward는

$$

h = x W_0 + x (BA) = x W_0 + (x A) B

$$

처럼 쓸 수 있다（행벡터 $x$ 관례）. 구현에서는 `x @ W0.T` 같은 열벡터 관례도 흔하니, **수식과 코드의 transpose를 한 번만 맞춰** 두면 된다.

## 작은 숫자로 보는 $\Delta W \approx BA$
$d_{\mathrm{in}}=4$, $d_{\mathrm{out}}=3$, $r=1$인 장난감 예를 본다.

$$

A = \begin{bmatrix} 1 & 0 & -1 & 2 \end{bmatrix}
\quad（1\times 4）

$$

$$

B = \begin{bmatrix} 0.5 \\ -1 \\ 2 \end{bmatrix}
\quad（3\times 1）

$$

그러면

$$

BA =
\begin{bmatrix}
0.5 \\ -1 \\ 2
\end{bmatrix}
\begin{bmatrix}
1 & 0 & -1 & 2
\end{bmatrix}
=
\begin{bmatrix}
0.5 & 0 & -0.5 & 1 \\
-1 & 0 & 1 & -2 \\
2 & 0 & -2 & 4
\end{bmatrix}

$$

관찰:

1. $BA$는 $3\times 4$로 **원래 $\Delta W$와 같은 shape**다.
2. 그런데 자유도는 $A$의 4개 + $B$의 3개 = **7개**뿐이다（전체 12개보다 적다）.
3. $BA$의 모든 행은 $A$의 배수다. 즉 **랭크가 최대 1**이다.

$r=2$로 올리면 행 공간이 2차원까지 넓어진다. rank가 커질수록 full $\Delta W$에 가까워지지만, **학습 파라미터도 거의 선형으로 증가**한다.

**스케일 관례:** 논문/구현에서는 종종 $\Delta W = \frac{\alpha}{r} BA$처럼 스케일을 둔다. $\alpha$는 learning rate와 비슷한 역할의 하이퍼파라미터다. 초보 실험에서는 $\alpha \approx r$로 시작해 “스케일 효과”와 “rank 효과”를 섞지 않게 하는 편이 안전하다.

## 학습 파라미터 수 — 세어 보기
한 Linear 층에 LoRA를 붙일 때:

$$

\#\mathrm{params}(A,B) = r\cdot d_{\mathrm{in}} + d_{\mathrm{out}}\cdot r = r(d_{\mathrm{in}}+d_{\mathrm{out}})

$$

전체 fine-tune:

$$

\#\mathrm{params}(\Delta W) = d_{\mathrm{in}}\cdot d_{\mathrm{out}}

$$

비율（대략）:

$$

\frac{r(d_{\mathrm{in}}+d_{\mathrm{out}})}{d_{\mathrm{in}} d_{\mathrm{out}}}

$$

### 예시（설명용 숫자）

$d_{\mathrm{in}}=d_{\mathrm{out}}=4096$, $r=8$이라고 하자.

- Full: $4096\times 4096 = 16{,}777{,}216$
- LoRA: $8\times(4096+4096)=65{,}536$
- 비율: 약 $0.39\%$（그 층만 기준）

모델 전체에 LoRA를 **모든** Linear에 다는 것은 아니다. 보통 Attention의 일부 투영만 고른다. 아래 절을 본다.

## 어디에 붙이는가
원 LoRA 논문과 후속 실무에서 자주 보는 선택:

| 모듈 | 흔한가? | 메모 |
|---|---|---|
| $W_q$, $W_v$ | 매우 흔함 | 원 논문 기본 조합 중 하나 |
| $W_k$, $W_o$ | 자주 | 표현력↑, 파라미터↑ |
| FFN (`up`/`down`/`gate`) | 선택 | 도메인 적응에 도움되는 경우 있음 |
| Embedding / lm_head | 드묾~선택 | vocab 특화 시 고려, 병합·배포 복잡도↑ |

원칙:

1. **먼저 $W_q,W_v$만**으로 베이스라인을 잡는다.
2. 데이터가 많고 목표가 어렵면 $W_o$, FFN으로 확장한다.
3. “모든 곳에 r=64”는 종종 **과한 자유도**다. SFT 데이터가 작으면 과적합으로 이어진다（제75강）.

## 구현 스케치 — 동결 + 어댑터
개념 코드（라이브러리 독립）:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """동결된 base Linear + 저랭크 어댑터."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        d_out, d_in = base.weight.shape
        self.r = r
        self.scaling = alpha / r

        # A: (r, d_in), B: (d_out, r)
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))  # 시작 시 ΔW=0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base: x @ W^T
        y = F.linear(x, self.base.weight, self.base.bias)
        # LoRA: (x @ A^T) @ B^T * scaling
        lora = (x @ self.A.T) @ self.B.T * self.scaling
        return y + lora
```

초기화 포인트:

- $B=0$이면 학습 시작 시 $\Delta W=0$이라 **사전학습 동작을 그대로** 유지한다.
- $A$는 작은 가우시안 등으로 둔다.
- Optimizer에는 `requires_grad=True`인 $A,B$만 넣는다.

```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=1e-4)
print(sum(p.numel() for p in trainable))
```

## 어댑터 병합(merge)
학습이 끝나면 추론마다 $BA$를 따로 더할 필요가 없다.

$$

W_{\mathrm{merged}} = W_0 + \frac{\alpha}{r} BA

$$

```python
@torch.no_grad()
def merge_lora_into_base(layer: LoRALinear) -> nn.Linear:
    W = layer.base.weight.data
    delta = (layer.B @ layer.A) * layer.scaling  # (d_out, d_in)
    merged = nn.Linear(W.shape[1], W.shape[0], bias=layer.base.bias is not None)
    merged.weight.copy_(W + delta)
    if layer.base.bias is not None:
        merged.bias.copy_(layer.base.bias.data)
    return merged
```

병합의 이점:

- 추론 그래프가 일반 Linear와 동일 → **지연·커널 단순**
- 배포 파일이 “베이스 + 작은 어댑터” 또는 “병합 가중치 하나”로 선택 가능

주의:

- 여러 도메인 어댑터를 **동시에** 쓰려면 병합 대신 **동적 로드**가 낫다.
- 병합 후 다시 학습하려면 어댑터 체크포인트를 별도 보관해야 한다.

## 학습·저장·실험 관점의 이득
| 항목 | Full FT | LoRA |
|---|---|---|
| Optimizer state | 거대 | 어댑터만 |
| 체크포인트 크기 | 모델 전체 | 보통 MB~소형 GB 수준（모델·r에 따름） |
| 다중 실험 | 비쌈 | 같은 베이스에 어댑터만 갈기 쉬움 |
| 표현력 상한 | 높음 | $r$에 묶임 |

**사실:** LoRA는 $\Delta W$를 랭크 $r$ 행렬곱으로 제약한다.  
**설명:** “작은 행렬이면 항상 충분하다”는 보장은 아니다. 과제가 어렵고 데이터가 풍부하면 $r$을 키우거나 full FT를 검토한다.

## SFT와의 연결
LoRA는 **손실 함수를 바꾸지 않는다**. SFT와 동일하게

- prompt(+instruction) 토큰은 loss mask
- response 토큰만 CE

를 쓰는 경우가 많다. 바뀌는 것은 **어느 파라미터가 움직이는가**뿐이다.

제76강 프로젝트에서는 Mini GPT에 작은 LoRA（또는 full SFT）를 붙여 before/after 생성을 비교한다. 미니 스케일에서는 full FT도 가능하므로, LoRA는 “습관”과 “실무 이식”을 위해 선택적으로 넣으면 된다.

## 흔한 실수
1. **베이스를 freeze하지 않음** → 사실상 full FT + 어댑터 중복
2. **$B$를 랜덤 초기화** → 시작 분포가 사전학습에서 벗어남
3. **lr을 full FT와 동일하게** → 어댑터는 종종 더 큰 lr을 씀（다만 과적합 주의）
4. **모든 모듈에 큰 $r$** → 데이터가 작을 때 style collapse（제75강）
5. **merge 후 어댑터 파일 삭제** → 재학습·A/B 실험 불가

## 수식 보강 — LoRA 업데이트

사전학습 가중치 $W_0\in\mathbb{R}^{d\times k}$를 고정하고 저랭크 적응항만 학습합니다.

$$
W = W_0 + \Delta W,\quad \Delta W = BA
$$

$$
B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\quad r\ll \min(d,k)
$$

순전파:

$$
h = W_0 x + B(Ax)
$$

학습 파라미터 수는 대략 $r(d+k)$로, 전체 $dk$보다 훨씬 작습니다.

작은 예: $d=4,k=4,r=1$, $A=[1,0,0,0]$, $B=[0.5,0,0,0]^\top$이면 $\Delta W$의 $(0,0)$만 $0.5$입니다.

### 스케일 $\alpha/r$까지

$$

\Delta W = \frac{\alpha}{r} BA
$$

$r$을 키울 때 $\alpha$를 고정하면 실효 스케일이 작아진다. $\alpha=r$로 두면 초기 실효 스케일을 맞추기 쉽다.

### 옵티마이저 상태 비교 (설명용)

파라미터 수 $P_{\mathrm{full}}=dk$, $P_{\mathrm{lora}}=r(d+k)$.  
Adam이 파라미터당 2개 모멘트를 FP32(4바이트)로 두면:

$$

M_{\mathrm{opt,full}} \approx 8 P_{\mathrm{full}},\qquad
M_{\mathrm{opt,lora}} \approx 8 P_{\mathrm{lora}}

$$

$d=k=4096$, $r=8$이면 $P_{\mathrm{lora}}/P_{\mathrm{full}}\approx 0.39\%$(한 층). 옵티마이저 메모리도 같은 비율로 줄어든다(그 층 기준).

### 그라디언트 경로

$W_0$ freeze면

$$

\frac{\partial L}{\partial W_0}=0,\quad
\frac{\partial L}{\partial A},\frac{\partial L}{\partial B}\neq 0
$$

(일반적으로). 체크포인트에 저장할 것도 $A,B$(+설정)면 충분하다.

## LLM에서는 어디에 사용될까?

이번 73강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- LoRA는 $\Delta W \approx BA$（또는 $\frac{\alpha}{r}BA$）로 저랭크 갱신만 학습한다.
- 파라미터 수는 대략 $r(d_{\mathrm{in}}+d_{\mathrm{out}})$이며, full 대비 크게 줄어든다.
- 보통 $W_q,W_v$ 등 선택 모듈에 붙인다.
- 학습 후 merge하면 추론은 일반 가중치와 동일해진다.
- 손실은 SFT와 같고, **움직이는 가중치의 집합**만 바뀐다.

## 용어 사전
| 용어 | 한 줄 의미 |
|---|---|
| LoRA | 저랭크 어댑터로 $\Delta W$를 학습하는 PEFT |
| Rank $r$ | $A,B$의 안쪽 차원; 표현력·파라미터 상한 |
| Freeze | $W_0$에 그라디언트를 막음 |
| Scaling $\alpha/r$ | 어댑터 출력 크기 조절 |
| Merge | $W_0+\Delta W$를 단일 행렬로 합침 |
| PEFT | Parameter-Efficient Fine-Tuning 총칭 |

## 연습문제
### 문제 1（숫자）

$d_{\mathrm{in}}=6$, $d_{\mathrm{out}}=4$, $r=2$일 때 LoRA 파라미터 수와 full $\Delta W$ 파라미터 수를 구하시오.

### 문제 2（행렬）

$A=\begin{bmatrix}1&0&1\end{bmatrix}$, $B=\begin{bmatrix}2\\-1\end{bmatrix}$일 때 $BA$를 계산하고 랭크를 말하시오.

### 문제 3（설계）

SFT 데이터가 500개뿐인데 Attention 전 모듈+FFN에 $r=64$를 켰다. 어떤 위험이 큰가?

### 문제 4（구현）

$B$를 zero로 초기화하는 이유를 한 문장으로 쓰시오.

### 문제 5（연결）

제74강에서 “베이스를 4-bit로 두고 LoRA만 학습”하면, 학습 가능한 파라미터는 여전히 무엇인가?

---

## 정답 및 해설
### 문제 1

LoRA: $2\times(6+4)=20$. Full: $6\times4=24$.

### 문제 2

$$

BA=\begin{bmatrix}2&0&2\\-1&0&-1\end{bmatrix}

$$

랭크 1（모든 행이 $[1,0,1]$의 배수）.

### 문제 3

어댑터 자유도가 데이터에 비해 과도해 **지시 과적합·문체 붕괴** 위험이 커진다.

### 문제 4

학습 시작 시 $\Delta W=0$이 되어 사전학습 forward를 그대로 유지하기 위함이다.

### 문제 5

여전히（주로） LoRA의 $A,B$（및 설정에 따라 일부 norm 등）이며, 양자화된 베이스 가중치 자체는 보통 동결한다.

## 다음 강의와 연결
LoRA는 “작은 행렬만 움직인다”. 그런데 베이스 모델 자체를 GPU에 올리는 비용은 여전히 크다.  
**제74강. QLoRA**에서는 베이스를 **4-bit로 올려 두고** 그 위에 LoRA를 학습하는 아이디어를 본다. NF4의 직관과, 학습·추론에서 메모리가 줄어드는 이유를 **사실/설명**으로 구분한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [72강. Chat Template과 Special Tokens](72강_Chat_Template과_Special_Tokens.md)
- **다음 강:** [74강. QLoRA](74강_QLoRA.md)

<!-- /LECTURE_NAV -->
