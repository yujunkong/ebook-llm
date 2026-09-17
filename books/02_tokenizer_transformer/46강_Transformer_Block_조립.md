# 제46강. Transformer Block 조립

> **학습 목표**
> - Pre-LN Block의 데이터 흐름을 그림·수식으로 쓰기
> - `x → Norm → Attn → + → Norm → FFN → +` 순서를 구현으로 옮기기
> - 왜 각 서브층이 `d_model`을 보존해야 하는지
> - Block을 $N$개 쌓아 “깊이”를 만드는 방법
> - NumPy 스케치와 PyTorch `nn.Module` 조립 코드
> - 제47~48강에서 Encoder/Decoder·Causal LM으로 확장하는 지점

---
## 1. 왜 이것을 배우는가

부품을 따로 알면 “이해한 기분”이 든다.  
실제로 한 블록을 조립해 보면 다음이 드러난다.

- Norm과 Residual의 순서
- dropout을 어디에 둘지
- causal 플래그를 Attn에만 전달하는 방법
- 위치 인코딩을 Block 밖(임베딩)에 둘지, Attn 안(RoPE)에 둘지

LLM의 “층이 32개”라는 말은 대개 **이 Block을 32번 반복**한다는 뜻이다.  
제48강 Causal LM, 제49~50강 Mini Transformer의 직접 선수과목이다.

## 2. 먼저 알아야 할 개념

체크리스트:

- [ ] MHA 입출력 shape `(B,T,C)` (제41강)
- [ ] Causal Mask (제40강)
- [ ] `y = x + F(LN(x))` Pre-LN (제44강)
- [ ] FFN이 position-wise (제45강)
- [ ] RoPE는 Q/K에 선택 적용 (제43강)

부족하면 해당 강의를 짧게 재독한다.

## 3. 핵심 개념 설명

### 3.1 Transformer Block이란?

**Transformer Block(또는 Layer)**는 Self-Attention 서브층과 FFN 서브층을 Residual·Norm과 함께 묶은 **반복 단위**다.

Pre-LN 형태(이 책의 기본):

$$

\begin{aligned}
h &= x + \mathrm{MHA}(\mathrm{LN}_1(x)) \\
y &= h + \mathrm{FFN}(\mathrm{LN}_2(h))
\end{aligned}

$$

Post-LN 형태(원 논문에 가깝게):

$$

\begin{aligned}
h &= \mathrm{LN}_1\big(x + \mathrm{MHA}(x)\big) \\
y &= \mathrm{LN}_2\big(h + \mathrm{FFN}(h)\big)
\end{aligned}

$$

우리는 **Pre-LN**을 기본으로 구현한다.  
깊은 스택에서 학습이 비교적 안정적이라는 실무 경험이 많기 때문이다(경험적 경향이며 절대 법칙은 아님).

### 3.2 왜 Pre-LN을 기본으로 하는가

설명(해석·경험):

- 서브층 입력이 정규화되어 스케일이 안정적
- Residual 본선이 끝까지 남아 표현이 덜 막힘
- 최적화(학습률·초기화)에 덜 민감한 경우가 많음

사실:

- 원 논문 서술과 그림은 Post-LN 쪽에 가깝다.
- GPT-2 이후 많은 구현이 Pre-LN을 채택했다.
- 현대 LLM은 Pre-LN + RMSNorm 조합도 흔하다.

교육용 코드는 `nn.LayerNorm`으로 두고, RMSNorm 교체 지점을 주석으로 남긴다.

### 3.3 차원이 보존되어야 하는 이유

Residual:

$$

x + F(x)

$$

이 성립하려면 $F(x)$의 마지막 차원이 $d_{\text{model}}$과 같아야 한다.

따라서:

- MHA: concat + $W^O$로 `C` 복구
- FFN: `d_ff → d_model`로 복구

이 제약이 곧 설계의 가이드레일이다.

### 3.4 위치 인코딩은 Block 안? 밖?

두 가지 패턴:

**패턴 A — Absolute PE (제42강)**

```text
tok_emb + pos_emb
  → Block × N
```

Block 자체는 위치를 모를 수 있다. 입력이 이미 위치를 담고 있다.

**패턴 B — RoPE (제43강)**

```text
tok_emb
  → Block × N
      └─ 각 Block의 MHA 안에서 Q/K 회전
```

이 책의 조립 코드는 먼저 **패턴 A로도 동작하는 Block**을 만들고,  
MHA 내부에 RoPE를 꽂을 수 있는 훅을 남긴다.

### 3.5 Dropout 위치 (최소 규칙)

교육용 최소치:

- Attention weight 뒤 (제41강)
- 각 서브층 출력 뒤, Residual에 더하기 전

```text
x + dropout(sublayer(norm(x)))
```

너무 많은 dropout은 소규모 실험에서 학습을 죽이기 쉽다.  
기본값 0.0~0.1에서 시작한다.

## 4. 직관적으로 이해하기

한 블록을 공장의 한 공정으로 본다.

```text
입력 팔레트 x
  │
  ├─ (1) 계량(Norm)
  ├─ (2) 회의(Attention: 토큰 간 통신)
  ├─ (3) 본선에 합류(Residual)
  ├─ (4) 다시 계량(Norm)
  ├─ (5) 개별 가공(FFN)
  └─ (6) 본선에 합류(Residual)
→ 다음 공정(다음 Block)으로
```

$N$번 반복하면 “깊은 추론 스택”이 된다.  
각 층이 다른 추상도에서 관계를 재구성한다는 해석이 가능하지만,  
층마다 역할을 사람이 이름 붙이는 것은 사후 해석에 가깝다.

## 5. 수학적으로 이해하기

### 5.1 한 블록의 완전 식 (Causal)

$$

\begin{aligned}
a &= \mathrm{LN}_1(x) \\
\alpha &= \mathrm{MHA}_{\mathrm{causal}}(a) \\
h &= x + \mathrm{Dropout}(\alpha) \\
b &= \mathrm{LN}_2(h) \\
\phi &= \mathrm{FFN}(b) \\
y &= h + \mathrm{Dropout}(\phi)
\end{aligned}

$$

### 5.2 스택

$$

x^{(0)} = E + P
\quad\text{(또는 RoPE 모델에서는 } x^{(0)}=E\text{)}

$$

$$

x^{(\ell)} = \mathrm{Block}^{(\ell)}\big(x^{(\ell-1)}\big),\quad \ell=1..N

$$

$$

z = \mathrm{LN}_{\mathrm{final}}\big(x^{(N)}\big)

$$

Causal LM이면 $z$에 LM Head를 곱해 logit을 만든다(제48강).

### 5.3 파라미터 스케치

한 블록 대략:

- MHA: $\sim 4 d^2$
- FFN(ratio 4): $\sim 8 d^2$
- Norm $\gamma,\beta$: $\sim 4d$ (두 개 LN)

총 $\sim 12 d^2$ 규모(편향·임베딩 제외).  
$N$층이면 $\sim 12 N d^2$.

이 숫자로 “왜 큰 모델이 메모리를 많이 쓰는지”를 가늠할 수 있다.

## 6. 작은 숫자로 흐름 따라가기

설정:

- $B=1,\ T=2,\ d_{\text{model}}=4$
- 수치는 상징적(실제 학습 값이 아님)

$$

x =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}

$$

1. $\mathrm{LN}_1(x)$: 각 행을 정규화 → $a$
2. $\mathrm{MHA}(a)$: 제41강처럼 head 분할·Causal·concat·$W^O$ → $\alpha$ (shape 동일)
3. $h = x + \alpha$
4. $\mathrm{LN}_2(h) \to b$
5. $\mathrm{FFN}(b)$: 4→16→4 (예) → $\phi$
6. $y = h + \phi$

확인 포인트는 단 하나다.

> 매 단계 후 shape가 `(1,2,4)`로 돌아오는가?

숫자 자체가 예쁘지 않아도, **shape 불변 + causal 확률**이 조립의 합격 기준이다.

## 7. 코드로 구현하기 — NumPy 조립

교육용으로만 사용한다.

```python
# transformer_block_numpy.py
"""Pre-LN Transformer Block (NumPy 스케치)."""

from __future__ import annotations

import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    B, T, C = x.shape
    D = C // n_heads
    return x.reshape(B, T, n_heads, D).transpose(0, 2, 1, 3)

def merge_heads(x: np.ndarray) -> np.ndarray:
    B, H, T, D = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)

def mha(
    x: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    n_heads: int,
    causal: bool = True,
) -> np.ndarray:
    B, T, C = x.shape
    D = C // n_heads
    q = split_heads(x @ Wq, n_heads)
    k = split_heads(x @ Wk, n_heads)
    v = split_heads(x @ Wv, n_heads)
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(D)
    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask[None, None], -1e9, scores)
    attn = softmax(scores, axis=-1)
    out = merge_heads(attn @ v) @ Wo
    return out

def ffn(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray):
    # ReLU FFN
    return np.maximum(0.0, x @ W1 + b1) @ W2 + b2

def transformer_block(
    x: np.ndarray,
    params: dict,
    n_heads: int,
    causal: bool = True,
) -> np.ndarray:
    # Pre-LN Attn
    a = layer_norm(x, params["ln1_g"], params["ln1_b"])
    h = x + mha(
        a,
        params["Wq"],
        params["Wk"],
        params["Wv"],
        params["Wo"],
        n_heads=n_heads,
        causal=causal,
    )
    # Pre-LN FFN
    b = layer_norm(h, params["ln2_g"], params["ln2_b"])
    y = h + ffn(b, params["W1"], params["b1"], params["W2"], params["b2"])
    return y

def init_params(d_model: int, d_ff: int, rng: np.random.Generator) -> dict:
    scale = 0.02
    return {
        "ln1_g": np.ones(d_model),
        "ln1_b": np.zeros(d_model),
        "ln2_g": np.ones(d_model),
        "ln2_b": np.zeros(d_model),
        "Wq": rng.normal(scale=scale, size=(d_model, d_model)),
        "Wk": rng.normal(scale=scale, size=(d_model, d_model)),
        "Wv": rng.normal(scale=scale, size=(d_model, d_model)),
        "Wo": rng.normal(scale=scale, size=(d_model, d_model)),
        "W1": rng.normal(scale=scale, size=(d_model, d_ff)),
        "b1": np.zeros(d_ff),
        "W2": rng.normal(scale=scale, size=(d_ff, d_model)),
        "b2": np.zeros(d_model),
    }

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, T, C, H = 2, 5, 8, 4
    x = rng.normal(size=(B, T, C))
    params = init_params(C, 4 * C, rng)
    y = transformer_block(x, params, n_heads=H, causal=True)
    print(y.shape)  # (2, 5, 8)
```

## 8. PyTorch로 구현하기 — 모듈 조립

```python
# transformer_block_torch.py
"""Pre-LN Transformer Block (PyTorch)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3,B,H,T,D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # RoPE를 쓴다면 여기서 q,k에 적용한다. (제43강)
        # q, k = apply_rope(q, pos), apply_rope(k, pos)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal:
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """Pre-LN: x + Attn(LN(x)), x + FFN(LN(x))."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion: int = 4,
        dropout: float = 0.0,
        # norm_cls=nn.LayerNorm  # RMSNorm으로 교체 가능 (제44강)
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, expansion=expansion, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), causal=causal))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

class MiniTransformerStack(nn.Module):
    """Block × N + final LN. Embedding은 제48강에서 연결."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, expansion, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, causal=causal)
        return self.ln_f(x)

if __name__ == "__main__":
    model = MiniTransformerStack(d_model=32, n_heads=4, n_layers=2)
    x = torch.randn(2, 8, 32)
    y = model(x, causal=True)
    print(y.shape)
    # 파라미터 수
    n = sum(p.numel() for p in model.parameters())
    print("params", n)
```

실행 확인:

- `y.shape == x.shape`
- `causal=True`일 때 학습·추론 경로가 미래 누수를 막는지(제40강 테스트 아이디어)
- `n_layers`를 늘려도 shape 불변

## 9. 조립 체크리스트 (디버깅)

구현이 안 되면 이 순서로 좁힌다.

1. **Shape**  
   블록 입출력 `(B,T,C)` 동일?

2. **약수**  
   `d_model % n_heads == 0`?

3. **Mask**  
   `scores`에 마스크가 더해지는 축이 `(T,T)`인가?

4. **Residual 경로**  
   `x + ...`에서 실수로 `ln(x)`만 앞으로 흘려보내지 않았는가?

5. **FFN 복귀**  
   마지막 Linear가 `d_model`인가?

6. **스택**  
   같은 `causal` 플래그가 모든 층에 전달되는가?

7. **Final LN**  
   스택 끝에 `ln_f`를 두었다면 LM Head 앞에 한 번 더 정규화되는가?

## 10. Encoder용 / Decoder용으로 스위치

제47강 예고:

```python
# Decoder-only GPT형
h = stack(x, causal=True)

# Encoder-only
h = stack(x, causal=False)
```

같은 `TransformerBlock`을 재사용한다.  
차이는 마스크와 (Encoder-Decoder라면) Cross-Attention 추가다.

제48강은 `causal=True` 스택 위에 Embedding과 LM Head를 얹는다.

## 11. 실제 LLM에서는 어떻게 사용하는가

사실:

- GPT형 모델은 이런 Block을 수십 층 쌓는다.
- 세부(Norm 종류, bias 유무, parallel attn+ffn 등)는 모델마다 다르다.
- 병렬형(예: Attn과 MLP를 동시에 쓰는 변형)도 존재하나, 기본 교육 모델은 직렬 Pre-LN이면 충분하다.

설명:

- “Block 그림 한 장”이 곧 LLM 논문의 반복 단위다.
- 최적화·병렬화·커널 퓨전은 5권 영역이다. 지금은 의미 구조를 고정한다.

연결 지도:

```text
제40강 Causal Mask
  → 제41강 MHA
    → 제44~45강 Norm/Residual/FFN
      → 제46강 Block   ← 지금
        → 제47강 Encoder/Decoder 지형
          → 제48강 Causal LM
```

## 12. 실습

1. `TransformerBlock` 하나에 랜덤 텐서를 통과시켜 shape를 확인하라.
2. `MiniTransformerStack(..., n_layers=4)`로 깊게 쌓아도 shape가 유지되는지 보라.
3. `causal=True`와 `False`에서 같은 입력의 첫 토큰 출력이 달라지는지 비교하라.  
   (첫 토큰은 causal에서도 과거가 없어 차이가 작을 수 있음. 마지막 토큰을 비교하는 편이 낫다.)
4. FFN expansion을 2와 4로 바꿔 파라미터 수 변화를 측정하라.
5. Residual을 주석 처리하고(`x = self.attn(...)`) 깊을 때 수치가 얼마나 불안정해지는지 관찰하라(학습 없이도 forward 분산으로 힌트를 얻을 수 있음).
6. (선택) MHA forward에 RoPE 훅을 실제로 연결해 보라.

## 13. 자주 하는 실수

1. **Post-LN 수식을 Pre-LN 코드에 혼입**  
   순서가 바뀌면 다른 모델이 된다.

2. **`x = attn(ln(x))` 후 residual 상실**  
   본선이 끊긴다.

3. **층마다 다른 `d_model`**  
   스택이 불가능하다. 폭은 공통이다.

4. **Causal를 첫 층만 적용**  
   모든 Self-Attention 층에 적용해야 Causal LM이 된다.

5. **Final LN 중복/누락**  
   설계를 하나로 고정한다.

6. **파라미터 init 스케일 무시**  
   너무 큰 초기화는 Softmax/Residual을 망가뜨린다. 작은 표준편차로 시작한다.

## 14. 핵심 정리

- Pre-LN Transformer Block =  
  `x + MHA(LN(x))` 후 `x + FFN(LN(x))`.
- MHA와 FFN은 모두 `d_model`을 보존해 Residual과 맞춘다.
- Causal Mask는 Block 안의 MHA에 전달되는 스위치다.
- Block × N이 LLM 깊이의 실체다.
- 위치 인코딩은 임베딩 add 또는 RoPE 훅으로 연결한다.
- 제48강은 이 스택 위에 LM Head를 얹은 결과다.

## 15. 핵심 용어

| 용어 | 설명 |
|---|---|
| Transformer Block / Layer | Attn+FFN+Norm+Residual 반복 단위 |
| Pre-LN | 서브층 앞 정규화 배치 |
| Stack / Depth $N$ | Block 반복 횟수 |
| Final LayerNorm | 스택 끝 정규화 |
| Residual Stream | 더하기 본선으로 흐르는 표현 |

## 16. 연습 문제
**문제 1.** Pre-LN Block의 두 줄 수식을 쓰라.

**문제 2.** 왜 MHA/FFN 출력이 `d_model`이어야 하는가?

**문제 3.** Causal LM에서 `causal` 플래그는 몇 개 층에 적용되는가?

**문제 4.** Absolute PE와 RoPE를 Block 조립 관점에서 한 줄씩 비교하라.

**문제 5.** `MiniTransformerStack`의 출력 직후(제48강 직전)에 보통 무엇이 붙는가?

**문제 6.** 원 논문 Post-LN과 이 책 Pre-LN의 순서 차이를 한 문장으로.

### 정답과 해설

1. $h=x+\mathrm{MHA}(\mathrm{LN}_1(x))$, $y=h+\mathrm{FFN}(\mathrm{LN}_2(h))$.

2. Residual 덧셈 shape를 맞추기 위해서.

3. Self-Attention이 있는 **모든** 층.

4. Absolute PE는 보통 스택 입구에서 더하고, RoPE는 각 층 MHA의 Q/K에 적용한다.

5. LM Head(선형층)로 vocabulary logit 생성. (그 앞에 final LN이 이미 있을 수 있음)

6. Post-LN은 residual 뒤에 Norm, Pre-LN은 서브층 앞에 Norm을 둔다.

## 17. 다음 강의와 연결

Block이라는 “벽돌”이 생겼다.  
제47강에서는 이 벽돌로 **Encoder 벽**을 쌓을지, **Decoder 탑**을 쌓을지, 둘을 다리(Cross-Attention)로 이을지 지형도를 그린다.

그다음 제48강에서

```text
Embedding → (Causal Block × N) → LM Head
```

로 GPT형 Causal LM 전체를 확정한다.  
제40강에서 배운 마스크가, 조립된 스택 전층에서 살아 숨 쉬는 순간이다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제45강. Feed Forward Network MLP](45강_Feed_Forward_Network_MLP.md)
- **다음 강:** [제47강. Encoder와 Decoder](47강_Encoder와_Decoder.md)

<!-- /LECTURE_NAV -->
