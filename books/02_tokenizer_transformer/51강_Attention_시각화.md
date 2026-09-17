# 51강. Attention 시각화
## 이번 강에서 배우는 내용

- Attention 가중치의 shape `[B, n_head, T, T]`를 해석한다.
- 작은 예제(수동 또는 학습 모델)에서 행렬을 표·히트맵으로 나타낸다.
- Causal mask가 만드는 하삼각 패턴을 그림에서 식별한다.
- 시각화의 한계(해석 과신)를 구분한다.

## 왜 중요한가?
수식만 보면 Attention은 $\mathrm{softmax}(QK^\top / \sqrt{d})V$일 뿐이다. 그러나 디버깅과 직관에는 **행렬을 보는 습관**이 도움이 된다.

- 마스크 버그: 상삼각에 큰 가중치가 보이면 위험 신호
- 특정 head가 항상 바로 이전 토큰만 보는지
- 구분 기호·반복 패턴에 주목이 모이는지

다만 시각화는 **설명 도구**이지 증명서가 아니다. “이 head가 문법 전담이다”처럼 단정하는 해석은 이 강의의 목표가 아니다.

## 선수 개념
1. Softmax Attention (제38강)
2. Causal Mask (제40강)
3. Multi-Head (제41강)
4. (권장) 제49~50강 Mini Transformer의 `last_attn` 훅

matplotlib이 있으면 히트맵이 편하다. 없어도 **숫자 표 인쇄**만으로 충분하다.

## Attention 가중치가 의미하는 것
위치 $i$(query)가 위치 $j$(key)를 얼마나 보는지:

$$

A_{i,j} = \mathrm{softmax}_j \left( \frac{q_i \cdot k_j}{\sqrt{d}} \right)

$$

행렬 $A \in \mathbb{R}^{T \times T}$에서

- **행 $i$**: “나는 누구를 보나” (합이 1)
- **열 $j$**: “누가 나를 보나” (합은 1이 아님)

Causal LM에서는 $j > i$인 곳이 0에 가깝다(마스크).

## 작은 수동 예제 — 표로 보기
토큰 4개, head 1개라고 가정하자. 마스크 적용 후 가중치가 다음과 같다고 하자(설명용 숫자).

```text
tokens:  [A, B, C, D]
         A      B      C      D
A      1.00   0.00   0.00   0.00
B      0.40   0.60   0.00   0.00
C      0.10   0.20   0.70   0.00
D      0.05   0.15   0.20   0.60
```

읽는 법:

- `C` 행은 과거 `A,B,C`만 본다. `D` 열은 0.
- `D`는 자기 자신(0.60)과 최근에 더 무게를 둔다.

이 표만으로도 “인과 마스크 + 행 softmax”가 눈에 들어온다.

## 코드 — Softmax Attention 가중치 추출
### 5.1 순수 PyTorch 미니 함수

```python
import math
import torch
import torch.nn.functional as F

def attention_weights(q, k, causal: bool = True):
    """q,k: [B, H, T, D] → weights [B, H, T, T]"""
    T = q.size(-2)
    scale = 1.0 / math.sqrt(q.size(-1))
    att = (q @ k.transpose(-2, -1)) * scale
    if causal:
        mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
    return F.softmax(att, dim=-1)

# 데모: 랜덤 Q,K
torch.manual_seed(0)
B, H, T, D = 1, 2, 5, 8
q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)
w = attention_weights(q, k, causal=True)
print(w.shape)
print(w[0, 0].numpy().round(3))
```

예상 출력 형태:

```text
torch.Size([1, 2, 5, 5])
[[1.    0.    0.    0.    0.   ]
 [0.412 0.588 0.    0.    0.   ]
 ...
]
```

상삼각(미래)이 0인지 확인한다.

### 5.2 Mini Transformer에서 꺼내기

제49강에서 저장해 둔 훅이 있다면:

```python
# model.eval() 후
logits = model(idx)
attn = model.blocks[0].attn.last_attn  # [B, H, T, T]
print(attn[0, 0])
```

훅이 없다면 `CausalSelfAttention.forward`에 한 줄 추가한다(제49강 선택 섹션).

## matplotlib 히트맵 (선택)
설치되어 있으면 다음으로 그림 파일을 남길 수 있다.

```python
# visualize_attn.py (선택)
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_attention(weights: torch.Tensor, tokens: list[str], path: str = "attn.png"):
    """weights: [T, T] (한 헤드)"""
    A = weights.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A, cmap="viridis", vmin=0.0, vmax=max(1e-6, A.max()))
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    ax.set_xlabel("key (j)")
    ax.set_ylabel("query (i)")
    ax.set_title("Attention weights")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("saved", path)

if __name__ == "__main__":
    # 가짜 하삼각 예제
    T = 6
    tokens = list("ABCDEF")
    A = torch.tril(torch.rand(T, T))
    A = A / A.sum(dim=-1, keepdim=True)
    plot_attention(A, tokens)
```

실행 후 `attn.png`를 연다. query 축이 위에서 아래로, key 축이 왼쪽에서 오른쪽인지만 확인하면 된다. 라이브러리 버전에 따라 스타일은 달라도 **하삼각 구조**가 보이면 성공이다.

matplotlib이 없다면:

```python
def print_attn(A, tokens):
    print("    ", " ".join(f"{t:>6}" for t in tokens))
    for i, row in enumerate(A):
        vals = " ".join(f"{v:6.2f}" for v in row.tolist())
        print(f"{tokens[i]:>4} {vals}")
```

## Multi-Head를 나란히 보기
헤드마다 패턴이 다를 수 있다.

```python
attn = model.blocks[0].attn.last_attn[0]  # [H, T, T]
for h in range(attn.size(0)):
    print("head", h)
    print(attn[h].detach().numpy().round(2))
```

관찰 포인트(설명 수준):

- 어떤 head는 **대각선 근처**(최근 토큰)에 무게
- 어떤 head는 **첫 토큰**에 무게가 몰림(소위 sink 현상과 유사하게 보일 수 있음)
- 층이 깊어지면 패턴이 달라질 수 있음

**사실과 해석 구분:** “첫 토큰에 무게가 보인다”는 관찰(사실·측정)이고, “따라서 이 head는 문장 경계를 담당한다”는 **가설**이다. 가설은 추가 실험 없이 단정하지 않는다.

## 학습 전후 비교 실험
제50강 체크포인트가 있다면:

1. 초기화 직후 Attention 히트맵 저장
2. 학습 후 같은 프롬프트로 다시 저장
3. 차이를 눈으로 비교

미니 데이터에서는 “언어학적으로 아름다운” 패턴보다 **마스크 준수 + 반복 구간에 무게**가 보이면 충분하다.

## 시각화의 한계
주의할 함정:

1. **가중치 ≠ 인과 기여의 전부**  
   Value·잔차·FFN이 최종 로짓에 더 크게 작용할 수 있다.
2. **한 샘플·한 층·한 head의 과해석**  
   일반화 주장으로 바로 이어지지 않는다.
3. **Softmax 희소성**  
   한 칸이 0.9면 나머지 정보가 잘 안 보인다. log scale을 쓰면 또 다른 왜곡이 생긴다.
4. **도구 의존**  
   예쁜 그림이 올바른 구현을 보장하지 않는다. 단위 테스트(상삼각 0)가 우선이다.

## 실습 체크리스트
- [ ] 랜덤 Q,K로 causal 가중치 상삼각이 0인지 확인
- [ ] 토큰 라벨이 있는 표 또는 히트맵 1장 생성
- [ ] Mini Transformer 1개 층의 head 2개 이상 인쇄
- [ ] (선택) 학습 전후 PNG 비교
- [ ] 관찰 한 줄을 “사실”과 “해석”으로 나눠 메모

## 수식 보강 — Attention map

$$
A=\mathrm{softmax}\Big(\frac{QK^{\top}}{\sqrt{d_k}}+M\Big)\in\mathbb{R}^{T\times T}
$$

시각화는 $A$의 행(쿼리)→열(키) 가중치를 히트맵으로 보는 일입니다. 행 합은 1입니다.


## 수학적으로 이해하기 — 행 Softmax와 하삼각

Causal Attention 가중치는 다음을 동시에 만족합니다.

$$

\alpha_{ij}
=
\frac{
\exp\!\big(s_{ij}\big)\,\mathbf{1}[j\le i]
}{
\sum_{j'=1}^{i}\exp\!\big(s_{ij'}\big)
}
,\qquad
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}}

$$

마스크를 $-\infty$로 넣은 뒤 Softmax를 하는 구현과, 위처럼 지시함수로 범위를 제한한 식은 **같은 분포**를 만듭니다（수치 안정화 항은 별도）.

행렬로 쓰면

$$

A = \mathrm{softmax}\big(\mathrm{mask}(QK^\top/\sqrt{d_k})\big)
\in\mathbb{R}^{T\times T},
\quad
\sum_{j=1}^{T} A_{ij}=1
\ \forall i

$$

입니다. 시각화는 바로 이 $A$의 **한 헤드 슬라이스**를 보는 일입니다.

### Multi-Head 슬라이스

$H$개 헤드가 있으면

$$

A^{(h)}\in\mathbb{R}^{T\times T},\quad h=1,\ldots,H

$$

이고, 저장 shape는 보통 `[B, H, T, T]`입니다. $B=1$일 때 메모리 대략

$$

\mathrm{bytes}(A) \approx 4\cdot H\cdot T^2
\quad(\mathrm{float32})

$$

예: $H=8$, $T=512$ → $8\cdot 512^2\cdot 4 \approx 8.4\times 10^6$ bytes ≈ **8MB/층**（가중치만, 배치 1）. 길이가 커지면 시각화용 materialize 자체가 부담이 됩니다（제52강）.

## 작은 숫자로 직접 계산하기 — 3토큰 히트맵

토큰 `A B C`, $d_k=1$（설명용）, 점수가

$$

S=\begin{bmatrix} 2 & 9 & 9 \\ 1 & 1 & 9 \\ 0 & 0 & 0 \end{bmatrix}
\quad\text{（마스크 전; 미래는 곧 차단）}

$$

라고 가정합니다. Causal mask 후:

$$

S'=\begin{bmatrix} 2 & -\infty & -\infty \\ 1 & 1 & -\infty \\ 0 & 0 & 0 \end{bmatrix}

$$

행 Softmax:

1. 행0: $\alpha=(1,0,0)$
2. 행1: $e^1/(e^1+e^1)=1/2$ → $\alpha=(0.5,0.5,0)$
3. 행2: 균등 $\alpha=(1/3,1/3,1/3)$

표:

```text
      A      B      C
A   1.00   0.00   0.00
B   0.50   0.50   0.00
C   0.33   0.33   0.33
```

히트맵에서 **상삼각이 비어 있음**이 곧 마스크 건강 신호입니다. 행2가 균등인 것은 “점수가 같아서”이지, 언어학적으로 특별하다는 뜻이 아닙니다.

## 직관적으로 이해하기 — 손전등 비유

Query 위치 $i$는 손전등입니다. Softmax 가중치는 빔의 밝기 분포입니다. Causal mask는 **미래의 벽**입니다.

```text
위치:  1    2    3    4
빔:   ■■■  ■■□  ■□□  （예: 최근 편향）
마스크: 뒤쪽(미래)은 항상 꺼짐
```

헤드가 여러 개면 손전등이 여러 개입니다. 어떤 빔은 바로 이전만, 어떤 빔은 문장 앞에 더 닿을 수 있습니다. **관찰은 OK, 기능 단정은 보류**합니다.

## 디버깅 체크리스트（시각화 우선순위）

1. **상삼각 ≈ 0?** — 아니면 mask 버그
2. **행 합 ≈ 1?** — 아니면 Softmax 축 오류
3. **NaN/Inf?** — scale·mask fill 값 점검
4. **모든 행이 one-hot 대각만?** — 학습 전에도 가능하지만, 점수 붕괴/온도 문제 의
5. **특정 열만 全程 밝음?** — sink 후보（관찰）, Value/잔차까지 보고 해석

단위 테스트 스케치:

```python
def assert_causal_attn(w, atol=1e-5):
    # w: [T, T]
    T = w.size(0)
    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    assert torch.all(w[upper] < atol)
    assert torch.allclose(w.sum(-1), torch.ones(T), atol=1e-4)
```

## 학습 전후 비교 — 정량 없이 보는 프로토콜

가짜 벤치를 만들지 않습니다. 대신 **동일 입력·동일 디코딩**으로 PNG/표를 저장합니다.

```text
artifacts/
  attn_init_block0_h0.png
  attn_step1000_block0_h0.png
  prompt.txt
```

비교 질문（예/아니오）:

- 마스크는 유지되는가?
- 반복 문자 구간으로 무게가 이동했는가?
- 특정 head만 바뀌고 나머지는 비슷한가?

“좋아졌다/나빠졌다”의 언어학 판정은 harness·생성 품질（3권）과 분리합니다.

## 수식 보강 — Attention sink를 “단정하지 않는” 정의

위치 $0$에 대한 평균 무게

$$

\bar\alpha_{:,0}
=
\frac{1}{T}\sum_{i=0}^{T-1} A_{i0}

$$

가 크면 **sink-like 패턴**이라고 부를 수 있습니다. 이는 측정이지, “이 헤드의 기능” 증명이 아닙니다. 실험 노트에는 숫자와 샘플 ID만 남깁니다.

## LLM 연결 — 서빙·긴 컨텍스트

- 긴 $T$에서는 $T\times T$를 매번 그리지 않습니다. 부분 창·특정 층만 샘플합니다.
- KV cache（제52강 미리보기）는 **가중치 히트맵과 다른 객체**입니다. 캐시는 K/V 텐서이고, 히트맵은 Softmax 결과입니다.
- 제품 설명에 “Attention이 이 단어를 봤다”를 그대로 쓰면 과해석 위험이 큽니다.

## 부록 A. ASCII 히트맵

```python
def ascii_attn(A, tokens, width=5):
    print(" " * 4 + "".join(f"{t:>{width}}" for t in tokens))
    for i, tok in enumerate(tokens):
        row = "".join(f"{A[i,j]*100:{width}.0f}" for j in range(len(tokens)))
        print(f"{tok:>4}{row}")
```

값은 퍼센트입니다. 교육용으로 matplotlib 없이도 충분합니다.

## 부록 B. Shape 치트시트

| 텐서 | Shape |
|---|---|
| Q, K, V | `[B, H, T, D]` |
| scores | `[B, H, T, T]` |
| weights $A$ | `[B, H, T, T]` |
| output | `[B, H, T, D]` → merge `[B, T, C]` |

## 부록 C. 자주 하는 시각화 실수

1. 축 라벨을 뒤집어 query/key를 오인합니다.
2. `vmin/vmax`를 고정하지 않아 학습 전후 색이 비교 불가가 됩니다.
3. 로그 스케일로 희소 패턴을 “풍부한 연결”로 오해합니다.
4. 한 샘플의 예쁜 그림으로 논문급 해석을 합니다.

## 부록 D. 연습 확장

프롬프트 `Hello!!`처럼 반복 기호가 있는 입력에서, 반복 구간 열의 평균 무게를 계산해 노트에 남기세요. 가설을 적되, 검증 실험 없이는 결론으로 승격하지 마세요.


<!-- enrich-block-51 -->
## Attention 맵을 수식으로 읽기

단일 헤드 Attention:

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

시각화에서 보는 행렬 $A$는 바로 소프트 가중치입니다.

$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)\in\mathbb{R}^{T\times T}
$$

행 $i$를 합하면 1입니다.

$$
\sum_{j=1}^{T} A_{ij} = 1
$$

### Causal Mask

미래 토큰을 $-\infty$로 막아  Softmax 후 0이 되게 합니다.

$$
\tilde{S}_{ij} =
\begin{cases}
S_{ij} & j\le i \\
-\infty & j>i
\end{cases}
$$

### 헤드 평균

멀티헤드면 헤드별 $A^{(h)}$를 평균하거나 대표 헤드만 그립니다.

$$
\bar{A} = \frac{1}{H}\sum_{h=1}^{H} A^{(h)}
$$

### 엔트로피로 “집중도” 측정

$$
H_i = -\sum_j A_{ij}\log(A_{ij}+\varepsilon)
$$

엔트로피가 낮으면 특정 키에 집중, 높으면 분산된 주의입니다.


<!-- enrich-extra-51 -->
## 실습 — Attention 가중치 추출

```python
# Causal attention 가중치 A 계산 (교육용)
import torch
import torch.nn.functional as F

B, T, d, d_k = 1, 6, 32, 8
Q = torch.randn(B, T, d_k)
K = torch.randn(B, T, d_k)
V = torch.randn(B, T, d_k)

scale = d_k ** 0.5
S = (Q @ K.transpose(-2, -1)) / scale   # (B,T,T)
# causal mask: j > i 이면 -inf
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
S = S.masked_fill(mask, float('-inf'))
A = F.softmax(S, dim=-1)
out = A @ V
print(A[0].sum(dim=-1))  # 각 행 합 ≈ 1
```

행 합이 1에 가깝지 않으면 Softmax 차원(`dim`)을 의심하세요.

### 시각화 팁

- 행 = query 위치, 열 = key 위치
- 밝기 = $A_{ij}$
- 대각 아래만 밝으면 causal이 잘 걸린 것

## LLM에서는 어디에 사용될까?

이번 51강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- Attention 시각화는 `[T,T]` 가중치에서 행=query, 열=key를 읽는 연습이다.
- Causal LM에서는 하삼각 패턴이 기본이다.
- Multi-Head는 같은 입력에도 다른 패턴을 보일 수 있다.
- 그림은 디버깅·직관용이며, 과해석을 경계한다.
- matplotlib은 선택; 표 인쇄만으로도 학습 목표는 달성한다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Attention weights | Softmax 후 $A_{i,j}$ 행렬 |
| Heatmap | 행렬 값을 색으로 표현한 그림 |
| Query axis | 행 — “누가 보는가” |
| Key axis | 열 — “누구를 보는가” |
| Attention sink | 특정 위치(종종 앞쪽)에 무게가 쏠리는 현상(관찰 용어) |
| Head diversity | 헤드마다 다른 패턴을 보이는 성질 |

## 연습문제
### 문제 1 (개념)

Attention 행렬에서 행의 합이 대략 1인 이유를 쓰시오.

### 문제 2 (디버깅)

히트맵 상삼각에 0.2 같은 값이 보인다. 무엇을 의심하는가?

### 문제 3 (shape)

`n_head=4`, `T=16`일 때 한 층의 `last_attn` shape를 쓰시오. (B=1)

### 문제 4 (해석 훈련)

“Head 2가 마침표에 강하게 반응한다”를 사실 문장과 가설 문장으로 각각 다시 쓰시오.

### 문제 5 (연결)

제52강 복잡도 논의에서, $T \times T$ 행렬을 실제로 materialize하는 것이 문제가 되는 이유를 한 문장으로 예고하시오.

---

## 정답 및 해설
### 문제 1

각 query마다 key 축에 softmax를 적용하므로 확률 분포가 되어 합이 1이다.

### 문제 2

Causal mask 누락·잘못된 broadcast·mask crop 실수.

### 문제 3

`[1, 4, 16, 16]`.

### 문제 4

사실 예: “샘플 X에서 head 2의 가중치가 마침표 열에 0.7 이상 집중되었다.” / 가설 예: “따라서 head 2는 문장 경계를 담당한다.” (후자는 미검증)

### 문제 5

길이가 커지면 $T^2$ 메모리·연산이 급증하여 긴 컨텍스트에서 병목이 된다.

## 다음 강의와 연결
그림을 봤다면 이제 **비용**을 본다. **제52강. 계산 복잡도와 메모리**에서는 Attention의 $O(T^2)$ 성격과 KV 메모리 미리보기를 다룬다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [50강. 프로젝트 — Mini Transformer 구현 (2)](50강_프로젝트_Mini_Transformer_구현_2.md)
- **다음 강:** [52강. 계산 복잡도와 메모리](52강_계산_복잡도와_메모리.md)

<!-- /LECTURE_NAV -->
