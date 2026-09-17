# 제51강. Attention 시각화

> **학습 목표**
> - Attention 가중치의 shape `[B, n_head, T, T]`를 해석한다.
> - 작은 예제(수동 또는 학습 모델)에서 행렬을 표·히트맵으로 나타낸다.
> - Causal mask가 만드는 하삼각 패턴을 그림에서 식별한다.
> - 시각화의 한계(해석 과신)를 구분한다.

---
## 1. 왜 이것을 배우는가

수식만 보면 Attention은 $\mathrm{softmax}(QK^\top / \sqrt{d})V$일 뿐이다. 그러나 디버깅과 직관에는 **행렬을 보는 습관**이 도움이 된다.

- 마스크 버그: 상삼각에 큰 가중치가 보이면 위험 신호
- 특정 head가 항상 바로 이전 토큰만 보는지
- 구분 기호·반복 패턴에 주목이 모이는지

다만 시각화는 **설명 도구**이지 증명서가 아니다. “이 head가 문법 전담이다”처럼 단정하는 해석은 이 강의의 목표가 아니다.

## 2. 먼저 알아야 할 개념

1. Softmax Attention (제38강)
2. Causal Mask (제40강)
3. Multi-Head (제41강)
4. (권장) 제49~50강 Mini Transformer의 `last_attn` 훅

matplotlib이 있으면 히트맵이 편하다. 없어도 **숫자 표 인쇄**만으로 충분하다.

## 3. Attention 가중치가 의미하는 것

위치 $i$(query)가 위치 $j$(key)를 얼마나 보는지:

$$

A_{i,j} = \mathrm{softmax}_j \left( \frac{q_i \cdot k_j}{\sqrt{d}} \right)

$$

행렬 $A \in \mathbb{R}^{T \times T}$에서

- **행 $i$**: “나는 누구를 보나” (합이 1)
- **열 $j$**: “누가 나를 보나” (합은 1이 아님)

Causal LM에서는 $j > i$인 곳이 0에 가깝다(마스크).

## 4. 작은 수동 예제 — 표로 보기

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

## 5. 코드 — Softmax Attention 가중치 추출

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

## 6. matplotlib 히트맵 (선택)

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

## 7. Multi-Head를 나란히 보기

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

## 8. 학습 전후 비교 실험

제50강 체크포인트가 있다면:

1. 초기화 직후 Attention 히트맵 저장
2. 학습 후 같은 프롬프트로 다시 저장
3. 차이를 눈으로 비교

미니 데이터에서는 “언어학적으로 아름다운” 패턴보다 **마스크 준수 + 반복 구간에 무게**가 보이면 충분하다.

## 9. 시각화의 한계

주의할 함정:

1. **가중치 ≠ 인과 기여의 전부**  
   Value·잔차·FFN이 최종 로짓에 더 크게 작용할 수 있다.
2. **한 샘플·한 층·한 head의 과해석**  
   일반화 주장으로 바로 이어지지 않는다.
3. **Softmax 희소성**  
   한 칸이 0.9면 나머지 정보가 잘 안 보인다. log scale을 쓰면 또 다른 왜곡이 생긴다.
4. **도구 의존**  
   예쁜 그림이 올바른 구현을 보장하지 않는다. 단위 테스트(상삼각 0)가 우선이다.

## 10. 실습 체크리스트

- [ ] 랜덤 Q,K로 causal 가중치 상삼각이 0인지 확인
- [ ] 토큰 라벨이 있는 표 또는 히트맵 1장 생성
- [ ] Mini Transformer 1개 층의 head 2개 이상 인쇄
- [ ] (선택) 학습 전후 PNG 비교
- [ ] 관찰 한 줄을 “사실”과 “해석”으로 나눠 메모

## 11. 핵심 정리

- Attention 시각화는 `[T,T]` 가중치에서 행=query, 열=key를 읽는 연습이다.
- Causal LM에서는 하삼각 패턴이 기본이다.
- Multi-Head는 같은 입력에도 다른 패턴을 보일 수 있다.
- 그림은 디버깅·직관용이며, 과해석을 경계한다.
- matplotlib은 선택; 표 인쇄만으로도 학습 목표는 달성한다.

## 12. 핵심 용어

| 용어 | 의미 |
|---|---|
| Attention weights | Softmax 후 $A_{i,j}$ 행렬 |
| Heatmap | 행렬 값을 색으로 표현한 그림 |
| Query axis | 행 — “누가 보는가” |
| Key axis | 열 — “누구를 보는가” |
| Attention sink | 특정 위치(종종 앞쪽)에 무게가 쏠리는 현상(관찰 용어) |
| Head diversity | 헤드마다 다른 패턴을 보이는 성질 |

## 13. 연습 문제
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

## 14. 다음 강의와 연결

그림을 봤다면 이제 **비용**을 본다. **제52강. 계산 복잡도와 메모리**에서는 Attention의 $O(T^2)$ 성격과 KV 메모리 미리보기를 다룬다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제50강. 프로젝트 — Mini Transformer 구현 (2)](50강_프로젝트_Mini_Transformer_구현_2.md)
- **다음 강:** [제52강. 계산 복잡도와 메모리](52강_계산_복잡도와_메모리.md)

<!-- /LECTURE_NAV -->
