# 제48강. Causal Language Model 구조

> **학습 목표**
> - CLM의 데이터 흐름을 `token ids → embed → (+pos) → N blocks → lm_head → logits`로 설명한다.
> - 텐서 shape를 `[B, T, C]` 관례로 추적한다.
> - Weight tying(임베딩–출력 공유)이 무엇인지, 왜 쓰는지 설명한다.
> - 제49~50강 Mini Transformer 프로젝트의 “빈 칸”이 어디인지 미리 본다.

---
## 1. 왜 이것을 배우는가

GPT 계열 모델은 “마법의 블랙박스”가 아니다. 구조는 놀라울 정도로 단순하다.

```text
토큰 ID
  → 벡터로 바꾸고
    → (위치를 섞어 주고)
      → Transformer Block을 N번 통과시키고
        → 어휘 크기만큼의 점수로 투영한다
```

이 한 줄이 **Causal LM**이다. 학습 목표는 제32강에서 본 **Next Token Prediction**이고, 마스크는 제40강의 **Causal Mask**다. 이번 강의는 “블록 하나”가 아니라 **모델 전체의 배선도**를 그린다.

제49~50강에서 코드를 칠 때, shape와 모듈 경계가 흔들리면 디버깅이 지옥이 된다. 지금 배선도를 머릿속에 고정해야 한다.

## 2. 먼저 알아야 할 개념

이번 강의 전에 다음이 준비되어 있어야 한다.

1. **Embedding** — 토큰 ID → 벡터 (제31강)
2. **Next Token Prediction / Softmax / CE** — 제32~34강
3. **Causal Self-Attention · Multi-Head** — 제39~41강
4. **Positional Encoding / RoPE** — 제42~43강
5. **LayerNorm · Residual · FFN · Block** — 제44~46강
6. **Decoder 쪽 인과 생성** — 제47강

아직 학습 루프 전체는 제50강에서 다시 조립한다. 지금은 **forward 골격**이 목표다.

## 3. 핵심 개념 — Causal LM이란

**Causal Language Model**은 “지금까지 본 토큰만으로 다음 토큰을 예측”하는 언어 모델이다.

- **Causal(인과적)**: 시점 $t$의 예측이 $t$ 이후 토큰을 보지 않는다.
- **Language Model**: 토큰 서열의 결합 분포를 조건부 분포의 곱으로 분해한다.

$$

P(x_1,\ldots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})

$$

구현에서는 보통 **한 번의 forward**로 모든 위치의 다음 토큰 logit을 동시에 얻는다. 위치 $t$의 출력은 $x_{t+1}$을 예측하도록 학습한다(시프트된 타깃).

## 4. 전체 구조 다이어그램

GPT-style Decoder-only CLM의 표준 골격은 다음과 같다.

```text
idx: [B, T]           # 토큰 ID (정수)
        │
        ▼
token_embedding       # nn.Embedding(vocab, C)
        │
        ▼
(+ positional)        # learned table / sin-cos / RoPE(보통 Attention 안)
        │
        ▼
x: [B, T, C]
        │
   ┌────┴────┐
   │ Block 1 │  Causal Self-Attn + FFN (+ Norm, Residual)
   └────┬────┘
   ┌────┴────┐
   │ Block 2 │
   └────┬────┘
        …  (× N)
   ┌────┴────┐
   │ Block N │
   └────┬────┘
        │
        ▼
(final LayerNorm)     # 많은 구현에서 존재
        │
        ▼
lm_head               # Linear(C → vocab)  또는 weight-tied Embedding
        │
        ▼
logits: [B, T, V]
```

기호 관례:

| 기호 | 의미 |
|---|---|
| $B$ | 배치 크기 (batch) |
| $T$ | 시퀀스 길이 (time / context) |
| $C$ | 채널·임베딩 차원 ($d_{\text{model}}$) |
| $V$ | 어휘 크기 (vocab_size) |
| $N$ | Transformer Block 개수 (n_layer) |

이 책에서는 shape를 **`[B, T, C]`**로 통일한다. PyTorch `nn.Linear`는 마지막 축에 작용하므로, `[B, T, C]` 텐서에 Linear를 적용하면 `[B, T, out]`이 된다.

## 5. 단계별 shape 추적

작은 숫자로 한 번 따라가 보자.

- $B=2$, $T=8$, $C=64$, $V=1000$, $N=4$

```text
idx          : [2, 8]        long
tok_emb(idx) : [2, 8, 64]
+ pos        : [2, 8, 64]    (더해도 shape 동일)
block × 4    : [2, 8, 64]    (각 블록 입출력 동일)
final LN     : [2, 8, 64]
lm_head      : [2, 8, 1000]  = logits
```

학습 시 타깃은 보통 한 칸 시프트한다.

```text
입력  : t0 t1 t2 t3 t4 t5 t6 t7
타깃  : t1 t2 t3 t4 t5 t6 t7 t8   (또는 패딩/무시 위치는 ignore_index)
```

Loss는 위치마다 Cross Entropy를 구한 뒤 평균한다(제34강). Softmax는 Loss 함수 안에서 수치 안정적으로 처리하는 것이 일반적이다(`CrossEntropyLoss`는 logit을 받는다).

## 6. 모듈별 역할

### 6.1 Token Embedding

`nn.Embedding(V, C)`는 정수 ID를 $C$차원 벡터로 바꾼다.

- 입력: `[B, T]`
- 출력: `[B, T, C]`
- 파라미터 수: 대략 $V \times C$

초기에는 비슷한 토큰이 꼭 비슷한 벡터일 필요는 없다. 학습이 의미를 만든다.

### 6.2 Positional 정보

Self-Attention 자체는 순열에 대해 대칭에 가깝다(마스크를 제외하면). 따라서 **위치**를 어딘가에 넣어야 한다.

선택지(요약):

| 방식 | 넣는 위치 | 이 책에서의 위치 |
|---|---|---|
| Learned absolute PE | embed 직후 가산 | 제42강, Mini Transformer에서 자주 사용 |
| Sinusoidal PE | embed 직후 가산 | 원조 Transformer |
| RoPE | Q/K에 회전 | 제43강, 현대 LLM에서 흔함 |

Mini Transformer(제49~50강)에서는 구현 단순화를 위해 **learned absolute position embedding**을 권장한다. RoPE는 개념은 이미 배웠고, 제품형 GPT는 3권에서 다시 만난다.

### 6.3 Transformer Block × N

제46강에서 조립한 블록을 $N$번 쌓는다. 각 블록의 입출력은 모두 `[B, T, C]`다.

전형적인 Pre-LN 블록(현대 구현에 가깝다):

```text
x = x + Attn(LN(x))     # causal self-attention
x = x + FFN(LN(x))
```

Post-LN(원조)도 가능하지만, 깊은 모델에서는 Pre-LN이 학습이 안정적인 경우가 많다. Mini 모델에서는 둘 다 동작한다. **한 가지를 고르고 일관**하면 된다.

### 6.4 Final LayerNorm

많은 Decoder-only 구현은 마지막 블록 뒤에 LayerNorm을 한 번 더 둔다. 없어도 이론상 CLM은 성립하지만, 관례와 안정성 때문에 두는 편이다.

### 6.5 LM Head

`lm_head`는 `[B, T, C] → [B, T, V]`로 투영하는 선형층이다. 각 위치에서 어휘 전체에 대한 **logit**을 만든다.

생성 시에는 보통 **마지막 위치**의 logit만 샘플링/그리디 선택에 쓴다.

```text
logits[:, -1, :]  →  다음 토큰 분포
```

## 7. Weight Tying

**Weight tying(가중치 공유)**은 token embedding 행렬과 lm_head 가중치를 **같은 파라미터**로 쓰는 기법이다.

직관:

- Embedding: ID → 벡터 ($V \times C$ 테이블에서 행을 집음)
- LM Head: 벡터 → vocab 점수 (이상적으로는 “어떤 임베딩과 가까운가”)

출력 쪽을 $W_{\text{out}} \in \mathbb{R}^{V \times C}$로 두면, logit은 대략 $x W_{\text{out}}^\top$ 형태가 된다. Embedding 가중치 $W_e$와 $W_{\text{out}}$을 공유하면 파라미터 수가 줄고, “입력 공간과 출력 공간”이 같은 좌표계를 쓰게 된다.

구현 스케치:

```python
self.tok_emb = nn.Embedding(vocab_size, n_embd)
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
# weight tying
self.lm_head.weight = self.tok_emb.weight
```

주의:

- `bias=False`인 경우가 많다(공유 시 bias를 어디에 둘지 애매해진다).
- tying을 **안 해도** CLM은 완전히 성립한다. Mini Transformer에서는 선택 사항으로 둔다.
- tying은 **사실(관행·동기)**과 **효과의 크기**를 구분해야 한다. “항상 성능이 N% 오른다” 같은 수치는 이 강의에서 단정하지 않는다. 동기(파라미터·공간 정렬)만 기억한다.

## 8. 코드로 보는 골격 (의사코드)

제49강에서 채울 뼈대다. 지금은 읽기만 해도 된다.

```python
class CausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # optional: self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)          # 내부에서 causal mask
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits
```

`Block` 안에는 제41·44·45·46강 내용이 그대로 들어간다. CLM “전체”는 사실 **임베딩 + 블록 스택 + 헤드**다.

## 9. Encoder-Decoder와의 위치

제47강에서 Encoder/Decoder를 구분했다. Causal LM(GPT 계열)은 보통 **Decoder-only**다.

| 구조 | 전형적인 용도 | 마스크 |
|---|---|---|
| Encoder-only | 분류·이해 (BERT 계열) | 양방향 |
| Decoder-only (CLM) | 생성·GPT | Causal |
| Encoder-Decoder | 번역·seq2seq | Enc 양방향 + Dec cross/causal |

이 책의 주 경로(3권 GPT Pretraining)는 Decoder-only CLM이다. Encoder는 “비교 좌표계”로 남겨 둔다.

## 10. 학습과 생성에서의 같은 골격

같은 `forward`가 두 모드를 먹는다.

**학습**

```text
logits [B,T,V]  vs  targets [B,T]
→ CrossEntropy (위치 평균)
→ backward → optimizer
```

**생성 (그리디 요약)**

```text
context [1, t]
→ logits
→ next_id = argmax(logits[0, -1])
→ context에 append
→ 반복
```

제50강에서 greedy generate를 실제로 붙인다. Temperature·top-k 샘플링은 3권에서 확장한다.

## 11. 자주 하는 실수

1. **Causal mask 누락**  
   미래 토큰을 보면 “커닝”이 되어 Loss는 쉽게 내려가지만, 생성 시 붕괴한다.

2. **타깃 시프트 실수**  
   입력과 타깃을 같은 위치로 두면 모델이 “자기 자신 복사”를 배운다.

3. **position 길이 초과**  
   `T > block_size`면 learned PE 테이블을 벗어난다. 추론 시 컨텍스트 길이 제한이 여기서 온다.

4. **shape 혼동**  
   Embedding 출력은 `[B,T,C]`인데, 실수로 `[B,C,T]`로 permute한 채 Attention에 넣으면 조용히 망가진다. 인쇄하라.

5. **weight tying 후 한쪽만 초기화/재할당**  
   공유 참조가 깨지면 파라미터가 두 벌이 된다. `is`로 동일 객체인지 확인한다.

## 12. 핵심 정리

- Causal LM은 token embed (+pos) → $N$ Transformer blocks → lm_head로 구성된 Decoder-only 언어 모델이다.
- 주 shape 관례는 `[B, T, C]`이며, logit은 `[B, T, V]`다.
- Causal mask와 next-token 타깃이 “인과 언어 모델”을 만든다.
- Weight tying은 embedding과 lm_head를 공유하는 선택적 관행이다.
- 제49~50강은 이 골격을 실행 가능한 Mini Transformer로 만든다.

## 13. 핵심 용어

| 용어 | 의미 |
|---|---|
| Causal LM (CLM) | 과거만 보고 다음 토큰을 예측하는 언어 모델 |
| `n_embd` / $C$ | 모델 채널(임베딩) 차원 |
| `block_size` / $T_{\max}$ | 최대 컨텍스트 길이 |
| `n_layer` / $N$ | Transformer Block 수 |
| `lm_head` | `[B,T,C]→[B,T,V]` 출력 투영 |
| Weight tying | tok_emb와 lm_head 가중치 공유 |
| Logits | Softmax 직전의 vocab 점수 |
| Pre-LN / Post-LN | Norm과 Residual의 배치 순서 |

## 14. 연습 문제
### 문제 1 (개념)

Causal LM의 forward 파이프라인을 다섯 단계 이상으로 나열하시오.

### 문제 2 (shape)

$B=4$, $T=16$, $C=128$, $V=5000$일 때 `lm_head` 출력 shape를 쓰시오.

### 문제 3 (개념)

Weight tying을 쓰는 동기 두 가지를 쓰시오. (성능 % 수치 금지)

### 문제 4 (연결)

학습 시 위치 $t$의 logit이 예측해야 하는 타깃 토큰은 무엇인가?

### 문제 5 (디버깅)

Loss는 잘 내려가는데 생성 문장이 엉망이다. Causal mask 관점에서 의심할 버그를 설명하시오.

---

## 정답 및 해설

### 문제 1

예: token id → token embedding → (+positional) → Block×N → (final LN) → lm_head → logits.

### 문제 2

`[4, 16, 5000]`.

### 문제 3

(1) 파라미터 수 감소 ($V\times C$ 한 벌) (2) 입력·출력 토큰 공간을 같은 임베딩 좌표계로 맞추려는 동기.

### 문제 4

보통 $x_{t+1}$ (다음 토큰). 시퀀스 끝은 패딩/무시 처리하거나 EOS 등으로 설계한다.

### 문제 5

학습 때 미래 토큰을 보게 되면(마스크 버그) 커닝으로 Loss가 내려간다. 생성은 미래가 없으므로 그 능력은 쓸모없고 품질이 무너진다.

## 15. 다음 강의와 연결

배선도를 그렸다. 다음은 **조립**이다.

**제49강. 프로젝트 — Mini Transformer 구현 (1)**에서는 config, 장난감 tokenizer, 모델 스켈레톤, forward shape 검증까지를 한 번에 만든다. 제50강에서 학습과 greedy 생성을 붙인다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제47강. Encoder와 Decoder](47강_Encoder와_Decoder.md)
- **다음 강:** [제49강. 프로젝트 — Mini Transformer 구현 (1)](49강_프로젝트_Mini_Transformer_구현_1.md)

<!-- /LECTURE_NAV -->
