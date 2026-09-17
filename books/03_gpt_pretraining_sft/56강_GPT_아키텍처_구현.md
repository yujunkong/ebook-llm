# 56강. GPT 아키텍처 구현
## 이번 강에서 배우는 내용

- Token Embedding · (학습된) Positional Embedding · Block 스택 · final LN · `lm_head`를 한 클래스로 조립한다
- 모든 중간 텐서를 `[B, T, C]` 관례로 추적하고, 최종 logits를 `[B, T, V]`로 낸다
- 2권의 Causal Self-Attention Block을 재사용(또는 동일 인터페이스로 이식)한다
- Weight tying 여부를 코드로 선택한다
- `forward`와 생성 루프가 같은 가중치를 쓰도록 경계를 나눈다
- 제57강 Loss 연결 전에 shape smoke test를 통과한다

## 왜 중요한가?
부품을 알아도 **한 파일에 경계를 긋지 않으면** 학습 루프에서 매번 흔들린다.

```text
임베딩은 맞는데 Block 입력이 [B,C,T]
Block은 맞는데 lm_head가 C가 아니라 T에 곱해짐
Loss는 내려가는데 generate가 다른 모듈을 호출
```

실무에서도 “모델 정의”와 “학습 목표”와 “디코딩”을 분리한다. 지금 `GPT` 클래스를 단단히 만들면 제57~59강은 **손실·샘플링만** 붙이면 된다.

제48강이 배선도였다면, 이번 강의는 **납땜**이다.

## 선수 개념
체크리스트:

- [ ] Embedding: id → 벡터 (제31강)
- [ ] Causal Mask + MHA (제40~41강)
- [ ] Positional Encoding 선택지 (제42~43강)
- [ ] Pre-LN Block (제46강)
- [ ] Causal LM 골격 (제48강)
- [ ] Mini Transformer 프로젝트 경험 (제49~50강)
- [ ] GPT 정의 (제55강)

특히 Block의 입출력이  alike `[B, T, C]`인지 확인한다. Residual이 성립하려면 차원이 보존되어야 한다.

## 핵심 개념
### 3.1 이 책이 구현하는 GPT 스택

표준 골격(교육용):

```text
idx: LongTensor [B, T]
        │
        ▼
wte: Embedding(V, C)          # token embedding
wpe: Embedding(T_max, C)      # learned absolute position (교육 기본)
        │
        ▼
x = wte(idx) + wpe(pos)       # [B, T, C]
x = dropout(x)                # 선택
        │
   ┌────┴────┐
   │ Block 1 │  Pre-LN Causal Self-Attn + FFN
   └────┬────┘
        …
   ┌────┴────┐
   │ Block N │
   └────┬────┘
        │
        ▼
ln_f: LayerNorm(C)            # final norm
        │
        ▼
lm_head: Linear(C, V, bias=False)
        │
        ▼
logits: [B, T, V]
```

기호:

| 기호 | 코드 이름 예 | 의미 |
|---|---|---|
| $B$ | `batch_size` | 배치 |
| $T$ | `T` / seq len | 현재 시퀀스 길이 |
| $T_{\max}$ | `block_size` | 최대 컨텍스트 |
| $C$ | `n_embd` | 채널·임베딩 차원 |
| $V$ | `vocab_size` | 어휘 크기 |
| $N$ | `n_layer` | Block 개수 |
| $H$ | `n_head` | Attention head 수 |

### 3.2 왜 learned positional embedding을 기본으로 하나

2권에서 sin-cos와 RoPE를 배웠다. 3권 초반 기본값은 **learned absolute PE**다.

이유(교육):

- 구현이 짧다: `nn.Embedding(block_size, n_embd)`
- shape 추적이 쉽다: 토큰 임베딩과 더하기만 하면 된다
- Mini GPT Pretraining에 충분하다

RoPE를 쓰고 싶다면 Attention 내부 Q/K에 적용하면 된다(제43강). 인터페이스만 유지하면 Block 교체가 가능하다.

### 3.3 Weight tying

**Weight tying(가중치 공유)**는 token embedding 행렬과 `lm_head` 가중치를 **같은 파라미터**로 쓰는 기법이다.

- Embedding: $[V, C]$
- `lm_head`: 보통 $[V, C]$ (Linear의 `out_features × in_features`)

공유하면 출력 logit이 “임베딩 공간과의 내적”에 가까워지고, 파라미터 수에서 $V\times C$ 한 벌을 아낀다.

이 책은 config 플래그로 on/off 한다. 끈 경우 `lm_head`를 독립 `nn.Linear`로 둔다.

### 3.4 Final LayerNorm

많은 GPT-style 구현은 Block 스택 끝에 **`ln_f`**를 둔다. Pre-LN Block만으로도 학습은 되지만, 최종 표현을 한 번 더 정규화한 뒤 lm_head에 넣는 관행이 흔하다.

### 3.5 재사용할 2권 Block

제46강 Pre-LN Block:

$$

\begin{aligned}
h &= x + \mathrm{MHA}(\mathrm{LN}_1(x)) \\
y &= h + \mathrm{FFN}(\mathrm{LN}_2(h))
\end{aligned}

$$

MHA 안에 Causal Mask가 들어 있어야 한다. 마스크가 빠지면 제48·54강에서 경고한 “커닝 학습”이 재발한다.

## 직관적으로 이해하기
GPT forward를 공장 라인으로 보면:

```text
정수 ID
  → 의미 벡터(+위치)
    → “문맥 섞기” N회 (Attention+FFN)
      → 어휘 점수표
```

각 위치 $t$의 점수표(logit)는 “다음에 올 토큰”에 대한 선호다.  
학습은 그 선호를 데이터에 맞추고(제57강), 생성은 점수표에서 하나를 고른다(제58강).

중요한 직관: **생성 전용 네트워크를 따로 두지 않는다.**  
같은 `forward`의 마지막 위치 logit만 사용한다.

## 수학적으로 이해하기
입력 토큰 $x_{1:T}$. 임베딩:

$$

\mathbf{e}_t = W_{\mathrm{tok}}[x_t] + W_{\mathrm{pos}}[t]

$$

블록 스택:

$$

\mathbf{h}^{(0)}_t=\mathbf{e}_t,\quad
\mathbf{h}^{(\ell)}= \mathrm{Block}^{(\ell)}(\mathbf{h}^{(\ell-1)})

$$

최종:

$$

\mathbf{z}_t = W_{\mathrm{lm}}\,\mathrm{LN}_f(\mathbf{h}^{(N)}_t) + \mathbf{b}
\quad(\mathbf{b}\text{는 종종 }0)

$$

$\mathbf{z}_t\in\mathbb{R}^{V}$가 logits다. Softmax는 Loss/디코딩 단계에서 적용한다.


$$
Weight tying 시 $W_{\mathrm{lm}}=W_{\mathrm{tok}}$ (전치 규약은 구현에 맞게).
$$

## 작은 숫자로 shape 추적하기
설정:

- $B=2$, $T=5$, $C=8$, $V=20$, $N=2$, $T_{\max}=8$, heads $=2$

```text
idx            [2, 5]
wte(idx)       [2, 5, 8]
wpe(pos)       [2, 5, 8]
sum + drop     [2, 5, 8]
block1         [2, 5, 8]
block2         [2, 5, 8]
ln_f           [2, 5, 8]
lm_head        [2, 5, 20]   ← logits
```

학습 시 타깃은 보통 한 칸 시프트(제57강):

```text
입력 위치:  0 1 2 3 4
예측 대상:  1 2 3 4 5(또는 EOS)
```

생성 시에는 보통 **마지막 위치** `logits[:, -1, :]`만 쓴다 → shape `[B, V]`.

## 코드로 구현하기 — Config
하이퍼파라미터를 흩뿌리지 말고 dataclass/dict로 묶는다.

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 1000
    block_size: int = 64      # T_max
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128        # C
    dropout: float = 0.0
    bias: bool = False       # Linear/LN bias (교육용 단순화)
    tie_weights: bool = True
```

제약이 있으면 `__post_init__`에서 검사한다.

```python
def __post_init__(self):
    assert self.n_embd % self.n_head == 0, "n_embd must divide n_head"
    assert self.block_size >= 1
```

## PyTorch로 구현하기
아래는 **교육용 스케치**다. 2권에서 만든 `CausalSelfAttention` / `MLP` / `Block`이 있다면 import 해서 채워 넣는다. 여기서는 인터페이스가 보이게 인라인으로 요약한다.

### 8.1 Attention · MLP · Block (요약 재사용)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        # causal mask: [1, 1, T, T]
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.split(self.n_embd, dim=2)
        # [B, nh, T, hd]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nh, T, T]
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # [B, nh, T, hd]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Pre-LN Transformer Block (2권과 동일 계열)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

### 8.2 GPT 본체

```python
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            # embedding weight와 lm_head weight 공유
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """
        idx: [B, T] long
        returns logits: [B, T, V]
        """
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.config.block_size}"
            )

        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # [1, T]
        x = self.wte(idx) + self.wpe(pos)  # broadcast → [B, T, C]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
```

### 8.3 파라미터 수 대략 계산

대략식(bias·LN 무시 시 감각용):

$$

\#\approx V C + T_{\max} C + N\left(4C^2 + 2C\cdot 4C\right) + C V

$$

Attention의 QKV·proj가 $4C^2$, MLP가 $8C^2$ 근사. Weight tying이면 마지막 $CV$를 제거.

```python
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

cfg = GPTConfig(vocab_size=1000, block_size=64, n_layer=4, n_head=4, n_embd=128)
model = GPT(cfg)
print(count_params(model))
```

숫자를 “사실인 규모”로 외우지 말고, **어느 항이 지배적인지** 본다. Vocab가 크면 $VC$가, 깊이가 깊으면 $N C^2$가 커진다.

### 8.4 Shape smoke test

```python
def smoke_test():
    cfg = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (3, 11))  # B=3, T=11
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (3, 11, cfg.vocab_size)
    print("ok", logits.shape)

smoke_test()
```

이 테스트가 통과하기 전에 Loss를 붙이지 않는다.

### 8.5 생성 훅의 자리만 마련

완전한 샘플링은 제58~59강이다. 여기서는 **같은 forward**를 쓰는 자리만 표시한다.

```python
@torch.no_grad()
def generate_greedy(model: GPT, idx, max_new_tokens: int):
    """idx: [B, T] → append greedy tokens."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits = model(idx_cond)          # [B, T', V]
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
        idx = torch.cat([idx, next_id], dim=1)
    return idx
```

## NumPy로 골격만 보기
프레임워크 없이 차원만 확인하는 스케치:

```python
import numpy as np

B, T, C, V, N = 2, 5, 8, 20, 2
idx = np.random.randint(0, V, size=(B, T))
# fake embeddings
Wte = np.random.randn(V, C) * 0.02
Wpe = np.random.randn(16, C) * 0.02
x = Wte[idx] + Wpe[np.arange(T)]  # [B, T, C]
# pretend blocks preserve shape
for _ in range(N):
    x = x + 0.0  # placeholder
# final linear
Wlm = np.random.randn(C, V) * 0.02
logits = x @ Wlm  # [B, T, V]
assert logits.shape == (B, T, V)
```

Attention 내부를 NumPy로 다시 칠 필요는 없다. 이미 2권에서 했다.

## LLM에서는 어디에 사용될까?
대규모 구현과의 대응:

| 이 책 | 흔한 실무 변형 |
|---|---|
| `nn.LayerNorm` | RMSNorm |
| learned `wpe` | RoPE (절대 테이블 없음) |
| MHA | MQA / GQA |
| dense Attention | FlashAttention 등 |
| `nn.ModuleList` Block | 동일 + pipeline/tensor parallel |
| 단일 `lm_head` | 동일 개념 (병렬 분할 가능) |

이름이 달라도 **입출력 계약**은 같다: 토큰 ID in, vocab logit out.

체크포인트 저장 시:

```python
torch.save({"model": model.state_dict(), "config": cfg.__dict__}, "ckpt.pt")
```

`config`를 같이 저장해야 나중에 구조를 복원할 수 있다.

## 실습
### 실습 1 — shape 표 채우기

$B=4$, $T=32$, $C=256$, $V=8000$, $N=6$일 때 다음 shape를 쓰시오.

1. `wte` 출력  
2. Block 통과 후  
3. `lm_head` 출력  
4. 생성 시 마지막 logit 슬라이스

### 실습 2 — tie_weights on/off

`tie_weights=True/False`로 모델을 두 개 만들고 `count_params` 차이를 확인한다. 차이가 대략 $V\times C$인지 본다.

### 실습 3 — block_size 초과

`T = block_size + 1`인 입력을 넣어 `ValueError`가 나는지 확인한다. 생성 루프에서 `idx_cond = idx[:, -block_size:]`가 왜 필요한지 주석으로 남긴다.

### 실습 4 — mask 버그 주입(의도적)

연습용으로 mask 줄을 주석 처리한 뒤, 같은 배치에서 Loss를 비교해 본다(제57강 이후). 지금은 “어디에 마스크가 있는지”만 표시해도 된다.

### 실습 5 — 2권 코드 이식

제46·48·50강에서 만든 Block이 있으면 `from ... import Block`으로 교체하고 smoke test를 다시 돌린다.

## 자주 하는 실수
1. **Causal mask 크기와 T 불일치**  
   버퍼는 `block_size`로 만들고 `[:T, :T]`로 잘라 쓴다. 현재 $T$로만 만들면 생성 중 길이가 변할 때 재생성 비용·버그가 생긴다.

2. **position id를 배치마다 잘못 생성**  
   `arange(T)`는 보통 0..T-1이다. 절대 위치가 아니라 “창 안 상대 위치”만 쓰면 PE 의미가 달라진다(창 슬라이드 시 주의).

3. **weight tying 후 Embedding을 재할당**  
   `self.wte = nn.Embedding(...)`를 다시 만들면 공유가 깨진다. `is`로 `lm_head.weight is wte.weight`를 확인한다.

4. **`[B, C, T]`로 permute한 채 방치**  
   Conv 습관이 남아 있으면 Linear/Attention이 조용히 틀린다. 인쇄 습관을 들인다.

5. **dropout을 eval에서 켜 둔 채 생성**  
   `model.eval()`을 빼면 생성 결과가 샘플마다 불필요하게 흔들린다.

6. **lm_head에 bias를 넣고 tying과 혼용**  
   공유 행렬과 bias 정책이 문서와 어긋나기 쉽다. 교육 기본은 `bias=False`.

7. **Block 수·head 수를 config와 모듈이 다르게 읽음**  
   생성자에서 `config`만 신뢰하고 매직넘버를 쓰지 않는다.

## 설계 선택 메모
교육용 기본값과 교체 지점:

| 항목 | 기본 | 교체 |
|---|---|---|
| Position | learned absolute | RoPE in Attn |
| Norm | LayerNorm | RMSNorm |
| Activation | GELU | SiLU 등 |
| Bias | False | True |
| Tying | True | False |
| Attn impl | manual matmul | SDPA / Flash |

교체해도 `forward(idx)->logits` 계약을 유지하면 제57강 이후 코드가 깨지지 않는다.

## LLM 연결 — Pretraining으로 가는 소켓
이 `GPT`는 제68강 Mini Pretraining 프로젝트의 모델 슬롯이다.

```text
Dataset → (x, y) → GPT.forward(x) → CE(y) → backward
                 ↘ generate_* (평가·샘플)
```

지금은 forward가 단단하면 충분하다. 데이터 파이프라인은 제60~61강, 목표는 제57강.

## 핵심 요약
- GPT 구현 = Embedding(+pos) + Block×N + final LN + lm_head
- 주 shape: `[B,T] → [B,T,C] → [B,T,V]`
- Block은 2권 Pre-LN Causal Block을 재사용한다
- Weight tying은 embedding과 lm_head 공유 옵션이다
- 생성은 별도 모델이 아니라 같은 logits의 마지막 위치를 사용한다
- smoke test 없이 Loss/데이터로 넘어가지 않는다

## 용어 사전
| 용어 | 의미 |
|---|---|
| `n_embd` / $C$ | 모델 채널 차원 |
| `block_size` | 최대 컨텍스트 길이 $T_{\max}$ |
| `wte` / `wpe` | token / position embedding |
| `ln_f` | 스택 끝 LayerNorm |
| `lm_head` | $C\to V$ 출력 투영 |
| Weight tying | `wte.weight`와 `lm_head.weight` 공유 |
| Smoke test | 학습 전 shape·실행 검증 |
| Pre-LN Block | Norm 후 Attn/FFN, Residual 본선 유지 |

## 연습문제
### 문제 1 (파이프라인)

GPT forward의 모듈 순서를 일곱 단계 이내로 쓰시오.

### 문제 2 (shape)

$B=1$, $T=64$, $C=512$, $V=32000$일 때 logits shape는?

### 문제 3 (개념)

Weight tying의 동기 두 가지를 쓰시오.

### 문제 4 (디버깅)

생성 중 `T`가 `block_size`를 넘어 ValueError가 난다. `generate`에서 고칠 한 줄을 설명하시오.

### 문제 5 (연결)

왜 `generate`가 `lm_head`를 직접 호출하지 않고 `model(idx)`를 호출해야 하는가?

### 문제 6 (2권)

Causal mask를 Attention 점수의 어디에, 어떤 값으로 넣는가?

---

## 정답 및 해설
### 문제 1

예: idx → wte → +wpe → dropout → Block×N → ln_f → lm_head → logits.

### 문제 2

`[1, 64, 32000]`.

### 문제 3

파라미터 수 감소; 입력·출력 토큰을 같은 임베딩 공간으로 맞춤.

### 문제 4

`idx_cond = idx[:, -block_size:]`처럼 컨텍스트를 잘라 모델에 넣는다.

### 문제 5

Embedding·Block·ln_f를 포함한 전체 표현이 필요하기 때문. `lm_head`만 호출하면 문맥 변환이 빠진다.

### 문제 6

Softmax 전 점수 행렬의 미래 위치($j>i$)에 `-inf`를 넣어 확률 0에 가깝게 만든다.

## 다음 강의와 연결
골격이 생겼다. 다음은 **무엇을 최대화하는가**다.

**제57강. Causal LM Training 목표**에서는 $\log p(x_t\mid x_{<t})$를 Cross Entropy로 올리는 법, 라벨 시프트, ignore_index, 작은 숫자 손계산까지 다룬다. 오늘 만든 `logits [B,T,V]`가 Loss의 입력이다.

이전 강의: [제55강. GPT란 무엇인가](./55강_GPT란_무엇인가.md)  
다음 강의: [제57강. Causal LM Training 목표](./57강_Causal_LM_Training_목표.md)

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [55강. GPT란 무엇인가](55강_GPT란_무엇인가.md)
- **다음 강:** [57강. Causal LM Training 목표](57강_Causal_LM_Training_목표.md)

<!-- /LECTURE_NAV -->
