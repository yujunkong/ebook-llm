# 제49강. 프로젝트 — Mini Transformer 구현 (1)

> **학습 목표**
> - `ch49_mini_transformer/` 디렉터리에 config·toy tokenizer·model 스켈레톤을 둔다.
> - `MiniTransformer.forward`가 `[B,T] → [B,T,V]`를 반환함을 shape로 검증한다.
> - Causal Self-Attention / Block / LM Head가 한 파일(또는 명확히 분리된 모듈)에서 연결됨을 확인한다.
> - 제50강에서 붙일 `train.py` / `generate`의 인터페이스를 고정한다.

---
## 1. 왜 이것을 배우는가

지금까지의 강의는 부품 단위였다.

| 부품 | 강의 |
|---|---|
| Tokenizer 감각 | 27~30 |
| Embedding / LM 목표 / CE | 31~34 |
| Attention 계열 | 35~41 |
| Position / Norm / FFN / Block | 42~46 |
| Causal LM 전체 골격 | 48 |

부품만 알면 “조립할 수 있다”는 착각이 생긴다. 실제로 조립하면 드러나는 것들:

- mask shape broadcast 실수
- `[B,T,C]` vs `[B,C,T]` 혼동
- position id 길이
- vocab와 lm_head 차원 불일치

1권 제25강 Mini NN이 “학습 루프 완주”였다면, 이번 프로젝트는 **언어 모델 forward 완주**다. 제50강에서 loss가 내려가고 글자가 이어지기 시작하면, 3권 GPT Pretraining으로 가는 다리가 생긴다.

## 2. 프로젝트 개요

### 2.1 목표 (1부 + 2부)

| 단계 | 강의 | 산출물 |
|---|---|---|
| 설정·토크나이저·모델 | 49 | `config.py`, `tokenizer.py`, `model.py`, shape smoke test |
| 학습·생성 | 50 | `data.py`, `train.py`, greedy `generate`, 예상 로그 |

성공 기준(전체):

1. 아주 작은 텍스트 코퍼스에서 training loss가 유의미하게 감소한다.
2. greedy 생성이 학습 데이터 패턴을 일부 흉내 낸다(완벽한 문장일 필요 없음).
3. 파일이 `model.py` / `train.py` 스타일로 나뉘어 있다.

### 2.2 모델 규모 (의도적으로 작게)

GPU 없이도 CPU에서 수 분 안에 돌아가게 한다.

| 하이퍼파라미터 | 권장값 | 의미 |
|---|---|---|
| `vocab_size` | toy tokenizer에 따름 (수십~수백) | 문자/단어 단위 |
| `block_size` | 64 | 최대 컨텍스트 |
| `n_embd` | 64 | $C$ |
| `n_head` | 4 | 헤드 수 (`n_embd % n_head == 0`) |
| `n_layer` | 2 | 블록 수 |
| `dropout` | 0.0~0.1 | 미니는 0도 가능 |

이 숫자는 “제품”이 아니다. **배선이 살아 있는지** 확인하는 장난감이다.

### 2.3 권장 디렉터리 구조

```text
ch49_mini_transformer/
├── config.py          # 하이퍼파라미터 dataclass / 간단 객체
├── tokenizer.py       # 문자 단위 toy tokenizer
├── model.py           # Causal Self-Attention, Block, MiniTransformer
├── data.py            # (50강) 텍스트 → 토큰 배치
├── train.py           # (50강) 학습 루프 + generate
└── README.md          # 실행 메모 (선택)
```

1부에서는 위 중 `config.py`, `tokenizer.py`, `model.py`와 shape 확인용 `__main__`까지 완성한다.

## 3. 먼저 알아야 할 개념 (체크리스트)

시작 전 다음을 체크한다.

- [ ] Causal mask: 미래 위치 `-inf` (제40강)
- [ ] Multi-Head: `C`를 head로 쪼개고 다시 합침 (제41강)
- [ ] Residual + LayerNorm + FFN (제44~45강)
- [ ] CLM 골격과 `[B,T,C]` (제48강)
- [ ] `nn.Module`, `CrossEntropyLoss` (1권·제34강)

부족하면 해당 강의를 짧게 재독하고 돌아온다.

## 4. `config.py` — 설정 한곳에

```python
# ch49_mini_transformer/config.py
"""Mini Transformer 하이퍼파라미터."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass
class MiniConfig:
    vocab_size: int = 128          # tokenizer.fit 후 덮어씀
    block_size: int = 64           # T_max
    n_embd: int = 64               # C
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    bias: bool = False             # Linear bias (미니는 False로 단순화 가능)

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must divide n_head"
        assert self.block_size >= 2

if __name__ == "__main__":
    cfg = MiniConfig()
    print(cfg)
```

예상 출력 예:

```text
MiniConfig(vocab_size=128, block_size=64, n_embd=64, n_head=4, n_layer=2, dropout=0.0, bias=False)
```

## 5. `tokenizer.py` — 문자 단위 toy tokenizer

본격 BPE(제29강) 대신, 프로젝트 속도를 위해 **문자(char) 단위**를 쓴다. 영어 소형 텍스트면 vocab가 수십 개로 충분하다.

```python
# ch49_mini_transformer/tokenizer.py
"""문자 단위 장난감 Tokenizer."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

if __name__ == "__main__":
    tok = CharTokenizer.from_text("hello world")
    ids = tok.encode("hello")
    print(tok.vocab_size, ids, tok.decode(ids))
```

예상 출력 예:

```text
8 [1, 2, 3, 3, 4] hello
```

(문자 집합·ID는 정렬 결과에 따라 달라질 수 있다. **decode(encode(s))==s**가 핵심이다.)

특수 토큰(`<pad>`, `<eos>`)은 미니 단계에서는 생략해도 된다. 필요하면 제30강 개념을 가져와 확장한다.

## 6. `model.py` — 전체 골격

한 파일에 Causal Attention · Block · MiniTransformer를 모두 둔다. 나중에 파일로 쪼개도 되지만, 학습용으로는 한 파일이 추적하기 쉽다.

### 6.1 Causal Self-Attention

```python
# ch49_mini_transformer/model.py
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MiniConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MiniConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # causal mask: [1, 1, T, T] 하삼각
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        B, T, C = x.shape
        qkv = self.c_attn(x)  # [B, T, 3C]
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scores: [B, n_head, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # [B, n_head, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
```

핵심 확인:

- mask는 `:T`로 잘라 현재 길이에 맞춘다.
- `-inf` → softmax 후 0에 가까운 가중치.
- 출력 shape는 입력과 같은 `[B,T,C]`.

### 6.2 MLP (FFN)와 Block

```python
class MLP(nn.Module):
    def __init__(self, cfg: MiniConfig):
        super().__init__()
        self.fc_in = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.fc_out = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = F.gelu(x)
        x = self.fc_out(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Pre-LN Transformer Block."""

    def __init__(self, cfg: MiniConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

### 6.3 MiniTransformer (Causal LM)

```python
class MiniTransformer(nn.Module):
    def __init__(self, cfg: MiniConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # optional weight tying (제48강)
        # self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [B, T]
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"T={T} > block_size={self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # [B, T, C]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits
```

Weight tying을 켜고 싶다면 `lm_head` 정의 직후 주석을 해제한다. 1부 smoke test에서는 tying 없이도 충분하다.

## 7. Forward shape smoke test

`model.py` 하단에 다음을 둔다.

```python
if __name__ == "__main__":
    from tokenizer import CharTokenizer

    text = "mini transformer toy text for shape check. " * 20
    tok = CharTokenizer.from_text(text)
    cfg = MiniConfig(vocab_size=tok.vocab_size, n_layer=2, n_embd=64, n_head=4)
    model = MiniTransformer(cfg)

    ids = torch.tensor([tok.encode(text[:40])], dtype=torch.long)  # [1, 40]
    logits = model(ids)
    print("vocab", tok.vocab_size)
    print("idx  ", tuple(ids.shape))
    print("logits", tuple(logits.shape))
    print("params", sum(p.numel() for p in model.parameters()))
```

예상 출력 예(숫자는 환경·vocab에 따라 약간 다름):

```text
vocab 24
idx   (1, 40)
logits (1, 40, 24)
params 100000전후
```

반드시 확인할 것:

1. `logits.shape[-1] == vocab_size`
2. `logits.shape[0:2] == idx.shape`
3. `assert` 없이 forward가 끝나는가
4. `T > block_size`면 assertion이 터지는가 (일부러 테스트)

## 8. 파라미터 수 감각 (설명)

대략적인 지배항:

- Embedding: $V C$ (+ position $T_{\max} C$)
- 각 블록: Attention($O(C^2)$) + MLP($O(C^2)$) 수준
- lm_head: $C V$ (tying 시 Embedding과 공유)

미니 설정에서는 수만~수십만 파라미터면 정상이다. “작다”는 느낌이 들어야 한다. 수억 파라미터를 지금 목표로 두지 않는다.

## 9. Attention 가중치를 남기고 싶다면 (선택)

제51강 시각화를 미리 준비하고 싶다면, `CausalSelfAttention.forward`에서 softmax 직후 `att`를 속성으로 저장할 수 있다.

```python
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach()  # [B, n_head, T, T]
        att = self.attn_dropout(att)
```

필수는 아니다. 학습 성능과 무관하며, 디버깅·시각화용이다.

## 10. 1부 완료 체크리스트

- [ ] `config.py` 실행 시 dataclass가 출력된다
- [ ] `tokenizer.py`에서 encode/decode round-trip이 된다
- [ ] `model.py` smoke test가 `logits [B,T,V]`를 인쇄한다
- [ ] `n_embd % n_head != 0`이면 config에서 막는다
- [ ] causal mask 버퍼 shape를 `print(model.blocks[0].attn.mask.shape)`로 확인했다

여기까지가 제49강의 완료 조건이다. **학습 루프는 제50강**이다.

## 11. 도전 과제 (1부)

### 도전 1 — Weight tying

`lm_head.weight`와 `tok_emb.weight`가 같은 객체인지 `is`로 확인하고, 파라미터 수가 줄어드는지 비교하라.

### 도전 2 — Post-LN Block

Pre-LN 대신 Post-LN(`x = LN(x + Attn(x))`)으로 바꿔 smoke test가 통과하는지 보라. (학습 안정성은 50강에서 체감)

### 도전 3 — RoPE 미리보기

절대 position embedding을 끄고, 제43강 RoPE를 Q/K에만 적용하는 변형을 시도하라. 실패해도 좋다. 로그를 남겨 3권에서 재도전한다.

### 도전 4 — 파일 분리

`attention.py`, `block.py`, `model.py`로 쪼개고 import가 순환하지 않게 정리하라.

## 12. 자주 하는 실수

1. **`mask`를 `[T,T]`로만 두고 broadcast 실패**  
   `[1,1,T,T]` 형태가 head·batch와 잘 맞는다.

2. **`view` 전 `transpose` 후 `contiguous` 누락**  
   일부 디바이스/버전에서 `view`가 실패한다. `contiguous()`를 습관화한다.

3. **`vocab_size`를 tokenizer와 불일치**  
   `CharTokenizer.from_text` 이후에 `cfg.vocab_size = tok.vocab_size`로 맞춘다.

4. **Dropout만 켜고 eval 모드를 잊음**  
   1부는 smoke test라 괜찮지만, 50강 생성 시 `model.eval()`이 필요하다.

## 13. 핵심 정리

- Mini Transformer 1부는 config · char tokenizer · Causal LM 모델 스켈레톤이다.
- Forward 계약은 `idx[B,T] → logits[B,T,V]`다.
- Block은 Pre-LN + Causal Self-Attention + MLP의 반복이다.
- Shape smoke test가 통과해야 2부(학습)로 넘어간다.
- 파일 이름 `config.py` / `tokenizer.py` / `model.py`를 유지한다.

## 14. 핵심 용어

| 용어 | 의미 |
|---|---|
| MiniConfig | 하이퍼파라미터 묶음 |
| CharTokenizer | 문자↔ID 맵 |
| CausalSelfAttention | 마스크된 MHA |
| `register_buffer` | 학습되지 않지만 state에 포함되는 텐서(마스크 등) |
| Pre-LN | Norm을 서브레이어 앞에 두는 배치 |
| Smoke test | 짧은 실행으로 파이프라인 생존 확인 |
| `block_size` | 모델이 허용하는 최대 $T$ |

## 15. 연습 문제
### 문제 1 (shape)

`n_embd=64`, `n_head=4`일 때 `head_dim`은?

### 문제 2 (코드)

`att.masked_fill(mask == 0, -inf)`에서 mask가 0인 위치의 의미는?

### 문제 3 (개념)

왜 `lm_head`의 출력 차원이 `n_embd`가 아니라 `vocab_size`인가?

### 문제 4 (연결)

제48강의 weight tying을 적용하면 줄어드는 파라미터 항은 무엇인가?

### 문제 5 (실험)

smoke test에서 `T=70`, `block_size=64`로 넣으면 어떤 일이 나야 하는가?

---

## 정답 및 해설

### 문제 1

$64 / 4 = 16$.

### 문제 2

미래(또는 금지) 위치. Softmax 전에 $-\infty$로 두어 가중치를 0에 가깝게 만든다.

### 문제 3

각 위치에서 어휘 전체에 대한 logit이 필요하므로 마지막 축이 $V$여야 한다.

### 문제 4

`lm_head`의 $C \times V$ (공유 시 Embedding과 한 벌로 합쳐짐).

### 문제 5

`assert T <= block_size`에 걸려 에러가 나야 한다. (에러 없이 돌아하면 가드가 빠진 것)

## 16. 다음 강의와 연결

뼈대가 섰다. **제50강. 프로젝트 — Mini Transformer 구현 (2)**에서 작은 텍스트로 배치를 만들고, 학습 루프를 돌리며, greedy 생성으로 “글자가 이어지는지”를 확인한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제48강. Causal Language Model 구조](48강_Causal_Language_Model_구조.md)
- **다음 강:** [제50강. 프로젝트 Mini Transformer 구현 2](50강_프로젝트_Mini_Transformer_구현_2.md)

<!-- /LECTURE_NAV -->
