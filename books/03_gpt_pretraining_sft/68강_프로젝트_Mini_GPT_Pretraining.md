# 3권. GPT Pretraining과 SFT

## 제68강. 프로젝트 — Mini GPT Pretraining

### 1. 이번 강의에서 배울 것

3권 55~67강에서 쌓은 부품(GPT 골격, Causal LM 목표, 생성·디코딩, 데이터·패킹, 학습 루프, 옵티마이저, 체크포인트, Validation, Perplexity)을 **하나의 실행 가능한 Mini GPT Pretraining**으로 조립한다.

이 강의를 마치면 다음을 할 수 있어야 한다.

- `ch68_mini_gpt_pretrain/`에 `config / data / model / train / generate`를 둔다.
- 아주 작은 텍스트 코퍼스에서 next-token CE로 학습하고 **loss가 내려가는 곡선**을 확인한다.
- 주기적으로 **val loss · PPL · 샘플 생성**을 같은 로그에 남긴다.
- (선택) 단순 시퀀스 패킹으로 pad를 줄여 본다.
- 실패 시 shape / 시프트 / eval 모드 / 토크나이저를 체계적으로 디버깅한다.

이번 강의는 3권 Pretraining 구간의 **실전 관문**이다. 코드를 읽기만 하지 말고 디렉터리를 만들어 실행한다. 완벽한 문장·공개 벤치마크 점수는 목표가 아니다.

### 2. 왜 이것을 배우는가

부품 강의만으로는 “Pretraining을 돌렸다”고 말하기 어렵다. 실제로 조립하면 드러나는 것들:

- 타깃 시프트 누락 → loss가 비정상적으로 낮음
- `block_size` crop 누락 → 생성 크래시
- train/val 누수 → “좋은 PPL” 착각
- `model.train()` 상태로 샘플링 → 흔들리는 생성
- checkpoint에 tokenizer 맵 미저장 → 로드 후 난수 문자열

1권 Mini NN, 2권 Mini Transformer가 “끝까지 학습되는 파이프라인”이었다면, 이번 프로젝트는 그 파이프라인을 **GPT Pretraining 용어와 평가 습관**으로 한 번 더 완주하는 것이다.

```text
tiny corpus
  → tokenize / (optional pack)
    → batches (x, y)
      → MiniGPT forward → CE loss
        → optimizer step
          → val_loss, ppl, sample generations
            → checkpoint
```

이후 제69강부터는 같은 모델에 **Instruction / SFT**를 얹는다. 지금 Pretraining 루프가 흔들리면 SFT도 흔들린다.

### 3. 프로젝트 개요

#### 3.1 목표와 성공 기준

| 항목 | 성공 기준 |
|---|---|
| 구조 | 파일이 역할별로 분리됨 |
| 학습 | training loss가 유의미하게 감소 |
| 평가 | val_loss · PPL이 로그에 출력 |
| 생성 | greedy(또는 temperature) 샘플이 학습 패턴을 일부 흉내 |
| 재현 | seed 고정 시 초기 loss 규모가 \(\ln V\) 근처 |

성공 기준은 **공개 리더보드 점수**가 아니다. 임의의 “정확도 XX%”를 주장하지 않는다.

#### 3.2 모델 규모 (의도적으로 작게)

GPU 없이도 CPU에서 수 분~십수 분 안에 돌아가게 한다.

| 하이퍼파라미터 | 권장값 | 의미 |
|---|---|---|
| `vocab_size` | char vocab에 따름 | 보통 수십~100대 |
| `block_size` | 64~128 | 컨텍스트 길이 \(T\) |
| `n_embd` | 64~128 | 임베딩/폭 \(C\) |
| `n_head` | 4 | `n_embd % n_head == 0` |
| `n_layer` | 2~4 | Transformer 블록 수 |
| `dropout` | 0.0~0.1 | 미니는 0도 가능 |
| `batch_size` | 16~32 | CPU면 작게 |
| `lr` | 3e-4 | AdamW 출발점 |
| `max_steps` | 500~3000 | 데이터 크기에 맞게 |

이 숫자는 제품이 아니다. **배선·루프·로그가 살아 있는지** 확인하는 장난감이다.

#### 3.3 권장 디렉터리 구조

```text
ch68_mini_gpt_pretrain/
├── config.py       # 하이퍼파라미터
├── data.py         # 코퍼스 로드, 토큰화, 배치, (선택) packing
├── model.py        # MiniGPT (Causal LM)
├── train.py        # 학습 루프 + val + PPL + 샘플 로그
├── generate.py     # 체크포인트 로드 후 생성
├── input.txt       # 아주 작은 학습 텍스트
└── README.md       # 실행 메모 (선택)
```

2권 `ch49_mini_transformer`와 닮았다. 차이는 **Pretraining 운영 습관**(val/PPL/ckpt/별도 generate)을 처음부터 넣는 것이다.

### 4. 먼저 알아야 할 개념 (체크리스트)

- [ ] Causal GPT / Decoder-only (제55·56강)
- [ ] Next-token CE 목표 (제57강)
- [ ] Greedy · Temperature (제58·59강)
- [ ] Dataset · Packing 개념 (제60·61강)
- [ ] Training loop · Optimizer (제62·63강)
- [ ] Checkpoint (제65강)
- [ ] Validation (제66강)
- [ ] Perplexity = exp(mean NLL) (제67강)
- [ ] 2권 Mini Transformer 조립 경험 (제49·50강)

부족하면 해당 강을 짧게 재독하고 돌아온다.

### 5. 학습 데이터 — 아주 작은 텍스트

완벽한 위키피디아가 필요 없다. **반복 패턴이 있는 짧은 텍스트**가 미니 LM에 친절하다.

`input.txt` 예:

```text
to be or not to be that is the question
to be or not to be that is the question
all the world's a stage and all the men and women merely players
to be or not to be that is the question
mini gpt pretraining mini gpt pretraining
the quick brown fox jumps over the lazy dog
to be or not to be that is the question
```

원칙:

- 처음엔 **1~10KB**면 충분하다.
- 너무 짧으면 금방 암기하지만, 파이프라인 검증에는 유효하다.
- train/val은 **문서 단위 또는 바이트 오프셋으로 뒤쪽 일부**를 떼어 누수를 줄인다.

한글 코퍼스를 써도 된다. char tokenizer면 한글 음절이 vocab에 그대로 들어간다. 다만 CPU에서 vocab·시퀀스가 커지지 않게 짧게 유지하라.

### 6. `config.py` — 설정 한곳에

```python
# ch68_mini_gpt_pretrain/config.py
"""Mini GPT Pretraining 하이퍼파라미터."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MiniGPTConfig:
    # model
    vocab_size: int = 128          # data.fit 후 덮어씀
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    bias: bool = False

    # train
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    eval_every: int = 100
    eval_batches: int = 20
    sample_every: int = 200
    ckpt_path: str = "mini_gpt_ckpt.pt"
    seed: int = 42
    device: str = "cpu"            # "cuda" 가능하면 변경
    pack_sequences: bool = False   # 선택: 단순 패킹
    val_fraction: float = 0.1

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0
        assert self.block_size >= 2
        assert 0.0 < self.val_fraction < 0.5


if __name__ == "__main__":
    print(MiniGPTConfig())
```

예상 출력 예:

```text
MiniGPTConfig(vocab_size=128, block_size=64, n_embd=64, n_head=4, n_layer=2, dropout=0.0, bias=False, batch_size=16, lr=0.0003, weight_decay=0.1, max_steps=1000, eval_every=100, eval_batches=20, sample_every=200, ckpt_path='mini_gpt_ckpt.pt', seed=42, device='cpu', pack_sequences=False, val_fraction=0.1)
```

### 7. `data.py` — 토큰 · 배치 · (선택) 패킹

#### 7.1 Char tokenizer

```python
# ch68_mini_gpt_pretrain/data.py
"""코퍼스 로드, 문자 토크나이저, 배치 샘플러, 선택적 패킹."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def fit(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def train_val_split_ids(
    ids: list[int],
    val_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(ids)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    data = torch.tensor(ids, dtype=torch.long)
    return data[:n_train], data[n_train:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """랜덤 윈도우. x[t] → y[t] = next token."""
    # data 길이 부족 시 크래시 방지
    hi = data.size(0) - block_size - 1
    if hi < 1:
        raise ValueError("corpus too short for block_size; shorten T or lengthen text")
    ix = torch.randint(0, hi, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def pack_ids(ids: list[int], block_size: int) -> torch.Tensor:
    """단순 패킹: 1D 스트림을 block_size 배수로 잘라 이어 붙임.

    문서 경계를 무시하는 미니 버전이다. 실전에서는 EOT/문서 경계를
    넣는 편이 안전하다. 여기서는 pad를 줄이는 감각만 익힌다.
    """
    data = torch.tensor(ids, dtype=torch.long)
    n = (data.numel() // block_size) * block_size
    return data[:n]


if __name__ == "__main__":
    text = load_text("input.txt")
    tok = CharTokenizer.fit(text)
    ids = tok.encode(text)
    train_ids, val_ids = train_val_split_ids(ids, 0.1)
    x, y = get_batch(train_ids, batch_size=2, block_size=16)
    print("vocab", tok.vocab_size)
    print("x", x.shape, "y", y.shape)
    print("x0", repr(tok.decode(x[0].tolist())))
    print("y0", repr(tok.decode(y[0].tolist())))
```

`python data.py` 예상 출력 패턴(텍스트에 따라 내용은 달라짐):

```text
vocab 37
x torch.Size([2, 16]) y torch.Size([2, 16])
x0 'to be or not to b'
y0 'o be or not to be'
```

`y`가 `x`보다 한 글자 앞이면 시프트가 맞다.

#### 7.2 Packing (선택)

`pack_sequences=True`이면 학습 스트림을 `block_size` 배수로 잘라 **남는 꼬리만 버리는** 단순 패킹을 쓸 수 있다. 미니 버전은 문서 경계를 무시하므로, 긴 실전 데이터에서는 EOT 토큰을 넣는 쪽이 맞다. 여기서 목적은 “pad로 채워 평균 NLL이 왜곡되는 감각”을 줄이는 것 정도다.

### 8. `model.py` — MiniGPT

2권 Mini Transformer와 동일한 Causal LM 골격이다. 핵심 shape: `[B,T] → [B,T,V]`.

```python
# ch68_mini_gpt_pretrain/model.py
"""작은 Causal GPT."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MiniGPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MiniGPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        # causal mask buffer
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: MiniGPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: MiniGPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: MiniGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 선택: weight tying
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"T={T} > block_size={self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # [B, T, V]


@torch.no_grad()
def generate(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits = model(idx_cond)[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


if __name__ == "__main__":
    cfg = MiniGPTConfig(vocab_size=50)
    m = MiniGPT(cfg)
    x = torch.randint(0, 50, (2, 16))
    logits = m(x)
    print("logits", tuple(logits.shape))
```

예상 smoke 출력:

```text
logits (2, 16, 50)
```

### 9. `train.py` — 학습 · Validation · PPL · 샘플

```python
# ch68_mini_gpt_pretrain/train.py
"""Mini GPT Pretraining 루프."""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from config import MiniGPTConfig
from data import (
    CharTokenizer,
    get_batch,
    load_text,
    pack_ids,
    train_val_split_ids,
)
from model import MiniGPT, generate


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def estimate_loss_ppl(
    model: MiniGPT,
    data: torch.Tensor,
    cfg: MiniGPTConfig,
) -> tuple[float, float]:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, cfg.batch_size, cfg.block_size, cfg.device)
        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
        losses.append(loss.item())
    mean_nll = sum(losses) / len(losses)
    return mean_nll, math.exp(mean_nll)


def main() -> None:
    cfg = MiniGPTConfig()
    set_seed(cfg.seed)
    device = cfg.device

    text = load_text(Path(__file__).parent / "input.txt")
    tok = CharTokenizer.fit(text)
    cfg.vocab_size = tok.vocab_size
    ids = tok.encode(text)
    if cfg.pack_sequences:
        ids = pack_ids(ids, cfg.block_size).tolist()

    train_ids, val_ids = train_val_split_ids(ids, cfg.val_fraction)
    train_ids = train_ids.to(device)
    val_ids = val_ids.to(device)

    model = MiniGPT(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    print(f"vocab={cfg.vocab_size} train_tokens={train_ids.numel()} val_tokens={val_ids.numel()}")
    print(f"params={sum(p.numel() for p in model.parameters())}")

    loss_curve: list[float] = []

    for step in range(1, cfg.max_steps + 1):
        model.train()
        x, y = get_batch(train_ids, cfg.batch_size, cfg.block_size, device)
        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_curve.append(loss.item())

        if step % cfg.eval_every == 0 or step == 1:
            tr, tr_ppl = estimate_loss_ppl(model, train_ids, cfg)
            va, va_ppl = estimate_loss_ppl(model, val_ids, cfg)
            print(
                f"step {step:4d} | "
                f"train_loss {tr:.3f} ppl {tr_ppl:.2f} | "
                f"val_loss {va:.3f} ppl {va_ppl:.2f}"
            )

        if step % cfg.sample_every == 0:
            prompt = "to be"
            idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
            out = generate(model, idx, max_new_tokens=80, temperature=0.8, top_k=20)
            print("--- sample ---")
            print(tok.decode(out[0].tolist()))
            print("--------------")

    # loss curve 저장 (도전 과제·디버깅용)
    Path("losses.txt").write_text(
        "\n".join(f"{i+1}\t{v:.6f}" for i, v in enumerate(loss_curve)),
        encoding="utf-8",
    )

    ckpt = {
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "stoi": tok.stoi,
        "itos": {int(k): v for k, v in tok.itos.items()},
    }
    torch.save(ckpt, cfg.ckpt_path)
    print(f"saved {cfg.ckpt_path}")


if __name__ == "__main__":
    main()
```

### 10. `generate.py` — 학습과 분리된 생성

```python
# ch68_mini_gpt_pretrain/generate.py
"""체크포인트에서 텍스트 생성."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from config import MiniGPTConfig
from data import CharTokenizer
from model import MiniGPT, generate


@dataclass
class Loaded:
    model: MiniGPT
    tok: CharTokenizer
    device: str


def load_ckpt(path: str, device: str = "cpu") -> Loaded:
    ckpt = torch.load(path, map_location=device)
    cfg = MiniGPTConfig(**{k: v for k, v in ckpt["config"].items() if k in MiniGPTConfig.__dataclass_fields__})
    tok = CharTokenizer(stoi=ckpt["stoi"], itos={int(k): v for k, v in ckpt["itos"].items()})
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return Loaded(model=model, tok=tok, device=device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="mini_gpt_ckpt.pt")
    p.add_argument("--prompt", default="to be")
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    bundle = load_ckpt(args.ckpt, args.device)
    idx = torch.tensor([bundle.tok.encode(args.prompt)], dtype=torch.long, device=args.device)
    out = generate(
        bundle.model,
        idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(bundle.tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
```

### 11. 실행 순서와 예상 출력

#### 11.1 준비

```bash
cd ch68_mini_gpt_pretrain
# input.txt 준비
python data.py
python model.py
python train.py
python generate.py --prompt "to be" --tokens 100
```

#### 11.2 초기 loss 규모

첫 eval에서 train/val loss가 대략 \(\ln V\) 근처면 정상 신호다. 예: \(V=40\)이면 \(\ln 40 \approx 3.69\).

```text
vocab=40 train_tokens=.... val_tokens=....
params=.....
step    1 | train_loss 3.71 ppl 40.85 | val_loss 3.74 ppl 42.10
```

(숫자는 환경·시드·텍스트에 따라 달라진다. **특정 최종 PPL을 암기할 필요 없다.**)

#### 11.3 학습이 살아 있을 때

step이 진행되며 train_loss가 내려가는 것이 1차 성공이다.

```text
step  100 | train_loss 2.45 ppl 11.59 | val_loss 2.60 ppl 13.46
step  200 | train_loss 1.92 ppl  6.82 | val_loss 2.21 ppl  9.12
...
--- sample ---
to be or not to be that is the question to be or ...
--------------
```

샘플이 학습 문장을 **부분적으로** 흉내 내면 충분하다. 완벽한 문장·사실 검증은 목표가 아니다.

#### 11.4 Loss 곡선 파일

`losses.txt` 앞부분 예:

```text
1	3.812345
2	3.790112
3	3.701004
...
```

꺾은선이 전반적으로 우하향이면 루프가 살아 있다. 평탄하거나 NaN이면 §13 디버깅으로.

### 12. 한 배치 Overfit 테스트 (필수 습관)

전체 학습 전에 **고정 배치 하나**만 수백 step 학습해 loss가 매우 낮아지는지 확인한다.

```python
# overfit_one_batch.py (발췌)
x, y = get_batch(train_ids, cfg.batch_size, cfg.block_size, device)
for step in range(300):
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(step, loss.item())
```

- loss가 거의 안 내려가면: 학습률·시프트·mask·step 누락을 의심
- 한 배치만 잘 되고 전체에서 실패하면: 데이터·평가·생성 쪽을 의심

### 13. 디버깅 가이드

```text
1) data.py에서 x/y decode가 한 글자 시프트인지
2) model.py smoke: logits [B,T,V]
3) 초기 loss ≈ ln V
4) 한 배치 overfit
5) 전체 학습 loss 곡선
6) generate 시 model.eval(), crop, tokenizer 일치
```

| 증상 | 의심 |
|---|---|
| shape RuntimeError | view/transpose, vocab 불일치 |
| loss NaN | lr 과대, mask 전체 -inf |
| loss 안 내려감 | 타깃 미시프트, `opt.step` 누락 |
| 초기 loss ≪ ln V | 시프트 버그·데이터 누수 |
| val ≪ train (과도) | val 분할 실수·누수 |
| 생성 반복 Junk | 학습 부족, temperature, 데이터 과짧음 |
| 로드 후 난수 문자 | stoi/itos 미저장·불일치 |

### 14. 완성 체크리스트

- [ ] `python data.py`가 vocab·shape·시프트 decode를 인쇄한다
- [ ] `python model.py`가 `logits (B,T,V)`를 인쇄한다
- [ ] `python train.py`가 step마다 train/val loss·PPL을 남긴다
- [ ] `sample_every`마다 샘플 문자열이 출력된다
- [ ] `losses.txt`와 `mini_gpt_ckpt.pt`가 생긴다
- [ ] `python generate.py --ckpt ... --prompt ...`가 동작한다
- [ ] (선택) `pack_sequences=True`로 한 번 더 돌려 본다

### 15. 도전 과제

#### 도전 1 — Validation을 문서 단위로

바이트 뒤쪽 절단 대신, 줄 단위로 문서를 나누고 마지막 10% 줄을 val로 두어 누수를 줄여 보라.

#### 도전 2 — Learning rate schedule

제63강 cosine decay를 `max_steps`에 붙여 train/val 곡선을 비교하라. “항상 N% 향상” 같은 주장은 하지 말고, **자기 실험 로그만** 기록한다.

#### 도전 3 — Greedy vs Temperature

같은 체크포인트에서 `temperature=0.1`近似 greedy와 `0.8` 샘플을 비교하고, PPL 숫자는 동일한데 체감이 다름을 문장으로 적는다. (제67강 연결)

#### 도전 4 — Packing on/off

`pack_sequences` True/False에서 **동일 eval 스크립트**로 val NLL을 비교하라. 미니 데이터에서는 차이가 작을 수 있다. 목적은 습관이다.

#### 도전 5 — 한글 미니 코퍼스

짧은 한글 동요·문장을 `input.txt`에 넣고 같은 파이프라인을 돌려 보라. vocab 구성을 인쇄해 영문과 차이를 확인한다.

#### 도전 6 — 생성 스크립트에 Top-p

제59강 nucleus sampling을 `generate`에 추가하라. 구현이 틀리면 분포가 깨지므로, 먼저 temperature-only와 비교한다.

### 16. 이 프로젝트가 증명하는 것

끝난 뒤 다음 문장을 말할 수 있어야 한다.

> 나는 작은 텍스트 코퍼스에서 Mini GPT를 next-token CE로 학습시키고, val loss·PPL을 추적하며, 체크포인트에서 프롬프트 이어쓰기를 할 수 있다.

이것이 3권 Pretraining 구간의 실전 관문이다. 규모·데이터를 키우면 “본격 Pretraining”으로 확장되고, 다음 장부터는 **같은 가중치에 지시 따르기를 심는 SFT**로 넘어간다.

### 17. 핵심 정리

- Mini GPT Pretraining = `config` + `data` + `model` + `train` + `generate`.
- 성공 기준은 loss 감소 + PPL 로깅 + 패턴을 흉내 내는 생성이다.
- 초기 loss ≈ \(\ln V\), 한 배치 overfit으로 구현 버그를 먼저 제거한다.
- Packing은 선택이며, 미니 버전은 문서 경계를 단순화한다.
- 벤치마크 허풍 없이, **재현 가능한 미니 실험**만 남긴다.

### 18. 핵심 용어

| 용어 | 의미 |
|---|---|
| Pretraining | 대량(여기선 소량) 텍스트 next-token 학습 |
| Mini GPT | 작은 Decoder-only Causal LM |
| Packing | 시퀀스를 이어 붙여 컨텍스트를 채우는 기법 |
| Val loss / PPL | 홀드아웃 평균 NLL과 그 지수 |
| Checkpoint | 가중치 + 설정 + tokenizer 맵 |
| Overfit test | 고정 배치 암기로 구현 검증 |
| Loss curve | step별 train loss 기록 |

### 19. 복습 문제

#### 문제 1 (개념)

`y`를 `x`와 동일하게 두면 어떤 잘못된 학습이 되는가?

#### 문제 2 (평가)

왜 train loss만 내려가고 val PPL이 올라갈 때 “성공”이라고 부르기 어려운가?

#### 문제 3 (코드)

`generate`에서 `idx[:, -block_size:]`로 crop하는 이유를 한 문장으로.

#### 문제 4 (연결)

제67강에 따르면, step 로그에 PPL만 남기고 샘플을 안 뽑을 때 놓치는 정보는?

#### 문제 5 (설계)

체크포인트에 `stoi`/`itos`를 저장하지 않으면 `generate.py`에서 어떤 실패가 나는가?

---

### 정답 및 해설

#### 문제 1

다음 토큰이 아니라 “현재 토큰 복사”를 맞추게 되어 loss가 비정상적으로 쉽게 내려가거나, 언어 모델 목표와 어긋난다.

#### 문제 2

학습 집합 암기·과적합 신호일 수 있어, 홀드아웃 예측력과 생성 일반화가 보장되지 않기 때문이다.

#### 문제 3

모델이 학습·구현상 `block_size`를 넘는 컨텍스트를 받지 못하도록(또는 positional 범위를 넘지 않도록) 입력을 자르기 위함이다.

#### 문제 4

디코딩 체감(반복, 붕괴, 유창성)과 프롬프트 관련성 등 **생성 품질 축**을 놓친다. PPL은 가능도 요약일 뿐이다.

#### 문제 5

토큰 ID ↔ 문자 매핑이 사라져 올바른 디코딩이 불가능하거나, 새로 fit한 vocab과 가중치가 어긋나 난수에 가까운 문자열이 나온다.

### 20. 다음 강의와 연결

이제 “텍스트를 많이 읽어 다음 말을 예측하는” **base LM**을 미니 규모로 손에 넣었다.

그런데 ChatGPT 같은 **assistant**는 base LM과 다르다. 다음 **제69강. Instruction Tuning의 개념**에서는 왜 지시 따르기 학습이 필요한지, SFT가 Pretraining과 RLHF 사이에 어디에 앉는지부터 정리한다.

> Pretraining으로 말을 잇게 했다면, 이제 “시키는 대로” 말하게 만드는 단계로 가자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제67강. Perplexity와 생성 품질](67강_Perplexity와_생성_품질.md)
- **다음 강:** [제69강. Instruction Tuning의 개념](69강_Instruction_Tuning의_개념.md)

<!-- /LECTURE_NAV -->
