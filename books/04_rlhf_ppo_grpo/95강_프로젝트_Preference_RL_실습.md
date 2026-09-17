# 95강. 프로젝트 — Preference / RL 실습
## 이번 강에서 배우는 내용

- `ch95_preference_rl/`에 선호 데이터·토이 LM·DPO 경로·（선택）RM+REINFORCE 경로를 둔다
- 초소형 preference JSONL로 chosen이 rejected보다 높은  implicit reward（DPO）또는 RM 점수를 확인한다
- 학습 전·후 메트릭을 `artifacts/`에 남긴다
- 제96강에서 다룰 실패 모드를 미니 스케일에서 일부러 관찰할 여지를 남긴다

## 왜 중요한가?
문서의 DPO 수식과 PPO 다이어그램은 매끄럽다. 손구현은 지저분하다.

| 읽기만 할 때 | 미니 구현에서 드러나는 것 |
|---|---|
| “chosen을 올린다” | 토큰 마스크·길이 정규화 |
| “rejected를 내린다” | 배치 평균·수치 안정 |
| “RM이 선호를 근사” | 라벨 노이즈에 과적합 |
| “REINFORCE로 정책 개선” | 보상 스케일·baseline |

제76강 Mini SFT가 “response mask”를 몸에 익혔듯, 이번 강은 **preference margin**을 몸에 익힌다.

## 두 경로（하나만 완료해도 성공）
```text
경로 A（권장 기본）:  Preference JSONL → DPO loss → before/after margin
경로 B（확장）:       Preference JSONL → Reward Model → REINFORCE on toy policy
```

| | 경로 A DPO | 경로 B RM+REINFORCE |
|---|---|---|
| 필요 모델 | policy (+ frozen ref) | RM + policy |
| 보상 모델 | 없음（암시적） | 명시적 스칼라 |
| 난이도 | 중 | 중상 |
| 4권 연결 | 90~91 | 85~86, 82 |

성공 기준은 **경로 A 완료**다. 경로 B는 시간이 남으면 켠다.

## 프로젝트 목표와 성공 기준
### 3.1 목표

1. 장난감 preference 20~40쌍 작성（train/eval 분리）
2. 초소형 Causal LM（임베딩+GRU 또는 tiny Transformer）준비
3. （선택）짧은 SFT 워밍으로 언어가 완전히 붕괴하지 않게 함
4. DPO 학습 루프 + 평가 스크립트
5. `artifacts/report.md`에 숫자·해석 기록

### 3.2 성공 기준

- [ ] `python train_dpo.py`가 에러 없이 끝난다
- [ ] train에서 DPO loss가 전반적으로 감소하는 로그가 있다
- [ ] eval에서 **mean margin** $ \log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x) $ 가 학습 전보다 커진다
- [ ]（경로 B）RM이 held-out 쌍에서 accuracy > 무작위（0.5）보다 높다
- [ ] `artifacts/`에 ckpt와 report가 남는다

“문장이 유창하다”는 필수 기준이 아니다. toy vocab에서는 유창성보다 **선호 방향**이 우선이다.

## 권장 디렉터리
```text
ch95_preference_rl/
├── README.md
├── config.py
├── toy_tokenizer.py
├── toy_lm.py
├── data_pref.py
├── train_sft_warm.py      # 선택: 워밍
├── train_dpo.py           # 경로 A
├── train_rm.py            # 경로 B
├── train_reinforce.py     # 경로 B
├── evaluate.py
├── data/
│   ├── pref_train.jsonl
│   └── pref_eval.jsonl
└── artifacts/
    ├── tok.json
    ├── ckpt_warm.pt
    ├── ckpt_dpo.pt
    ├── ckpt_rm.pt
    ├── metrics_dpo.json
    └── report.md
```

외부 거대 체크포인트는 쓰지 않는다. CPU에서 수분 안에 끝나도록 규모를  фикси한다.

## 초소형 Preference 데이터
### 5.1 형식

한 줄 JSON:

```json
{"id": "p01", "prompt": "Q: 1+1=\nA:", "chosen": " 2", "rejected": " 3"}
{"id": "p02", "prompt": "감정: 좋다 →", "chosen": " 긍정", "rejected": " 부정"}
{"id": "p03", "prompt": "영어로 cat:", "chosen": " cat", "rejected": " dog"}
```

규칙:

1. **짧은** prompt/응답（토이 토크나이저·짧은 context）
2. chosen/rejected 차이가 **한두 토큰**으로도 드러나게
3. train과 eval은 paraphrase（동일 문장 복사 금지）
4. 위험·혐오 문장은 넣지 않는다

### 5.2 샘플 `pref_train.jsonl`（발췌）

실습 시 파일에 20줄 이상 넣는다. 아래는 패턴 예시 8쌍이다.

```json
{"id": "t01", "prompt": "Q: 2+2=\nA:", "chosen": " 4", "rejected": " 5"}
{"id": "t02", "prompt": "Q: 3+1=\nA:", "chosen": " 4", "rejected": " 2"}
{"id": "t03", "prompt": "색: 하늘 →", "chosen": " 파랑", "rejected": " 빨강"}
{"id": "t04", "prompt": "색: 피 →", "chosen": " 빨강", "rejected": " 파랑"}
{"id": "t05", "prompt": "예/아니오: 2는 짝수?", "chosen": " 예", "rejected": " 아니오"}
{"id": "t06", "prompt": "예/아니오: 3은 짝수?", "chosen": " 아니오", "rejected": " 예"}
{"id": "t07", "prompt": "번역: hi →", "chosen": " 안녕", "rejected": " 사과"}
{"id": "t08", "prompt": "번역: apple →", "chosen": " 사과", "rejected": " 안녕"}
```

eval은 숫자·색·예아니오·번역을 **다른 표면 문장**으로 4~8쌍.

## `config.py`
```python
# ch95_preference_rl/config.py
from dataclasses import dataclass

@dataclass
class CFG:
    seed: int = 7
    data_train: str = "data/pref_train.jsonl"
    data_eval: str = "data/pref_eval.jsonl"
    artifacts: str = "artifacts"
    vocab_min_freq: int = 1
    d_model: int = 64
    n_layers: int = 1  # GRU layers
    max_len: int = 48
    batch_size: int = 8
    warm_steps: int = 200
    dpo_steps: int = 400
    rm_steps: int = 300
    rl_steps: int = 200
    lr: float = 3e-3
    beta: float = 0.1          # DPO β
    max_new_tokens: int = 4
    device: str = "cpu"
```

규모를 더 줄여도 된다. loss가 NaN이면 `lr`을 낮춘다.

## 토이 토크나이저
공백 없는 한글을 위해 **문자 단위**가 안전하다. 영어·숫자는 문자/기호 단위로 묶어도 된다.

```python
# ch95_preference_rl/toy_tokenizer.py
"""Character-level tokenizer for the ch95 toy preference project."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

class CharTokenizer:
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, stoi: dict[str, int]):
        self.stoi = dict(stoi)
        self.itos = {i: s for s, i in self.stoi.items()}

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 1) -> "CharTokenizer":
        cnt: Counter[str] = Counter()
        for t in texts:
            cnt.update(list(t))
        specials = [cls.PAD, cls.BOS, cls.EOS, cls.UNK]
        vocab = specials + sorted([c for c, n in cnt.items() if n >= min_freq and c not in specials])
        stoi = {s: i for i, s in enumerate(vocab)}
        return cls(stoi)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.UNK]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(c, self.unk_id) for c in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            s = self.itos.get(int(i), self.UNK)
            if s in (self.PAD, self.BOS, self.EOS):
                continue
            if s == self.UNK:
                out.append("?")
            else:
                out.append(s)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.stoi, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        stoi = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(stoi)
```

## 토이 LM
설명용으로 **Embedding + GRU + Linear**면 충분하다.（Transformer로 바꿔도 인터페이스만 같으면 된다.）

```python
# ch95_preference_rl/toy_lm.py
"""Tiny causal LM for preference / DPO / REINFORCE demos."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T] -> logits [B, T, V]
        x = self.emb(input_ids)
        h, _ = self.rnn(x)
        return self.head(h)

    def nll_on_continuation(
        self,
        prompt_ids: torch.Tensor,
        cont_ids: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        """Per-example mean NLL on continuation tokens (prompt context included).

        prompt_ids: [B, Tp], cont_ids: [B, Tc] (already padded).
        Returns: [B] average NLL over non-pad continuation positions.
        """
        B, Tp = prompt_ids.shape
        Tc = cont_ids.size(1)
        full = torch.cat([prompt_ids, cont_ids], dim=1)  # [B, Tp+Tc]
        logits = self.forward(full)  # [B, Tp+Tc, V]
        # predict cont tokens at positions Tp-1 .. Tp+Tc-2  → labels cont
        pred = logits[:, Tp - 1 : Tp + Tc - 1, :]  # [B, Tc, V]
        logp = F.log_softmax(pred, dim=-1)
        gather = logp.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)  # [B, Tc]
        mask = (cont_ids != pad_id).float()
        # avoid div0
        denom = mask.sum(dim=1).clamp_min(1.0)
        return -(gather * mask).sum(dim=1) / denom
```

평균 NLL（또는 평균 logprob）을 쓰는 이유: 길이 차만으로 margin이 지배되지 않게 하기 위함이다. 합（sum）logprob를 쓰는 변형도 있으며, report에 **어느 쪽인지** 명시한다.

## 데이터 로더
```python
# ch95_preference_rl/data_pref.py
"""Load preference JSONL and build padded batches."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

class PrefDataset(Dataset):
    def __init__(self, path: str | Path, tok, max_len: int = 48):
        self.rows = load_jsonl(path)
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        return {
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }

def _pad(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    T = max(len(s) for s in seqs)
    out = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out

def collate_pref(batch: list[dict], tok, max_len: int) -> dict[str, torch.Tensor | list]:
    prompts, chosens, rejecteds = [], [], []
    for ex in batch:
        p = tok.encode(ex["prompt"])[-max_len:]
        c = tok.encode(ex["chosen"], add_eos=True)[:max_len]
        rj = tok.encode(ex["rejected"], add_eos=True)[:max_len]
        # prompt must be at least length 1 for indexing Tp-1
        if len(p) == 0:
            p = [tok.bos_id]
        prompts.append(p)
        chosens.append(c if c else [tok.eos_id])
        rejecteds.append(rj if rj else [tok.eos_id])
    return {
        "prompt_ids": _pad(prompts, tok.pad_id),
        "chosen_ids": _pad(chosens, tok.pad_id),
        "rejected_ids": _pad(rejecteds, tok.pad_id),
        "prompts": [b["prompt"] for b in batch],
    }
```

## 경로 A — DPO
### 10.1 손실（복습을 코드에 고정）

참조 정책 $\pi_{\mathrm{ref}}$（보통 SFT/워밍 스냅샷）와 학습 정책 $\pi_\theta$에 대해:

$$

L_{\mathrm{DPO}}
=
-\mathbb{E}\log\sigma(
\beta\big[
(\log\pi_\theta(y_w|x)-\log\pi_{\mathrm{ref}}(y_w|x))
-
(\log\pi_\theta(y_l|x)-\log\pi_{\mathrm{ref}}(y_l|x))
\big]
)

$$

여기서는 구현 편의를 위해 **평균 토큰 logprob**를 $\log\pi$ 자리에 넣는다. 논문의 토큰 합과 스케일이 다르므로, $\beta$는 toy에 맞게 다시 고른다.

### 10.2 `train_dpo.py`

```python
# ch95_preference_rl/train_dpo.py
"""Toy DPO training loop on character-level preference data."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from data_pref import PrefDataset, collate_pref, load_jsonl
from toy_lm import ToyLM
from toy_tokenizer import CharTokenizer

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

def build_tokenizer() -> CharTokenizer:
    rows = load_jsonl(CFG.data_train) + load_jsonl(CFG.data_eval)
    texts = []
    for r in rows:
        texts += [r["prompt"], r["chosen"], r["rejected"]]
    tok = CharTokenizer.build(texts, min_freq=CFG.vocab_min_freq)
    Path(CFG.artifacts).mkdir(parents=True, exist_ok=True)
    tok.save(Path(CFG.artifacts) / "tok.json")
    return tok

def dpo_loss(
    model: ToyLM,
    ref: ToyLM,
    batch: dict,
    pad_id: int,
    beta: float,
) -> torch.Tensor:
    p, cw, cl = batch["prompt_ids"], batch["chosen_ids"], batch["rejected_ids"]
    # nll = - mean logpi  →  mean logpi = -nll
    with torch.no_grad():
        logp_ref_w = -ref.nll_on_continuation(p, cw, pad_id)
        logp_ref_l = -ref.nll_on_continuation(p, cl, pad_id)
    logp_w = -model.nll_on_continuation(p, cw, pad_id)
    logp_l = -model.nll_on_continuation(p, cl, pad_id)
    logits = beta * ((logp_w - logp_ref_w) - (logp_l - logp_ref_l))
    return -F.logsigmoid(logits).mean()

@torch.no_grad()
def mean_margin(model: ToyLM, loader: DataLoader, pad_id: int) -> float:
    margins = []
    for batch in loader:
        p, cw, cl = batch["prompt_ids"], batch["chosen_ids"], batch["rejected_ids"]
        logp_w = -model.nll_on_continuation(p, cw, pad_id)
        logp_l = -model.nll_on_continuation(p, cl, pad_id)
        margins.append((logp_w - logp_l).mean().item())
    return sum(margins) / max(len(margins), 1)

def main() -> None:
    set_seed(CFG.seed)
    tok = build_tokenizer()
    model = ToyLM(len(tok.stoi), CFG.d_model, CFG.n_layers)
    warm = Path(CFG.artifacts) / "ckpt_warm.pt"
    if warm.exists():
        model.load_state_dict(torch.load(warm, map_location="cpu"))
        print("loaded warm", warm)
    ref = copy.deepcopy(model)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    ds_tr = PrefDataset(CFG.data_train, tok, CFG.max_len)
    ds_ev = PrefDataset(CFG.data_eval, tok, CFG.max_len)
    collate = lambda b: collate_pref(b, tok, CFG.max_len)
    ld_tr = DataLoader(ds_tr, batch_size=CFG.batch_size, shuffle=True, collate_fn=collate)
    ld_ev = DataLoader(ds_ev, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate)

    before = mean_margin(model, ld_ev, tok.pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    model.train()
    step = 0
    logs = []
    while step < CFG.dpo_steps:
        for batch in ld_tr:
            loss = dpo_loss(model, ref, batch, tok.pad_id, CFG.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 20 == 0:
                print(f"step={step} dpo_loss={loss.item():.4f}")
                logs.append({"step": step, "loss": float(loss.item())})
            if step >= CFG.dpo_steps:
                break

    after = mean_margin(model, ld_ev, tok.pad_id)
    ckpt = Path(CFG.artifacts) / "ckpt_dpo.pt"
    torch.save(model.state_dict(), ckpt)
    metrics = {
        "eval_margin_before": before,
        "eval_margin_after": after,
        "beta": CFG.beta,
        "steps": CFG.dpo_steps,
        "logprob": "mean_token",
    }
    (Path(CFG.artifacts) / "metrics_dpo.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("saved", ckpt)
    print(metrics)

if __name__ == "__main__":
    main()
```

### 10.3 기대 출력（형태）

정확한 숫자는 시드·데이터에 따라 달라진다. **형태**만 고정한다.

```text
step=20 dpo_loss=0.6xxx
step=40 dpo_loss=0.5xxx
...
saved artifacts/ckpt_dpo.pt
{'eval_margin_before': -0.02..., 'eval_margin_after': 0.15..., ...}
```

성공 판정: `eval_margin_after > eval_margin_before`.  
실패 시 §16 챌린지를 본다.

## （선택）워밍 SFT
preference만으로 처음부터 돌리면 toy LM이 노이즈에 가깝다. chosen 응답에 짧은 CE 워밍을 넣을 수 있다.

```python
# ch95_preference_rl/train_sft_warm.py
"""Optional CE warm-up on chosen responses only."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from data_pref import PrefDataset, collate_pref
from toy_lm import ToyLM
from toy_tokenizer import CharTokenizer
from train_dpo import build_tokenizer, set_seed

def ce_on_chosen(model: ToyLM, batch: dict, pad_id: int) -> torch.Tensor:
    # reuse mean NLL as loss
    return model.nll_on_continuation(batch["prompt_ids"], batch["chosen_ids"], pad_id).mean()

def main() -> None:
    set_seed(CFG.seed)
    tok = build_tokenizer()
    model = ToyLM(len(tok.stoi), CFG.d_model, CFG.n_layers)
    ds = PrefDataset(CFG.data_train, tok, CFG.max_len)
    ld = DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pref(b, tok, CFG.max_len),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    model.train()
    step = 0
    while step < CFG.warm_steps:
        for batch in ld:
            loss = ce_on_chosen(model, batch, tok.pad_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"warm step={step} loss={loss.item():.4f}")
            if step >= CFG.warm_steps:
                break
    path = Path(CFG.artifacts) / "ckpt_warm.pt"
    torch.save(model.state_dict(), path)
    print("saved", path)

if __name__ == "__main__":
    main()
```

권장 실행 순서:

```bash
cd ch95_preference_rl
python train_sft_warm.py
python train_dpo.py
python evaluate.py
```

## 경로 B — Reward Model + REINFORCE
### 12.1 RM

프롬프트+응답을 이어 붙여 스칼라 점수를 낸다. BT 손실:

$$

L_{\mathrm{RM}} = -\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)

$$

```python
# ch95_preference_rl/train_rm.py
"""Bradley-Terry reward model on toy preferences."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from data_pref import PrefDataset, collate_pref
from toy_tokenizer import CharTokenizer
from train_dpo import build_tokenizer, set_seed

class ToyRM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.score = nn.Linear(d_model, 1)

    def forward_score(self, prompt_ids: torch.Tensor, resp_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        full = torch.cat([prompt_ids, resp_ids], dim=1)
        x = self.emb(full)
        h, _ = self.rnn(x)
        # last non-pad hidden
        mask = (full != pad_id)
        lengths = mask.sum(dim=1).clamp_min(1) - 1
        last = h[torch.arange(h.size(0)), lengths]
        return self.score(last).squeeze(-1)  # [B]

def main() -> None:
    set_seed(CFG.seed)
    tok = CharTokenizer.load(Path(CFG.artifacts) / "tok.json") if (
        Path(CFG.artifacts) / "tok.json"
    ).exists() else build_tokenizer()
    rm = ToyRM(len(tok.stoi), CFG.d_model)
    ds_tr = PrefDataset(CFG.data_train, tok, CFG.max_len)
    ds_ev = PrefDataset(CFG.data_eval, tok, CFG.max_len)
    collate = lambda b: collate_pref(b, tok, CFG.max_len)
    ld_tr = DataLoader(ds_tr, batch_size=CFG.batch_size, shuffle=True, collate_fn=collate)
    ld_ev = DataLoader(ds_ev, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(rm.parameters(), lr=CFG.lr)
    rm.train()
    step = 0
    while step < CFG.rm_steps:
        for batch in ld_tr:
            rw = rm.forward_score(batch["prompt_ids"], batch["chosen_ids"], tok.pad_id)
            rl = rm.forward_score(batch["prompt_ids"], batch["rejected_ids"], tok.pad_id)
            loss = -F.logsigmoid(rw - rl).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"rm step={step} loss={loss.item():.4f}")
            if step >= CFG.rm_steps:
                break

    # eval accuracy
    rm.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in ld_ev:
            rw = rm.forward_score(batch["prompt_ids"], batch["chosen_ids"], tok.pad_id)
            rl = rm.forward_score(batch["prompt_ids"], batch["rejected_ids"], tok.pad_id)
            correct += (rw > rl).sum().item()
            total += rw.numel()
    acc = correct / max(total, 1)
    path = Path(CFG.artifacts) / "ckpt_rm.pt"
    torch.save(rm.state_dict(), path)
    (Path(CFG.artifacts) / "metrics_rm.json").write_text(
        json.dumps({"eval_pref_acc": acc, "steps": CFG.rm_steps}, indent=2),
        encoding="utf-8",
    )
    print("saved", path, "eval_pref_acc", acc)

if __name__ == "__main__":
    main()
```

기대: `eval_pref_acc`가 0.5를 안정적으로 넘는 것. 데이터 10쌍 미만이면 변동이 크다.

### 12.2 REINFORCE（초미니）

프롬프트에서 짧게 샘플링 → RM 점수 → $(r-b)\nabla\log\pi$.

```python
# ch95_preference_rl/train_reinforce.py
"""REINFORCE with frozen ToyRM scores (educational, high variance)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from data_pref import PrefDataset, collate_pref
from toy_lm import ToyLM
from toy_tokenizer import CharTokenizer
from train_rm import ToyRM

@torch.no_grad()
def sample(model: ToyLM, prompt_ids: torch.Tensor, pad_id: int, max_new: int) -> torch.Tensor:
    B = prompt_ids.size(0)
    cur = prompt_ids
    gens = []
    for _ in range(max_new):
        logits = model(cur)[:, -1, :]
        dist = torch.distributions.Categorical(logits=logits)
        tok = dist.sample()
        gens.append(tok)
        cur = torch.cat([cur, tok.unsqueeze(1)], dim=1)
    return torch.stack(gens, dim=1)  # [B, max_new]

def continuation_logprob_sum(
    model: ToyLM, prompt_ids: torch.Tensor, cont_ids: torch.Tensor
) -> torch.Tensor:
    Tp = prompt_ids.size(1)
    full = torch.cat([prompt_ids, cont_ids], dim=1)
    logits = model(full)
    pred = logits[:, Tp - 1 : Tp + cont_ids.size(1) - 1, :]
    logp = F.log_softmax(pred, dim=-1)
    token_lp = logp.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum(dim=1)  # [B]

def main() -> None:
    tok = CharTokenizer.load(Path(CFG.artifacts) / "tok.json")
    model = ToyLM(len(tok.stoi), CFG.d_model, CFG.n_layers)
    warm = Path(CFG.artifacts) / "ckpt_warm.pt"
    if warm.exists():
        model.load_state_dict(torch.load(warm, map_location="cpu"))
    rm = ToyRM(len(tok.stoi), CFG.d_model)
    rm.load_state_dict(torch.load(Path(CFG.artifacts) / "ckpt_rm.pt", map_location="cpu"))
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)

    ds = PrefDataset(CFG.data_train, tok, CFG.max_len)
    ld = DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pref(b, tok, CFG.max_len),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr * 0.5)
    model.train()
    step = 0
    while step < CFG.rl_steps:
        for batch in ld:
            prompts = batch["prompt_ids"]
            cont = sample(model, prompts, tok.pad_id, CFG.max_new_tokens)
            with torch.no_grad():
                r = rm.forward_score(prompts, cont, tok.pad_id)
                b = r.mean()
                adv = r - b
            logπ = continuation_logprob_sum(model, prompts, cont)
            loss = -(adv.detach() * logπ).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 20 == 0:
                print(f"rl step={step} loss={loss.item():.4f} mean_r={r.mean().item():.4f}")
            if step >= CFG.rl_steps:
                break
    path = Path(CFG.artifacts) / "ckpt_rl.pt"
    torch.save(model.state_dict(), path)
    print("saved", path)

if __name__ == "__main__":
    main()
```

경로 B의 성공은 “생성 문장이 아름다움”이 아니라 **RM이 선호하는 쪽 토큰이 샘플에 더 자주 등장**하는 조짐이다. 분산이 커서 실패하기 쉽다 — 그 실패 자체가 학습 목표다.

## `evaluate.py`
```python
# ch95_preference_rl/evaluate.py
"""Summarize DPO margins and write artifacts/report.md."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CFG
from data_pref import PrefDataset, collate_pref
from toy_lm import ToyLM
from toy_tokenizer import CharTokenizer
from train_dpo import mean_margin

def main() -> None:
    art = Path(CFG.artifacts)
    tok = CharTokenizer.load(art / "tok.json")
    ds = PrefDataset(CFG.data_eval, tok, CFG.max_len)
    ld = DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_pref(b, tok, CFG.max_len),
    )

    def load_model(name: str) -> ToyLM | None:
        p = art / name
        if not p.exists():
            return None
        m = ToyLM(len(tok.stoi), CFG.d_model, CFG.n_layers)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        return m

    lines = ["# ch95 preference RL report", ""]
    metrics_path = art / "metrics_dpo.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        lines.append("## DPO metrics")
        lines.append("```json")
        lines.append(json.dumps(metrics, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    for label, fname in [("warm", "ckpt_warm.pt"), ("dpo", "ckpt_dpo.pt")]:
        m = load_model(fname)
        if m is None:
            continue
        margin = mean_margin(m, ld, tok.pad_id)
        lines.append(f"- `{label}` eval mean margin: **{margin:.4f}**")

    rm_metrics = art / "metrics_rm.json"
    if rm_metrics.exists():
        lines.append("")
        lines.append("## RM metrics")
        lines.append("```json")
        lines.append(rm_metrics.read_text(encoding="utf-8"))
        lines.append("```")

    lines += [
        "",
        "## Interpretation notes",
        "- margin = mean_token_logπ(chosen) - mean_token_logπ(rejected)",
        "- DPO success: after > before on held-out pairs",
        "- Do not treat toy fluency as alignment quality",
        "",
    ]
    report = art / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", report)

if __name__ == "__main__":
    main()
```

## `README.md`（프로젝트 폴더용 초안）
```markdown
# ch95_preference_rl

4권 제95강 미니 프로젝트.

## Quick start

```bash
python train_sft_warm.py
python train_dpo.py
python evaluate.py
```

Optional path B:

```bash
python train_rm.py
python train_reinforce.py
```

## Success

- `artifacts/metrics_dpo.json` with eval_margin_after > eval_margin_before
- `artifacts/report.md`
```

## 챌린지 · 실패 실험（권장）
완벽 성공만 기록하지 말고, 아래를 **하나씩** 깨본다.

| ID | 실험 | 관찰할 것 |
|---|---|---|
| C1 | chosen/rejected를 의도적으로 50% 뒤집음 | margin↑이 무의미해짐 / RM acc≈0.5 |
| C2 | $\beta$를 극단적으로 키움 | 수치 불안정·과격한 확률 이동 |
| C3 | ref를 복사하지 않고 랜덤 초기화 | DPO 해석이 무너짐 |
| C4 | 길이만 다른 chosen/rejected（내용 유사） | sum-logprob vs mean 차이와 길이 bias |
| C5 | REINFORCE에서 baseline 제거 | 분산↑, 학습 출렁임 |
| C6 | train 문장을 eval에 그대로 복사 | “일반화”로 착각하는 누수 |

제96강（reward hacking, sycophancy 등）의 **축소판 감각**을 남기는 것이 목적이다.

## 채점 루브릭（자가）
| 점수 | 기준 |
|---|---|
| 필수 | 경로 A 실행 + margin 개선 + report |
| 가점 | 경로 B RM acc>0.5 + REINFORCE 로그 |
| 가점 | 챌린지 2개 이상 기록 |
| 감점 | 외부 대형 모델 API에 의존 |
| 감점 | 벤치마크 숫자를 책 밖으로 날조해 report에 기입 |

## 4권 이론과의 대응표
| 코드 | 강의 |
|---|---|
| preference JSONL | 84 |
| ToyRM + BT loss | 85 |
| DPO loss | 90~91 |
| ref policy freeze | 89（KL/참조의 사촌） |
| REINFORCE | 82 |
| margin 평가 | 본강 + 96의 평가 한계 예고 |
| （확장）outcome verifier | 93~94 — 별 트랙 |

Reasoning outcome RL을 이 폴더에 억지로 넣지 않아도 된다. 여력이 있으면 `outcome_reward`로 숫자 퀴즈만 검증하는 스크립트를 **별도 파일**로 추가하라.

## LLM에서는 어디에 사용될까?

이번 95강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- 미니 프로젝트의 완결 조건은 유창한 챗봇이 아니라 **선호 방향 메트릭의 이동**이다.
- 경로 A（DPO）만으로 4권 중반의 수식을 코드에 고정할 수 있다.
- 경로 B는 “RM이 곧 보상”인 RLHF 스케치이며 분산과 해킹 여지가 드러난다.
- 챌린지 실험이 제96강으로 가는 다리이다.


## 용어 사전

| 용어 | 설명 |
|---|---|
| (이 강 핵심어) | 본문에서 강조한 용어를 다시 적어 보세요. |

## 연습문제
### 문제 1

DPO에서 `ref = copy.deepcopy(model)` 후 `requires_grad_(False)`를 하는 이유를 쓰시오.

### 문제 2

이 프로젝트의 `mean_margin` 정의를 수식으로 쓰시오.

### 문제 3

경로 B에서 RM accuracy가 0.5 근처이면 REINFORCE를 돌려도 안 되는 이유를 쓰시오.

### 문제 4

sum logprob와 mean logprob 중, 길이 bias에 더 민감한 쪽은?

### 문제 5

챌린지 C1（라벨 뒤집기）이 제96강의 어떤 주제와 맞닿는지 한 줄로 쓰시오.

---

## 정답 및 해설
### 문제 1


$$
DPO 항이 $\pi_\theta$와 $\pi_{\mathrm{ref}}$의 로그비에 의존하므로, 참조 분포를 학습 중 움직이지 않게 고정하기 위함이다.
$$

### 문제 2

$\frac{1}{N}\sum_i\big(\overline{\log\pi}(y_w^{(i)}|x^{(i)})-\overline{\log\pi}(y_l^{(i)}|x^{(i)})\big)$ （막대는 토큰 평균）.

### 문제 3

보상이 선호를 구분하지 못하면 정책 경사에 쓸 신호가 노이즈에 가깝다.

### 문제 4

sum logprob.

### 문제 5

보상/선호 신호 자체가 오염되면 최적화는 “틀린 목표”를 잘 푸는 쪽으로 간다（reward misspecification·해킹의 입구）.

## 다음 강의와 연결
**제96강. Alignment의 한계와 부작용**에서 reward hacking, sycophancy, over-refusal, 분포 이동, 평가 한계를 정리한다. 제95강 report의 실패 로그가 있으면 사례로 삼아라.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [94강. Reasoning Training](94강_Reasoning_Training.md)
- **다음 강:** [96강. Alignment의 한계와 부작용](96강_Alignment의_한계와_부작용.md)

<!-- /LECTURE_NAV -->
