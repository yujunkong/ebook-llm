# 3권. GPT Pretraining과 SFT

## 제76강. 프로젝트 — Mini GPT + SFT

### 1. 이번 강의에서 배울 것

제68강 Mini GPT Pretraining으로 **다음 토큰 사전학습**의 미니 루프를 닫았다. 이번 프로젝트는 그 체크포인트（또는 동일 아키텍처의 소형 모델）위에 **아주 작은 instruction set**으로 SFT를 올린다.

이 강의를 마치면 다음을 할 수 있어야 한다.

- `ch76_mini_gpt_sft/`에 데이터·마스크·학습·생성·비교 스크립트를 둔다.
- prompt 구간은 loss에서 제외하고 response만 CE로 학습한다.
- SFT 전·후 생성을 **동일 프롬프트**로 저장·비교한다.
- 제75강식의 초미니 harness（규칙 2~3개）로 회귀를 확인한다.

완벽 챗봇이 목표가 아니다. **“지시→응답” 학습 신호가 코드로 흐른다**는 것을 증명한다.

### 2. 왜 이것을 배우는가

문서만 읽으면 SFT는 “데이터 포맷을 바꾼 pretraining”처럼 보인다. 손으로 구현하면 차이가 드러난다.

| Pretraining (68강) | SFT (이번) |
|---|---|
| 긴 코퍼스를 패킹 | (instruction, response) 쌍 |
| 거의 모든 토큰에 loss | **response만** loss |
| 문장 이어쓰기 | 질문에 답하는 형식 |
| PPL 중심 | 지시 준수·before/after |

LoRA/QLoRA（제73~74강）는 이 프로젝트에서 **선택**이다. 미니 모델은 full SFT가 가능하므로, 기본 경로는 full（또는 전 레이어 소형 LoRA）로 두고, 시간이 남으면 어댑터 ablation을 추가한다.

### 3. 프로젝트 목표와 성공 기준

#### 3.1 목표

1. 장난감 instruction JSONL（20~50쌍）작성
2. Chat template으로 토큰열 구성
3. Label mask로 prompt 토큰 `-100` 처리
4. 짧은 학습 루프（CPU 수분~십수분 수준을 목표로 규모 조절）
5. Before/after 생성 리포트 + 초미니 eval

#### 3.2 성공 기준

- [ ] 학습 중 response-only loss가 전반적으로 감소
- [ ] 학습에 **없는** 형식 지시 1개 이상에서 after가 before보다 형식 준수
- [ ] mask 버그 단위 테스트 통과（prompt 위치 loss 기여 ≈ 0）
- [ ] `artifacts/before_after.md` 생성

“모든 답을 완벽히”는 기준이 아니다.

### 4. 권장 디렉터리

```text
ch76_mini_gpt_sft/
├── README.md
├── config.py
├── model.py          # 68강 Mini GPT 재사용 또는 축소 복사
├── tokenizer.py      # 문자/단어 toy 또는 단순 SentencePiece 래퍼
├── data_sft.py       # JSONL → tensors + labels
├── train_sft.py
├── generate_compare.py
├── eval_mini.py
├── data/
│   ├── instruct_train.jsonl
│   └── instruct_eval.jsonl
└── artifacts/
    ├── ckpt_pretrained.pt   # (있다면) 68강 가중치
    ├── ckpt_sft.pt
    └── before_after.md
```

제68강 산출물이 있으면 `model.py`·tokenizer·pretrained ckpt를 재사용한다. 없으면 **더 작은** Causal LM을 이 폴더에 두고, 먼저 수 분 pretrain（장난감 코퍼스）후 SFT해도 된다.

### 5. 초소형 Instruction 데이터

#### 5.1 형식

```json
{"id": "t01", "instruction": "다음 문장을 대문자로 바꾸세요.", "input": "hello world", "output": "HELLO WORLD"}
{"id": "t02", "instruction": "불릿 2개로 장점만 적으세요.", "input": "자전거 출퇴근", "output": "- 비용이 적다\n- 운동이 된다"}
{"id": "t03", "instruction": "예/아니오로만 답하세요.", "input": "1+1은 2인가?", "output": "예"}
```

규칙:

1. **train / eval 분리**（제75강）. eval은 표현을 paraphrase.
2. 형식 제약 문항을 의도적으로 넣는다（예/아니오, 불릿 n개, JSON）.
3. 한국어·영어를 섞어도 되나, toy tokenizer면 **문자 집합**을 맞춘다.

#### 5.2 샘플 생성 팁

- 같은 스킬을 다른 지시문으로 3개
- 일부러 짧은 output（과적합·style 관찰이 쉬움）
- 위험 문장은 넣지 않는다（교육용 미니셋）

### 6. Chat template → 토큰 → 마스크

#### 6.1 문자열 템플릿

미니 프로젝트용 단순 템플릿:

```text
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

`input`이 비면 Input 블록을 생략해도 된다. **학습과 생성에서 동일 템플릿**을 쓴다.

#### 6.2 마스크 규칙

토큰화된 전체 서열을 $x_0,x_1,\ldots,x_{T-1}$라 하고, 다음 토큰 예측 라벨을 $y_t = x_{t+1}$로 둔다（마지막은 무시/EOS）.

Response 시작 위치 $t_r$ 이후만 학습:

$$
L = \frac{1}{|\mathcal{R}|}\sum_{t:\, t+1 \in \mathcal{R}} \mathrm{CE}(f_\theta(x_{\le t}), y_t)
$$

구현 관례（PyTorch）:

```python
IGNORE = -100
labels = input_ids.clone()
labels[:, :response_start] = IGNORE
# 선택: prompt 끝의 "### Response:\n" 경계는 데이터 함수에서 정확히 계산
```

#### 6.3 `data_sft.py` 핵심

```python
# ch76_mini_gpt_sft/data_sft.py
"""Instruction JSONL → (input_ids, labels) with response-only loss."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def format_example(ex: dict, for_training: bool = True) -> str:
    inst = ex["instruction"].strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()

    if inp:
        head = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        head = f"### Instruction:\n{inst}\n\n### Response:\n"

    if for_training:
        return head + out
    return head  # 생성 시에는 response를 비움


@dataclass
class SFTItem:
    input_ids: torch.Tensor  # [T]
    labels: torch.Tensor     # [T]


class SFTJsonlDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_len: int = 256):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> SFTItem:
        ex = self.rows[idx]
        prompt = format_example(ex, for_training=False)
        full = format_example(ex, for_training=True)

        prompt_ids = self.tok.encode(prompt)
        full_ids = self.tok.encode(full)[: self.max_len]

        labels = full_ids.copy()
        cut = min(len(prompt_ids), len(full_ids))
        for i in range(cut):
            labels[i] = IGNORE_INDEX

        return SFTItem(
            input_ids=torch.tensor(full_ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
        )


def collate_sft(batch: list[SFTItem], pad_id: int) -> dict[str, torch.Tensor]:
    T = max(x.input_ids.numel() for x in batch)
    B = len(batch)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    labels = torch.full((B, T), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros(B, T, dtype=torch.long)
    for i, ex in enumerate(batch):
        n = ex.input_ids.numel()
        input_ids[i, :n] = ex.input_ids
        labels[i, :n] = ex.labels
        attn[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
```

**단위 테스트 아이디어:** `labels`에서 `IGNORE_INDEX`가 아닌 위치가 모두 response 문자에 해당하는지, 디코드해 육안 확인한다.

### 7. 모델과 손실

Causal LM forward가 logits `[B,T,V]`를 낸다고 가정（제56~57·68강）.

```python
import torch.nn.functional as F

def sft_loss(logits, labels):
    # logits: [B,T,V], labels: [B,T] with IGNORE_INDEX
    # next-token: 보통 logits[:, :-1] vs labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
```

마스크가 없다면 모델은 instruction 문장을 **그대로 복사·예측**하는 쪽에 용량을 쓴다. 미니셋에서는 특히 빨리 망가진다.

### 8. `train_sft.py` 골격

```python
# ch76_mini_gpt_sft/train_sft.py
"""Mini GPT SFT loop — response-only CE."""

import torch
from torch.utils.data import DataLoader

from config import CFG
from data_sft import SFTJsonlDataset, collate_sft, IGNORE_INDEX
from model import MiniGPT  # 68강 호환
from tokenizer import ToyTokenizer


def main():
    tok = ToyTokenizer.load(CFG.tokenizer_path)
    model = MiniGPT(CFG)
    if CFG.pretrained_ckpt:
        model.load_state_dict(torch.load(CFG.pretrained_ckpt, map_location="cpu"))

    ds = SFTJsonlDataset(CFG.train_jsonl, tok, max_len=CFG.block_size)
    loader = DataLoader(
        ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sft(b, pad_id=tok.pad_id),
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG.lr,
    )

    model.train()
    step = 0
    for epoch in range(CFG.epochs):
        for batch in loader:
            logits = model(batch["input_ids"])  # [B,T,V]
            loss = sft_loss(logits, batch["labels"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 10 == 0:
                # 디버그: 마스크되지 않은 토큰 수
                n = (batch["labels"][:, 1:] != IGNORE_INDEX).sum().item()
                print(f"epoch={epoch} step={step} loss={loss.item():.4f} resp_tok={n}")

    torch.save(model.state_dict(), CFG.sft_ckpt)
    print("saved", CFG.sft_ckpt)


def sft_loss(logits, labels):
    import torch.nn.functional as F
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


if __name__ == "__main__":
    main()
```

하이퍼파라미터 감각（미니·설명용）:

| 항목 | 시작점 |
|---|---|
| epochs | 5~30（데이터가 작으면 early stop） |
| lr | $1\mathrm{e}{-4}$~$3\mathrm{e}{-4}$（미니 full FT） |
| batch | 4~16 |
| block_size | 128~256 |

데이터가 30쌍인데 epoch 200이면 제75강의 overfitting을 **의도적으로** 재현할 수도 있다. 한 번은 과하게 돌려 보고, harness가 망가지는 모습을 기록해 보라.

### 9. Before / After 생성 비교

#### 9.1 생성 시 입력

학습과 동일하게 `format_example(..., for_training=False)`까지만 넣고, `### Response:\n` 뒤에서 토큰을 이어 생성한다.

```python
@torch.no_grad()
def greedy_complete(model, tok, prompt: str, max_new: int = 64) -> str:
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    model.eval()
    for _ in range(max_new):
        logits = model(ids)[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1).item())
        if next_id == tok.eos_id:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
    return tok.decode(ids[0].tolist())
```

#### 9.2 `generate_compare.py`

```python
# 의사코드 흐름
pretrained = load_model(CFG.pretrained_ckpt)
sft = load_model(CFG.sft_ckpt)

lines = ["# Before / After\n"]
for ex in eval_rows:
    prompt = format_example(ex, for_training=False)
    before = greedy_complete(pretrained, tok, prompt)
    after = greedy_complete(sft, tok, prompt)
    lines += [
        f"## {ex['id']}\n",
        f"**prompt**\n\n```\n{prompt}\n```\n",
        f"**before**\n\n```\n{before}\n```\n",
        f"**after**\n\n```\n{after}\n```\n",
    ]
Path("artifacts/before_after.md").write_text("".join(lines), encoding="utf-8")
```

비교 시 볼 것:

1. 형식 마커（불릿, JSON, 예/아니오）
2. instruction 무시 여부
3. 학습셋 문장 복사 징후

### 10. 초미니 Eval harness

```python
# ch76_mini_gpt_sft/eval_mini.py
import json
import re


def score_yes_no(text: str) -> float:
    t = text.strip().splitlines()[0] if text.strip() else ""
    return float(t in {"예", "아니오", "Yes", "No"})


def score_bullet_n(text: str, n: int) -> float:
    bullets = re.findall(r"^\s*[-*]\s+\S+", text, flags=re.M)
    return float(len(bullets) == n)


def score_json_keys(text: str, keys: list[str]) -> float:
    try:
        # 첫 { ... } 시도
        m = re.search(r"\{.*\}", text, flags=re.S)
        obj = json.loads(m.group(0) if m else text)
        return float(all(k in obj for k in keys))
    except Exception:
        return 0.0
```

리포트 예:

```text
id=e_yes  score=1.0
id=e_bul2 score=1.0
id=e_json score=0.0   # after도 실패 → 데이터/용량 부족 기록
mean=0.67
```

점수가 낮아도 **프로세스**가 맞으면 프로젝트 성공이다. 다만 mask 버그가 있으면 거의 항상 형식 학습이 안 된다.

### 11. （선택）LoRA ablation

미니 모델에 LoRA를 붙이는 연습:

1. `q_proj`,`v_proj`만 $r=4$
2. full SFT와 동일 step으로 harness 비교
3. merge 후 생성 지연이 거의 같은지 확인

파라미터가 원래 작으면 LoRA의 VRAM 이득은 미미하다. 목적은 **제73강 습관의 이식**이다.

### 12. 디버깅 가이드

| 증상 | 의심 | 확인 |
|---|---|---|
| loss가 거의 0으로 급락, 생성 쓰레기 | label이 input과 동일·시프트 오류 | `labels` 디코드 |
| loss 안 감 | lr·동결·데이터 빈 response | trainable 수, 배치 출력 |
| before≈after | ckpt 경로 혼동, epoch 부족 | state_dict 해시 |
| 형식만 암기 | overfitting | held-out paraphrase |
| JSON만 실패 | 이스케이프/길이 | max_new_tokens |

### 13. 권장 실험 순서（체크리스트）

1. [ ] train 20~40 / eval 8~15 JSONL 작성
2. [ ] mask 시각화（prompt=`#`, response=`R`）
3. [ ] 1 epoch smoke train
4. [ ] before/after 3문항 수동 확인
5. [ ] full train + `eval_mini.py`
6. [ ] （선택）epoch 과다로 overfitting 재현
7. [ ] （선택）LoRA vs full

### 14. 핵심 정리

- Mini GPT SFT의 핵은 **response-only mask**와 **동일 템플릿 생성**이다.
- 작은 데이터로도 before/after·harness를 남기면 실험이 축적된다.
- Loss 감소는 필요조건이고, 형식 준수·held-out가 성공에 가깝다.
- 이 프로젝트가 3권 SFT 실습의 관문이며, 제77강에서 역할 이론으로 되돌아간다.

### 15. 핵심 용어

| 용어 | 한 줄 의미 |
|---|---|
| Instruction set | (지시, 입력, 응답) 지도 데이터 |
| Response-only loss | 응답 토큰에만 CE |
| IGNORE_INDEX | 마스크용 라벨 센티널（흔히 -100） |
| Before/after | SFT 전후 동일 프롬프트 생성 비교 |
| Mini harness | 규칙 기반 초소형 평가기 |

### 16. 복습 문제

#### 문제 1（마스크）

`labels` 앞부분을 `-100`으로 두는 이유를 학습 신호 관점에서 쓰시오.

#### 문제 2（템플릿）

학습 때는 `### Response:` 뒤 정답을 붙이고, 생성 때는 비우는 이유를 쓰시오.

#### 문제 3（평가）

train에만 있는 지시문 그대로 eval하면 제75강 기준 무엇이 위험한가?

#### 문제 4（구현）

`cross_entropy`에 `ignore_index`를 넣지 않고 prompt 라벨을 0으로 채우면 생기는 문제를 쓰시오.

#### 문제 5（설계）

데이터가 25쌍일 때 epoch를 매우 크게 잡은 실험을 “실패 재현”으로 쓰는 목적을 한 줄로 쓰시오.

---

### 정답 및 해설

#### 문제 1

모델이 instruction/prompt 토큰을 맞추는 데 용량을 쓰지 않게 하고, 응답 생성에 학습 신호를 집중하기 위함이다.

#### 문제 2

학습은 teacher forcing으로 정답 응답을 보여 주고, 추론은 모델이 응답을 이어서 쓰게 해야 하므로 입력 분포를 맞추되 정답을 미리 넣지 않는다.

#### 문제 3

Instruction overfitting을 見逃하고 일반화를 과대평가할 위험（사실상 누수에 가까운 평가）.

#### 문제 4

토큰 id 0을 “정답”으로 배우게 되어 학습이 오염된다. `-100` ignore와 완전히 다르다.

#### 문제 5

과적합·style collapse 증상을 harness로 관찰해, 조기 종료·데이터 다양성의 필요를 체감하기 위함이다.

### 17. 다음 강의와 연결

미니 SFT를 한 바퀴 돌렸다.  
**제77강. Pretraining과 SFT의 역할 정리**에서는 두 단계가 각각 **할 수 있는 일 / 할 수 없는 일**을 표로 고정하고, 4권 Post-Training이 왜 필요한지 위치를 잡는다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제75강. SFT 평가와 실패 사례](75강_SFT_평가와_실패_사례.md)
- **다음 강:** [제77강. Pretraining과 SFT의 역할 정리](77강_Pretraining과_SFT의_역할_정리.md)

<!-- /LECTURE_NAV -->
