# 2권. Tokenizer와 Transformer

## 제50강. 프로젝트 — Mini Transformer 구현 (2)

### 1. 이번 강의에서 배울 것

제49강에서 세운 Mini Transformer 뼈대에 **데이터 · 학습 루프 · greedy 생성**을 붙인다. 이번 강의(2부)가 끝나면 “작은 텍스트를 외우며 다음 글자를 이어 쓰는” 장난감 LM이 동작해야 한다.

이 강의를 마치면 다음을 할 수 있어야 한다.

- `data.py`로 텍스트를 토큰 시퀀스·미니배치 `(x, y)`로 만든다.
- `train.py`에서 Cross Entropy로 학습하고 loss 감소를 확인한다.
- greedy `generate`로 프롬프트 뒤 문자를 이어 쓴다.
- 실패 시 shape / mask / 타깃 시프트 / eval 모드를 체계적으로 디버깅한다.

### 2. 왜 이것을 배우는가

Forward만 되는 모델은 아직 “언어 모델”이 아니다. 언어 모델은 **Loss가 정의되고, 파라미터가 갱신되며, 생성 API가 닫히는** 순간 완성된다.

3권 GPT Pretraining은 이 루프의 확대판이다.

```text
텍스트 코퍼스
  → 토큰 배치 (x, y)
    → logits = model(x)
      → CE(logits, y)
        → backward / step
          → (가끔) generate로 샘플 점검
```

지금 CPU 미니 규모로 이 루프를 한 번 완주해야, 이후 데이터셋만 커져도 당황하지 않는다.

### 3. 프로젝트 상태 확인 (1부에서 이어짐)

디렉터리는 동일하다.

```text
ch49_mini_transformer/
├── config.py
├── tokenizer.py
├── model.py          # 49강에서 완성
├── data.py           # 이번 강
├── train.py          # 이번 강
└── README.md         # 선택
```

1부 smoke test가 실패하면 2부를 시작하지 않는다. `logits.shape == (B, T, V)`를 다시 확인하라.

### 4. 학습 데이터 — 아주 작은 텍스트

완벽한 위키피디아가 필요 없다. **반복 패턴이 있는 짧은 텍스트**가 미니 LM에 더 친절하다.

예: `input.txt`에 다음을 넣는다(직접 만들어도 됨).

```text
to be or not to be that is the question
to be or not to be that is the question
all the world's a stage and all the men and women merely players
to be or not to be that is the question
```

또는 더 단순하게:

```text
abababababababab
hello hello hello hello
mini transformer mini transformer
```

원칙:

- 너무 짧으면(수 십 자) 금방 암기하지만, 파이프라인 검증에는 충분하다.
- 너무 길면 CPU에서 느리다. 처음엔 **1~5KB**면 넉넉하다.

### 5. `data.py` — 배치 만들기

Next-token 학습의 핵심은 **한 칸 시프트**다.

```text
원문 토큰: t0 t1 t2 t3 t4
x (입력):  t0 t1 t2 t3
y (타깃):  t1 t2 t3 t4
```

```python
# ch49_mini_transformer/data.py
"""작은 텍스트 → 토큰 텐서와 랜덤 배치."""

from __future__ import annotations

from pathlib import Path

import torch

from tokenizer import CharTokenizer


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_tokenizer_and_data(text: str):
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    return tok, data


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device | str = "cpu",
):
    """랜덤 시작 위치에서 (x, y) 배치를 뽑는다.

    x, y shape: [B, T]  (T=block_size)
    """
    # 마지막에 y로 쓸 +1이 필요하므로 상한은 len-block_size-1
    hi = data.size(0) - block_size - 1
    if hi < 1:
        raise ValueError("텍스트가 block_size에 비해 너무 짧습니다.")
    starts = torch.randint(0, hi, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    # 파일 없으면 인라인 텍스트로 테스트
    sample = ("to be or not to be that is the question\n") * 50
    tok, data = build_tokenizer_and_data(sample)
    x, y = get_batch(data, batch_size=2, block_size=16)
    print("vocab", tok.vocab_size, "data", data.shape)
    print("x", x.shape, "y", y.shape)
    print("decode x0:", repr(tok.decode(x[0].tolist())))
    print("decode y0:", repr(tok.decode(y[0].tolist())))
```

예상 출력 예:

```text
vocab 18 data torch.Size([2000])
x torch.Size([2, 16]) y torch.Size([2, 16])
decode x0: 'to be or not to '
decode y0: 'o be or not to b'
```

`y`가 `x`보다 한 글자씩 앞서 있음을 눈으로 확인한다. 이 확인을 건너뛰면 Loss 디버깅이 어려워진다.

### 6. Loss — logits와 타깃 연결

`CrossEntropyLoss`는 보통 `(N, V)` logit과 `(N,)` 클래스 인덱스를 기대한다. `[B,T,V]`를 펼친다.

```python
logits = model(x)                     # [B, T, V]
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)), # [B*T, V]
    y.view(-1),                       # [B*T]
)
```

제34강 내용 그대로다. Softmax는 Loss 안에서 처리되므로 `softmax`를 미리 씌우지 않는다.

### 7. `train.py` — 학습 루프와 greedy 생성

#### 7.1 Greedy generate

```python
# ch49_mini_transformer/train.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from config import MiniConfig
from data import build_tokenizer_and_data, get_batch, load_text
from model import MiniTransformer


@torch.no_grad()
def generate_greedy(
    model: MiniTransformer,
    idx: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """idx: [B, t] → [B, t+max_new_tokens], greedy."""
    model.eval()
    for _ in range(max_new_tokens):
        # crop to block_size
        idx_cond = idx[:, -model.cfg.block_size :]
        logits = model(idx_cond)          # [B, t', V]
        logits = logits[:, -1, :]         # 마지막 위치
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
```

포인트:

- 컨텍스트가 `block_size`를 넘으면 **뒤쪽만** 잘라 모델에 넣는다.
- 예측은 **마지막 위치** logit만 사용한다.
- `argmax`는 greedy다. temperature 샘플링은 3권에서.

#### 7.2 학습 메인

```python
def train(
    text: str,
    steps: int = 800,
    batch_size: int = 16,
    lr: float = 3e-3,
    print_every: int = 100,
    device: str = "cpu",
):
    tok, data = build_tokenizer_and_data(text)
    cfg = MiniConfig(
        vocab_size=tok.vocab_size,
        block_size=64,
        n_embd=64,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )
    model = MiniTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(data, batch_size, cfg.block_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % print_every == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # 짧은 생성 샘플
    prompt = text[:16]
    start = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = generate_greedy(model, start, max_new_tokens=80)
    print("--- prompt ---")
    print(prompt)
    print("--- greedy ---")
    print(tok.decode(out[0].tolist()))
    return model, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="path to input.txt")
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()

    if args.input:
        text = load_text(args.input)
    else:
        text = ("to be or not to be that is the question\n") * 80
        text += ("all the world's a stage\n") * 40

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| chars:", len(text))
    train(text, steps=args.steps, device=device)


if __name__ == "__main__":
    main()
```

실행:

```bash
cd ch49_mini_transformer
python train.py --steps 800
# 또는
python train.py --input input.txt --steps 1000
```

### 8. 예상 출력

환경마다 숫자는 다르다. **경향**이 맞으면 성공이다.

```text
device: cpu | chars: 3200
step    1 | loss 2.85xx
step  100 | loss 1.9xxx
step  200 | loss 1.2xxx
step  400 | loss 0.7xxx
step  800 | loss 0.3xxx
--- prompt ---
to be or not to 
--- greedy ---
to be or not to be that is the question
to be or not to be that is the question
...
```

해석 가이드:

| 관찰 | 의미 |
|---|---|
| step 1 loss ≈ $\ln V$ 근처 | 랜덤 초기화로 그럴듯함 (대략) |
| loss가 단조에 가깝게 감소 | 학습 신호 존재 |
| greedy가 학습 문장 조각을 반복 | 암기·패턴 학습 (미니에서는 정상) |
| loss↓인데 생성 난수 같음 | mask/시프트/eval 버그 의심 |

문자 vocab가 20이면 $\ln 20 \approx 3.0$ 근처가 초기 CE의 대략적 감각이다. 엄밀한 하한은 아니므로 “근처”로만 보라.

### 9. 한 배치 overfit 테스트 (강력 추천)

전체 루프 전에, **고정된 배치 하나**만 반복 학습해 보라.

```python
x, y = get_batch(data, batch_size=4, block_size=cfg.block_size, device=device)
for step in range(200):
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(step, loss.item())
```

충분히 작은 모델·배치에서 loss가 거의 0 근처로 안 내려가면, 데이터 문제가 아니라 **구현 버그** 확률이 높다.

### 10. 체크포인트 저장 (선택)

```python
torch.save(
    {
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "stoi": tok.stoi,
    },
    "mini_ckpt.pt",
)
```

불러올 때는 같은 `MiniConfig`·tokenizer 맵으로 모델을 재구성한 뒤 `load_state_dict`한다. 1권 제21·25강 패턴과 같다.

### 11. 디버깅 순서

```text
1) data.py에서 x/y decode가 한 글자 시프트인지
2) model smoke test logits [B,T,V]
3) 한 배치 overfit
4) 전체 학습 loss 곡선
5) generate 시 model.eval(), crop, argmax 위치
```

증상별 힌트:

| 증상 | 의심 |
|---|---|
| `RuntimeError` shape | view/transpose, vocab 불일치 |
| loss가 NaN | lr 과대, mask에 0 행 전체 -inf |
| loss 안 내려감 | 타깃 미시프트, step 누락, 잘못된 device |
| 생성 즉시 반복  Junk | 학습 부족 또는 데이터 너무 짧음 |
| 생성 시 크래시 | `T > block_size` crop 누락 |

### 12. 완성 체크리스트 (2부)

- [ ] `python data.py`가 x/y shape와 decode를 인쇄한다
- [ ] `python train.py --steps 800`이 loss 감소 로그를 남긴다
- [ ] greedy 출력이 프롬프트를 포함하고 뒤가 이어진다
- [ ] (선택) `mini_ckpt.pt` 저장·로드
- [ ] (선택) weight tying on/off loss 비교 메모

### 13. 도전 과제 (2부)

#### 도전 1 — Validation split

데이터 뒤 10%를 val로 고정하고, `print_every`마다 val loss를 함께 출력하라. train만 내려가고 val이 올라가면 암기 신호다.

#### 도전 2 — Temperature 샘플링

`argmax` 대신 `logits / T` 후 `multinomial`로 샘플링하라. `T=1`과 `T=0.5` 차이를 관찰한다. (사실: temperature는 분포 평탄도를 바꾼다. “항상 품질 +N%” 같은 주장은 하지 말 것.)

#### 도전 3 — 단어 단위 tokenizer

공백 분리 vocab로 바꿔 같은 `model.py`를 재사용하라. `vocab_size`만 커진다.

#### 도전 4 — 학습 곡선 저장

step별 loss를 `losses.txt`에 쓰고, (선택) matplotlib로 꺾은선을 그려라. 제51강 시각화와 연결된다.

#### 도전 5 — 스크립트 정리

`train.py`에서 generate를 `generate.py`로 분리하고, `python generate.py --ckpt mini_ckpt.pt --prompt "to be"` 형태로 실행되게 하라.

### 14. 이 프로젝트가 증명하는 것

끝난 뒤 다음 문장을 말할 수 있어야 한다.

> 나는 토큰 ID 배치를 Causal Transformer에 넣어 next-token CE로 학습시키고, greedy 디코딩으로 문자열을 생성할 수 있다.

이것이 2권의 실전 관문이다. 규모만 키우면 3권 Mini GPT / Pretraining으로 이어진다.

### 15. 핵심 정리

- 2부는 `data.py`(시프트 배치) + `train.py`(루프·greedy)다.
- Loss는 `logits[B,T,V]`와 `y[B,T]`의 Cross Entropy다.
- 생성은 마지막 위치 logit → argmax → append 반복이다.
- 한 배치 overfit 테스트로 구현 버그를 먼저 걸러라.
- 성공 기준은 완벽한 문장이 아니라 **loss 감소 + 패턴을 흉내 내는 생성**이다.

### 16. 핵심 용어

| 용어 | 의미 |
|---|---|
| Next-token batch | 입력 $x$와 한 칸 앞 타깃 $y$ |
| `cross_entropy` | logit·클래스 ID용 다중분류 손실 |
| Greedy decoding | 매 스텝 최고 확률 토큰 선택 |
| Context crop | 긴 컨텍스트를 `block_size`로 절단 |
| Overfit test | 고정 배치 암기로 구현 검증 |
| AdamW | 미니 LM에 흔히 쓰는 Optimizer |
| Checkpoint | `state_dict` + 설정 + tokenizer 맵 |

### 17. 복습 문제

#### 문제 1 (개념)

`y = data[i+1 : i+1+T]`인 이유를 Next Token Prediction으로 설명하시오.

#### 문제 2 (코드)

`logits[:, -1, :]`만 사용하는 생성 단계에서, 학습 때 모든 위치 logit을 쓰는 이유와 모순되지 않는가?

#### 문제 3 (실험)

초기 loss가 $\ln V$보다 훨씬 작다면 의심할 만한 버그 하나를 쓰시오.

#### 문제 4 (연결)

제48강 weight tying을 켠 채로 학습할 때 저장해야 할 가중치 별칭 이슈를 한 문장으로 쓰시오.

#### 문제 5 (디버깅)

overfit 테스트에서 loss가 0.01까지 내려갔는데, 전체 데이터 학습 후 generate가 프롬프트와 무관한 이유 후보 두 가지를 쓰시오.

---

### 정답 및 해설

#### 문제 1

위치 $t$의 출력이 $x_{t+1}$을 맞추도록 감독 신호를 주기 위해, 타깃 시퀀스를 한 칸 앞으로 둔다.

#### 문제 2

모순 없음. 학습은 모든 위치를 병렬 감독해 샘플 효율을 높이고, 생성은 이미 확정된 prefix 뒤의 **다음 한 토큰**만 필요할 뿐이다. 같은 forward 함수를 쓴다.

#### 문제 3

예: 타깃이 입력과 동일(시프트 누락)해 “복사”로 쉽게 맞추는 경우, 또는 데이터 누수·마스크 커닝.

#### 문제 4

`state_dict`에 같은 storage가 두 키로 잡히거나 로드 시 한쪽만 복원될 수 있으므로, 저장/로드 후 `lm_head.weight is tok_emb.weight`(또는 동등성)를 재확인한다.

#### 문제 5

예: (1) 프롬프트 문자가 train 분포 밖 (2) `model.train()` 상태로 dropout이 켜짐 (3) crop/디바이스 불일치 (4) tokenizer 맵 불일치.

### 18. 다음 강의와 연결

동작하는 Mini Transformer를 손에 넣었다. **제51강. Attention 시각화**에서는 학습된(또는 작은 예제) Attention 가중치를 그림으로 읽어, 모델이 “어디에 주목하는지”를 관찰한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제49강. 프로젝트 — Mini Transformer 구현 (1)](49강_프로젝트_Mini_Transformer_구현_1.md)
- **다음 강:** [제51강. Attention 시각화](51강_Attention_시각화.md)

<!-- /LECTURE_NAV -->
