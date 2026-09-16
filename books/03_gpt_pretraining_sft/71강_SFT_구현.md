# 3권. GPT Pretraining과 SFT

## 제71강. SFT 구현

### 1. 이번 강의에서 배울 것

제69~70강에서 Instruction Tuning의 개념과 데이터 형식을 고정했다. 이번 강의는 **SFT 학습 루프**를 코드 수준으로 구현한다. 핵심은 한 문장이다.

> **Loss는 assistant(응답) 토큰에만 적용하고, user/system(프롬프트)은 마스크한다.**

이 강의를 마치면 다음을 할 수 있어야 한다.

- Pretraining 목표와 SFT 목표를 같은 Causal LM 위에서 비교해 설명하기
- `labels`에 `ignore_index(-100)`로 프롬프트를 가리는 텐서를 만들기
- 미니 SFT train step을 작성하고 loss가 응답 구간에만 의존함을 검증하기
- 흔한 버그(전체 시프트 실수, 마스크 전부 -100, 템플릿 불일치)를 점검하기

제72강에서 Chat Template·특수 토큰으로 “문자열 조립”을 더 단단히 고정한다. 이번 강은 **손실 마스크**에 집중한다.

### 2. 왜 이것을 배우는가

제68강 Mini GPT Pretraining의 루프는 다음과 같았다.

```text
모든 위치 t에 대해: 예측 x_{t+1}
loss = mean CE(전체 유효 토큰)
```

SFT에서 같은 루프를 그대로 쓰면:

```text
user 발화까지 “다음 토큰 맞추기” 연습
→ 질문 문체 생성 편향, 지시 수행 신호 희석
```

마스크 SFT는 같은 Transformer를 쓰되 **감독 신호의 지지 집합(support)** 만 바꾼다. 이 한 줄 차이가 base와 instruct의 학습 정의 차이다.

### 3. 먼저 알아야 할 개념

- Causal LM forward: `[B,T] → [B,T,V]` (제56·68강)
- Cross Entropy + `ignore_index` (2권 제34강)
- Instruction / messages 스키마 (제70강)
- Next-token 시프트: logits\[t\]가 타깃 token\[t+1\](또는 labels 정렬 방식)을 맞춤
- Pretraining 루프 (제62·68강)

### 4. 핵심 개념 설명

#### 4.1 목표 함수 비교

**Pretraining**

$$
\mathcal{L}_{\mathrm{PT}}
=
-\frac{1}{|\mathcal{T}|}
\sum_{t\in\mathcal{T}}
\log p_\theta(x_t\mid x_{<t})
$$

\(\mathcal{T}\): 패딩을 제외한 거의 모든 위치.

**SFT (응답 마스크)**

$$
\mathcal{L}_{\mathrm{SFT}}
=
-\frac{1}{|\mathcal{T}_{\mathrm{resp}}|}
\sum_{t\in\mathcal{T}_{\mathrm{resp}}}
\log p_\theta(x_t\mid x_{<t})
$$

\(\mathcal{T}_{\mathrm{resp}}\): assistant 응답에 해당하는 타깃 위치.

공통점:

- 같은 Causal 분해
- 같은 Softmax + CE
- 같은 optimizer 루프 (`zero_grad → backward → step`)

차이점:

- 데이터 분포(문서 vs 지시-응답)
- loss에 포함되는 위치
- (대개) 학습 step 수·학습률·정규화가 더 작거나 보수적

#### 4.2 labels와 ignore_index

PyTorch `F.cross_entropy(..., ignore_index=-100)` 관례:

```text
input_ids:  모델에 넣는 토큰
labels:     각 위치에서 “맞춰야 할” 토큰 ID
            단, -100 이면 그 위치 loss 제외
```

정렬 방식은 두 가지가 자주 보인다.

**방식 A — 시프트는 loss 함수에 맡김**

많은 예제가 `labels`를 `input_ids`와 같은 길이로 두고, 모델 문서에 따라 내부에서 shift한다. 직접 구현 미니 LM에서는 다음이 더 명시적이다.

**방식 B — 명시적 시프트 (제68강과 동일)**

```text
x = tokens[:, :-1]
y = tokens[:, 1:]
y_masked = y.clone()
y_masked[prompt_positions] = -100
```

이 책 미니 코드는 **방식 B**를 기본으로 한다. “한 칸 시프트 + 마스크”가 눈에 보인다.

#### 4.3 프롬프트 구간 결정

문자열 단계에서 경계를 안다.

```text
prompt_text + response_text
```

토큰화 후:

```text
n_prompt = len(tok.encode(prompt_text))
# 주의: 단순 concat 토큰화 vs 각각 토큰화 후 합치기
# 특수 토큰·공백 때문에 길이가 달라질 수 있음 → 제72강
```

미니 실험에서는 **구분자 특수 문자열**을 고정해 경계를 찾는다.

```text
<|user|>...<|assistant|>RESPONSE<|end|>
                 └ border
```

### 5. 직관적으로 이해하기

교실 비유:

- Pretraining: 교과서를 한 줄씩 읽으며 **다음 단어를 맞추는 받아쓰기**
- SFT: 시험지에서 **문제(프롬프트)는 이미 인쇄**되어 있고, 학생이 채워야 할 칸(응답)만 채점

문제 칸까지 채점하면, 학생은 “문제 베끼기”에 점수를 딴다. 마스크는 **답안 칸만 빨간 펜**으로 보는 것과 같다.

### 6. 작은 숫자로 직접 계산하기

토큰 ID 시퀀스(가상):

```text
ids:     [U1, U2, A1, A2, A3]   # U=user, A=assistant
targets: [U2, A1, A2, A3, EOS]  # 한 칸 시프트 가정 (길이 맞춤은 구현에 따라 조정)
mask:    [0,  1,  1,  1,  1]   # U2를 맞출지 여부는 경계 정의에 달림
```

경계 정의가 “assistant 첫 토큰부터”라면, `U2` 위치는 마스크 0이어야 한다.  
실무에서는 **응답 시작 토큰의 위치**를 템플릿으로 정확히 잰다.

가정: 세 위치의 NLL이 \(2.0, 1.0, 0.5\)이고 모두 응답이며 평균하면:

$$
\mathcal{L}= (2.0+1.0+0.5)/3 = 1.1667
$$

프롬프트 위치 NLL이 아무리 커도(예: \(10.0\)) 평균에 **들어가지 않는다**.

### 7. 코드로 구현하기 — 마스크 텐서

```python
# sft_mask.py
"""프롬프트/응답 경계로 labels 마스크 만들기."""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def build_sft_batch(
    prompt_ids: list[int],
    response_ids: list[int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """단일 샘플 → (x, y) with y masked on prompt.

    시퀀스: prompt + response
    x = seq[:-1], y = seq[1:], prompt 구간은 -100
    """
    seq = prompt_ids + response_ids
    if len(seq) < 2:
        raise ValueError("sequence too short")
    # 미니: 앞쪽 truncate (실전은 다양한 전략)
    if len(seq) > block_size + 1:
        seq = seq[-(block_size + 1) :]

    seq_t = torch.tensor(seq, dtype=torch.long)
    x = seq_t[:-1]
    y = seq_t[1:].clone()

    # prompt 길이 n_p → x의 앞 n_p-1 개(있다면) + 경계 처리는 정의에 따름
    n_p = len(prompt_ids)
    # x[i]가 예측하는 것은 seq[i+1]
    # seq[i+1]이 prompt 안에 있으면 마스크
    for i in range(len(y)):
        target_pos = i + 1  # seq index
        if target_pos < n_p:
            y[i] = IGNORE_INDEX

    # 길이 block_size로 패딩 (선택)
    if x.numel() < block_size:
        pad_len = block_size - x.numel()
        x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
        y = torch.cat([y, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
    else:
        x = x[:block_size]
        y = y[:block_size]

    return x, y


if __name__ == "__main__":
    prompt = [10, 11, 12]       # 가짜 ID
    response = [20, 21, 22]
    x, y = build_sft_batch(prompt, response, block_size=8)
    print("x", x.tolist())
    print("y", y.tolist())
```

예상 출력 패턴:

```text
x [10, 11, 12, 20, 21, 22, 0, 0]
y [-100, -100, 20, 21, 22, -100, -100, -100]
```

해석:

- 타깃 `20`(응답 첫 토큰)부터 CE가 켜진다.
- 패딩은 `-100`.
- (마지막 토큰 이후 pad 타깃도 `-100`)

실제 출력은 pad 위치·truncate 정책에 따라 조금 달라질 수 있다. **응답 ID가 y에 살아 있고 프롬프트 타깃이 -100인지**를 보면 된다.

### 8. PyTorch로 구현하기 — SFT train step

```python
# sft_step.py
"""마스크 CE 한 스텝."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def sft_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """logits [B,T,V], y [B,T] with -100 ignore."""
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * T, V),
        y.reshape(B * T),
        ignore_index=IGNORE_INDEX,
    )


def sft_train_step(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    model.train()
    logits = model(x)
    loss = sft_loss(logits, y)
    if torch.isnan(loss):
        raise RuntimeError("NaN loss — check mask (all ignored?)")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.item())
```

#### 8.1 “마스크가 진짜로 먹는가” 스모크 테스트

응답을 고정하고 프롬프트만 바꿔 보면, **올바른 마스크**에서는 loss가 거의 같아야 한다(동일 response, 동일 모델 상태 가정 — 실제로는 문맥이 바뀌면 조건부 분포가 달라져 loss도 변할 수 있음).  
더 직접적인 테스트:

```python
# y 전 구간 -100 → cross_entropy는 0에 가까운 값 또는 에러/경고
# 유효 토큰 0개면 구현에 따라 nan
```

유효 타깃 개수:

```python
n = int((y != IGNORE_INDEX).sum())
assert n > 0, "no supervised tokens"
```

#### 8.2 미니 학습 루프 스케치

```python
for step in range(max_steps):
    x, y = collate_sft_batch(samples)  # build_sft_batch 여러 개 stack
    loss = sft_train_step(model, opt, x.to(device), y.to(device))
    if step % 50 == 0:
        print(step, loss)
```

제68강 `train.py`와의 차이:

| 항목 | Pretrain | SFT |
|---|---|---|
| 데이터 | `input.txt` 스트림 | instruction JSONL |
| 배치 | 랜덤 윈도우 | 샘플 단위(+패딩/패킹) |
| labels | 전체 시프트 | 응답만 |
| 평가 | val PPL | 지시 준수 샘플 + (선택) loss |

### 9. Pretraining 코드에서 최소 변경으로 가는 길

이미 Mini GPT가 있다면:

1. `data_sft.py`: JSONL → `(prompt_ids, response_ids)`
2. `build_sft_batch`로 `(x,y)`
3. `train_sft.py`: 제68강 루프에서 `get_batch`만 교체
4. 생성 시에는 **프롬프트만** 넣고 assistant 구간을 이어서 샘플

전체 프로젝트화는 **제76강. Mini GPT + SFT**에서 묶는다. 이번 강은 손실·마스크의 정합성이 목표다.

### 10. 실제 LLM에서는 어떻게 사용하는가

- Hugging Face `Trainer` / 각종 SFTTrainer가 `labels` 마스크를 자동·반자동으로 처리하는 경우가 많다.
- 그래도 **템플릿·토크나이저·마스크**가 어긋나면 조용히 잘못된 학습이 된다. 자동을 믿기 전에 한 샘플의 `labels`를 decode해 보라.
- 긴 대화는 응답만 남겨 truncate하거나, 앞쪽 턴을 자르는 정책이 필요하다.
- Packing SFT: 여러 짧은 대화를 이어 붙일 때 **문서 경계·마스크**가 더 까다롭다. (제61강 패킹 개념의 SFT 버전)

임의 벤치마크 향상 수치를 제시하지 않는다. 구현이 맞는지면 `labels` 시각화와 loss 감소로 확인한다.

### 11. 실습

#### 실습 1 — labels 인쇄

§7 코드를 실행해 `y`에 `-100`과 응답 ID가 기대한 대로인지 확인하라.

#### 실습 2 — 유효 토큰 수

배치에서 `(y != -100).sum()`을 매 step 로그하라. 0이 나오면 즉시 중단하는 assert를 넣어 보라.

#### 실습 3 — Pretrain vs SFT loss 위치

같은 시퀀스에 대해 (1) 마스크 없음 (2) 응답만 마스크의 loss를 비교하라. 값은 다를 수 있으나, (2)의 gradient가 프롬프트 임베딩에만 “간접 문맥”으로 흐르고 타깃은 응답임을 설명으로 정리하라.

#### 실습 4 — 한 샘플 overfit

instruction 3~5개만으로 SFT overfit을 시도하고, 프롬프트를 넣었을 때 모범 응답 근처로 이어지는지 본다. (미니 모델·tiny 데이터 한정)

### 12. 자주 하는 실수

1. **`y` 전체를 -100으로 만듦**  
   loss 0/NaN, 학습 없음.

2. **시프트 방향 반대**  
   응답이 한 칸 밀려 프롬프트 끝 문자를 맞춤.

3. **문자열 길이로 마스크**  
   토큰 경계와 문자 경계는 다르다. 반드시 토큰 ID 기준.

4. **pad를 0으로 두고 ignore 안 함**  
   pad 토큰을 “정답 0번”으로 학습.

5. **평가 때 마스크 없이 PPL만 보고 SFT 성공 판정**  
   지시 준수 샘플을 함께 본다. (제67·75강)

6. **학습 템플릿 ≠ 추론 템플릿**  
   제72강에서 더 깊게 다룬다.

### 13. 핵심 정리

- SFT는 Causal LM CE를 **응답 토큰 지지 집합**으로 제한한 학습이다.
- `ignore_index(-100)`로 프롬프트·패딩을 labels에서 제외한다.
- Pretraining 루프와 옵티마이저는 같고, 데이터·마스크가 다르다.
- 유효 타깃 개수 assert와 labels 디코드가 최고의 디버거이다.
- 본격 프로젝트 조립은 제76강, 템플릿 고정은 제72강으로 이어진다.

### 14. 핵심 용어

| 용어 | 의미 |
|---|---|
| SFT | 정답 응답에 대한 Supervised Fine-Tuning |
| Prompt mask | 프롬프트 위치 loss 제외 |
| `ignore_index` | CE에서 무시할 label 값(보통 -100) |
| Response tokens | 감독 신호가 살아있는 토큰 집합 |
| Label shift | next-token 정렬을 위한 타깃 시프트 |
| Collate | 여러 샘플을 패딩·스택해 배치로 만듦 |
| Overfit test | 소수 샘플 암기로 구현 검증 |

### 15. 복습 문제

#### 문제 1 (수식)

\(\mathcal{L}_{\mathrm{SFT}}\)에서 합의 범위가 \(\mathcal{T}_{\mathrm{resp}}\)인 이유를 한 문장으로.

#### 문제 2 (코드)

`y`가 모두 `-100`일 때 어떤 현상이 위험한가?

#### 문제 3 (비교)

Pretraining과 SFT의 **공통점 하나**와 **차이점 하나**를 쓰시오.

#### 문제 4 (디버깅)

마스크를 문자 인덱스로 잡았을 때 생길 수 있는 버그를 쓰시오.

#### 문제 5 (연결)

제72강 Chat Template이 이번 마스크 구현에 왜 필수적으로 맞물리는가?

---

### 정답 및 해설

#### 문제 1

모델이 배워야 할 생성 구간이 응답이며, 프롬프트는 조건으로만 쓰기 위해서이다.

#### 문제 2

유효 타깃이 없어 loss가 0·NaN이 되거나 학습이 사실상 멈추는 위험이 있다.

#### 문제 3

공통: Causal next-token CE. 차이: SFT는 응답 위치만 감독(및 지시 데이터 분포).

#### 문제 4

서브워드 토큰 경계와 어긋나 응답 앞뒤가 잘못 가려지거나, 특수 토큰이 타깃에 섞인다.

#### 문제 5

역할 경계·특수 토큰이 템플릿으로 고정되어야 프롬프트/응답 구간을 토큰 단위로 일관되게 마스크하고, 추론 때도 같은 문자열을 재현할 수 있기 때문이다.

### 16. 다음 강의와 연결

마스크까지 이해했다면, 남은 큰 구멍은 **문자열을 어떻게 일관되게 조립·토큰화하는가**다.

다음 **제72강. Chat Template과 Special Tokens**에서는 `apply_chat_template` 개념, 역할 특수 토큰, 학습·추론 일치 규칙을 다룬다. 그다음 **제73강. LoRA**에서 전체 가중치를 건드리지 않는 효율적 SFT로 확장한다.

> Loss를 어디에 줄지 알았으면, 이제 “어떤 문자열에” 줄지를 템플릿으로 고정하자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제70강. Instruction Dataset 형식](70강_Instruction_Dataset_형식.md)
- **다음 강:** [제72강. Chat Template과 Special Tokens](72강_Chat_Template과_Special_Tokens.md)

<!-- /LECTURE_NAV -->
