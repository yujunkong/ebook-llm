# 제70강. Instruction Dataset 형식

> **학습 목표**
> - Alpaca-like 필드(`instruction` / `input` / `output`)를 읽고 단일 프롬프트로 조립하기
> - Chat messages 형식(`system` / `user` / `assistant`)의 의미
> - 멀티턴 대화에서 역할이 어떻게 교차하는지
> - Prompt masking이 왜 필요한지 미리보기 (구현은 제71강)
> - 잘못된 JSON·역할 누락·응답 누락을 걸러내는 체크리스트

---
## 1. 왜 이것을 배우는가

SFT 코드 버그의 상당수는 모델이 아니라 **데이터 스키마 불일치**에서 온다.

```text
학습 때: ### Instruction: ... ### Response: ...
추론 때: <|user|>...<|assistant|>
→ 모델 입장에서는 다른 언어
```

또는:

```text
messages에는 assistant가 있는데
loss를 전체 시퀀스에 줌
→ 사용자 발화까지 “예측 연습”을 해 버림
```

형식은 취향이 아니다. **학습과 추론이 공유하는 계약**이다.

## 2. 먼저 알아야 할 개념

- Instruction Tuning / SFT 목표 (제69강)
- JSON / JSONL (한 줄에 샘플 하나) 기본
- Special tokens 감각 (2권 제30강, 제72강에서 확장)
- Causal LM이 하나의 긴 토큰 시퀀스를 본다는 사실

## 3. 핵심 개념 설명

### 3.1 두 가지 대표 스키마

실습·공개 데이터에서 자주 만나는 큰 줄기는 둘이다.

| 스키마 | 대표 필드 | 특징 |
|---|---|---|
| Alpaca-like | `instruction`, `input`, `output` | 단일턴 과제에 단순 |
| Messages / Chat | `messages: [{role, content}, ...]` | 멀티턴·system에 자연스러움 |

둘은 서로 변환 가능하다. 중요한 것은 **최종적으로 모델에 들어가는 문자열(또는 토큰) 규칙이 유일한가**이다.

### 3.2 Alpaca-like 형식

한 샘플 예:

```json
{
  "instruction": "다음 문장을 한국어로 번역하라.",
  "input": "Life is short.",
  "output": "인생은 짧다."
}
```

`input`이 비는 경우:

```json
{
  "instruction": "1부터 5까지의 합을 구하라.",
  "input": "",
  "output": "15"
}
```

프롬프트 조립 관례(예시 템플릿):

```text
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

`input`이 없으면 Input 절을 생략하는 변형도 흔하다. **데이터셋마다 템플릿 문자열을 문서화**해야 한다.

필드 의미:

| 필드 | 의미 |
|---|---|
| `instruction` | 해야 할 일 |
| `input` | 지시의 대상 텍스트(선택) |
| `output` | 모범 응답 |

### 3.3 Messages 형식

```json
{
  "messages": [
    {"role": "system", "content": "당신은 간결한 조수다. 한 문장만 답한다."},
    {"role": "user", "content": "대한민국의 수도는?"},
    {"role": "assistant", "content": "서울입니다."}
  ]
}
```

역할:

| role | 의미 |
|---|---|
| `system` | 전역 규칙·페르소나(선택, 보통 맨 앞) |
| `user` | 사용자 발화 |
| `assistant` | 모델 응답(학습 타깃의 중심) |
| (기타) | `tool` 등 확장 역할 — 고급, 여기선 최소만 |

멀티턴:

```json
{
  "messages": [
    {"role": "user", "content": "2+2는?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "거기에 3을 더하면?"},
    {"role": "assistant", "content": "7"}
  ]
}
```

SFT 시 보통 **모든 assistant 턴**을 타깃으로 삼거나, 마지막 assistant만 타깃으로 삼는다. 데이터셋 문서에 정책을 명시하라.

### 3.4 JSONL

대용량에서는 JSON 배열 대신 **JSONL**이 흔하다.

```text
{"messages":[...]}
{"messages":[...]}
{"instruction":"...","input":"...","output":"..."}
```

한 줄 = 한 샘플. 스트리밍·샤딩에 유리하다.

### 3.5 Prompt masking 미리보기

학습 시 시퀀스를 한 줄로 이어 붙인다고 하자.

```text
[system+][user][assistant]
```

토큰으로:

```text
ids:  t0 t1 t2 t3 t4 t5 t6 t7
mask: 0  0  0  0  1  1  1  1
                 └─ assistant 구간만 1
```

- `mask=0` (또는 label=`ignore_index`): loss 미기여
- `mask=1`: next-token CE 적용

이렇게 하는 이유(제69강 복습):

> 사용자 말을 “맞추는” 연습이 아니라, **조수답게 답하는** 연습을 한다.

상세 구현·학습 루프는 제71강. 이번 강의에서는 **데이터에 역할 경계가 드러나야 마스크를 칠 수 있다**는 점만 고정한다. Alpaca 템플릿이면 `### Response:` 뒤가 응답 구간이다. Messages면 `role==assistant` 콘텐츠가 응답 구간이다.

## 4. 직관적으로 이해하기

데이터 형식 = **연극 대본의 서식**.

- Alpaca: “지문(instruction) + 소품(input) + 대사(output)” 단막극
- Messages: 여러 막의 역할 대사

배우(모델)는 서식에 익숙해진다. 공연(추론) 날 다른 서식을 주면 헤맨다. 제72강 Chat Template은 이 서식을 **토크나이저 함수로 고정**하는 장치다.

## 5. 스키마 변환

Alpaca → messages (개념 코드):

```python
def alpaca_to_messages(ex: dict) -> list[dict]:
    instr = ex["instruction"].strip()
    inp = (ex.get("input") or "").strip()
    if inp:
        user = f"{instr}\n\n{inp}"
    else:
        user = instr
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": ex["output"].strip()},
    ]
```

messages → 학습 문자열은 **반드시 동일한 `apply_chat_template`**로 (제72강). 여기서 임의로 문자열을 섞어 붙이면 학습/추론 불일치가 생긴다.

## 6. 작은 숫자·토큰 예

역할 경계를 특수 기호로 표시하는 미니 템플릿:

```text
<|user|>한 단어로. 수도?<|assistant|>서울<|end|>
```

문자 단위로 보면 응답 구간은 `서울`뿐이라고 치고, 그 앞은 마스크 0일 수 있다.  
(실제 토크나이저는 서브워드라 경계가 토큰 중간에 걸리지 않게 **특수 토큰 ID**로 역할을 끊는 편이 안전하다. 제72강.)

## 7. 코드로 다루기 — 로드와 검증

```python
# inspect_sft_data.py
"""Instruction JSONL 기초 검증."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_ALPACA = {"instruction", "output"}
VALID_ROLES = {"system", "user", "assistant"}

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"line {i}: {e}") from e
    return rows

def validate_alpaca(ex: dict[str, Any]) -> None:
    missing = REQUIRED_ALPACA - ex.keys()
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if not str(ex["instruction"]).strip():
        raise ValueError("empty instruction")
    if not str(ex["output"]).strip():
        raise ValueError("empty output")

def validate_messages(ex: dict[str, Any]) -> None:
    msgs = ex.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("messages must be non-empty list")
    roles = [m.get("role") for m in msgs]
    for r in roles:
        if r not in VALID_ROLES:
            raise ValueError(f"invalid role: {r}")
    if "assistant" not in roles:
        raise ValueError("no assistant message (nothing to train)")
    if roles[-1] != "assistant":
        # 정책에 따라 경고만 할 수도 있음
        raise ValueError("last role should be assistant for SFT")

if __name__ == "__main__":
    # 데모
    demo = {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    validate_messages(demo)
    print("ok", demo)
```

## 8. 품질·다양성 체크리스트 (형식 너머)

형식만 맞아도 내용이 나쁘면 SFT는 틀린 선생을 복사한다.

| 검사 | 질문 |
|---|---|
| 공백 응답 | output이 비어 있지 않은가? |
| 지시-응답 불일치 | 물어본 것과 다른 답을 쓰지 않았는가? |
| 언어 혼선 | 한글로 시키면 한글 응답인가? |
| 중복 | 동일 샘플이 과도하게 반복되지 않는가? |
| 유해 | 위험 지시를 무비판적으로 수행하라고 가르치지 않는가? |
| 길이 | 극단적으로 긴 응답만 있지 않은가? |

이 책은 특정 공개 데이터셋의 “점수”를 주장하지 않는다. **자기 미니 JSONL을 깨끗하게 만드는 습관**이 목표다.

## 9. 실제 LLM에서는 어떻게 사용하는가

- 공개 SFT 데이터는 Alpaca-like와 ShareGPT/messages 계열이 공존한다.
- 많은 학습 프레임워크가 messages를 받아 내부에서 chat template을 적용한다.
- System prompt 정책(항상 넣기 / 가끔 넣기 / 모델 기본 system)은 제품마다 다르다.
- 합성 데이터(모델이 만든 지시-응답)를 쓸 때는 **형식 일관성 + 필터**가 사실상 필수다.

추론 시에는 학습에 쓴 것과 **같은 템플릿**으로 messages를 렌더링한다. (제72강)

## 10. 실습

### 실습 1 — 미니 JSONL 작성

다음 세 과제를 messages JSONL로 직접 만들어 `tiny_sft.jsonl`에 저장하라.

1. 한 자리 덧셈
2. 한 문장 영→한 번역
3. “세 단어로 자기소개”

### 실습 2 — Alpaca 변환

실습 1을 Alpaca 필드로도 저장하고, §6 함수로 왕복 변환이 의미상 같은지 확인하라. (문자열 템플릿까지 같아야 토큰이 같아진다.)

### 실습 3 — 마스크 경계 표시

한 샘플을 골라 종이에(또는 주석으로) `prompt | response` 경계를 긋고, 어떤 문자열이 response인지 표시하라. 제71강에서 `labels`에 `-100`을 채울 구간이다.

### 실습 4 — 잘못된 샘플 탐지

고의로 `assistant` 없는 messages, 빈 `output`, 잘못된 role을 넣고 `validate_*`가 잡는지 확인하라.

## 11. 자주 하는 실수

1. **`input`과 `instruction`을 중복 서술**  
   모델이 어느 쪽을 따라야 할지 모호해진다.

2. **멀티턴인데 중간 assistant를 비움**  
   대화 구조가 깨진다.

3. **학습 템플릿과 평가 프롬프트가 다름**  
   “모델이 지시를 못 따른다”는 착각의 원인 1위급.

4. **JSON 배열을 한 줄로 강제**  
   JSONL과 JSON 배열을 혼동하면 로더가 실패한다.

5. **system에 과도한 비밀·장문 정책**  
   미니 실험에선 짧게. 실전에서도 토큰 예산을 생각하라.

## 12. 핵심 정리

- Instruction 데이터는 주로 Alpaca-like 또는 messages 스키마로 표현된다.
- `system` / `user` / `assistant`는 역할이며, SFT 타깃의 중심은 assistant다.
- Prompt masking을 하려면 역할·템플릿 경계가 데이터에서 명확해야 한다.
- 형식 검증과 내용 품질 검증을 분리해서 본다.
- 학습·추론 템플릿 일치가 Instruction following의 전제 조건이다.

## 13. 핵심 용어

| 용어 | 의미 |
|---|---|
| Alpaca-like | instruction/input/output 필드 스키마 |
| Messages | role·content 리스트로 된 대화 스키마 |
| System prompt | 전역 규칙·페르소나 메시지 |
| JSONL | 줄단위 JSON 샘플 저장 형식 |
| Prompt masking | 프롬프트 구간을 loss에서 제외 |
| Response span | assistant(또는 ### Response) 토큰 구간 |
| Schema validation | 필수 필드·역할·비어 있음 검사 |

## 14. 연습 문제
### 문제 1 (형식)

Alpaca 샘플에서 `input`이 비어 있을 때 프롬프트 조립 시 흔한 처리를 쓰시오.

### 문제 2 (역할)

messages에서 `system`이 없는 샘플도 合法일 수 있는가?

### 문제 3 (마스킹)

왜 user 토큰까지 loss에 넣으면 SFT 목표가 흐려질 수 있는가?

### 문제 4 (변환)

Alpaca → messages 변환 후에도 학습 문자열이 달라질 수 있는 이유를 쓰시오.

### 문제 5 (연결)

제71강 구현에 넘기기 위해, 각 샘플에서 반드시 식별 가능해야 하는 두 구간은?

---

## 정답 및 해설

### 문제 1

`### Input:` 절을 생략하거나, 빈 Input 절 없이 instruction만 넣는 방식이 흔하다. 데이터셋마다 하나로 고정해야 한다.

### 문제 2

가능하다. system은 선택이다. 다만 제품이 항상 system을 넣는다면 학습 분포에도 비율을 맞춰야 한다.

### 문제 3

모델이 사용자 발화를 “생성해야 할 텍스트”로 학습해, 응답 대신 질문 문체를 이어 쓰거나 지시 수행 신호가 희석될 수 있다.

### 문제 4

최종 렌더 템플릿(구분자 문자열·특수 토큰·공백 규칙)이 다르면 같은 의미라도 토큰 시퀀스가 달라진다.

### 문제 5

프롬프트(조건) 구간과 응답(타깃) 구간.

## 15. 다음 강의와 연결

데이터 모양이 정해졌다. 이제 **학습 루프에서 마스크를 실제로 칠 차례**다.

다음 **제71강. SFT 구현**에서는 Pretraining 루프와의 차이를 코드로 고정하고, assistant 토큰에만 loss를 주는 SFT를 구현한다.

> 형식을 정했으면, 이제 “어디에 loss를 줄지”를 코드로 못 박자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [69강. Instruction Tuning의 개념](69강_Instruction_Tuning의_개념.md)
- **다음 강:** [71강. SFT 구현](71강_SFT_구현.md)

<!-- /LECTURE_NAV -->
