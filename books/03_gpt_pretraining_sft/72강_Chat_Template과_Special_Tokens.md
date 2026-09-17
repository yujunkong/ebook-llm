# 72강. Chat Template과 Special Tokens
## 이번 강에서 배우는 내용

- Chat template이 messages → 단일 문자열/토큰 시퀀스로 가는 규칙임을 설명하기
- 역할·경계에 쓰이는 special tokens의 역할 (`bos`/`eos`/role 마커 등)
- `apply_chat_template` 개념(학습·추론 동일 적용)을 코드로 스케치하기
- train/infer 템플릿 불일치가 Instruction following 실패로 이어지는 이유
- 다음 제73강 LoRA로 넘어가기 전, SFT 입력 파이프라인의 마지막 퍼즐 맞추기

## 왜 중요한가?
같은 messages라도 사람이 붙이는 문자열이 달라지면 **다른 데이터**다.

```text
# A
User: 안녕
Assistant: 안녕하세요

# B
<|im_start|>user
안녕<|im_end|>
<|im_start|>assistant
안녕하세요<|im_end|>
```

의미는 비슷해 보여도 토큰 ID 시퀀스는 다르다. 모델은 의미를 읽는 것이 아니라 **토큰 분포**를 배운다.

따라서:

```text
학습: 템플릿 A + 마스크
추론: 템플릿 B
→ “지시 못 따름”처럼 보이는 자기 함정
```

Chat template은 이 함정을 막기 위한 **단일 렌더러**다.

## 선수 개념
- Special tokens 기초 (2권 제30강)
- messages 스키마 (제70강)
- SFT 프롬프트 마스크 (제71강)
- Tokenizer: `encode` / `decode`, vocab 확장

## 핵심 개념
### 3.1 Chat Template이란?

**Chat Template**은 `messages`(역할·내용 리스트)를 모델이 기대하는 **단일 프롬프트 문자열 또는 토큰 시퀀스**로 변환하는 규칙이다. 보통 다음을 포함한다.

- 역할 이름 표기 (`user`, `assistant`, `system`)
- 메시지 시작/끝 마커
- 공백·개행 규칙
- (선택) generation prompt: 추론 시 assistant 시작 마커만 열고 내용은 비움

의사코드:

```text
apply_chat_template(messages, add_generation_prompt=False|True)
  → string or token ids
```

### 3.2 Special Tokens이란?

**Special Tokens**는 일반 텍스트 단어가 아니라 **제어·경계**를 위해 vocab에 넣어 둔 토큰이다. 예:

| 종류 | 예(가명) | 역할 |
|---|---|---|
| BOS/EOS | `<s>`, `</s>` | 문장/시퀀스 시작·끝 |
| PAD | `<pad>` | 배치 길이 맞추기 |
| UNK | `<unk>` | 미지 문자 |
| Role/Chat | `<\|user\|>`, `<\|assistant\|>` | 대화 역할 경계 |
| EOT / IM end | `<\|im_end\|>` | 메시지 종료 |
| (도구 등) | `<\|tool\|>` … | 확장 인터페이스 |

이름은 모델 가족마다 다르다. **중요한 것은 문자열 자체가 아니라, 학습 때 그 ID가 어떤 의미로 쓰였는가**다.

Special token을 vocab에 추가하면:

1. `tokenizer`에 심볼 등록
2. 임베딩 행 추가(또는 평균 초기화 등)
3. (필요 시) LM head 행 추가
4. 템플릿 문자열에 그 심볼만 사용

### 3.3 `apply_chat_template` 개념

실무(예: Hugging Face Tokenizer)에서는 대략:

```python
ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # 추론 시 assistant 헤더까지
    return_tensors="pt",
)
```

학습 시에는 보통:

```python
ids = tokenizer.apply_chat_template(
    messages,  # 마지막이 assistant 정답 포함
    tokenize=True,
    add_generation_prompt=False,
)
```

그 다음 제71강처럼 assistant 구간만 labels로 남긴다. 일부 도구는 템플릿과 함께 **마스킹 헬퍼**를 제공한다. 원리를 모르면 헬퍼도 디버깅하기 어렵다.

### 3.4 Generation prompt

추론:

```text
messages = [
  {system...},
  {user: "2+2?"}
]
# add_generation_prompt=True  →  ... <assistant|>   (여기서 생성 시작)
```

학습:

```text
messages = [
  {user: "2+2?"},
  {assistant: "4"}
]
# 정답 토큰까지 포함, generation prompt 불필요
```

이 한 플래그를 학습/추론에 섞어 쓰면 경계가 밀린다.

### 3.5 Train / Infer 일관성

체크리스트:

| 항목 | 학습 | 추론 |
|---|---|---|
| 템플릿 버전 | vX | **동일 vX** |
| special tokens | 동일 맵 | 동일 맵 |
| system 사용 여부 | 분포 반영 | 동일 정책 |
| 개행/공백 | 템플릿에 위임 | 손으로 다시 짜지 않기 |
| 마스크 | 응답만 | (생성은 마스크 없음) |

“프롬프트를 예쁘게” 수동 수정하는 습관이 불일치의 주범이다.

## 직관적으로 이해하기
Chat template은 **연극 대본의 서식 파일(CSS가 아니라 각본 서식)** 이다. Special tokens는 **막간을 알리는 방울 소리**다.

배우(모델)는 방울 소리 다음에 대사가 나온다고 학습했다. 공연 날 방울을 다른 악기 소리로 바꾸면, 대사를 언제 시작할지 모른다.

## 작은 템플릿 예제
미니 교재용 초간단 템플릿(교육용; 실제 대형 모델 템플릿과는 이름이 다를 수 있음):

```text
{% for message in messages %}
<|{{ message.role }}|>
{{ message.content }}<|end|>
{% endfor %}
```

렌더 예:

```text
<|system|>
짧게 답하라.<|end|>
<|user|>
수도는?<|end|>
<|assistant|>
서울<|end|>
```

추론 시 `add_generation_prompt`:

```text
... (system, user까지 동일) ...
<|assistant|>
```

여기서 생성을 시작한다.

## 코드로 구현하기 — 미니 `apply_chat_template`
Jinja 없이 동작하는 교육용 구현:

```python
# mini_chat_template.py
"""교육용 chat template. 실전은 토크나이저 내장 템플릿을 우선한다."""

from __future__ import annotations

from typing import Literal

Role = Literal["system", "user", "assistant"]

def apply_chat_template(
    messages: list[dict],
    add_generation_prompt: bool = False,
) -> str:
    parts: list[str] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"bad role: {role}")
        parts.append(f"<|{role}|>\n{content}<|end|>\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)

def find_assistant_spans(text: str) -> list[tuple[int, int]]:
    """문자 오프셋 예시. 실전 마스크는 토큰 인덱스에서 하라."""
    spans = []
    start_tag = "<|assistant|>\n"
    end_tag = "<|end|>"
    pos = 0
    while True:
        s = text.find(start_tag, pos)
        if s < 0:
            break
        content_s = s + len(start_tag)
        e = text.find(end_tag, content_s)
        if e < 0:
            break
        spans.append((content_s, e))
        pos = e + len(end_tag)
    return spans

if __name__ == "__main__":
    msgs = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    train_s = apply_chat_template(msgs, add_generation_prompt=False)
    infer_s = apply_chat_template(msgs[:1], add_generation_prompt=True)
    print("TRAIN:\n", train_s)
    print("INFER:\n", infer_s)
    print("spans", find_assistant_spans(train_s))
```

예상 출력:

```text
TRAIN:
 <|user|>
2+2?<|end|>
<|assistant|>
4<|end|>

INFER:
 <|user|>
2+2?<|end|>
<|assistant|>

spans [(..., ...)]
```

문자 오프셋은 직관용이다. 제71강 마스크는 **같은 템플릿으로 만든 뒤 토큰화**한 ID 배열에서 잡는 것이 정석이다.

## Special tokens를 tokenizer에 등록하는 스케치
```python
# register_specials.py
"""개념 스케치 — 사용하는 토크나이저 API에 맞게 조정."""

SPECIALS = {
    "additional_special_tokens": [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
    ]
}

# Hugging Face 예시(개념):
# tokenizer.add_special_tokens(SPECIALS)
# model.resize_token_embeddings(len(tokenizer))
```

주의:

- 이미 템플릿이 있는 공개 chat 모델을 쓸 때는 **임의로 심볼을 바꿔 덮지 말 것**
- 미니 GPT(제68강 char tokenizer)에서는 문자 단위라 `<|`가 여러 토큰으로 쪼개질 수 있다. 교육용으로는 **역할 마커를 단일 문자 제어코드로 두거나**, 단어 단위 vocab에 마커를 통째로 넣는 편이 낫다.

## 마스크와 템플릿의 결합
권장 파이프라인:

```text
messages
  → apply_chat_template(..., add_generation_prompt=False)
    → tokenize
      → assistant token span 계산
        → labels = input_ids.clone(); labels[:span] / non-asst = -100
          → SFT loss
```

추론:

```text
messages (정답 없음)
  → apply_chat_template(..., add_generation_prompt=True)
    → tokenize
      → generate
        → decode (특수 토큰 스킵 옵션 일관되게)
```

이 두 경로가 **같은 `apply_chat_template` 구현**을 import해야 한다. train 스크립트에 문자열을 하드코딩하고 generate에 다른 하드코딩을 두지 말 것.

## 수식 보강 — Chat template

메시지 열 $(m_1,\ldots,m_k)$를 특수 토큰으로 직렬화합니다.

$$
\text{string}=\mathrm{Template}(m_1,\ldots,m_k)
$$

토크나이저 후 ID 열이 모델 입력이 됩니다. 역할(user/assistant/system) 경계가 학습 신호입니다.


## 수학적으로 이해하기 — 템플릿은 결정적 맵

Chat template $\mathcal{T}$는 메시지 리스트를 토큰열로 보내는 결정적（또는 규약상 결정적）함수입니다.

$$

\mathcal{T}:\ (m_1,\ldots,m_K)\ \mapsto\ x_{1:T}\in\{1,\ldots,V\}^T

$$

학습과 추론이 다른 $\mathcal{T}$를 쓰면, 모델이 보는 조건 분포가 달라집니다.

$$

p_\theta(y\mid \mathcal{T}_{\mathrm{train}}(c))
\;\neq\;
p_\theta(y\mid \mathcal{T}_{\mathrm{infer}}(c))
\quad\text{（일반적으로）}

$$

“지시 실패”의 상당수는 최적화 실패가 아니라 **좌표 변환 실패**입니다.

### Special token과 vocab 확장

역할 마커를 $k$개 추가하면

$$

V \leftarrow V+k
$$

이고, 임베딩·lm_head에 $k$행이 늘어납니다. 초기화는 평균 임베딩 복사 등이 흔합니다. **토큰 문자열만 바꾸고 ID를 안 넣으면** 마커가 서브워드로 쪼개져 경계가 붕괴합니다.

## 작은 숫자 예 — 두 템플릿의 토큰 수

동일 messages에 대해（설명용 가정）:

| 템플릿 | 렌더 길이 T | assistant 시작 위치 |
|---|---|---|
| A: `User:/Assistant:` | 24 | 18 |
| B: `<\|im_start\|>...` | 28 | 21 |

마스크를 위치 18 기준으로 칠했는데 추론은 B를 쓰면, 경계가 어긋납니다.  
단위 테스트는 **같은 messages → 같은 ids**를 학습/추론 경로에서 assert 합니다.

```python
ids_train = tok.apply_chat_template(msgs, add_generation_prompt=False)
ids_infer_prefix = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True)
# msgs 마지막이 assistant 정답일 때, prefix는 그 직전까지와 일치해야 함
```

## 직관적으로 이해하기 — 연극 무대 장치

Special tokens는 조명 큐입니다. `assistant` 조명이 켜진 뒤에만 배우가 말을 잇습니다.  
템플릿 불일치는 **다른 무대에서 리허설하고 본공연에 서는 것**과 같습니다.

## Generation prompt

추론 시

```text
... (user message) ... <assistant>
```

처럼 **빈 assistant 헤더**까지 붙이는 옵션이 `add_generation_prompt=True`입니다. 학습 때 본 패턴과 같아야 모델이 “이제 응답 토큰을 낼 차례”임을 통계적으로 압니다.

## 부록 A. 템플릿 불일치 증후군

| 관찰 | 점검 |
|---|---|
| 역할을 따라 씀（`User:`를 생성） | generation prompt/마sk |
| 갑자기 영어 메타 문구 | 학습 데이터 스타일·system |
| 특수기호를 글자로 분해 | special token 미등록 |
| 학습 loss↓·추론 실패 | $\mathcal{T}$ 불일치 최우선 |

## 부록 B. 최소 자체 템플릿 스케치

```python
def apply_chat_template(messages, add_generation_prompt=False):
    parts = []
    for m in messages:
        parts.append(f"<|{m['role']}|>\n{m['content']}<|end|>\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)
```

교육용입니다. 실전에서는 모델 카드의 공식 템플릿을 따릅니다.

## 부록 C. 수식 카드

$$

\begin{aligned}
&\text{train:}&& x=\mathcal{T}(c+\text{assistant answer})\\
&\text{infer:}&& x=\mathcal{T}(c,\ \text{add_generation_prompt}=1)\\
&\text{loss:}&& \sum_t m_t(-\log p_\theta(x_t\mid x_{<t}))
\end{aligned}

$$


<!-- enrich-batch2-72 -->
## Special Tokens

$$
\mathcal{S}=\{\mathrm{BOS},\mathrm{EOS},\mathrm{PAD},\mathrm{USR},\mathrm{AST}\}
$$

템플릿이 바뀌면 같은 모델도 분포가 달라집니다.

```python
tmpl = "<|user|>{q}<|assistant|>"
print(tmpl.format(q="hi"))
```

$$
p_\theta(y\mid \mathrm{tmpl}(x)) \neq p_\theta(y\mid x)
$$


<!-- enrich-pass-1f64 -->
## 수식 전개 — 템플릿은 결정적 맵

메시지 리스트 $M=((r_i,c_i))_{i=1}^{n}$에 대해 템플릿 $\mathcal{T}$는 문자열（또는 토큰열）을 만듭니다.

$$
x=\mathcal{T}(M)
$$

학습과 서빙에서 $\mathcal{T}$가 다르면 분포가 어긋납니다.

특수 토큰 집합 $\mathcal{S}=\{\texttt{bos},\texttt{eos},\texttt{im\_start},\ldots\}$에 대해

$$
\mathrm{id}:\mathcal{S}\to\mathbb{Z}
$$

가 tokenizer vocabulary에 등록되어 있어야 합니다. 미등록 문자열을 그대로 넣으면 **조각 토큰**으로 쪼개져 의미가 깨집니다.

### 마스크와의 결합

assistant 스팬만 남기는 마스크는 템플릿이 찍은 경계에 의존합니다.

$$
m_t=1\iff t\in\mathrm{span}_{\mathcal{T}}(\text{assistant})
$$

## Shape 표 — apply_chat_template

| 입출력 | 예 |
|---|---|
| messages | `List[{role, content}]` |
| rendered string | `str` |
| `input_ids` | `(T,)` |
| special token ids | scalar씩 |

## 구현 스케치 — 경계 기록

```python
def render_with_spans(messages, tok):
    ids, spans, roles = [], {}, []
    for msg in messages:
        piece = format_role(msg)  # 역할별 특수 토큰 포함
        start = len(ids)
        ids.extend(tok(piece))
        end = len(ids)
        spans.setdefault(msg["role"], []).append((start, end))
    return ids, spans
```

spans로 labels를 만들면 제71강 마스크와 정확히 맞습니다.

## 실패 모드 — Chat Template

| 실패 | 증상 | 처방 |
|---|---|---|
| train≠serve 템플릿 | 지시 준수 실패 | 단일 소스 |
| special token 미등록 | 조각화 | `add_special_tokens` |
| 시스템 롤 무시 | 톤 불안정 | 템플릿에 명시 |
| 공백/개행 불일치 | 토큰 어긋남 | 스냅샷 테스트 |

## 실습 코드 — 스냅샷 테스트

```python
def test_template_snapshot(apply_fn, messages, expected):
    got = apply_fn(messages)
    assert got == expected, repr(got)
```

템플릿을 바꾸면 스냅샷을 의도적으로 갱신합니다. 조용히 바꾸지 마세요.

## 수식 보강 — 멀티턴 길이

$$
T
=
\big|\mathcal{T}(M)\big|
\le T_{\max}
$$

초과 시 오래된 턴부터 자르는 truncation 정책을 문서화하세요. 응답 스팬이 잘리면 학습 신호가 사라집니다.

## LLM에서는 어디에 사용될까?
- 모델 카드에 `chat_template`(종종 Jinja)가 배포되는 경우가 많다.
- 토크나이저 설정과 가중치가 한 쌍이다. 토크나이저만 다른 버전으로 바꾸면 특수 토큰 ID가 어긋난다.
- 멀티모달·도구 호출 모델은 역할이 더 늘어나고 템플릿이 길어진다. 원리는 동일: **단일 렌더러 + 일관된 specials**.
- 서빙 엔진(5권 vLLM 등)도 템플릿을 알아야 stop token·역할 경계를 맞춘다.

특정 상용 모델의 숨은 템플릿을 추측해 “공식”처럼 적지 않는다. 사용 중인 체크포인트의 문서를 따른다.

## 실습
### 실습 1 — Train/Infer 문자열 비교

§7 코드로 train/infer 문자열을 출력하고, infer가 train의 접두(정답 제외)와 일치하는지 확인하라.

### 실습 2 — 불일치 실험 (의도적)

학습은 `<|user|>` 템플릿, 추론은 `User:` 평문을 써  generat ion 품질이 어떻게 깨지는지 미니 모델로 관찰하라. (제69강 개념의 실험 버전)

### 실습 3 — Stop tokens

생성 시 `<|end|>` 또는 EOS에서 멈추도록 stop 규칙을 추가하라. 멈추지 않으면 다음 user 마커를 환각할 수 있다.

### 실습 4 — labels 디코드

마스크 후 `labels != -100`인 ID만 decode해 “모델이 외워야 할 문자열”이 응답 내용과 일치하는지 눈으로 확인하라.

## 자주 하는 실수
1. **학습·추론 템플릿 분기 구현**  
   복사-수정 순간 불일치가 생긴다. 함수 하나로.

2. **special token을 일반 텍스트로만 취급**  
   BPE가 `<|user|>`를 조각내면 경계가 흐려진다. 반드시 special로 등록하거나 통짜 토큰화.

3. **`add_generation_prompt` 혼동**  
   학습 데이터에 빈 assistant 헤더가 중복되거나, 추론에 헤더가 없어 역할이 안 열린다.

4. **decode 시 special 제거 정책 불일치**  
   평가 문자열이 학습 때와 달라 보인다.

5. **system을 추론에만 추가**  
   학습 분포에 없던 전역 규칙이 갑자기 나타나면 행동이 불안정해질 수 있다.

## 핵심 요약
- Chat template은 messages를 모델 입력 시퀀스로 만드는 **단일 계약**이다.
- Special tokens는 역할·경계·종료를 vocab에 고정한다.
- `apply_chat_template` + (추론 시) `add_generation_prompt` 패턴을 이해한다.
- SFT 마스크는 템플릿이 만든 토큰 시퀀스 위에서만 정확하다.
- Train/Infer 일관성이 Instruction following의 전제다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Chat template | 대화 messages → 프롬프트 변환 규칙 |
| `apply_chat_template` | 템플릿을 적용하는 API/함수 개념 |
| Special tokens | 제어·경계용 vocab 심볼 |
| Generation prompt | 추론 시 assistant 시작 헤더를 붙이는 옵션 |
| Role marker | user/assistant/system 구간을 표시하는 토큰·문자열 |
| Stop token | 생성 종료를 유발하는 토큰 |
| Train/infer consistency | 학습과 추론의 템플릿·토크나이저 일치 |

## 연습문제
### 문제 1 (개념)

Chat template이 필요한 이유를 “토큰 분포” 관점에서 한 문장으로.

### 문제 2 (API)

`add_generation_prompt=True`를 쓰는 시점과 `False`를 쓰는 시점을 구분하시오.

### 문제 3 (특수 토큰)

Role marker를 special로 등록하지 않고 일반 BPE에 맡기면 어떤 문제가 생기는가?

### 문제 4 (디버깅)

모델이 응답 끝에 곧바로 `<|user|>`를 생성하기 시작할 때, 템플릿/stop 측면에서 의할 후보 두 가지를 쓰시오.

### 문제 5 (연결)

제71강 마스크, 이번 템플릿, 다음 제73강 LoRA는 SFT 파이프라인에서 각각 어떤 층을 담당하는가?

---

## 정답 및 해설
### 문제 1

모델은 의미가 아니라 렌더된 토큰 시퀀스의 분포를 배우므로, messages를 항상 같은 규칙으로 토큰열에 옮겨야 학습과 추론이 같은 분포에 있기 때문이다.

### 문제 2

True: 정답 assistant가 없는 추론 입력에서 assistant 헤더까지 열어 생성을 시작할 때. False: 정답이 포함된 학습 시퀀스를 만들 때.

### 문제 3

마커가 여러 서브워드로 쪼개져 역할 경계가 불안정해지고, 마스크·stop·해석이 어려워진다.

### 문제 4

예: (1) 학습 데이터에서 메시지 종료 토큰을 충분히/일관되게 쓰지 않음 (2) 생성 시 end/eos stop 미설정 (3) generation prompt/개행 불일치.

### 문제 5

마스크=손실 지지 집합, 템플릿=입력 렌더 계약, LoRA=전체 가중치 대신 저랭크 어댑터로 효율적 미세조정.

## 다음 강의와 연결
이제 SFT의 **데이터 형식 → 손실 마스크 → 템플릿/특수 토큰**이 한 줄로 연결되었다.

다음 **제73강. LoRA**에서는 모든 파라미터를 업데이트하지 않고도 Instruction Tuning을 수행하는 **저랭크 어댑터**를 배운다. 그 위 QLoRA(제74강), SFT 평가(제75강), Mini GPT+SFT 프로젝트(제76강)로 이어진다.

> 입력을 한 방식으로 고정했다면, 이제 “무엇을 얼마나 업데이트할지”를 효율적으로 설계하자.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [71강. SFT 구현](71강_SFT_구현.md)
- **다음 강:** [73강. LoRA](73강_LoRA.md)

<!-- /LECTURE_NAV -->
