# 2권. Tokenizer와 Transformer

## 제47강. Encoder와 Decoder

### 1. 이번 강의에서 배울 것

제46강에서 Transformer Block 하나를 조립했다.  
이번 강의는 그 블록을 **어떤 지형도**로 쌓는지 본다. 핵심은 세 가지 가족이다.

1. **Encoder-only**
2. **Decoder-only**
3. **Encoder-Decoder**

이 강의를 마치면 다음을 말할 수 있어야 한다.

- Encoder Self-Attention과 Decoder Causal Self-Attention의 차이
- Cross-Attention이 필요한 이유
- 원 논문 Transformer(번역)가 Encoder-Decoder인 이유
- **GPT는 Decoder-only Causal LM**이라는 사실
- BERT류 Encoder-only와 GPT류 Decoder-only의 학습 목표 차이(개요)
- 제40강 Causal Mask → 제48강 Causal LM으로 이어지는 지도

### 2. 왜 이것을 배우는가

“Transformer”라는 말만 들으면 구조가 하나인 것처럼 느껴진다.  
실제로는 **마스크와 입출력 포트**가 다른 여러 제품군이 있다.

LLM(특히 챗 모델)을 이해하려면 최소한 다음 문장을 고정해야 한다.

> 오늘날 많은 GPT형 LLM은 Encoder가 없는 **Decoder-only** 스택이며, Self-Attention에 **Causal Mask**를 건다.

제48강에서 Causal LM 전체 구조를 그리기 직전의 지도 제작 강의다.

### 3. 먼저 알아야 할 개념

- Self-Attention / MHA (제39~41강)
- Causal Mask (제40강)
- Transformer Block (제46강)
- Next Token Prediction (제32강)

### 4. 핵심 개념 설명

#### 4.1 Encoder

**Encoder**는 입력 시퀀스 전체를 양방향으로 보며 맥락화한 표현을 만든다.

특징:

- Self-Attention에 **Causal Mask가 없음** (기본)
- 위치 \(i\)의 토큰이 왼쪽·오른쪽을 모두 참조 가능
- 출력은 “각 입력 토큰의 맥락 벡터” 서열

대표 사용:

- BERT류 masked language model
- 분류·문장 임베딩 등 (세부 과제는 다양)

직관:

```text
문장 전체를 한 번에 읽고 각 단어의 뜻을 채워 넣는다
```

#### 4.2 Decoder

**Decoder**는 (원 논문 맥락에서) 이미 생성한 출력 토큰을 보며 다음을 예측하는 쪽이다.

표준 Decoder 블록의 Attention은 두 종류일 수 있다.

1. **Masked (Causal) Self-Attention**  
   미래 출력을 보지 못함 (제40강)

2. **Cross-Attention** (Encoder-Decoder일 때)  
   Query는 Decoder 쪽, Key/Value는 Encoder 출력

Decoder-only 모델에서는 Cross-Attention이 없고, Causal Self-Attention만 반복한다.

#### 4.3 Encoder-Decoder

원 논문 “Attention Is All You Need”의 기계번역기:

```text
소스 문장 ──► Encoder 스택 ──► 메모리(맥락 벡터열)
                                 │
타깃 문장(교사강제) ──► Decoder 스택 ──► 다음 타깃 토큰
                      (causal self + cross)
```

왜 둘이 필요한가(설명):

- Encoder: 원문 전체를 양방향으로 이해
- Decoder: 번역문을 왼쪽부터 생성하며, 필요 시 원문 메모리를 조회(Cross-Attention)

#### 4.4 Cross-Attention

$$
Q = X_{\mathrm{dec}} W^Q,\quad
K = H_{\mathrm{enc}} W^K,\quad
V = H_{\mathrm{enc}} W^V
$$

- Decoder 상태가 “무엇을 물어볼지”(Query)
- Encoder 상태가 “어디에 정보가 있는지”(Key/Value)

Self-Attention과 수식은 같고, **Q의 출처와 K/V의 출처가 다르다**.

#### 4.5 Decoder-only Causal LM (GPT형)

```text
토큰 임베딩 (+ 위치: PE 또는 RoPE)
  → (Causal Transformer Block) × N
    → LayerNorm
      → LM Head → vocab logits
```

각 블록:

- Pre-LN
- Causal MHA
- Residual
- FFN
- Residual

사실:

- GPT-2/3, LLaMA 등 다수는 **Encoder 없는 Decoder-only**.
- 학습 목표는 다음 토큰 예측(Causal Language Modeling).

설명:

- Encoder-Decoder도 언어 생성에 쓸 수 있으나, 대규모 범용 LLM에서는 Decoder-only가 단순 스케일링·엔지니어링에서 유리하다는 선택이 지배적이었다.
- “Decoder-only가 항상 우월”은 만능 법칙이 아니다. 번역 등에서는 Encoder-Decoder가 여전히 유력한 선택일 수 있다.

### 5. 직관적으로 이해하기

세 가족을 한 장면으로:

| 가족 | 비유 |
|---|---|
| Encoder-only | 책을 통째로 읽고 각 문장에 주석을 달기 |
| Decoder-only | 이야기를 한 글자씩 이어 쓰기 (앞부분만 참고) |
| Encoder-Decoder | 원서를 옆에 두고 번역문을 한 줄씩 쓰기 |

GPT형 LLM은 세 번째가 아니라 **두 번째**에 가깝다.  
챗 인터페이스가 붙어도, 핵심 엔진은 Causal Decoder 스택이다.

### 6. 수학적으로 이해하기 (마스크 관점)

시퀀스 길이 \(T\)의 Attention score에 더해지는 마스크 \(M\):

**Encoder Self-Attention**

$$
M_{ij} = 0 \quad (\text{패딩 제외})
$$

**Decoder Causal Self-Attention**

$$
M_{ij} =
\begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

**Cross-Attention**

- Decoder 길이 \(T_{\mathrm{dec}}\), Encoder 길이 \(T_{\mathrm{enc}}\)
- score shape `(T_dec, T_enc)`
- 기본은 Encoder 패딩만 가림. “미래” 개념은 Decoder self 쪽에 있음

이 차이가 아키텍처 이름의 실체다.

### 7. 작은 비교 예제

문장: `BOS A B C` (학습 시 다음 토큰 예측)

Decoder-only:

- 위치 `A`는 `BOS, A`만 봄
- 위치 `B`는 `BOS, A, B`만 봄
- 목표: `A→B`, `B→C`, …

Encoder-only (MLM 스케치):

- `B`를 `[MASK]`로 가리고 양방향 맥락으로 `B`를 맞추기
- 마스크 토큰 예측이지, 왼→오른쪽 생성이 기본 목표는 아님

같은 “Transformer Block”이라도 **마스크와 손실**이 달라지면 제품이 달라진다.

### 8. 코드로 구조 스케치

```python
# architecture_families.py
"""세 가족의 차이만 드러내는 의사코드형 스케치."""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block  # 제46강 TransformerBlock 가정

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x, causal=True)


class BidirectionalBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x, causal=False)


class EncoderDecoderLayer(nn.Module):
    def __init__(self, self_attn, cross_attn, ffn, norm1, norm2, norm3):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ffn = ffn
        self.norm1, self.norm2, self.norm3 = norm1, norm2, norm3

    def forward(self, y: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # y: decoder stream, memory: encoder output
        y = y + self.self_attn(self.norm1(y), causal=True)
        y = y + self.cross_attn(self.norm2(y), kv=memory, causal=False)
        y = y + self.ffn(self.norm3(y))
        return y
```

완전한 구현은 제49~50강 프로젝트에서, Causal Decoder-only를 우선한다.

### 9. 실제 LLM에서는 어떻게 사용하는가

사실:

- 원 논문 Transformer: Encoder-Decoder (번역)
- BERT: Encoder-only
- GPT 계열·LLaMA 계열: Decoder-only Causal LM
- T5: Encoder-Decoder를 텍스트-투-텍스트로 일반화

설명:

- 이름이 “GPT”라고 해서 원 논문 Decoder와 내부가 완전히 같다고 단정하지 않는다.  
  Cross-Attention 유무, Norm 종류, 위치 인코딩이 다르다.
- 다만 **Causal Self-Attention 스택으로 다음 토큰을 예측**한다는 큰 그림은 공유한다.

제40강에서 배운 마스크가, 제48강 Causal LM의 기본 법칙이 되는 지점이 바로 이 Decoder-only 선택이다.

### 9.5 한 장으로 보는 선택 가이드

과제별로 자주 고르는 가족(경향이지 법칙 아님):

| 과제 | 흔한 선택 | 이유 스케치 |
|---|---|---|
| 기계번역 | Encoder-Decoder | 원문 양방향 + 타깃 생성 |
| 문장 분류/이해 | Encoder-only | 전체 문맥 양방향 |
| 챗/코드/일반 LM | Decoder-only | 다음 토큰 예측으로 스케일링 |
| 텍스트-투-텍스트 범용 | Encoder-Decoder (T5류) | 입력을 인코딩하고 출력을 생성 |

이 책이 3권에서 다루는 GPT Pretraining은 **Decoder-only** 경로다.  
따라서 제40강 Causal Mask가 “옵션”이 아니라 **기본 규칙**이 된다.

### 9.6 제46강 Block을 재사용하는 법

```text
TransformerBlock(causal=True)  × N  → GPT형 Decoder-only
TransformerBlock(causal=False) × N  → BERT형 Encoder-only
DecoderBlock = Causal Self + Cross + FFN → 원 논문 Decoder
```

Cross-Attention만 새로 추가하면 Encoder-Decoder가 된다.  
제49~50강 프로젝트는 먼저 Decoder-only를 완성한다.

### 10. 실습

1. 길이 4 시퀀스에서 Encoder용 마스크(모두 0)와 Causal 마스크를 각각 출력하라.
2. Cross-Attention의 score shape가 `(T_dec, T_enc)`가 되는 이유를 shape로 설명하라.
3. GPT형 모델을 Encoder-Decoder라고 부르는 문장이 왜 부정확한지 한 줄로 쓰라.
4. 제46강 Block에 `causal=True/False` 스위치가 있다면, Encoder-only/Decoder-only를 어떻게 재사용하는지 스케치하라.
5. (선택) 번역 데이터에서 Encoder 입력과 Decoder 입력이 어떻게 갈라지는지 도식으로 그려 보라.

### 11. 자주 하는 실수

1. **Transformer = 무조건 Encoder-Decoder**  
   원 논문만의 설정이다.

2. **GPT에 Cross-Attention이 있다고 가정**  
   표준 GPT형 Decoder-only에는 없다.

3. **Encoder가 ‘생성 못 함’으로 단정**  
   Encoder-only는 생성 목표로 보통 학습하지 않을, 불가능 선언과는 다르다.

4. **Causal Mask를 Encoder에 잘못 적용**  
   양방향 맥락이 깨진다.

5. **제목의 Decoder와 GPT Decoder-only를 혼동**  
   원 논문 Decoder는 Encoder 메모리를 본다. GPT는 그 메모리가 없다.

### 12. 핵심 정리

- Encoder는 양방향 Self-Attention으로 입력 맥락을 만든다.
- Decoder(원 논문)는 Causal Self-Attention + Cross-Attention으로 출력을 생성한다.
- Decoder-only는 Causal Self-Attention 스택만으로 언어 모델을 만든다.
- GPT형 LLM은 Decoder-only Causal LM이다.
- 제40강 마스크가 제48강 전체 아키텍처의 핵심 규칙으로 확장된다.

### 13. 핵심 용어

| 용어 | 설명 |
|---|---|
| Encoder | 양방향 맥락화 스택 |
| Decoder | 생성 측 스택(원 논문은 cross 포함) |
| Encoder-Decoder | 둘을 연결한 seq2seq Transformer |
| Decoder-only | Causal Self-Attention만 쌓은 LM |
| Cross-Attention | Decoder Query × Encoder Key/Value |
| Causal LM | 왼쪽 맥락으로 다음 토큰을 예측하는 언어 모델 |

### 14. 복습 문제

**문제 1.** Encoder Self-Attention에 Causal Mask가 기본으로 있는가?

**문제 2.** Cross-Attention에서 Q와 K/V의 출처는?

**문제 3.** GPT형 LLM의 가족 분류는?

**문제 4.** 원 논문 Transformer의 주 과제 설정은?

**문제 5.** 제40강과 제48강을 잇는 한 문장을 쓰라.

#### 정답과 해설

1. 없다(패딩 마스크 등은 있을 수 있음).

2. Q는 Decoder, K/V는 Encoder 출력.

3. Decoder-only Causal LM.

4. 기계번역 등 Encoder-Decoder 시퀀스 변환.

5. 제40강의 미래 차단 규칙이 Decoder-only 스택 전체에 적용되어 제48강 Causal LM이 된다.

### 15. 다음 강의와 연결

지도가 완성되었다.  
제48강 **Causal Language Model 구조**에서는 Embedding → (Block × N) → LM Head를 한 장의 설계도로 고정한다.

```text
제40강 Causal Mask
제41~46강 부품과 Block
제47강 Encoder/Decoder 지형도
  → 제48강 Causal LM 전체 구조
    → 제49~50강 Mini Transformer 구현
```

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제46강. Transformer Block 조립](46강_Transformer_Block_조립.md)
- **다음 강:** [제48강. Causal Language Model 구조](48강_Causal_Language_Model_구조.md)

<!-- /LECTURE_NAV -->
