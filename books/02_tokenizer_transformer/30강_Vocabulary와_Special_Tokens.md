# 2권. Tokenizer와 Transformer

## 제30강. Vocabulary와 Special Tokens

### 1. 이번 강의에서 배울 것

29강까지로 텍스트를 Subword로 자르고 id로 바꾸는 법을 익혔다. 그런데 Vocabulary에는 일반 토큰만 있는 것이 아니다. 모델과 학습 루프를 위해 **예약된 기호**가 들어간다.

이 강의를 마치면 다음을 말할 수 있어야 한다.

- **Vocabulary size**가 Embedding·LM Head 파라미터에 미치는 영향
- `<unk>`, `<pad>`, `<bos>`, `<eos>` 등 **Special Tokens**의 역할
- 패딩과 Attention Mask가 왜 한 세트인지
- 챗/지시 모델에서 대화 템플릿 special token이 하는 일
- 토큰 id 공간을 설계할 때 흔히 하는 실수

### 2. 왜 이것을 배우는가

같은 “토크나이저”라도 special token 처리가 다르면 학습이 달라진다.

```text
문장 A: "Hello"
문장 B: "Hello world, this is longer"
```

배치로 묶으려면 길이를 맞춰야 하고, 그때 `<pad>`가 필요하다.  
생성 종료를 알려면 `<eos>`가 필요하다.  
대화에서 역할이 바뀌는 지점을 표시하려면 `<|im_start|>` 같은 마커가 필요하다.

Special Token은 “장식”이 아니라 **제어 신호**이다.

### 3. 먼저 알아야 할 개념

- Token ID / Vocabulary (27강)
- Subword / BPE (28·29강)
- 배치 학습과 Tensor shape `[B, T]` (1권 22강)
- Softmax가 vocab 전체에 걸린다는 감각 (33강에서 엄밀화)
- Embedding 행렬 shape `[V, d]` (31강에서 상세)

### 4. 핵심 개념 설명

#### 4.1 Vocabulary (어휘 사전) 다시 보기

**Vocabulary**는 모델이 입·출력으로 사용할 수 있는 토큰의 유한 집합이다.

$$
V = \{t_0, t_1, \ldots, t_{|V|-1}\}
$$

각 토큰은 고유 id를 갖는다.

$$
\text{id}: V \rightarrow \{0,1,\ldots,|V|-1\}
$$

LLM의 마지막 층은 대개 각 위치에서 \(|V|\)개의 점수를 낸다. (logit)

```text
logits shape = [batch, seq_len, vocab_size]
```

따라서 Vocabulary는 Tokenizer만의 문제가 아니라 **모델 출력 차원**의 문제이기도 하다.

#### 4.2 Vocabulary Size의 비용

Embedding:

$$
\#\text{params}_{\text{emb}} \approx |V| \times d
$$

Language Model Head (출력 투영):

$$
\#\text{params}_{\text{lm head}} \approx d \times |V|
$$

(가중치 공유를 쓰면 Embedding과 LM Head를 묶어 절약하기도 한다.)

예:

```text
|V| = 50,000
d   = 768
emb ≈ 50,000 × 768 ≈ 38.4M parameters
```

Vocab을 두 배로 키우면 이 부분만으로도 수천만 파라미터가 늘어날 수 있다.  
반대로 vocab이 너무 작으면 토큰 서열이 길어져 Transformer 계산이 비싸진다. (28강 트레이드오프)

#### 4.3 Special Tokens (특수 토큰)

**Special Tokens(스페셜 토큰, 특수 토큰)**는 일반 텍스트 조각이 아니라, **파이프라인·학습·생성 제어**를 위해 Vocabulary에 예약된 토큰이다.

대표 목록:

| 토큰 | 이름 | 역할 |
|---|---|---|
| `<unk>` | Unknown | 사전에 없는 입력을 대체 |
| `<pad>` | Padding | 배치 길이 맞춤용 무의미(마스크될) 자리 |
| `<bos>` | Beginning Of Sequence | 서열 시작 표시 |
| `<eos>` | End Of Sequence | 서열 종료 표시 |
| `<sep>` | Separator | 문장/구간 구분 (BERT 등) |
| `<cls>` | Classification | 분류용 집약 토큰 (BERT 등) |
| `<mask>` | Mask | MLM용 가림 토큰 |

GPT식 Causal LM에서는 `<bos>`/`<eos>` 명칭 대신 다른 문자열을 쓰기도 한다.  
중요한 것은 **이름보다 역할**이다.

#### 4.4 `<unk>` — Unknown Token

**UNK(Unknown Token)**는 Vocabulary로 표현할 수 없는 입력을 대체하는 특수 토큰이다.

Word Tokenizer에서는 빈번하다.  
Byte-level BPE에서는 이론상 거의 불필요하다. (모든 바이트가 vocab에 있으면)

그래도 실무에서 UNK가 남는 경우:

- 구형 word/char vocab
- 일부 SentencePiece 설정
- 잘못된 전처리로 생긴 이상한 심볼을 강제로 매핑

UNK가 많으면 모델은 “무언가 있다”만 보고 세부 정보를 잃는다.

#### 4.5 `<pad>` — Padding Token

**PAD(Padding Token)**는 배치 내 서열 길이를 맞추기 위해 뒤에(또는 앞에) 채우는 토큰이다.

```text
ids_A = [10, 20, 30]
ids_B = [10, 20, 30, 40, 50]
패딩 후 (pad_id=0):
batch = [
  [10, 20, 30,  0,  0],
  [10, 20, 30, 40, 50],
]
```

PAD는 의미가 없다. 따라서:

1. **Loss**에서 PAD 위치는 무시해야 한다. (`ignore_index`)
2. **Attention**에서 PAD 위치는 마스크해야 한다. (`attention_mask`)

PAD를 실제 토큰처럼 학습에 포함하면, 모델이 “패딩을 예측”하느라 신호가 오염된다.

#### 4.6 `<bos>` / `<eos>`

**BOS(Beginning Of Sequence)**는 서열의 시작을 알린다.  
**EOS(End Of Sequence)**는 서열의 끝을 알린다.

생성 루프 (개념):

```text
ids = [BOS] 또는 프롬프트 ids
loop:
  logits = model(ids)
  next = sample(logits[-1])
  if next == EOS: break
  ids.append(next)
```

학습 데이터에도 EOS를 붙여, 모델이 “여기서 멈추는 법”을 배운다.

모델/프레임워크마다:

- BOS를 항상 붙이는 경우
- 프롬프트만 넣고 EOS만 쓰는 경우
- 챗 템플릿의 eot 토큰이 EOS 역할을 겸하는 경우

문서와 `tokenizer` 설정을 확인해야 한다.

#### 4.7 Attention Mask

**Attention Mask**는 “이 위치를 볼 수 있는가?”를 표시하는 텐서이다.

패딩과 함께 쓰는 기본 형태:

```text
input_ids:      [10, 20, 30, 0, 0]
attention_mask: [ 1,  1,  1, 0, 0]
```

- `1`: 유효 토큰
- `0`: 패딩 (무시)

Causal LM에서는 여기에 **미래 토큰을 가리는 causal mask**가 추가로 걸린다. (이후 Attention 강의)

#### 4.8 챗/LLM Special Tokens

지시 튜닝·챗 모델은 역할 경계를 special token으로 표시하는 경우가 많다.

개념 예 (모델마다 문자열 다름):

```text
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
안녕?<|im_end|>
<|im_start|>assistant
```

왜 필요한가?

1. 모델이 “지금 누구 말인가”를 토큰 수준에서 구분한다.
2. 생성 종료(`im_end` / `eot`)를 명확히 학습한다.
3. 시스템 프롬프트와 사용자 입력이 한데 섞여 생기는 모호함을 줄인다.

Special Token을 일반 텍스트처럼 이스케이프하지 않으면, 사용자가 같은 문자열을 입력해 **템플릿을 위조**할 위험이 있다. 프로덕션에서는 토크나이저/템플릿 레이어에서 막는다.

### 5. 직관적으로 이해하기

Vocabulary를 도시 전화번호부에 비유한다.

- 일반 토큰: 실제 가입자 번호
- `<unk>`: “조회 불가” 안내석
- `<pad>`: 빈 칸 채우기용 더미 번호 (실제 통화에 쓰면 안 됨)
- `<bos>` / `<eos>`: 통화 시작음 / 종료음
- 챗 special tokens: “운영자 멘션”, “고객 멘션”처럼 **채널을 바꾸는 신호**

모델은 이 번호들의 서열만 본다. 예약 번호의 의미를 Embedding이 학습한다.

### 6. 작은 숫자로 직접 보기

작은 vocab:

```text
0: <pad>
1: <unk>
2: <bos>
3: <eos>
4: i
5: love
6: cats
7: .
```

문장 두 개:

```text
A: "i love cats ."
B: "i love ."
```

BOS/EOS를 붙인 id:

```text
A: [2, 4, 5, 6, 7, 3]
B: [2, 4, 5, 7, 3]
```

배치 패딩 (max_len=6 이미 A가 6, B는 pad):

```text
B_padded: [2, 4, 5, 7, 3, 0]
mask_B:   [1, 1, 1, 1, 1, 0]
```

학습 타깃(next-token) 스케치:

```text
입력:  [2, 4, 5, 6, 7]
정답:  [4, 5, 6, 7, 3]
```

PAD 위치의 정답은 `ignore_index`로 두어 Loss에 안 넣는다.

### 7. 코드로 구현하기

```python
# special_tokens_batch.py
from typing import Dict, List, Tuple

import torch


class Vocab:
    def __init__(self, tokens: List[str]):
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    def encode_word_level(self, text: str, add_bos_eos: bool = True) -> List[int]:
        ids = []
        if add_bos_eos:
            ids.append(self.bos_id)
        for t in text.lower().split():
            ids.append(self.token_to_id.get(t, self.unk_id))
        if add_bos_eos:
            ids.append(self.eos_id)
        return ids


def pad_batch(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """오른쪽 패딩 + attention mask."""
    max_len = max(len(s) for s in seqs)
    batch = []
    mask = []
    for s in seqs:
        pad_len = max_len - len(s)
        batch.append(s + [pad_id] * pad_len)
        mask.append([1] * len(s) + [0] * pad_len)
    return (
        torch.tensor(batch, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
    )


def make_lm_labels(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Causal LM용 타깃: 한 칸 shift.
    패딩 위치는 -100 (PyTorch CrossEntropy ignore_index 관례).
    """
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    labels[input_ids == pad_id] = -100
    # shift 후 생긴 타깃에서도 pad를 무시
    # (간단히 input pad 위치를 기준으로 처리; 정교화 가능)
    return labels


if __name__ == "__main__":
    vocab = Vocab(
        ["<pad>", "<unk>", "<bos>", "<eos>", "i", "love", "cats", "dogs", "."]
    )
    s1 = vocab.encode_word_level("i love cats .")
    s2 = vocab.encode_word_level("i love .")
    s3 = vocab.encode_word_level("i love birds .")  # birds → unk

    ids, mask = pad_batch([s1, s2, s3], vocab.pad_id)
    labels = make_lm_labels(ids, vocab.pad_id)

    print("input_ids:\n", ids)
    print("attn_mask:\n", mask)
    print("labels:\n", labels)
    print("unk example ids:", s3)
```

실행하면 배치 shape과 ignore된 label(`-100`)을 눈으로 확인할 수 있다.

### 8. PyTorch Embedding과의 연결 (미리보기)

```python
import torch.nn as nn

V, d = 9, 4
emb = nn.Embedding(num_embeddings=V, embedding_dim=d, padding_idx=0)
# padding_idx=0 이면 <pad> 벡터는 학습 중 0으로 유지되는 관례적 설정
```

`padding_idx`를 지정하면 해당 행의 grad가 쌓이지 않는다.  
PAD를 “의미 있는 토큰”으로 배우고 싶지 않을 때 유용하다.

상세한 Embedding 자체는 31강에서 다룬다.

### 9. 실제 LLM에서는 어떻게 사용하는가

Hugging Face식 속성 예:

```text
tokenizer.pad_token
tokenizer.eos_token
tokenizer.bos_token
tokenizer.unk_token
tokenizer.pad_token_id
...
```

체크할 것:

1. `pad_token`이 `None`인 모델이 있다. (GPT-2 등)  
   → 학습 시 `eos`를 pad로 임시 지정하는 관례가 있으나, **마스크를 반드시** 걸어야 한다.
2. `model.resize_token_embeddings(len(tokenizer))`  
   → special token을 추가하면 Embedding/LM Head 크기를 늘려야 한다.
3. Chat template  
   → `tokenizer.apply_chat_template(...)`가 special token을 자동 삽입한다.

사전학습 데이터 포맷 예:

```text
문서1 토큰들 ... <eos> 문서2 토큰들 ... <eos> ...
```

긴 스트림을 고정 길이로 자를 때 EOS가 문서 경계를 배우는 힌트가 된다.

추론 시:

```text
stop sequences = [eos_token, <|eot_id|>, ...]
```

EOS를 무시하고 계속 샘플링하면 무한 생성·환각적 연장이 길어진다.

### 10. Vocab 설계 체크리스트

실무에서 Vocabulary를 만지거나 고를 때:

1. **크기**: 모델 예산(`V × d`)과 토큰 효율을 같이 본다.
2. **특수 토큰 예약**: id 앞쪽을 고정하면 디버깅이 쉽다.
3. **추가 토큰**: 도메인 심볼, 툴콜 마커, 이미지 placeholder 등.
4. **하위 호환**: 기존 모델에 토큰을 추가하면 새 행은 랜덤 초기화 → 추가 학습 필요.
5. **다국어**: 특정 언어 스크립트가 과소표현되면 토큰 낭비가 커진다.
6. **보안**: 사용자 입력이 special token 문자열을 그대로 주입하지 못하게 한다.

### 11. 실습

#### 실습 1. 패딩과 마스크

서로 다른 길이의 문장 3개를 encode하고 `pad_batch`로 묶어, mask 합이 원래 토큰 수와 같은지 검증하시오.

#### 실습 2. UNK 비율

작은 word vocab으로 위키 문장 10개를 encode해 `<unk>` 비율을 계산하시오.  
같은 문장을 29강 BPE로 처리하면 비율이 어떻게 바뀌는지 관찰하시오.

#### 실습 3. Special Token 추가 시뮬레이션

vocab 끝에 `<|tool|>`를 추가했다고 가정하고:

- 새 id
- Embedding 행렬에 추가해야 할 행 수
- 기존 id가 유지되어야 하는 이유

를 쓰시오.

#### 실습 4. 챗 템플릿 분해

공개 모델 하나의 chat template 문자열을 찾아, special token이 어디에 삽입되는지 위치를 표시하시오.

### 12. 자주 하는 실수

1. **PAD를 Loss에 포함**  
   모델이 패딩 id를 “정답”처럼 배운다.

2. **PAD를 Attention에서 가리지 않음**  
   의미 없는 위치가 문맥을 오염시킨다.

3. **`pad_token = eos_token`으로 두고 마스크 생략**  
   종료 신호와 빈칸이 섞인다.

4. **Special Token 추가 후 Embedding resize 잊음**  
   index out of range 또는 잘못된 공유 가중치.

5. **챗 마커를 일반 텍스트로만 넣음**  
   토크나이저가 한 토큰으로 안 묶고 조각내면 역할 신호가 약해진다. (전용 id로 추가하는 편이 낫다)

6. **BOS/EOS 정책을 학습·추론에서 불일치**  
   학습 때 붙이던 EOS를 추론 stop에서 빼먹으면 길이가 폭주한다.

### 13. 핵심 정리

- Vocabulary 크기 \(|V|\)는 Embedding·LM Head 파라미터와 Softmax 차원에 직결된다.
- Special Tokens는 제어 신호이다: UNK/PAD/BOS/EOS 및 챗 마커.
- PAD는 Mask·Loss ignore와 함께 써야 한다.
- 챗 LLM은 역할 경계를 special token으로 명시하는 경우가 많다.
- Tokenizer와 모델의 vocab 공간은 항상 동기화되어야 한다.

### 14. 핵심 용어

| 용어 | 의미 |
|---|---|
| Vocabulary size \|V\| | 토큰 종류 수 |
| Special Token | 제어·예약 토큰 |
| UNK | 미등록 대체 토큰 |
| PAD | 길이 맞춤 토큰 |
| BOS / EOS | 시작 / 종료 토큰 |
| Attention Mask | 유효 위치 표시 |
| `ignore_index` | Loss에서 무시할 label id |
| Chat Template | 대화 역할을 토큰열로 직렬화하는 형식 |

### 15. 복습 문제

#### 문제 1 (계산)

\(|V|=32000\), \(d=4096\)일 때 Embedding 파라미터 수는?

#### 문제 2 (개념)

`<pad>`를 Loss에 넣으면 안 되는 이유를 쓰시오.

#### 문제 3 (구분)

`<unk>`와 `<eos>`의 역할 차이를 한 문장씩 쓰시오.

#### 문제 4 (코드)

`input_ids=[2,4,5,6,3,0,0]`이고 pad=0일 때, 기본 `attention_mask`는?

#### 문제 5 (LLM 연결)

챗 모델에서 `<|im_start|>assistant` 같은 마커가 필요한 이유를 설명하시오.

---

### 정답 및 해설

#### 문제 1

\(32000 \times 4096 = 131{,}072{,}000\) (약 1.31억)

#### 문제 2

패딩은 의미가 없는 채움 기호인데, Loss에 넣으면 모델이 그 id를 예측하도록 학습되어 신호와 평가가 왜곡된다.

#### 문제 3

`<unk>`: 표현 불가능한 입력을 대체.  
`<eos>`: 서열/생성의 종료를 표시.

#### 문제 4

`[1,1,1,1,1,0,0]`

#### 문제 5

모델이 현재 생성 주체(assistant)와 구간 경계를 토큰 수준에서 인식하게 하여, 역할 혼동을 줄이고 종료·형식 학습을 돕는다.

### 16. 다음 강의와 연결

이번 강의에서 Vocabulary와 Special Tokens라는 “번호부의 예약석”을 정리했다.

다음 **제31강. Embedding — 토큰을 벡터로**에서는, id가 실제로 신경망이 계산할 수 있는 **밀집 벡터**로 바뀌는 과정, one-hot과의 차이, Embedding 행렬 lookup, 학습되는 파라미터를 다룬다.

이전 강의: **제29강. BPE Tokenizer 직접 구현**  
다음 강의: **제31강. Embedding — 토큰을 벡터로**

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제29강. BPE Tokenizer 직접 구현](29강_BPE_Tokenizer_직접_구현.md)
- **다음 강:** [제31강. Embedding — 토큰을 벡터로](31강_Embedding_토큰을_벡터로.md)

<!-- /LECTURE_NAV -->
