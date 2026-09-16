# 3권. GPT Pretraining과 SFT

## 제61강. Tokenization Pipeline과 Dataset Packing

### 1. 이번 강의에서 배울 것

제60강에서 Pretraining Dataset의 **문서 구성·필터·분할**을 다뤘다. 이번 강의는 그 문서를 모델이 먹을 수 있는 형태, 즉 **토큰 ID 시퀀스**로 바꾸고, 고정 길이 `block_size`에 맞게 **Packing(패킹)** 하는 파이프라인을 고정한다.

이 강의를 마치면 다음을 할 수 있어야 한다.

- 문서 → tokenize → token ids → 고정 길이 청크까지의 흐름을 설명한다.
- BOS / EOS가 packing에서 어떤 역할을 하는지 구분한다.
- Packing과 Padding의 차이를 표로 정리한다.
- 문서 경계를 넘는 Attention이 왜 문제인지, 어떻게 막는지를 말한다.
- 작은 코드로 packing과 label/mask를 직접 만든다.

### 2. 왜 이것을 배우는가

Pretraining의 학습 단위는 “파일”이 아니라 **고정 길이 토큰 윈도우**다. GPU는 `[B, T]` 텐서를 원한다. 문서는 길이가 제각각이다.

```text
문서 A (짧음)   문서 B (김)        문서 C (중간)
   │               │                  │
   └───────────────┴──────────────────┘
                     │
              tokenize → ids
                     │
         packing / padding → [B, T]
                     │
              Causal LM forward
```

여기서 설계를 잘못하면 두 가지가 동시에 망가진다.

1. **효율**: 패딩만 쓰면 유효 토큰 비율이 떨어져 같은 GPU 시간에 보는 토큰 수가 줄어든다.
2. **신호**: 문서를 이어 붙일 때 경계를 무시하면, 앞 문서의 끝이 뒤 문서의 “과거 맥락”이 되어 버린다.

제62강 Training Loop는 이 파이프라인이 만든 배치를 전제로 돈다. 지금 경계를 분명히 해야 한다.

### 3. 먼저 알아야 할 개념

1. **Tokenizer / Vocabulary / Special Tokens** — 제27~30강
2. **Causal LM / Next Token Prediction** — 제32, 48, 57강
3. **Causal Mask / Attention Mask** — 제40강, 제30강
4. **Cross Entropy와 `ignore_index`** — 제34강
5. **Pretraining Dataset 구성** — 제60강

아직 Optimizer·AMP·Checkpoint는 제63~65강이다. 이번 강의의 산출물은 **정수 텐서 배치**다.

### 4. 핵심 개념 — Tokenization Pipeline

**Tokenization Pipeline**이란 원문 텍스트를 모델 입력 정수 열로 바꾸는 일련의 단계다.

전형적인 단계는 다음과 같다.

```text
1. 문서 로드 (jsonl / parquet / shard)
2. (선택) 정규화: 줄바꿈, 제어문자, 인코딩 정리
3. tokenize(text) → List[int]
4. special token 삽입 (BOS/EOS 등, 정책에 따름)
5. packing 또는 padding으로 길이 T 맞춤
6. input_ids / labels / attention_mask 생성
7. DataLoader가 [B, T] 배치로 공급
```

#### 4.1 용어 정의

| 용어 | 의미 |
|---|---|
| **문서(document)** | 학습 말뭉치의 논리적 단위(웹 페이지, 책 챕터, 코드 파일 등) |
| **토큰 ID** | Vocabulary 상의 정수 인덱스 |
| **시퀀스 길이 $T$** | 한 샘플(윈도우)의 토큰 수. 보통 `block_size` / `seq_len` |
| **Packing** | 여러 짧은 문서를 이어 붙여 길이 $T$를 채우는 방식 |
| **Padding** | 짧은 시퀀스 뒤를 PAD로 채워 길이 $T$를 맞추는 방식 |
| **EOS** | End-Of-Sequence. 문서(또는 생성) 종료 표지 |
| **BOS** | Beginning-Of-Sequence. 시작 표지(쓰는 토크나이저만) |

사실(관행): GPT-2 계열 학습에서는 문서 사이에 EOS를 넣고 packing하는 패턴이 흔하다. BOS를 항상 쓰는지는 토크나이저·레시피마다 다르다. “모든 LLM이 BOS를 쓴다”는 사실이 아니다.

### 5. 문서에서 토큰 ID까지

#### 5.1 한 문서의 tokenize

개념 코드:

```python
# 가정: tokenizer.encode(text) -> List[int]
# eos_id: 문서 종료 토큰 ID (예: tokenizer.eos_token_id)

def tokenize_document(text: str, tokenizer, eos_id: int, add_eos: bool = True) -> list[int]:
    """문서 하나를 토큰 ID 리스트로 변환한다."""
    ids = tokenizer.encode(text)  # 구현체에 따라 add_special_tokens 옵션이 다름
    if add_eos:
        # 문서 끝에 EOS를 붙여 "여기서 끊김"을 모델에 알린다.
        if len(ids) == 0 or ids[-1] != eos_id:
            ids = ids + [eos_id]
    return ids
```

포인트:

- `encode`가 이미 special token을 넣는 구현도 있다. **이중 삽입**을 피하려면 토크나이저 문서를 확인해야 한다.
- 빈 문서는 스킵하는 편이 안전하다.
- 매우 긴 문서는 tokenize 후 **자르거나** packing 단계에서 여러 윈도우로 나눈다.

#### 5.2 타깃(labels)과의 관계

Causal LM에서는 보통:

```text
input_ids:  t0  t1  t2  t3  ...  t_{T-1}
labels:     t1  t2  t3  t4  ...  t_T      (한 칸 시프트)
```

구현에서는 같은 `ids`를 두고 loss 함수 안에서 shift하거나, 미리 labels를 만들어 둔다. 어느 쪽이든 **PAD나 문서 경계에서 무시할 위치**를 `ignore_index`(흔히 `-100`)로 표시하는 패턴이 많다.

### 6. Packing이란 무엇인가

**Packing(시퀀스 패킹)**은 여러 문서의 토큰을 이어 붙여 하나의 길이 $T$ 윈도우를 채우는 기법이다.

직관:

```text
문서A: [a1 a2 a3 EOS]
문서B: [b1 b2 b3 b4 b5 EOS]
문서C: [c1 c2 EOS]

T = 12 로 packing한 한 샘플 예:
[a1 a2 a3 EOS b1 b2 b3 b4 b5 EOS c1 c2]
```

장점(설명):

- 패딩 비율을 줄여 **유효 토큰/초**를 높이기 쉽다.
- 짧은 문서가 많은 말뭉치에서 GPU 활용이 좋아진다.

대가(설명):

- 문서 경계를 Attention·Loss에서 **명시적으로 다루지 않으면** 잘못된 맥락이 섞인다.
- 구현·디버깅이 padding-only보다 복잡하다.

### 7. Padding이란 무엇인가

**Padding**은 한 시퀀스가 $T$보다 짧을 때 남는 칸을 PAD로 채우는 것이다.

```text
문서A만 (길이 4), T=8:
[a1 a2 a3 EOS PAD PAD PAD PAD]
attention_mask:
[ 1  1  1  1   0   0   0   0]
labels (PAD 위치 ignore):
[a2 a3 EOS -100 -100 -100 -100 -100]  # 시프트 방식에 따라 세부 인덱스는 달라짐
```

장점:

- 구현이 단순하다. “문서 하나 = 샘플 하나”로 생각하기 쉽다.
- 경계 혼동이 packing보다 적다(문서가 샘플을 넘지 않는 경우).

대가:

- 짧은 문서가 많으면 PAD가 대부분인 배치가 나와 **계산 낭비**가 커진다.
- 긴 문서는 여전히 truncate가 필요하다.

### 8. Packing vs Padding 비교

| 항목 | Packing | Padding |
|---|---|---|
| 한 윈도우의 내용 | 여러 문서(또는 문서 조각) | 보통 한 문서(+PAD) |
| 유효 토큰 비율 | 높게 만들기 쉬움 | 짧은 문서에서 낮아지기 쉬움 |
| 구현 복잡도 | 경계 마스크·문서 ID 추적 필요 | 상대적으로 단순 |
| 경계 이슈 | **문서 간 Attention 누수** 위험 | 샘플 단위로 분리되면 적음 |
| 언제 쓰나 | 대용량 Pretraining에서 흔함 | 짧은 시퀀스·프로토타입·일부 SFT |

둘은 배타적이지 않다. packing 후에도 마지막 윈도우가 $T$에 못 미치면 **끝부분만 padding**하는 혼합이 일반적이다.

### 9. BOS / EOS와 문서 경계

#### 9.1 EOS의 역할

Pretraining packing에서 EOS는 대개 다음을 신호한다.

- “이 문서가 여기서 끝났다.”
- 생성 시 “멈춰도 되는 지점”을 학습하는 데 도움이 될 수 있다(설명; 항상 보장되지는 않음).

문서 사이에 EOS 없이 이어 붙이면, 모델 입장에서는 한 긴 문서처럼 보인다.

#### 9.2 BOS의 역할

BOS를 쓰는 설정에서는 문서(또는 시퀀스) 시작을 표시한다. 쓰지 않는 토크나이저도 많다.

사실 vs 설명:

- **사실**: special token 집합과 사용 여부는 토크나이저/학습 레시피에 종속한다.
- **설명**: BOS가 있으면 “새 문서 시작”을 명시적 토큰으로 학습하기 쉽다는 해석이 가능하다. 그러나 BOS 없이도 EOS+경계 마스크만으로 운영하는 파이프라인이 존재한다.

#### 9.3 작은 예시

```text
문서1: "hi"   → [h i EOS]
문서2: "bye"  → [b y e EOS]

packing (T=8), BOS 미사용:
[h i EOS b y e EOS PAD]

packing, 문서마다 BOS 사용(가정 bos=B):
[B h i EOS B b y e]   # 길이에 따라 truncate/pad
```

### 10. Attention과 Packing 경계

Causal LM의 Attention은 기본적으로 **같은 시퀀스 안의 과거 토큰**을 본다. packing하면 “같은 시퀀스” 안에 **서로 다른 문서**가 공존한다.

문제 상황:

```text
위치:  0   1   2   3   4   5
토큰:  a1  a2 EOS  b1  b2  b3
```

위치 4(`b2`)가 causal mask만 있으면 `a1,a2,EOS,b1`을 본다. 즉 **앞 문서가 뒤 문서의 맥락**이 된다.

이것이 항상 치명적 버그인지는 레시피·목표에 따라 논의가 있다. 다만 다음을 구분해야 한다.

| 접근 | 의미 |
|---|---|
| **경계 무시 packing** | 문서가 이어진 채 causal mask만 적용. 구현 단순, 경계 누수 허용 |
| **문서 블록 마스크** | 다른 문서 토큰에는 Attention을 막음(또는 loss만 차단하는 변형) |
| **reset / segment id** | 문서 ID를 두고 Attention을 같은 문서 내로 제한 |

#### 10.1 Attention 차단의 직관

문서 블록 마스크의 아이디어:

```text
허용: 같은 문서의 과거 위치
금지: 다른 문서의 모든 위치 (과거라도)
```

의사 코드:

```python
def document_aware_causal_mask(doc_ids, T):
    """
    doc_ids: [T] 각 위치가 속한 문서 번호
    반환: [T, T] True면 attend 가능
    """
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    same_doc = doc_ids[:, None] == doc_ids[None, :]
    return causal & same_doc
```

#### 10.2 Loss만 경계에서 다루기

어떤 구현은 Attention은 그대로 두고, **문서가 바뀌는 첫 토큰 예측**만 `ignore_index`로 제외한다. 이는 “앞 문서 다음에 뒤 문서 첫 토큰”을 강제 학습하지 않게 하려는 약한 경계 처리다.

주의: Attention 누수와 Loss 마스킹은 **다른 층위**의 문제다. 용어를 섞지 말 것.

### 11. Packing 알고리즘 (개념 구현)

아래는 교육용 단순 packing이다. 대용량에서는 메모리 맵·샤드·비동기 로더를 쓴다.

```python
from typing import Iterator

def pack_token_streams(
    doc_id_lists: list[list[int]],
    block_size: int,
    pad_id: int,
) -> Iterator[dict]:
    """
    여러 문서의 토큰 리스트를 이어 붙여 block_size 길이 청크로 만든다.
    마지막 청크가 짧으면 PAD로 채운다.

    반환 키:
      input_ids: List[int] 길이 block_size
      doc_ids:   List[int] 각 토큰의 문서 번호 (-1 = PAD)
      attention_mask: List[int] 1=유효, 0=PAD
    """
    buffer: list[int] = []
    doc_buffer: list[int] = []

    for doc_idx, ids in enumerate(doc_id_lists):
        for tok in ids:
            buffer.append(tok)
            doc_buffer.append(doc_idx)
            if len(buffer) == block_size:
                yield {
                    "input_ids": buffer,
                    "doc_ids": doc_buffer,
                    "attention_mask": [1] * block_size,
                }
                buffer, doc_buffer = [], []

    if buffer:
        n = len(buffer)
        pad_n = block_size - n
        yield {
            "input_ids": buffer + [pad_id] * pad_n,
            "doc_ids": doc_buffer + [-1] * pad_n,
            "attention_mask": [1] * n + [0] * pad_n,
        }


def make_labels(input_ids: list[int], pad_id: int, ignore_index: int = -100) -> list[int]:
    """다음 토큰 예측용 labels. PAD 위치와 시퀀스 끝은 ignore."""
    labels = input_ids[1:] + [ignore_index]
    labels = [ignore_index if t == pad_id else t for t in labels]
    # 입력이 PAD인 위치의 label도 무시 (시프트 정합)
    labels = [
        ignore_index if src == pad_id else lab
        for src, lab in zip(input_ids, labels)
    ]
    return labels
```

사용 예:

```python
docs = [
    [10, 11, 12, 2],       # ... EOS=2
    [20, 21, 2],
    [30, 31, 32, 33, 2],
]
for batch in pack_token_streams(docs, block_size=8, pad_id=0):
    labels = make_labels(batch["input_ids"], pad_id=0)
    print(batch["input_ids"])
    print(batch["doc_ids"])
    print(labels)
```

### 12. Dataset / DataLoader로 연결

제22강의 Dataset 패턴을 토큰 스트림에 적용한다.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class PackedLMDataset(Dataset):
    def __init__(self, chunks: list[dict], pad_id: int):
        self.chunks = chunks
        self.pad_id = pad_id

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        c = self.chunks[idx]
        input_ids = torch.tensor(c["input_ids"], dtype=torch.long)
        attn = torch.tensor(c["attention_mask"], dtype=torch.long)
        labels = torch.tensor(
            make_labels(c["input_ids"], self.pad_id),
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            # 선택: doc_ids를 쓰면 커스텀 attention에 활용
            "doc_ids": torch.tensor(c["doc_ids"], dtype=torch.long),
        }


def build_loader(chunks, pad_id, batch_size=4, shuffle=True):
    ds = PackedLMDataset(chunks, pad_id)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
```

온라인 packing(스트리밍)에서는 `__getitem__` 대신 **iterable dataset**이 더 자연스럽다. 원리는 같다: 버퍼를 채우다 $T$가 되면 yield.

### 13. 실무에서 자주 하는 설계 선택

1. **문서 단위 EOS 강제**  
   원문에 없어도 학습용으로 EOS를 붙일지 결정한다.

2. **최대 문서 길이**  
   토큰화 전·후 중 어디서 truncate할지. 너무 공격적으로 자르면 말뭉치 통계가 왜곡된다.

3. **셔플 단위**  
   문서 셔플 vs 이미 packing된 청크 셔플. 청크만 셔플하면 인접 청크 상관이 남을 수 있다.

4. **멀티모달/코드**  
   파일 경계·언어 경계도 “문서 경계”와 유사하게 취급할지 정한다.

5. **SFT와의 차이(미리보기)**  
   Instruction Tuning(제69~71강)에서는 packing보다 **샘플 단위 padding + loss mask(프롬프트 구간 무시)** 가 더 흔하다. Pretraining packing 습관을 그대로 가져가면 안 된다.

### 14. 흔한 버그

1. **EOS 이중 삽입**  
   토크나이저가 이미 EOS를 넣었는데 파이프라인에서 또 붙임.

2. **PAD를 label에 포함**  
   모델이 PAD id를 “정답 토큰”으로 학습한다. Loss가 가짜로 낮아질 수 있다.

3. **attention_mask와 causal mask 혼동**  
   padding mask는 “유효 위치”, causal mask는 “미래 차단”. 둘 다 필요할 수 있다.

4. **packing 후 doc 경계 미추적**  
   디버깅 시 어느 문서가 섞였는지 재현 불가.

5. **길이 off-by-one**  
   `block_size=1024`인데 labels shift 때문에 실효 예측 위치가 1023이 되는 점을 로그/토큰 집계에 반영하지 않음.

6. **디토크나이즈 없이 파이프라인만 믿음**  
   샘플 몇 개를 `decode`해 사람이 읽어야 한다.

### 15. 미니 점검 체크리스트

파이프라인을 닫기 전에:

- [ ] 랜덤 문서 3개를 encode→decode 왕복해 보았는가?
- [ ] packing된 `input_ids`에서 EOS 위치가 문서 끝과 일치하는가?
- [ ] PAD 비율을 로그로 찍는가? (padding 위주 파이프라인일수록 중요)
- [ ] labels의 `ignore_index` 비율이 비정상적으로 크지 않은가?
- [ ] (경계 마스크 사용 시) 문서가 바뀌는 위치 전후로 attend 금지인지 단위 테스트가 있는가?

### 16. 핵심 정리

- Tokenization Pipeline은 문서 → token ids → 고정 길이 텐서 배치까지의 생산 라인이다.
- Packing은 효율을 위해 문서를 이어 붙이고, Padding은 빈칸을 PAD로 채운다.
- EOS(및 선택적 BOS)는 문서 경계를 토큰 수준에서 표시한다.
- packing 시 Attention/Loss에서 경계를 어떻게 다룰지 **명시적으로** 정해야 한다.
- 제62강은 이 배치를 받아 `forward → loss → backward → step`을 돌린다.

### 17. 핵심 용어

| 용어 | 의미 |
|---|---|
| Tokenization Pipeline | 텍스트를 학습 텐서로 바꾸는 단계열 |
| Packing | 여러 문서를 이어 길이 $T$를 채움 |
| Padding | 짧은 시퀀스를 PAD로 길이 맞춤 |
| EOS / BOS | 종료 / 시작 special token |
| `block_size` | 학습 윈도우 토큰 길이 |
| `attention_mask` | 유효 토큰 위치 표시 |
| Document boundary | packing된 시퀀스 안의 문서 경계 |
| `ignore_index` | Loss에서 무시할 label 값 |

### 18. 복습 문제

#### 문제 1 (개념)

Packing과 Padding을 한 문장씩 정의하고, Pretraining에서 packing을 쓰는 주된 이유를 쓰시오.

#### 문제 2 (설계)

다음 토큰열이 $T=10$으로 packing되었다. 문서 경계는 어디인가?

```text
[11, 12, 2, 21, 22, 23, 2, 31, 2, 0]
```

단, EOS=2, PAD=0.

#### 문제 3 (Attention)

위 시퀀스에서 위치 4(0-index, 토큰 22)가 **순수 causal mask만** 있을 때 볼 수 있는 위치 집합을 쓰고, 문서 블록 마스크를 쓰면 어떻게 달라지는지 설명하시오.

#### 문제 4 (코드)

PAD label을 마스킹하지 않으면 어떤 잘못된 학습 신호가 생기는가?

#### 문제 5 (연결)

제60강의 “문서 필터/샤드”가 이번 파이프라인의 어느 단계 앞에 와야 하는지, 이유를 포함해 쓰시오.

---

### 정답 및 해설

#### 문제 1

Packing: 여러 문서 토큰을 이어 고정 길이를 채우는 것.  
Padding: 빈 칸을 PAD로 채워 길이를 맞추는 것.  
주된 이유: 유효 토큰 비율을 높여 동일 연산으로 더 많은 실 토큰을 학습에 쓰기 위함(효율).

#### 문제 2

EOS(2) 뒤가 경계 후보. 문서1 끝=인덱스2, 문서2 끝=인덱스6, 문서3 끝=인덱스8, 인덱스9는 PAD.

#### 문제 3

순수 causal: 위치 `{0,1,2,3,4}` (과거+현재; 구현에 따라 자기 자신 포함).  
문서 블록 마스크: 토큰 22는 두 번째 문서이므로 `{3,4}`(및 같은 문서 과거)만 허용하고 `{0,1,2}`는 차단하는 식이 된다.

#### 문제 4

모델이 PAD id를 다음 토큰으로 예측하도록 학습되어, 의미 없는 채움 기호에 확률 질량을 쓰게 된다. 평가·생성도 왜곡된다.

#### 문제 5

필터/샤드는 **tokenizeize·packing 이전**(또는 tokenize와 병행하되 packing 입력 전)에 와야 한다. 이미 packing된 청크만 있으면 문서 단위 품질 제어·재셔플·라이선스 단위 제거가 어렵다.

### 19. 다음 강의와 연결

토큰 배치가 준비되었다.

다음 **제62강. Training Loop 설계**에서는 이 배치를 받아 `zero_grad → forward → loss → backward → clip → step`으로 이어지는 **학습 스텝**을 설계한다. epoch과 token budget, 로깅 항목도 함께 고정한다.

이전 강의: **제60강. Pretraining Dataset 구성**  
다음 강의: **제62강. Training Loop 설계**

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제60강. Pretraining Dataset 구성](60강_Pretraining_Dataset_구성.md)
- **다음 강:** [제62강. Training Loop 설계](62강_Training_Loop_설계.md)

<!-- /LECTURE_NAV -->
