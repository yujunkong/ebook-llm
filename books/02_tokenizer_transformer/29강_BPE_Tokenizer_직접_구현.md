# 제29강. BPE Tokenizer 직접 구현

> **학습 목표**
> - BPE의 학습(merge)과 추론(encode)이 어떻게 다른지
> - Merge Table(병합 표)이 무엇인지, 왜 순서가 중요한지
> - 초소형 코퍼스에서 pair 빈도를 세고 병합을 반복하는 절차
> - encode / decode를 직접 작성하고, 결과가 왕복되는지 검증하기
> - GPT식 Byte-level BPE가 같은 아이디어의 어디에 해당하는지

---
## 1. 왜 이것을 배우는가

라이브러리(`tiktoken`, `sentencepiece`, Hugging Face `tokenizers`)만 쓰면 `encode`는 한 줄이다.  
그러나 다음 질문이 남는다.

- Vocabulary는 누가, 어떤 기준으로 만들었는가?
- 왜 `"lowest"`가 `low` + `est`로 쪼개지는가?
- 체크포인트와 함께 저장되는 `merges.txt`는 무엇인가?

BPE를 한 번 구현해 두면:

1. Tokenizer 버그(공백, 바이트, special token)를 디버깅할 수 있다.
2. Embedding 행 수 $V$가 어디서 오는지 설명한다.
3. “토큰 효율”을 개선하려면 **코퍼스와 merge 횟수**를 손봐야 한다는 감각이 생긴다.

## 2. 먼저 알아야 할 개념

- Token / Vocabulary / OOV (27·28강)
- Python `dict`, `Counter`, 문자열 슬라이싱
- “빈도(frequency)”가 통계적 학습의 가장 단순한 신호라는 직관
- 학습(training)과 적용(inference)의 분리

BPE 학습은 신경망 학습이 아니다. **통계적 병합 규칙 학습**이다.  
Gradient도 Loss도 없다. 그래도 LLM 파이프라인의 입구를 결정한다.

## 3. 핵심 개념 설명

### 3.1 BPE (Byte Pair Encoding)

**BPE(Byte Pair Encoding, 바이트 쌍 인코딩)**는 원래 데이터 압축 기법에서 출발해, NLP에서는 **빈도가 높은 기호 쌍을 반복적으로 병합**해 Subword Vocabulary를 만드는 알고리즘이다.

기본 아이디어:

```text
1. 텍스트를 아주 작은 단위(글자 또는 바이트)로 초기화한다
2. 인접 쌍(pair)의 빈도를 센다
3. 가장 자주 나온 쌍을 하나의 새 기호로 합친다
4. 원하는 횟수만큼 2~3을 반복한다
```

결과물:

- **Merge Table**: `(a, b) → ab` 병합을 적용한 **순서 목록**
- **Vocabulary**: 최종적으로 살아남은 기호(토큰) 집합

### 3.2 왜 “쌍”을 합치는가

언어에는 반복되는 덩어리가 있다.

```text
e + r  → er
l + o  → lo
lo + w → low
```

자주 붙는 조각을 미리 합쳐 두면:

- 같은 패턴을 한 토큰으로 공유한다
- 문장 길이가 과도하게 길어지는 것을 완화한다
- 희귀 단어도 익숙한 조각의 조합으로 표현한다

### 3.3 Training vs Encoding

| 단계 | 입력 | 출력 | 하는 일 |
|---|---|---|---|
| Training | 코퍼스 + merge 횟수 | merges, vocab | 어떤 쌍을 어떤 순서로 합칠지 결정 |
| Encoding | 새 텍스트 + merges | token ids | **이미 정해진** merges를 순서대로 적용 |
| Decoding | token ids + vocab | 텍스트 | id→토큰 문자열 연결 |

실수 포인트: 인코딩 중에 “지금 문장에서 제일 잦은 쌍”을 다시 고르지 않는다.  
**학습 때 고정된 전역 merge 순서**를 따른다.

### 3.4 Word-end Marker `</w>`

교육용 BPE 구현에서는 단어 끝을 표시하기 위해 `</w>`를 붙이는 경우가 많다.

```text
"low" → l o w </w>
```

이유:

- `"er"`가 단어 중간에 있는지, 끝에 있는지 구분할 수 있다
- 공백으로 나뉜 단어 경계를 학습 신호로 유지한다

실전 GPT-2식 Byte-level BPE는 공백을 `Ġ` 같은 기호로 표현하는 등 **다른 경계 처리**를 쓴다. 아이디어는 같다: **경계를 기호로 드러낸다**.

### 3.5 Merge Table (병합 표)

**Merge Table**은 우선순위가 있는 병합 규칙 목록이다.

```text
1순위: ('e', 's') → 'es'
2순위: ('es', 't') → 'est'
3순위: ('l', 'o') → 'lo'
...
```

인코딩 시:

```text
항상 1순위부터 가능한 위치에 적용
→ 그다음 2순위
→ ...
```

순서를 바꾸면 분할 결과가 달라질 수 있다. 그래서 Tokenizer 파일에는 merges가 **순서 그대로** 저장된다.

## 4. 직관적으로 이해하기

어린아이가 글자를 익히는 과정에 비유할 수 있다.

1. 처음에는 글자만 안다: `c a t`
2. 자주 보이는 `c a`를 `ca`로 묶는다
3. 그다음 `ca t`를 `cat`으로 묶는다

BPE는 “의미”를 이해하지 않는다. **빈도**만 본다.  
그래도 자연언어의 통계가 의미를 어느 정도 반영하기 때문에, 결과 조각이 종종 접두·접미·어근처럼 보인다.

## 5. 수학적으로 / 절차적으로 이해하기

코퍼스를 단어 빈도 맵으로 둔다.

$$

\text{word\_freq}(w) = \text{코퍼스에서 단어 } w \text{의 등장 횟수}

$$

각 단어를 기호열로 둔다.

$$

w = (s_1, s_2, \ldots, s_n)

$$

인접 쌍 $(s_i, s_{i+1})$의 전역 빈도:

$$

\text{count}(a,b)
=
\sum_{w} \text{word\_freq}(w) \cdot \#\{i : (s_i,s_{i+1})=(a,b)\}

$$

매 스텝:

$$

(a^\star, b^\star)
=
\arg\max_{(a,b)} \text{count}(a,b)

$$

모든 단어 기호열에서 $a^\star b^\star$를 하나의 기호로 치환한다.

이를 $N_{\text{merge}}$번 반복한다.

초기 vocab 크기(대략 알파벳+`</w>`)에 merge 횟수를 더하면 최종 vocab 크기의 감각이 나온다.

$$

|V| \approx |V_0| + N_{\text{merge}}

$$

(실제로는 special token, byte fallback 등으로 더 늘어난다.)

## 6. 작은 숫자로 직접 계산하기

초소형 코퍼스:

```text
low low low low lowest
newer newer wider wider newer
```

단어 빈도:

```text
low: 5
lowest: 1
newer: 3
wider: 2
```

### 6.1 초기 분할

```text
low:    l o w </w>           ×5
lowest: l o w e s t </w>     ×1
newer:  n e w e r </w>       ×3
wider:  w i d e r </w>       ×2
```

### 6.2 첫 번째 빈번 쌍 찾기 (스케치)

몇 가지 쌍만 세어 보자.

- `(e, r)` : newer에서 3, wider에서 2 → 5  
- `(e, r)` 뒤에 `</w>`가 오므로 실제 기호열은 `e r </w>`  
  쌍 `(e, r)` 빈도 5, 쌍 `(r, </w>)` 빈도 5
- `(l, o)` : low×5 + lowest×1 → 6
- `(o, w)` : low×5 + lowest×1 → 6

동점이면 구현마다 tie-break가 다르다. 여기서는 사전식 순서로 고른다고 가정하지 말고, **코드를 단일 기준으로 고정**하자.  
아래 구현은 `max`가 같은 빈도일 때 **먼저 나타난 키**에 의존할 수 있으므로, 안정적 tie-break를 명시한다.

교육 계산을 위해 첫 merge를 `(e, r)`로 두자. (빈도 5 — 실제 코퍼스에선 `(l,o)`/`(o,w)`가 6이라 더 높을 수 있음. 아래는 **절차 시연용**으로 한 스텝을 풀어쓴다.)

**(정확한 손으로 한 바퀴)**  
`(l, o)` 빈도 6이 최고라고 하자.

Merge 1: `l o` → `lo`

```text
low:    lo w </w>           ×5
lowest: lo w e s t </w>     ×1
newer:  n e w e r </w>       ×3
wider:  w i d e r </w>       ×2
```

Merge 2: `(lo, w)` 빈도 6 → `low`

```text
low:    low </w>             ×5
lowest: low e s t </w>       ×1
...
```

Merge 3: 후보 `(e, r)` = 5 → `er`

```text
newer: n er </w> 가 아니라 n e w er </w> 전 단계:
newer: n e w e r </w> → n e w er </w>
wider: w i d er </w>
```

이런 식으로 반복하면 `low</w>`, `est</w>`, `er</w>` 같은 조각이 생긴다.

손으로 전부 따라가는 것보다 **코드를 실행해 merge 목록을 인쇄**하는 편이 정확하다. 다음 절의 구현이 그 역할을 한다.

## 7. 코드로 구현하기 — 학습

아래는 의존성 없는 교육용 BPE이다. 실전 성능·유니코드 엣지케이스보다 **알고리즘 투명성**을 우선한다.

```python
# bpe_train.py
"""작은 코퍼스로 BPE merge table을 학습한다."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

Pair = Tuple[str, str]

def get_word_freqs(corpus: str) -> Dict[str, int]:
    """공백 기준 단어 빈도. 교육용으로 소문자만 가정."""
    freqs: Dict[str, int] = Counter()
    for line in corpus.strip().splitlines():
        for word in line.strip().lower().split():
            freqs[word] += 1
    return dict(freqs)

def word_to_symbols(word: str) -> List[str]:
    """단어 → 초기 기호열 (끝에 </w>)."""
    return list(word) + ["</w>"]

def get_pair_counts(splits: Dict[str, List[str]], freqs: Dict[str, int]) -> Counter:
    """모든 단어 기호열에서 인접 쌍 빈도를 누적."""
    counts: Counter = Counter()
    for word, seq in splits.items():
        f = freqs[word]
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i + 1])] += f
    return counts

def merge_pair(pair: Pair, splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """모든 단어 기호열에서 pair를 하나의 기호로 병합."""
    a, b = pair
    merged = a + b
    new_splits: Dict[str, List[str]] = {}
    for word, seq in splits.items():
        out: List[str] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        new_splits[word] = out
    return new_splits

def train_bpe(corpus: str, num_merges: int) -> Tuple[List[Pair], Dict[str, int]]:
    """
    BPE 학습.
    returns:
      merges: 우선순위 순 merge 목록
      vocab: token → id
    """
    freqs = get_word_freqs(corpus)
    splits = {w: word_to_symbols(w) for w in freqs}

    merges: List[Pair] = []

    for step in range(num_merges):
        pair_counts = get_pair_counts(splits, freqs)
        if not pair_counts:
            break

        # 안정적 tie-break: (-빈도, 사전식 pair)
        best_pair = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        best_count = pair_counts[best_pair]
        merges.append(best_pair)
        splits = merge_pair(best_pair, splits)
        print(f"merge#{step+1:02d}  {best_pair}  count={best_count}")

    # vocab 구성: 초기 문자 + </w> + 병합으로 생긴 토큰
    vocab_tokens = set()
    for w in freqs:
        vocab_tokens.update(list(w))
    vocab_tokens.add("</w>")
    for a, b in merges:
        vocab_tokens.add(a + b)

    # 결정적 id 부여
    vocab = {tok: i for i, tok in enumerate(sorted(vocab_tokens))}
    return merges, vocab

DEMO_CORPUS = """
low low low low lowest
newer newer wider wider newer
"""

if __name__ == "__main__":
    merges, vocab = train_bpe(DEMO_CORPUS, num_merges=10)
    print("\n=== merges ===")
    for i, p in enumerate(merges, 1):
        print(i, p)
    print("\nvocab size:", len(vocab))
```

실행하면 merge 순서가 인쇄된다. 이 목록이 Tokenizer의 “유전자”이다.

## 8. 코드로 구현하기 — Encode / Decode

```python
# bpe_codec.py
"""학습된 merges로 encode/decode."""

from __future__ import annotations

from typing import Dict, List, Tuple

Pair = Tuple[str, str]

def apply_merges_to_word(symbols: List[str], merges: List[Pair]) -> List[str]:
    """한 단어 기호열에 merge를 우선순위대로 적용."""
    seq = list(symbols)
    for a, b in merges:
        merged = a + b
        out: List[str] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return seq

class BPETokenizer:
    def __init__(self, merges: List[Pair], vocab: Dict[str, int]):
        self.merges = list(merges)
        self.token_to_id = dict(vocab)
        self.id_to_token = {i: t for t, i in vocab.items()}
        if "<unk>" not in self.token_to_id:
            # 교육용: 없을 수 있음. 없으면 동적 추가
            unk_id = len(self.token_to_id)
            self.token_to_id["<unk>"] = unk_id
            self.id_to_token[unk_id] = "<unk>"

    def encode_word(self, word: str) -> List[int]:
        symbols = list(word.lower()) + ["</w>"]
        pieces = apply_merges_to_word(symbols, self.merges)
        unk = self.token_to_id["<unk>"]
        return [self.token_to_id.get(p, unk) for p in pieces]

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in text.strip().split():
            ids.extend(self.encode_word(word))
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        </w>를 공백으로 바꿔 단어열을 복원한다.
        완벽히 원문과 같지 않을 수 있다(대소문자 등).
        """
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        words: List[str] = []
        buf = ""
        for t in tokens:
            if t.endswith("</w>"):
                buf += t[: -len("</w>")]
                words.append(buf)
                buf = ""
            elif t == "</w>":
                words.append(buf)
                buf = ""
            else:
                buf += t
        if buf:
            words.append(buf)
        return " ".join(words)

# 빠른 자가 검증 예시 (train_bpe와 연결)
if __name__ == "__main__":
    from bpe_train import DEMO_CORPUS, train_bpe

    merges, vocab = train_bpe(DEMO_CORPUS, num_merges=12)
    tok = BPETokenizer(merges, vocab)

    text = "low newer widest"
    ids = tok.encode(text)
    print("text:", text)
    print("ids :", ids)
    print("pieces:")
    for word in text.split():
        symbols = list(word) + ["</w>"]
        print(" ", word, "→", apply_merges_to_word(symbols, merges))
    print("decoded:", tok.decode(ids))
```

`"widest"`처럼 학습 때 없던 단어도, `w`, `i`, `d`, `est</w>` 같은 조각으로 분해되면 `<unk>` 없이 처리될 수 있다.  
이것이 28강에서 말한 Subword의 OOV 내성이다.

## 9. 한 파일로 합친 미니 실험

학습·인코딩·왕복 검증을 한 스크립트로 묶는다.

```python
# bpe_mini_lab.py
from collections import Counter
from typing import Dict, List, Tuple

Pair = Tuple[str, str]

def train_and_roundtrip() -> None:
    corpus = (
        "the cat sat on the mat\n"
        "the cat ate the rat\n"
        "a rat sat on a mat\n"
    )
    num_merges = 20

    # ---- train (축약 버전) ----
    freqs = Counter()
    for line in corpus.splitlines():
        for w in line.split():
            freqs[w] += 1

    splits = {w: list(w) + ["</w>"] for w in freqs}
    merges: List[Pair] = []

    for _ in range(num_merges):
        counts: Counter = Counter()
        for w, seq in splits.items():
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += freqs[w]
        if not counts:
            break
        best = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        merges.append(best)
        a, b = best
        merged = a + b
        new_splits = {}
        for w, seq in splits.items():
            out, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            new_splits[w] = out
        splits = new_splits

    vocab_tokens = set("</w>")
    for w in freqs:
        vocab_tokens.update(list(w))
    for a, b in merges:
        vocab_tokens.add(a + b)
    vocab = {t: i for i, t in enumerate(sorted(vocab_tokens))}
    vocab.setdefault("<unk>", len(vocab))

    # ---- encode ----
    def encode(text: str) -> List[int]:
        ids = []
        for word in text.split():
            seq = list(word) + ["</w>"]
            for a, b in merges:
                merged = a + b
                out, i = [], 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                seq = out
            ids.extend(vocab.get(p, vocab["<unk>"]) for p in seq)
        return ids

    sample = "the cat sat"
    ids = encode(sample)
    print("merges used:", len(merges))
    print("sample:", sample)
    print("ids:", ids)
    print("final word splits (training words):")
    for w, seq in sorted(splits.items()):
        print(f"  {w:5s} → {seq}")

if __name__ == "__main__":
    train_and_roundtrip()
```

## 10. Byte-level BPE로 가는 다리

교육용 구현은 “단어 문자 + `</w>`”에서 시작했다.  
GPT-2식 **Byte-level BPE**는 대략 다음이 다르다.

1. 입력을 Unicode 문자 대신 **UTF-8 바이트(0~255)**로 본다.
2. 임의의 텍스트를 바이트로 떨어뜨리므로, 이론상 `<unk>`가 필요 없다.
3. 공백·문장부호도 바이트/맵핑 규칙으로 표현한다.
4. 학습된 merges + byte vocab이 Tokenizer 파일로 저장된다.

공통점:

```text
작은 단위에서 출발
  → 빈도 높은 쌍 병합 반복
    → merge 순서로 encode
```

차이점:

```text
초기 알파벳이 "문자"인가 "바이트"인가
경계 마커를 어떻게 표현하는가
특수 토큰을 vocab 앞에 어떻게 예약하는가
```

실전 Tokenizer를 읽을 때 `merges.txt` / `tokenizer.json`이 보이면, 오늘 구현한 목록의 대형 버전이라고 보면 된다.

## 11. 실제 LLM에서는 어떻게 사용하는가

사전학습 준비 파이프라인 (단순화):

```text
원시 텍스트 코퍼스
  → (선택) 정규화·필터
    → BPE/SentencePiece 학습 → tokenizer 아티팩트
      → 전체 코퍼스를 token id 스트림으로 변환
        → 고정 길이 블록으로 잘라 학습 배치 구성
```

모델 코드 쪽:

```python
# 개념 스케치
input_ids = tokenizer.encode(text)          # List[int]
logits = model(torch.tensor([input_ids]))   # [1, T, V]
next_id = sample(logits[0, -1])
text_out = tokenizer.decode(input_ids + [next_id])
```

중요한 실무 규칙:

1. **모델과 Tokenizer는 함께 버전 관리**한다.
2. merge 순서를 바꾸면 같은 텍스트의 id가 바뀌어 Embedding이 무의미해진다.
3. vocab size $V$가 바뀌면 출력 Linear 층 `[d, V]`도 다시 맞춰야 한다.

## 12. 실습

### 실습 1. merge 횟수 실험

`DEMO_CORPUS`에서 `num_merges`를 5, 10, 20으로 바꿔 보고:

- vocab size
- `"lowest"`의 분할
- `"newer"`의 분할

을 표로 정리하시오.

### 실습 2. 왕복 검증

학습 코퍼스에 있는 문장과 없는 문장 각각에 대해:

```text
decode(encode(text))
```

가 얼마나 원문에 가까운지 비교하시오. (소문자화 때문에 완벽히 같지는 않을 수 있다.)

### 실습 3. tie-break 바꾸기

동점 pair가 있을 때:

- 사전식 최소
- 사전식 최대

를 각각 적용해 merge 순서가 달라지는지 확인하시오. 인코딩 결과가 달라지면 그 예를 하나 적으시오.

### 실습 4. (도전) 바이트 초기화

`list(word)` 대신 `list(word.encode("utf-8"))`처럼 바이트에서 시작해, 한글 한 단어가 어떻게 쪼개지는지 관찰하시오.

## 13. 자주 하는 실수

1. **인코딩 중에 pair 빈도를 다시 계산해 병합**  
   학습과 추론을 섞은 오류이다. 추론은 고정 merges만 적용한다.

2. **merge 순서를 집합으로만 저장**  
   순서를 잃으면 분할이 재현되지 않는다.

3. **단어 경계를 무시하고 문장 전체에 BPE**  
   가능하지만, 교육용/고전 BPE는 단어 단위로 적용하는 설명이 많다. 실전 규칙은 구현을 확인한다.

4. **`</w>`를 디코딩에서 깜빡함**  
   단어 복원이 망가진다.

5. **vocab id를 학습마다 랜덤 셔플**  
   저장·재현이 깨진다. 결정적 정렬이 안전하다.

6. **실전 Tokenizer와 교육용 BPE를 동일시**  
   정규화, regex pre-tokenize, byte map, special tokens가 추가된다.

## 14. 핵심 정리

- BPE는 빈도 높은 인접 쌍을 반복 병합해 Subword를 만든다.
- 학습 결과물은 **순서 있는 Merge Table**과 Vocabulary이다.
- Encoding은 새 텍스트에 **같은 merges를 순서대로** 적용하는 과정이다.
- 희귀 단어도 작은 조각 조합으로 표현되어 OOV를 줄인다.
- GPT의 Byte-level BPE는 초기 단위가 바이트인 같은 가족이다.
- LLM에서 Tokenizer 아티팩트는 모델 가중치만큼 중요한 인터페이스이다.

## 15. 핵심 용어

| 용어 | 의미 |
|---|---|
| BPE | 빈번 쌍 병합 기반 서브워드 알고리즘 |
| Merge / Merge Table | (a,b)→ab 규칙과 그 우선순위 목록 |
| Symbol sequence | 단어의 현재 기호열 |
| `</w>` | 단어 끝 마커 (교육용 구현) |
| Byte-level BPE | UTF-8 바이트에서 시작하는 BPE |
| encode / decode | 텍스트↔id 변환 |
| vocab size $V$ | Embedding·LM head 차원과 직결 |

## 16. 연습 문제
### 문제 1 (개념)

BPE 학습과 인코딩의 차이를 한 문장씩 쓰시오.

### 문제 2 (절차)

초기 기호가 `a b c`이고 merge `("a","b")`가 1순위일 때, 한 번 적용한 결과는?

### 문제 3 (이유)

Merge Table에서 순서가 중요한 이유를 예를 들어 설명하시오.

### 문제 4 (코드)

교육용 구현에서 단어 `"cat"`의 초기 기호열을 쓰시오.

### 문제 5 (LLM 연결)

모델 학습 중 Tokenizer merges를 임의로 교체하면 어떤 문제가 생기는가?

### 문제 6 (계산)

초기 vocab에 문자 26개 + `</w>`가 있고 merge를 100번 했다면, 대략적 vocab 크기는? (special/`<unk>` 제외)

---

## 정답 및 해설

### 문제 1

학습: 코퍼스 통계로 merge 순서를 결정한다.  
인코딩: 이미 고정된 merge 순서를 새 텍스트에 적용해 토큰열/id를 만든다.

### 문제 2

`ab c` (기호열 `["ab", "c"]`)

### 문제 3

같은 쌍 집합이라도 적용 순서가 다르면 중간 기호열이 달라져 최종 분할이 달라질 수 있다. 예: 먼저 `a b→ab`를 할지 `b c→bc`를 할지에 따라 `abc`의 결과가 달라진다.

### 문제 4

`["c", "a", "t", "</w>"]`

### 문제 5

같은 텍스트가 다른 id 서열이 되어, 이미 학습된 Embedding 행의 의미가 깨진다. 사실상 다른 언어로 모델을 읽는 것과 같다.

### 문제 6

$26 + 1 + 100 = 127$

## 17. 다음 강의와 연결

이번 강의에서 BPE의 merge·encode·decode를 밑바닥에서 만들었다.

다음 **제30강. Vocabulary와 Special Tokens**에서는, vocab size가 모델 파라미터에 미치는 영향과 `<unk>`, `<pad>`, `<bos>`, `<eos>`, 챗 special token이 LLM 대화 형식에서 하는 일을 정리한다.  
Tokenizer가 “조각”을 만들었다면, 30강은 “사전에 어떤 예약석을 두는가”이다.

이전 강의: **제28강. Character / Word / Subword Tokenization**  
다음 강의: **제30강. Vocabulary와 Special Tokens**

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제28강. Character / Word / Subword Tokenization](28강_Character_Word_Subword_Tokenization.md)
- **다음 강:** [제30강. Vocabulary와 Special Tokens](30강_Vocabulary와_Special_Tokens.md)

<!-- /LECTURE_NAV -->
