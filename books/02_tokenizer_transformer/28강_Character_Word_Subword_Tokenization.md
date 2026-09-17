# 제28강. Character / Word / Subword Tokenization

> **학습 목표**
> - Character / Word / Subword Tokenization의 정의와 차이
> - OOV(Out-of-Vocabulary)가 왜 문제인지
> - 토큰 수가 길어지면 생기는 비용(길이·계산·문맥 창)
> - 현대 LLM이 왜 Subword(특히 BPE 계열)를 주로 쓰는지

---
## 1. 왜 이것을 배우는가

Tokenizer는 “전처리 옵션”이 아니라 **모델의 입력 언어**를 정의한다.

같은 문장 `"unhappiness"`도 분할 방식에 따라 모델이 보는 서열이 달라진다.

```text
Character: u n h a p p i n e s s     → 길이 11
Word:      unhappiness               → 길이 1 (사전에 있다면)
Subword:   un happiness  또는 un happi ness
```

길이와 Vocabulary 크기는 서로 반대 방향으로 당긴다.

```text
Vocab을 키우면 → 문장이 짧아질 수 있음 (한 토큰에 더 많은 정보)
Vocab을 줄이면 → 문장이 길어질 수 있음 (조각이 잘게 쪼개짐)
```

LLM의 계산량·메모리·Context Window는 **토큰 길이**에 민감하므로, 이 트레이드오프를 이해하지 못하면 이후 Attention·학습 비용 이야기도 공중에 뜬다.

## 2. 먼저 알아야 할 개념

- Token / Token ID / Vocabulary (27강)
- OOV의 존재 (27강 실습에서 `<unk>`로 맛봄)
- 서열 길이 $T$가 커지면 Self-Attention 비용이 대략 $O(T^2)$로 커진다는 감각 (상세는 이후 Attention 강의)
- “압축”이란 같은 정보를 더 짧은 기호열로 나타내는 것

## 3. 핵심 개념 설명

### 3.1 Character Tokenization (문자 단위)

**Character Tokenization(캐릭터 토큰화)**은 텍스트를 **글자(또는 코드포인트) 단위**로 나눈다.

```text
"cats" → ["c", "a", "t", "s"]
"안녕" → ["안", "녕"]
```

장점:

1. Vocabulary가 작다. (알파벳·한글 음절·구두점 등)
2. OOV가 사실상 없다. 모르는 “단어”도 글자로 분해된다.
3. 구현이 단순하다.

단점:

1. 서열이 길다. 같은 문장이 토큰을 많이 쓴다.
2. 모델이 “단어” 수준의 패턴을 처음부터 다시 배워야 한다.
3. 장거리 의존을 학습하기 더 어려워질 수 있다. (같은 의미가 더 먼 위치에 흩어짐)

직관: 레고를 **1×1 브릭만**으로만 조립하는 것과 같다. 무엇이든 만들 수 있지만, 조립 단계가 길다.

### 3.2 Word Tokenization (단어 단위)

**Word Tokenization(워드 토큰화)**은 공백·구두점 규칙으로 **단어**를 단위로 삼는다.

```text
"I love cats." → ["I", "love", "cats", "."]  (규칙에 따라 다름)
```

장점:

1. 사람이 읽기 자연스럽다.
2. 자주 쓰는 단어는 한 토큰으로 끝나 서열이 짧다.
3. 고전 NLP(Count Vector, 초기 임베딩)와 잘 맞았다.

단점:

1. **OOV**가 심각하다. 신조어, 오타, 희귀 고유명사.
2. Vocabulary가 매우 커질 수 있다. (수백만 단어)
3. 형태소 변형마다 별개 토큰이 되기 쉽다. (`run` / `running` / `runs`)
4. 언어마다 “단어” 경계가 다르다. (한국어·중국어·독일어 합성어 등)

직관: 레고를 **완성된 큰 모듈**만 쓰는 것이다. 모듈이 있으면 빠르지만, 없는 모양은 조립 자체가 실패한다 (`<unk>`).

### 3.3 Subword Tokenization (부분 단어)

**Subword Tokenization(서브워드 토큰화)**은 단어보다 작고 글자보다 큰 **자주 나오는 부분 문자열**을 단위로 삼는다.

```text
"unhappiness" → ["un", "happiness"]
또는           → ["un", "happi", "ness"]
```

대표 알고리즘:

| 이름 | 핵심 아이디어 |
|---|---|
| **BPE (Byte Pair Encoding)** | 자주 붙는 쌍을 반복 병합 |
| **WordPiece** | 우도(likelihood) 기준으로 병합/분할 (BERT 계열) |
| **Unigram LM** | 후보 부분문자열의 확률 모델로 분할 (SentencePiece) |

현대 GPT 계열은 대개 **Byte-level BPE** 계열을 쓴다. (29강에서 BPE를 직접 구현)

장점:

1. OOV를 크게 줄인다. 희귀 단어도 익숙한 조각으로 분해.
2. Vocabulary 크기와 서열 길이의 균형을 맞출 수 있다.
3. 공통 접두·접미 (`un`, `ing`, `ness`)를 공유해 통계를 모은다.

단점:

1. 토큰 경계가 사람 직관과 다를 수 있다.
2. 학습·디버깅 시 “왜 이렇게 잘렸지?”를 추적해야 한다.
3. 언어·도메인에 따라 토큰 효율이 달라진다. (한국어 vs 영어 등)

직관: 자주 쓰는 패턴은 큰 레고 블록으로, 드문 패턴은 작은 블록으로 섞어 쓰는 것이다.

### 3.4 OOV (Out-of-Vocabulary)

**OOV(Out-of-Vocabulary, 미등록 어휘)**는 Vocabulary에 없는 토큰/단어가 입력에 나타나는 현상이다.

Word Tokenizer 예:

```text
vocab = {I, love, cats, .}
입력: "I love pterodactyls ."
→ "pterodactyls" 는 OOV → <unk>
```

문제가 무엇인가?

1. 정보 손실: 모델은 그 단어의 정체를 못 본다.
2. 학습 불안정: `<unk>`가 너무 많으면 신호가 뭉개진다.
3. 생성 한계: 사전에 없는 단어를 정확히 생성하기 어렵다. (워드 단위일 때)

Subword는 OOV를 “글자/바이트 조각”으로 **흡수**한다.

```text
"pterodactyls"
→ ["p", "ter", "od", "act", "yls"]  # 예시적 분할
```

모든 조각이 Vocabulary에 있으면 `<unk>` 없이 인코딩 가능하다. (byte-level이면 사실상 임의의 UTF-8 가능)

### 3.5 Compression (압축)과 토큰 효율

여기서 **Compression(압축)**은 “같은 텍스트를 더 적은 토큰으로 나타내는 능력”을 말한다. (정보이론의 무손실 압축과 방향은 비슷하나, Tokenizer의 1차 목표는 학습 친화적 이산 단위이다.)

측정 감각:

```text
토큰 효율 ≈ (문자 수) / (토큰 수)
또는        (바이트 수) / (토큰 수)
```

값이 클수록 “한 토큰이 더 많은 텍스트를 담는다”.

왜 중요한가?

- Context Window가 8k 토큰이면, 토큰 효율이 낮은 언어/도메인은 **실제 들어갈 문자 수**가 줄어든다.
- 학습·추론 FLOPs는 대략 토큰 수에 비례해 증가한다.
- API 과금도 보통 토큰 단위이다.

다만 “무조건 짧게”가 목표는 아니다. Vocabulary가 과도하게 크면 Embedding 파라미터가 폭증한다.

```text
Embedding 파라미터 ≈ V × d
```

- $V$: vocab size  
- $d$: embedding dimension  

균형점이 Subword의 존재 이유이다.

## 4. 직관적으로 이해하기

세 방식을 택배 포장에 비유한다.

| 방식 | 비유 | 실패 모드 |
|---|---|---|
| Character | 모래알 단위로 포장 | 상자(문맥 창)가 빨리 참 |
| Word | 완제품만 포장 | 목록에 없는 제품은 배송 불가(OOV) |
| Subword | 자주 나가는 세트 + 부품 혼합 | 세트 구성이 직관과 다를 수 있음 |

LLM 사전학습 코퍼스는 거대하고 잡다하다. 신조어·코드·다국어·오타가 섞인다.  
이 조건에서 Word는 너무 깨지기 쉽고, Character는 너무 길다. Subword가 실무 기본값이 된 배경이다.

## 5. 작은 숫자로 비교하기

문장:

```text
"lower lowest"
```

가정 Vocabulary (개념 예시):

**Character vocab** (일부):

```text
l o w e r s t   + 공백
```

분할:

```text
l o w e r   l o w e s t
토큰 수 ≈ 11 (공백 포함 여부에 따라 달라짐)
```

**Word vocab**:

```text
{"lower":1, "lowest":2}
→ ["lower", "lowest"]  토큰 수 2
```

그런데 입력이 `"lowing"`이면 Word는 OOV.

**Subword vocab** (BPE로 학습했다고 가정):

```text
병합으로 "lo", "w", "er", "est", "ing" 등이 생김
"lower"  → lo + w + er   또는 low + er
"lowest" → lo + w + est  또는 low + est
"lowing" → lo + w + ing  (OOV 없이 처리 가능)
```

토큰 수는 Word보다 길 수 있지만, **미지 단어를 분해**할 수 있다.

숫자로 비용을 감각화:

```text
같은 문서 10,000 글자
Character: 토큰 ~10,000
Word:      토큰 ~1,800  (언어·도메인 가정)
Subword:   토큰 ~2,500
```

Attention이 $O(T^2)$라면 Character는 Word 대비  Roughly $(10000/1800)^2 ≈ 30$배 무거운 자기주의 비용을 치를 수 있다. (상수·구현·근사 제외한 스케치)

## 6. 코드로 구현하기

세 방식을 같은 인터페이스로 비교한다.

```python
# compare_tokenizers.py
from typing import List

def char_tokenize(text: str) -> List[str]:
    # 공백도 하나의 토큰으로 취급
    return list(text)

def word_tokenize(text: str) -> List[str]:
    # 교육용: 공백 split만 수행
    return text.split()

def naive_subword_tokenize(text: str, merges: List[tuple]) -> List[str]:
    """
    아주 단순한 "이미 학습된 merge 목록" 적용기.
    29강 BPE의 축소판이다.
    예: merges = [("l", "o"), ("lo", "w"), ("er", "</w>")]
    """
    # 단어별로 글자 분해 후 끝표시 </w>
    pieces: List[str] = []
    for word in text.split():
        chars = list(word) + ["</w>"]
        for a, b in merges:
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                    new_chars.append(a + b)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        # </w> 표시는 보기 쉽게 제거/치환
        cleaned = [c.replace("</w>", "") for c in chars if c != "</w>"]
        # 'low</w>'처럼 붙은 경우도 처리
        cleaned = [c.replace("</w>", "") for c in cleaned]
        pieces.extend([c for c in cleaned if c])
    return pieces

def report(name: str, tokens: List[str]) -> None:
    print(f"[{name}] n={len(tokens)} -> {tokens}")

if __name__ == "__main__":
    text = "lower lowest lowing"

    report("char", char_tokenize(text))
    report("word", word_tokenize(text))

    # 가정된 merge 결과 (실제 학습은 29강)
    merges = [
        ("l", "o"),
        ("lo", "w"),
        ("e", "r"),
        ("e", "s"),
        ("es", "t"),
        ("i", "n"),
        ("in", "g"),
    ]
    report("subword-ish", naive_subword_tokenize(text, merges))
```

이 코드의 목적은 “완벽한 Tokenizer”가 아니라 **분할 철학의 차이**를 화면에 고정하는 것이다.

## 7. 한국어·코드·다국어에서의 차이

영어 공백 단어는 비교적 또렷하다. 한국어는 다르다.

```text
"나는 학교에 간다"
```

- 공백 단어: `["나는", "학교에", "간다"]`
- 형태소: `나/는/학교/에/가/ㄴ다` 에 가까운 분석이 필요할 수 있음
- Subword/BPE: 빈도 기반으로 `학교`, `에`, `간다` 등이 덩어리로 학습됨

코드 텍스트:

```python
variable_name = 1
```

Word Tokenizer는 `variable_name`을 통째로 희귀 단어 취급하기 쉽다.  
Subword는 `variable`, `_`, `name`처럼 조각내 통계를 공유한다.

결론: “어떤 단위가 정답인가?”는 언어·도메인·모델 목적에 따라 달라진다.  
범용 LLM에서는 Subword가 평균적으로 가장 실용적이다.

## 8. 실제 LLM에서는 어떻게 사용하는가

대표적 선택:

| 모델 계열 | Tokenizer 경향 |
|---|---|
| GPT-2 / GPT-3 계열 | Byte-level BPE |
| LLaMA / 많은 오픈소스 LLM | SentencePiece(BPE/Unigram) 계열 |
| BERT | WordPiece |
| 초기 연구/실험 | Character, Word도 사용 |

서비스에서 체감되는 현상:

1. **같은 문장의 토큰 수가 모델마다 다름**  
   → Context에 들어가는 “실제 글자 수”가 달라짐
2. **한국어가 영어보다 토큰을 더 쓰는 경우**가 많음  
   → 동일 창에서 소화 가능한 한국어 문서량이 줄어들 수 있음
3. **챗 템플릿의 special token**도 토큰 예산을 소비함

개발자가 할 일:

```text
배포 모델의 tokenizer로
실제 프롬프트의 token 수를 측정하라
```

추정만으로 Context Overflow를 논하면 자주 틀린다.

## 9. 실습

### 실습 1. 세 방식 토큰 수 비교

다음 문장을 char / word / (가능하면 tiktoken 또는 HuggingFace tokenizer)로 나눠 토큰 수를 비교하시오.

```text
"Tokenization is the process of splitting text into tokens."
"토큰화는 텍스트를 토큰으로 나누는 과정이다."
```

### 실습 2. OOV 재현

작은 word vocab으로 뉴스 한 문장을 encode하고 `<unk>` 비율을 세시오.  
같은 문장을 글자 단위로 인코딩하면 `<unk>`가 사라지는지 확인하시오.

### 실습 3. 압축비 계산

```text
compression ≈ len(text) / len(tokens)
```

영어 문장과 한국어 문장의 값을 비교하고, Context Window 관점에서 한 단락 해석을 쓰시오.

## 10. 자주 하는 실수

1. **“Subword가 항상 더 짧다”고 믿음**  
   Word보다 길어질 수 있다. 목표는 OOV·vocab 크기의 균형이다.

2. **토큰 수 = 단어 수라고 착각**  
   서브워드·특수기호·공백 규칙 때문에 다르다.

3. **한국어를 영어식 공백 단어로만 처리**  
   조사·어미가 붙은 형태를 별개 희귀 단어로 만들 수 있다.

4. **OOV를 전부 `<unk>`로 묻고 넘어감**  
   정보 손실이 학습을 망가뜨릴 수 있다.

5. **모델 비교 시 토큰 수를 무시**  
   “8k context”가 모든 언어에 같은 문자 용량을 의미하지 않는다.

## 11. 핵심 정리

- Tokenization 단위는 Character / Word / Subword로 나뉜다.
- Character는 OOV에 강하지만 서열이 길다.
- Word는 짧을 수 있지만 OOV와 거대 Vocabulary에 취약하다.
- Subword는 빈도가 높은 조각을 학습해 **길이와 적용 범위의 균형**을 만든다.
- LLM 실무의 기본값은 Subword(BPE/WordPiece/Unigram)이다.
- 토큰 효율은 Context·비용·다국어 성능과 직접 연결된다.

## 12. 핵심 용어

| 용어 | 의미 |
|---|---|
| Character Tokenization | 글자/코드포인트 단위 분할 |
| Word Tokenization | 단어 단위 분할 |
| Subword Tokenization | 부분 문자열 단위 분할 |
| OOV | 사전에 없는 어휘가 나타나는 현상 |
| BPE | 빈도 높은 쌍을 반복 병합하는 서브워드 알고리즘 |
| WordPiece | BERT 계열의 서브워드 알고리즘 |
| Unigram LM | 후보 조각의 확률로 분할하는 방식 |
| Compression / 토큰 효율 | 텍스트량 대비 토큰 수 효율 |

## 13. 연습 문제
### 문제 1 (개념)

Character Tokenization의 장점과 단점을 각각 한 가지씩 쓰시오.

### 문제 2 (개념)

Word Tokenization에서 OOV가 생기는 이유를 예와 함께 설명하시오.

### 문제 3 (비교)

Subword가 Word와 Character 사이에서 어떤 균형을 맞추는지 두 문장으로 쓰시오.

### 문제 4 (계산)

문장 길이 120자, 토큰 30개일 때 `len(text)/len(tokens)`는?

### 문제 5 (LLM 연결)

Context Window가 토큰 기준인 이유와, 토큰 효율이 낮은 언어에서 생기는 실질적 제약을 설명하시오.

---

## 정답 및 해설

### 문제 1

장점 예: OOV가 거의 없음 / vocab이 작음.  
단점 예: 서열이 길어져 계산·문맥 비용이 커짐.

### 문제 2

단어 사전은 유한하다. 학습/구축 때 못 본 단어(신조어·오타·고유명사)가 오면 매핑할 id가 없어 `<unk>`가 된다.

### 문제 3

자주 나오는 패턴은 긴 조각으로 묶어 서열을 과도하게 늘리지 않고, 드문 패턴은 작은 조각으로 분해해 OOV를 줄인다. Vocabulary 크기와 문장 길이의 절충안이다.

### 문제 4

$120 / 30 = 4$

### 문제 5

모델·인프라가 다루는 단위가 토큰이기 때문이다. 토큰 효율이 낮으면 같은 창에 들어가는 실제 문자·문장 수가 줄어, 긴 문서 이해·대화 기억에 불리해질 수 있다.

## 14. 다음 강의와 연결

이번 강의에서 세 가지 분할 철학과 OOV·압축 트레이드오프를 비교했다.

다음 **제29강. BPE Tokenizer 직접 구현**에서는, 작은 코퍼스로 **merge table을 학습**하고, encode/decode까지 밑바닥에서 구현한다.  
28강이 “왜 Subword인가”라면, 29강은 “BPE는 기계적으로 무엇을 하는가”이다.

이전 강의: **제27강. 텍스트가 숫자가 되는 과정**

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [27강. 텍스트가 숫자가 되는 과정](27강_텍스트가_숫자가_되는_과정.md)
- **다음 강:** [29강. BPE Tokenizer 직접 구현](29강_BPE_Tokenizer_직접_구현.md)

<!-- /LECTURE_NAV -->
