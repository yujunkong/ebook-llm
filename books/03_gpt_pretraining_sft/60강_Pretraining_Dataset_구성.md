# 3권. GPT Pretraining과 SFT

## 제60강. Pretraining Dataset 구성

### 1. 이번 강의에서 배울 것

제55~59강에서 GPT의 정의·구조·학습 목표·생성 규칙을 고정했다. 이번 강의는 Pretraining의 **연료**, 즉 데이터가 어떤 층위로 구성되는지를 다룬다.

이 강의를 마치면 다음을 말할 수 있어야 한다.

- Pretraining Corpus(말뭉치)의 전형적인 출처 유형을 분류한다
- Cleaning(정제) · Filtering(필터) · Deduplication(중복 제거)의 목적과 차이를 구분한다
- Document boundary(문서 경계)가 왜 모델 신호에 영향을 주는지 설명한다
- Train/Validation 분할의 누수(leakage)를 경계한다
- “인터넷 전체”를 숫자로 흉내 내지 않고, **작은 재현 가능 코퍼스**로 실습 설계를 한다
- 제61강 Tokenization · Packing으로 넘길 산출물이 무엇인지 고정한다

이 강의는 특정 공개 데이터셋의 **토큰 수·용량을 사실처럼 암기**하지 않는다. 규모는 개념으로만 다루고, 절차와 함정에 집중한다.

### 2. 왜 이것을 배우는가

모델과 Loss가 같아도 데이터가 바뀌면 **배우는 언어의 통계**가 바뀐다.

```text
코드만 가득한 코퍼스  → 코드에 강한 base
대화만 가득한 코퍼스  → 채팅 말투에 치우침
중복·보일러플레이트多 → 같은 문장 암기·평가 오염
문서 경계 무시        → 앞글이 뒷글의 가짜 문맥
```

Pretraining을 “큰 파일을 넣으면 된다”로 생각하면, 제61강 packing·제68강 프로젝트에서 반드시 발목을 잡힌다.  
오늘은 **문서를 고르고 깨끗이 나누는 층**을 만든다.

### 3. 먼저 알아야 할 개념

- Causal LM 목표: 다음 토큰 CE (제57강)
- Tokenizer가 텍스트→id로 바꾼다는 사실 (2권 제27~30강)
- Dataset / DataLoader 추상 (제22강)
- 학습·검증 분리와 과적합 (제24강)
- 생성 품질 ≠ Loss만 (제58~59강, 제67강 예고)

아직 토큰 packing·BOS/EOS 배치 디테일은 제61강이다. 오늘은 **문서 단위 이전 단계**다.

### 4. 핵심 개념 설명

#### 4.1 Pretraining Dataset이란?

**Pretraining Dataset**은 특정 지시-응답 쌍에 맞추기 전에, 언어 모델이 next-token 목표로 먼저 학습하도록 모은 **대규모(상대적) 텍스트 집합**이다.

구성 단위를 흔히 이렇게 부른다.

| 층위 | 의미 | 예 |
|---|---|---|
| Corpus | 수집·혼합된 말뭉치 전체 | 웹+책+코드 혼합 |
| Document | 논리적 문서 한 편 | 위키 문서, 파일, 이슈 글 |
| Shard / Split | 저장·스트리밍 단위 | `train_00042.jsonl` |
| Example / Window | 모델이 실제로 먹는 고정 길이 조각 | `block_size` 토큰 (제61강) |

이 강의의 초점은 **Corpus → Document(정제된)** 까지다.

#### 4.2 코퍼스 유형(유형론)

출처를 절대 목록으로 외울 필요는 없다. **신호의 성격**으로 묶는다.

1. **Web crawl(웹 크롤)**  
   광범위·다양·잡음 많음. 광고·메뉴·중복·유해 콘텐츠 비율이 이슈.

2. **Books / long-form**  
   문장 호흡이 길고 문체가 안정적인 편. 저작권·라이선스가 민감.

3. **Wikipedia / encyclopedia**  
   상대적으로 정제된 설명문. 도메인 편향(백과 문체)이 있다.

4. **Code repositories**  
   문법·들여쓰기·API 패턴. 자연어와 토큰 통계가 크게 다르다.

5. **Academic / papers**  
   전문 용어·수식·인용. OCR·PDF 추출 잡음 가능.

6. **Conversational / social**  
   구어체·짧은 턴. PII(개인정보)·독성 이슈.

7. **Domain mixtures**  
   위 유형을 **비율로 섞은 것**. 실무 Pretraining의 기본 형태에 가깝다.

교육용 Mini GPT는 보통 (3)(4)의 **아주 작은 공개 샘플** 또는 직접 만든 텍스트로 충분하다.

#### 4.3 Cleaning(정제)

**Cleaning**은 문서에서 학습에 해로운 **형식적 잡음**을 줄이는 과정이다.

예:

- HTML 태그·스크립트 잔재 제거
- 이상한 인코딩/깨진 유니코드 정규화
- 과도한 공백·제어 문자 정리
- 보일러플레이트(헤더·푸터·“쿠키 동의”) 제거
- 추출 실패로 생긴 반복 문자열 축소

Cleaning은 “의미를 검열”한다기보다 **추출 파이프라인의 찌꺼기**를 깎는 일에 가깝다.

#### 4.4 Filtering(필터링)

**Filtering**은 문서 단위로 **남길지 버릴지** 결정하는 단계다.

기준 예(개념):

- 너무 짧거나 너무 긴 문서
- 언어 식별(원하는 언어만)
- 품질 점수(분류기·휴리스틱)
- 유해·성인·불법 조장 콘텐츠 정책
- 개인식별정보(PII) 위험
- 라이선스·로봇 배제·이용약관 위반 가능성

필터는 윤리·법적·품질 축이 겹친다. 이 책은 법률 자문을 제공하지 않는다. **실습에서는 허가된 텍스트만** 쓴다.

#### 4.5 Deduplication(중복 제거)

**Deduplication**은 동일·유사 문서를 줄여 다음을 막는 과정이다.

- 같은 페이지가 미러·신디케이션으로 반복
- 평가 세트가 학습 세트에 섞이는 **contamination**
- 모델이 “다음 토큰” 대신 **통째 암기**에 치우침

수준:

| 수준 | 아이디어 |
|---|---|
| Exact | 해시가 같으면 동일 문서 |
| Near-duplicate | shingle / MinHash 등으로 유사도 |
| Substring | 긴 부분 문자열 반복 제거(고급) |

Mini 실습에서는 exact hash만으로도 “왜 필요한지”를 체험할 수 있다.

#### 4.6 Document boundary

**Document boundary(문서 경계)**는 한 문서의 끝과 다음 문서의 시작이 맞닿는 지점이다.

packing(제61강)에서 문서를 이어 붙이면, Causal Attention이 **앞 문서 끝**을 **뒤 문서의 과거**로 볼 수 있다. 원치 않으면:

- 문서 사이에 특수 토큰(EOS/EOT) 삽입
- Attention/Loss에서 경계를 넘기지 못하게 마스크
- 아예 패딩으로 문서를 창에 단독 배치(비효율)

오늘은 “경계가 신호”라는 사실만 고정한다. 구현은 다음 강의.

#### 4.7 Split: train / validation

Pretraining에도 **held-out**가 필요하다.

원칙:

- 문서를 통째로 한쪽에만 넣기 (문장 단위로 섞어 누수 만들지 않기)
- 전처리 후 분할할지, 분할 후 동일 전처리를 적용할지 파이프라인을 고정
- 중복 제거를 분할 **전에** 하면 평가 오염을 줄이는 데 도움이 되는 경우가 많음(설계 의존)

제66~67강에서 val loss / PPL을 말할 때, “무엇에 대해 잰 값인지”가 이 분할에 달려 있다.

### 5. 직관적으로 이해하기

Pretraining 데이터를 **식재료**로 보면:

```text
산지(출처 유형)
  → 씻기(cleaning)
    → 상한 것 버리기(filter)
      → 같은 재료 중복 빼기(dedup)
        → 그릇에 담기(document list)
          → 칼로 일정 크기(cutting/packing, 제61강)
            → 요리(CE training)
```

고급 레스토랑(상용 모델)은 산지 비율·알레르기 정책이 복잡하다.  
집밥(Mini GPT)은 **깨끗한 재료 소량**으로 레시피를 익히는 것이 목적이다.

### 6. 수학적으로? — 데이터와 목표의 연결

모델이 최대화하는 것은 데이터 분포 $\mathcal{D}$ 위 로그우도다.

$$
\max_\theta\ \mathbb{E}_{x\sim\mathcal{D}}\sum_t\log P_\theta(x_t\mid x_{<t})
$$

$\mathcal{D}$를 바꾸면 최적 $\theta$가 가리키는 **언어**가 바뀐다.  
중복이 많은 $\mathcal{D}$는 사실상 일부 $x$에 재가중치를 주는 것과 비슷하다.

명시적 수식보다 중요한 실무 문장:

> 데이터 혼합 비율은 곧 학습되는 분포의 사전(prior)이다.

### 7. 작은 숫자로 파이프라인 흉내

장난감 문서 5개:

```text
D0: "alpha beta gamma"
D1: "alpha beta gamma"          # exact duplicate of D0
D2: "alpha beta gamma !!!"      # near copy (실습에선 다른 취급 가능)
D3: "<html>hello</html>"
D4: "def add(a, b): return a+b"
```

절차 예:

1. Cleaning: D3 → `"hello"`
2. Filter: 길이 < 2 토큰이면 제거(가정) — 없음
3. Exact dedup: D1 제거 (D0와 동일)
4. 남음: D0, D2, D3', D4
5. Split: train={D0,D3',D4}, val={D2}

이 규모에서는 “통계적 성능”이 아니라 **파이프라인이 문서를 올바르게 다루는지**만 본다.

### 8. 코드로 구현하기 — 문서 리스트 다루기

#### 8.1 문서 표현

```python
from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, List


@dataclass
class Document:
    doc_id: str
    text: str
    source: str = "unknown"
```

#### 8.2 간단한 cleaning

```python
_TAG_RE = re.compile(r"<[^>]+>")

def clean_text(text: str) -> str:
    text = _TAG_RE.sub(" ", text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

실전 HTML 제거는 라이브러리·도메인 규칙이 필요하지만, 교육용으로는 “정제 훅이 있다”는 점이 중요하다.

#### 8.3 길이 필터

```python
def keep_document(text: str, min_chars: int = 8, max_chars: int = 100_000) -> bool:
    n = len(text)
    return min_chars <= n <= max_chars
```

#### 8.4 Exact dedup

```python
def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_dedup(docs: Iterable[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        h = text_hash(d.text)
        if h in seen:
            continue
        seen.add(h)
        out.append(d)
    return out
```

#### 8.5 파이프라인 조립

```python
def build_corpus(raw_docs: List[Document]) -> List[Document]:
    cleaned = []
    for d in raw_docs:
        t = clean_text(d.text)
        if not keep_document(t):
            continue
        cleaned.append(Document(doc_id=d.doc_id, text=t, source=d.source))
    return exact_dedup(cleaned)
```

#### 8.6 Train/Val 분할(문서 단위)

```python
import random

def split_docs(docs: List[Document], val_ratio: float = 0.1, seed: int = 0):
    docs = list(docs)
    rng = random.Random(seed)
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * val_ratio)) if len(docs) > 1 else 0
    val = docs[:n_val]
    train = docs[n_val:]
    return train, val
```

문서가 극히 적을 때는 val을 억지로 쪼개기보다, **고정 held-out 문장**을 따로 두는 편이 낫다.

### 9. JSONL로 저장·로드

실습에서 다루기 쉬운 형식:

```json
{"doc_id": "wiki_1", "source": "toy", "text": "알파벳의 첫 글자는 A이다."}
{"doc_id": "code_1", "source": "toy", "text": "def hello():\n    print('hi')\n"}
```

```python
import json
from pathlib import Path

def save_jsonl(path: str, docs: List[Document]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> List[Document]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(**obj))
    return docs
```

제61강은 이 JSONL의 `text`를 tokenize한다.

### 10. 실제 LLM에서는 어떻게 구성하는가

공개 보고·블로그에서 반복적으로 보이는 **개념적** 패턴:

```text
다중 소스
  → 추출
    → 언어/품질/안전 필터
      → 대규모 dedup
        → 혼합 비율로 샘플
          → 토큰화·packing
            → 스트리밍 학습
```

세부 도구명·토큰 총량·GPU 년수는 출처마다 다르고 시간이 지나면 바뀐다.  
이 책은 그 숫자를 **암기 사실로 적지 않는다.**  
대신 “혼합 비율이 곧 커리큘럼”이라는 설계 감각을 남긴다.

라이선스·로봇 배제·개인정보·유해성은 연구·제품 모두에서 **데이터 엔지니어링의 일부**다. Mini 실습은 안전한 텍스트로 절차만 익힌다.

### 11. 실습

#### 실습 1 — 유형 태그

자신이 쓸 Mini 코퍼스 문서 10개에 `source` 태그를 `web|wiki|code|other` 중 하나로 달고 비율을 센다.

#### 실습 2 — 중복 주입

같은 문서를 3번 넣은 뒤 `exact_dedup` 전후 개수를 확인한다.

#### 실습 3 — HTML 찌꺼기

`<div>안녕</div>`이 cleaning 후 어떻게 되는지 보고, 태그를 제거하지 않으면 토크나이저가 무엇을 조각내는지 관찰한다(제61강과 연계 가능).

#### 실습 4 — 누수 시나리오

한 문장을 train/val에  alike 쪼개 넣었을 때 val loss가 왜 낙관적으로 나올 수 있는지 문장으로 설명하라.

#### 실습 5 — 산출물 고정

`train.jsonl`, `val.jsonl`을 만들고 README에 “라이선스: 직접 작성 / 허용된 샘플”을 한 줄 기록한다.

### 12. 자주 하는 실수

1. **용량만 키우고 중복을 방치**  
   유일 문서 수는 적은데 토큰 수만 커 보인다.

2. **평가 문서를 학습에 포함**  
   PPL이 좋아 보여도 일반화가 아니다.

3. **문서 경계를 무시**  
   packing 후 이상한 “문서 간 예언”이 학습된다.

4. **라이선스 없는 크롤을 실습에 사용**  
   교육 환경에서도 피한다.

5. **필터를 과도하게 적용**  
   다양성이 사라져 생성만 비슷해질 수 있다. 기준을 로그로 남긴다.

6. **상용 코퍼스 규모 숫자를 지어내어 메모**  
   출처 없는 수치는 쓰지 않는다.

7. **SFT 지시 데이터와 Pretraining 코퍼스를 같은 폴더에 섞음**  
   단계가 헷갈린다. 디렉터리를 분리한다.

### 13. Mini Pretraining을 위한 권장 범위

제68강 프로젝트를 염두에 둔 **교육용** 가이드:

- 문서는 직접 작성하거나 명백히 사용 가능한 짧은 텍스트
- 먼저 **한 도메인**(예: 짧은 이야기 또는 초간단 코드)으로 Loss↓를 확인
- 그다음 소량 혼합으로 다양성 실험
- “웹 전체를 흉내 낸 가짜 통계”는 목표가 아님

성공 기준은 테라바이트가 아니라:

```text
재현 가능한 파일
  → 정제·분할 스크립트
    → 토큰화 가능
      → Loss가 의미 있게 움직임
        → 생성이 데이터 패턴을 일부 반영
```

### 14. 핵심 정리

- Pretraining 데이터는 출처 유형 · 정제 · 필터 · 중복 제거 · 분할의 층으로 구성된다
- 문서 경계는 packing·Attention과 맞물리는 신호다
- 중복과 평가 누수는 숫자 착시를 만든다
- Mini 실습은 안전·재현 가능한 소량 코퍼스로 절차를 익힌다
- 다음 단계는 문서를 토큰 창으로 바꾸는 Tokenization Pipeline이다

### 15. 핵심 용어

| 용어 | 의미 |
|---|---|
| Corpus | 학습에 쓰는 말뭉치 전체 |
| Document | 논리적으로 한 덩어리인 텍스트 |
| Cleaning | 형식 잡음 제거 |
| Filtering | 문서 단위 채택/폐기 |
| Deduplication | 동일·유사 문서 제거 |
| Document boundary | 문서와 문서 사이 경계 |
| Contamination | 평가 데이터가 학습에 섞임 |
| Mixture | 출처·도메인 혼합 비율 |
| JSONL | 줄 단위 JSON 저장 형식 |
| Held-out | 학습에 쓰지 않는 평가용 분할 |

### 16. 복습 문제

#### 문제 1 (층위)

Corpus · Document · Training window의 차이를 한 줄씩 쓰시오.

#### 문제 2 (절차)

Cleaning과 Filtering의 차이를 예와 함께 설명하시오.

#### 문제 3 (중복)

Exact dedup이 막는 문제 두 가지를 쓰시오.

#### 문제 4 (경계)

문서 경계를 무시하고 이어 붙이면 Causal LM이 잘못 배울 수 있는 패턴을 쓰시오.

#### 문제 5 (설계)

Mini GPT용 데이터를 고를 때 “규모 숫자” 대신 우선할 기준 세 가지를 쓰시오.

---

### 정답 및 해설

#### 문제 1

Corpus=전체 집합, Document=논리 문서, Training window=모델이 한 번에 먹는 고정 길이 토큰 조각(제61강).

#### 문제 2

Cleaning은 텍스트 안 잡음 제거(예: HTML 태그). Filtering은 문서 자체를 탈락(예: 너무 짧음, 잘못된 언어).

#### 문제 3

예: 반복 암기 편향, 평가 세트 오염(동일 문서가 train/val에 중복).

#### 문제 4

앞 문서의 끝 문장이 뒤 문서 첫 토큰을 “문맥상 정답”처럼 예측하게 되어, 실제 단절된 문서를 이어서 읽는 잘못된 의존을 학습할 수 있다.

#### 문제 5

예: 라이선스/안전, 재현 가능성, 정제·분할 가능성(또는 도메인 적합성).

### 17. 다음 강의와 연결

문서 리스트가 준비되었다. 다음은 **토큰 ID와 고정 길이 창**이다.

**제61강. Tokenization Pipeline과 Dataset Packing**에서는 문서→tokenize ids→`block_size` packing, BOS/EOS, 패딩과의 차이, 경계 마스크를 다룬다. 오늘 만든 `train.jsonl` / `val.jsonl`이 입력이 된다.

이전 강의: [제59강. Temperature, Top-K, Top-P](./59강_Temperature_TopK_TopP.md)  
다음 강의: [제61강. Tokenization Pipeline과 Dataset Packing](./61강_Tokenization_Pipeline과_Dataset_Packing.md)

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제59강. Temperature, Top-K, Top-P](59강_Temperature_TopK_TopP.md)
- **다음 강:** [제61강. Tokenization Pipeline과 Dataset Packing](61강_Tokenization_Pipeline과_Dataset_Packing.md)

<!-- /LECTURE_NAV -->
