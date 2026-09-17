# 제84강. Preference Dataset

> **학습 목표**
> - Chosen / Rejected 쌍이 무엇인지, 왜 단일 정답 라벨과 다른지
> - Human preference(인간 선호)가 어떤 절차로 수집되는지
> - Pairwise 형식(프롬프트 + 두 응답 + 선호)의 스키마
> - 품질 문제: 애매함, 편향, 불일치, 길이 편향, 라벨 노이즈
> - 제85강 Reward Model이 이 데이터를 어떻게 먹는지에 대한 예고

---
## 1. 왜 이것을 배우는가

SFT만으로는 한계가 있다.

```text
SFT:  (prompt, gold_response)
→ 모델은 gold를 모방한다
→ 하지만 “금보다 은보다 나은 미세한 차이”는 배우기 어렵다
```

현실의 응답 품질은 연속적이다.

```text
같은 질문에 대해
A: 정확하지만 불친절
B: 친절하지만 일부 틀림
C: 길고 장황하지만 안전
D: 짧고 명확하지만 위험 정보 포함
```

어느 것이 “정답”인가? 도메인·정책·사용자에 따라 달라진다. 그래서 정렬 파이프라인은 **절대 정답 하나** 대신 **상대 선호**를 모은다.

```text
3권 SFT 데이터          4권 Preference 데이터
─────────────────       ─────────────────────
instruction → output    prompt → (yw, yl)
“이렇게 답하라”          “yw가 yl보다 낫다”
```

제85강에서 이 쌍으로 Reward Model을 학습하고, 제86~88강에서 그 점수로 정책을 업데이트한다. **데이터가 흔들리면 RM·PPO·DPO 전부가 흔들린다.**

## 2. 먼저 알아야 할 개념

- SFT / Instruction Dataset (3권 제69·70강)
- Chat messages 형식 (`user` / `assistant`)
- 제79강 Post-Training 지도: SFT → Preference → RL
- 제80~83강: Reward, Policy, Advantage의 자리 (아직 PPO 수식은 제87강)
- Pairwise comparison: 둘 중 하나를 고르는 비교 실험 설계

아직 깊게 들어가지 않는 것:

- Bradley-Terry 손실의 전체 유도 (제85강)
- PPO clip (제87·88강)
- DPO가 preference를 직접 쓰는 방식 (제90강)
- 특정 공개 벤치마크의 “SOTA 점수” 수치

## 3. 핵심 개념 설명

### 3.1 Preference Dataset이란?

**Preference Dataset(선호 데이터셋)**은 각 샘플이 대략 다음을 담는 집합이다.

| 필드 | 의미 |
|---|---|
| `prompt` (또는 `messages`의 user 구간) | 질문·지시 |
| `chosen` ($y_w$, winner) | 선호된 응답 |
| `rejected` ($y_l$, loser) | 덜 선호된 응답 |
| (선택) `margin`, `confidence`, `annotator_id` | 확신·주석자 메타 |

표기:

- $x$: 프롬프트
- $y_w$: chosen (preferred / winning response)
- $y_l$: rejected (dispreferred / losing response)

한 샘플의 의미는 “$y_w$가 정답이다”가 아니라:

\[
y_w \succ y_l \mid x
\]

즉, **조건 $x$에서 $y_w$가 $y_l$보다 선호된다**.

### 3.2 Chosen / Rejected

**Chosen**은 주석자(또는 규칙·모델 심사자)가 더 낫다고 고른 응답이다.  
**Rejected**는 비교에서 진 쪽이다.

중요:

1. Rejected가 “완전히 틀린 응답”일 필요는 없다. 단지 **상대적으로 덜 나은** 응답이다.
2. Chosen이 “완벽한 응답”일 필요도 없다. 쌍 안에서의 승자일 뿐이다.
3. 같은 응답이 다른 프롬프트·다른 쌍에서는 rejected가 될 수 있다.

```text
예: x = "파이썬으로 리스트 합을 구해 줘"

chosen:   "sum([1,2,3]) → 6 입니다. ..."
rejected: "for 루프로 더하세요. (코드 없음)"
```

둘 다 “도움”일 수 있으나, 과제 수행력에서 chosen이 앞선다.

### 3.3 Human preference

**Human preference(인간 선호)**는 사람이 두(또는 여러) 후보를 보고 순위를 매기거나 승자를 고르는 신호다. RLHF 문헌에서 자주 등장하는 수집 절차는 대략 다음과 같다.

```text
1. SFT 정책(또는 후보 정책들)에서 같은 x에 대해 응답 샘플링
2. 주석자에게 A/B(또는 순위) 제시
3. 가이드라인에 따라 승자 선택 (또는 tie)
4. (prompt, chosen, rejected)로 저장
```

가이드라인 예시(개념):

- 정확성 · 도움됨 · 유해하지 않음 · 지시 준수 · 과도한 장황함 지양
- 충돌 시 우선순위(예: 안전 > 정확 > 스타일)를 문서화

**사실:** 실제 대규모 정렬 데이터는 조직·제품마다 가이드라인과 검수 절차가 다르다.  
**설명:** 이 책은 특정 회사의 내부 수치나 “몇 %가 안전 개선” 같은 숫자를 발명하지 않는다. 구조와 실패 모드에 집중한다.

### 3.4 Pairwise 형식

가장 흔한 학습용 형식은 **pairwise(쌍 비교)**다.

JSONL 한 줄 예:

```json
{
  "prompt": "광합성의 정의를 두 문장으로 설명해 줘.",
  "chosen": "광합성은 식물이 빛 에너지로 이산화탄소와 물을 포도당과 산소로 바꾸는 과정이다. 엽록체에서 일어나며 생명계의 에너지 유입에 핵심이다.",
  "rejected": "광합성은 식물이 숨 쉬는 것이다. 밤에만 한다."
}
```

Messages 스타일로 확장한 예:

```json
{
  "prompt": [
    {"role": "user", "content": "광합성의 정의를 두 문장으로 설명해 줘."}
  ],
  "chosen": [
    {"role": "assistant", "content": "광합성은 ..."}
  ],
  "rejected": [
    {"role": "assistant", "content": "광합성은 식물이 숨 쉬는 ..."}
  ]
}
```

멀티턴이면 `prompt`에 이전 대화가 포함된다. 중요한 계약은 하나다.

> **비교되는 두 응답은 같은 대화 문맥 $x$를 공유해야 한다.**

문맥이 다르면 “선호”가 아니라 “다른 과제”를 비교하는 셈이다.

### 3.5 Pointwise · Listwise와의 관계

| 형식 | 내용 | 비고 |
|---|---|---|
| Pairwise | $y_w \succ y_l$ | RM·DPO에서 가장 흔함 |
| Pointwise | 응답에 절대 점수(1~5 등) | 점수 척도 보정이 어려움 |
| Listwise | $y_1 \succ y_2 \succ \cdots$ | 쌍으로 분해해 쓰기도 함 |

Pointwise 점수가 있어도 학습 시 pairwise로 바꾸는 경우가 많다. 이유: 주석자마다 점수 스케일이 다르고, “4점 vs 5점”의 의미가 불안정하기 쉽다.

### 3.6 데이터가 생기는 경로

Preference 쌍을 만드는 대표 경로:

1. **Human vs Human 후보**  
   여러 모델·온도로 샘플 → 사람이 고름
2. **Human rewrite**  
   나쁜 응답을 사람이 고쳐 chosen으로 둠
3. **AI feedback (RLAIF 계열)**  
   강한 심사 모델이 선호를 달음 (비용↓, 편향 전이 위험↑)
4. **규칙·검증기**  
   단위 테스트 통과/실패, 형식 준수 등 (제93강 RLVR과 연결)

이 책의 RLHF 본체(제85~88강)는 주로 **사람 또는 그에 준하는 pairwise 라벨**을 전제로 한다.

### 3.7 품질 이슈 — 왜 “그냥 많이”가 위험한가

Preference 데이터는 라벨이 주관적이다. 품질이 나쁘면 RM이 **잘못된 취향**을 학습한다.

주요 실패 모드:

| 문제 | 증상 | 결과 |
|---|---|---|
| 애매한 쌍 | 거의 비슷한 두 응답 | 노이즈 라벨 |
| 길이 편향 | 긴 쪽이 자주 chosen | 장황함 보상 |
| 아첨 편향 | 동의·칭찬 응답 선호 | sycophancy |
| 주석자 불일치 | 같은 쌍에 다른 승자 | 학습 신호 상쇄 |
| 분포 편중 | 쉬운 과제만 많음 | 어려운 과제에서 RM 붕괴 |
| 유해 누락 | 안전 가이드 미적용 | 위험 응답이 승자가 됨 |
| 정답 누출 없음의 착각 | rejected가 너무 터무니없음 | 쉬운 분류만 학습 |

**길이 편향**은 특히 유명하다. 주석자가 “자세히 쓴 쪽”을 무의식적으로 고르면, RM은 “길면 높은 점수”를 배우고, PPO는 장황한 정책을 강화한다.

```text
실제 선호의 일부 ≠ 우리가 원하는 정책
데이터에 섞인 편향 → RM → 정책으로 증폭
```

### 3.8 Tie와 “둘 다 나쁨”

주석 UI에는 종종 다음이 있다.

- A 승 / B 승 / Tie(비슷) / Both bad

처리 관례(개념):

- Tie: 학습에서 제외하거나, 약한 margin으로 처리
- Both bad: 새 응답을 다시 샘플링하거나, 별도 안전 데이터로 보강

초보 실습에서는 **명확한 승패 쌍만** 쓰는 것이 안전하다.

### 3.9 SFT 데이터와의 공존

Preference Dataset이 SFT를 대체하지는 않는다.

```text
권장 순서(개념적):
1) SFT로 “형식·기본 지시 수행”을 세움
2) Preference로 “미세 선호·안전·스타일”을  refinement
```

SFT 없이 약한 base에서 preference만 돌리면, 비교 대상 응답 자체가 너무 나빠 **상대 선호가 의미를 잃는다**.

## 4. 직관적으로 이해하기

식당 리뷰에 비유하자.

```text
SFT:
  “이 메뉴 레시피는 이것이다” (정답 레시피)

Preference:
  “같은 재료로 만든 A접시가 B접시보다 낫다”
```

셰프(정책)는 레시피로 기본을 익힌 뒤, 손님 선호로 간을 맞춘다.  
손님마다 취향이 다르므로 **가이드라인**이 필요하고, 잘못된 리뷰(봇·악의·대충)가 많으면 맛이 이상해진다.

또 다른 직관: 스포츠 토너먼트.

```text
모델 A 응답 vs 모델 B 응답
주석자 = 심판
chosen = 승자
데이터셋 = 경기 기록 모음
RM = 선수 실력 점수 추정기
```

심판이 뇌물을 받거나(편향), 규칙이 없으면(가이드 부재) 랭킹이 무의미해진다.

## 5. 수학적으로 이해하기

Preference를 확률 모델로 쓰는 가장 흔한 출발점은 **Bradley-Terry**다. (유도·학습은 제85강에서 본격화한다.)

응답에 스칼라 점수 $r(x,y)$가 있다고 가정하면:

\[
P(y_w \succ y_l \mid x)
=
\sigma\big(r(x,y_w) - r(x,y_l)\big)
=
\frac{1}{1+e^{-(r_w - r_l)}}
\]

여기서 $\sigma$는 sigmoid다.

해석:

- 점수 차이가 크면 승 확률이 1에 가까움
- 점수가 같으면 승 확률 0.5 (동전)

데이터셋 $\mathcal{D}=\{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}$는 이 확률 모델의 **관측 표본**이다.  
RM 학습은 “관측된 승패를 잘 설명하는 $r$”를 찾는 일이다.

이 강의에서 기억할 최소 수학:

\[
\text{데이터 한 줄} \;\equiv\; \text{사건 }\{y_w \succ y_l \mid x\}
\]

절대 점수 라벨이 없어도, 상대 비교만으로 $r$의 **차이**를 학습할 수 있다. 절대 스케일은 나중에 KL·정규화로 묶는다(제89강).

## 6. 작은 숫자로 직접 계산하기

가상의 주석 결과 3쌍이 있다고 하자. (교육용 숫자)

| i | $r_w$ | $r_l$ | $r_w-r_l$ | $P(y_w\succ y_l)=\sigma(\Delta)$ |
|---|---|---|---|---|
| 1 | 2.0 | 0.0 | 2.0 | $\sigma(2)\approx 0.88$ |
| 2 | 0.5 | 0.4 | 0.1 | $\sigma(0.1)\approx 0.525$ |
| 3 | 1.0 | 3.0 | -2.0 | $\sigma(-2)\approx 0.12$ |

해석:

- 쌍 1: 점수 차이가 커서 “거의 확실히 chosen 승”
- 쌍 2: 거의 동점 → 라벨이 바뀌기 쉬운 **애매 쌍**
- 쌍 3: 현재 점수 함수가 라벨과 **반대** → 강한 학습 신호(또는 라벨 오류)

데이터 품질 관점의 숫자 감각:

```text
애매 쌍 비율이 높으면
→ 평균 |Δ|가 작음
→ RM accuracy가 55~60%대에 머물기 쉬움 (개념적 경향)
→ “정확도만 보고 좋다”고 말하기 위험
```

**사실:** 특정 공개 RM의 정확도 숫자를 여기서 단정하지 않는다.  
**설명:** 애매 쌍이 많으면 분류 상한이 낮아진다는 **구조적** 이야기만 한다.

길이 편향 미니 시나리오:

```text
응답 A: 40토큰, 정확
응답 B: 120토큰, 정확+군더더기
주석 10명 중 8명이 B 선택 (자세히 보여서)
→ 데이터: B = chosen
→ RM: 길이 ↑ → reward ↑ 경향
```

## 7. 코드로 구현하기 — 스키마와 검증

학습 전에 **스키마 검증**을 두는 것이 실무적으로 중요하다.

```python
from typing import Any

REQUIRED = ("prompt", "chosen", "rejected")

def as_text(field: Any) -> str:
    """prompt/chosen/rejected가 str 또는 messages  alike인지 정규화."""
    if isinstance(field, str):
        return field.strip()
    if isinstance(field, list):
        # [{"role": "...", "content": "..."}, ...]
        parts = []
        for m in field:
            role = m.get("role", "")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts).strip()
    raise TypeError(f"unsupported field type: {type(field)}")

def validate_preference_row(row: dict) -> list[str]:
    errors = []
    for k in REQUIRED:
        if k not in row:
            errors.append(f"missing:{k}")
            continue
        try:
            text = as_text(row[k])
        except TypeError as e:
            errors.append(str(e))
            continue
        if not text:
            errors.append(f"empty:{k}")

    if not errors:
        c = as_text(row["chosen"])
        r = as_text(row["rejected"])
        if c == r:
            errors.append("chosen==rejected")
        if len(c) < 5 or len(r) < 5:
            errors.append("too_short_response")
    return errors
```

필터링 파이프라인 스케치:

```python
def filter_dataset(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    keep, drop = [], []
    for row in rows:
        errs = validate_preference_row(row)
        if errs:
            drop.append({"row": row, "errors": errs})
        else:
            keep.append(row)
    return keep, drop
```

## 8. PyTorch로 구현하기 — Dataset 골격

토크나이저·템플릿은 프로젝트마다 다르므로, 여기서는 **쌍을 텐서로 묶는 골격**만 둔다. RM 학습 루프는 제85강.

```python
import torch
from torch.utils.data import Dataset

class PreferenceDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=512):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def _encode_pair(self, prompt: str, answer: str):
        # 실제로는 chat template을 써야 한다 (3권 제72강)
        text = f"User: {prompt}\nAssistant: {answer}"
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return enc["input_ids"], enc["attention_mask"]

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = row["prompt"] if isinstance(row["prompt"], str) else as_text(row["prompt"])
        chosen = row["chosen"] if isinstance(row["chosen"], str) else as_text(row["chosen"])
        rejected = row["rejected"] if isinstance(row["rejected"], str) else as_text(row["rejected"])

        c_ids, c_mask = self._encode_pair(prompt, chosen)
        r_ids, r_mask = self._encode_pair(prompt, rejected)
        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_mask,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_mask,
        }

def collate_preference(batch, pad_id=0):
    def pad(seqs):
        m = max(len(s) for s in seqs)
        out = torch.full((len(seqs), m), pad_id, dtype=torch.long)
        mask = torch.zeros(len(seqs), m, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return out, mask

    c_ids, c_mask = pad([b["chosen_input_ids"] for b in batch])
    r_ids, r_mask = pad([b["rejected_input_ids"] for b in batch])
    return {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
    }
```

체크리스트:

- chosen/rejected에 **동일 템플릿** 적용
- 프롬프트만 인코딩하고 응답을 빼먹는 버그 금지
- 패딩 토큰이 점수에 섞이지 않게 mask 전달 (제85강)

## 9. 실제 LLM에서는 어떻게 사용하는가

산업·연구 파이프라인에서의 위치:

```text
SFT 정책 π_SFT
    │
    ├─ 프롬프트 세트 X에서 y ~ π_SFT (또는 여러 정책)
    │
    ▼
주석(사람/AI/규칙) → Preference Dataset D
    │
    ├─ Reward Model 학습          (제85강)
    ├─ DPO 등 직접 preference 학습 (제90강)
    └─ 평가용 pairwise win-rate
```

실무에서 자주 보는 운영 이슈:

1. **지속적 수집:** 제품 로그의 사용자 피드백(👍/👎)을 pairwise로 재구성
2. **재라벨:** 정책이 바뀌면 옛 데이터의 rejected/chosen 분포가 어긋남
3. **안전 전용 세트:** 일반 도움됨 데이터와 유해/거절 데이터를 분리 관리
4. **골드 세트:** 고신뢰 주석으로 RM·정책의 회귀를 감지

공개 데이터셋을 쓸 때:

- 라이선스·개인정보·유해 콘텐츠 포함 여부를 확인한다
- 영어 중심 데이터로 한국어 정책을 학습하면 **언어·문화 불일치**가 생긴다
- “유명 데이터셋 이름”만으로 품질을 보장하지 않는다

## 10. 실습

### 실습 A — 미니 선호 데이터 10쌍 작성

주제: “초보에게 파이썬 함수 설명”.

요구:

1. 동일 `prompt`에 chosen/rejected를 직접 작성
2. rejected는 “완전히 헛소리”가 아니라 **미묘하게 나쁜** 응답 포함 (최소 3쌍)
3. `validate_preference_row`로 검증
4. chosen==rejected가 없게

### 실습 B — 주석 가이드 한 장

다음 충돌을 우선순위로 문서화하라.

```text
정확성 vs 친절
간결함 vs 상세
안전 거절 vs 최대한 도와주기
```

각 항목에 예시 쌍 1개씩.

### 실습 C — 길이 편향 측정

작성한 10쌍에서

\[
\Delta_{\text{len}} = \mathrm{len}(chosen) - \mathrm{len}(rejected)
\]

의 부호가 +인 비율을 세라. 80% 이상이면 의도적으로 짧은 chosen 쌍을 추가해 재균형하라.

### 실습 D — SFT 샘플과 혼동하지 않기

같은 프롬프트로

- SFT 샘플 1개 `(prompt, response)`
- Preference 샘플 1개 `(prompt, chosen, rejected)`

를 나란히 두고, 학습 목표 문장을 각각 한 줄로 쓰라.

## 11. 자주 하는 실수

1. **Rejected를 항상 쓰레기로 만들기**  
   너무 쉬운 분류만 학습되어 미세 선호를 못 배운다.

2. **프롬프트 불일치**  
   chosen과 rejected가 다른 질문을 보고 답한 쌍을 합침.

3. **템플릿 불일치**  
   한쪽만 system 프롬프트가 붙음 → RM이 내용이 아니라 포맷을 학습.

4. **주석 가이드 없음**  
   주석자마다 “좋은 응답” 정의가 달라 불일치↑.

5. **Tie를 억지로 승패로 변환**  
   동전 라벨이 들어간다.

6. **평가 세트가 학습 세트와 중복**  
   RM accuracy가 과대평가된다.

7. **언어·도메인 불일치 무시**  
   코딩 선호 데이터로 의료 챗봇을 정렬하려 함.

8. **Preference를 SFT gold처럼 취급**  
   chosen만 모아 SFT하면 “상대 정보”가 사라진다. (쓰더라도 목적이 다름을 명시)

## 12. 핵심 정리

- Preference Dataset은 $(x, y_w, y_l)$ 형태의 **상대 선호** 기록이다.
- Chosen/Rejected는 절대 진리가 아니라 **쌍 안 승패**다.
- Human preference는 가이드라인·검수·불일치 관리가 데이터 품질의 핵심이다.
- Pairwise 형식이 RM·DPO의 표준 입력이다.
- 길이 편향·아첨·애매 쌍·분포 편중이 정책 실패로 증폭될 수 있다.
- SFT 데이터와 목적이 다르며, 보통 SFT 이후에 쌓는다.
- 다음 강의에서 이 데이터로 **스칼라 보상 함수**를 학습한다.

## 13. 핵심 용어

| 용어 | 의미 |
|---|---|
| Preference Dataset | 선호 비교가 라벨인 데이터셋 |
| Chosen ($y_w$) | 선호된 응답 |
| Rejected ($y_l$) | 덜 선호된 응답 |
| Pairwise | 두 응답을 비교하는 형식 |
| Human preference | 사람이 매긴 선호 신호 |
| Annotator disagreement | 주석자 간 승패 불일치 |
| Length bias | 긴 응답을 과도하게 선호하는 편향 |
| Tie | 거의 동등하여 승부를 내지 못함 |
| Bradley-Terry | 점수 차이로 승 확률을 모델링하는 틀(제85강) |
| RLAIF | AI가 선호 라벨을 다는 계열 접근 |

## 14. 연습 문제
### 문제 1（형식）

SFT 샘플과 Preference 샘플의 필드 차이를 한 문장으로 쓰시오.

### 문제 2（의미）

Rejected 응답이 사실적으로 맞을 수도 있는 이유를 쓰시오.

### 문제 3（품질）

길이 편향이 PPO 정책에 어떤 행동으로 나타날 수 있는가?

### 문제 4（스키마）

`chosen == rejected`인 행을 학습에 넣으면 생기는 문제를 쓰시오.

### 문제 5（연결）

제85강 Reward Model이 Preference Dataset에서 학습하려는 함수 $r(x,y)$의 출력 형태는?

### 문제 6（구분）

Pointwise 5점 척도와 pairwise 승패 중, 주석자 스케일 불일치에 더 민감한 쪽은?

---

## 정답 및 해설

### 문제 1

SFT는 보통 `(prompt, response)` 정답 한 개이고, Preference는 `(prompt, chosen, rejected)` 상대 비교다.

### 문제 2

선호는 상대 비교이므로, 둘 다 그럴듯해도 스타일·완전성·안전 등에서 한쪽이 질 수 있다.

### 문제 3

보상·선호가 길이를 좋아하면 정책이 불필요하게 장황한 답변을 생성하도록 강화될 수 있다.

### 문제 4

비교 신호가 0이라 학습이 무의미하거나, 수치적으로 불안정한 쌍이 되어 노이즈가 된다.

### 문제 5

스칼라 점수(실수 보상). 확률 자체가 아니라 보상에 가깝다. (승 확률은 점수 차이로 유도)

### 문제 6

Pointwise 5점 척도. 사람마다 점수 기준이 달라지기 쉽다.

## 15. 다음 강의와 연결

제83강에서 Advantage로 “평균 대비 얼마나 나았는지”를 배웠다면, 이번 강의는 그 보상의 **데이터 원천**을 LLM 정렬 맥락에서 정의한 셈이다.

다음 **제85강. Reward Model 구현**에서는 Preference 쌍을 입력으로 받아, Bradley-Terry 목표로 $r_\phi(x,y)$를 학습하는 방법을 수식·숫자·코드로 구현한다. 출력이 스칼라 보상이 되는 순간, 제86강 RLHF 파이프라인이 닫히기 시작한다.

> 정답을 하나 고르는 데이터가 아니라, 승패를 고르는 데이터다. 그 승패로 점수 함수를 만드는 일이 바로 Reward Model이다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제83강. Advantage](83강_Advantage.md)
- **다음 강:** [제85강. Reward Model 구현](85강_Reward_Model_구현.md)

<!-- /LECTURE_NAV -->
