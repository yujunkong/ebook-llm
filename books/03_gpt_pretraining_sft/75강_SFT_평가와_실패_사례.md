# 3권. GPT Pretraining과 SFT

## 제75강. SFT 평가와 실패 사례

### 1. 이번 강의에서 배울 것

SFT 학습 곡선이 예쁘게 내려가도, 제품·연구 관점에서는 **실패**일 수 있다. 이번 강의는 “무엇을 재는가”와 “무엇이 자주 무너지는가”를 고정한다.

이 강의를 마치면 다음을 할 수 있어야 한다.

- instruction overfitting, style collapse, data leakage를 사례로 구분한다.
- 작은 **eval harness**（고정 프롬프트 세트 + 규칙/채점）아이디어를 스케치한다.
- automatic metric과 human/LLM-judge의 역할을 남용 없이 배치한다.
- 제76강 프로젝트에서 before/after를 비교할 때 무엇을 볼지 체크리스트로 만든다.

### 2. 왜 이것을 배우는가

Pretraining의 perplexity（제67강）는 “다음 토큰을 얼마나 잘 맞추는가”에 가깝다. SFT는 목표가 다르다.

> 주어진 지시에 **알맞은 형식·내용·거절/준수**로 응답하는가?

Loss만 보면 다음이 전부 “성공”으로 보인다.

- 학습셋 답을 암기
- 항상 같은 말투로 도배
- 평가 문항이 학습 데이터에 섞임

실패를 이름 붙이지 않으면, LoRA/QLoRA（제73~74강）를 얼마나 우아하게 써도 **잘못된 최적점**으로 달려간다.

### 3. 먼저 알아야 할 개념

1. SFT의 response-only loss mask（제71강）
2. Chat template / special tokens（제72강）
3. Train/val 분리와 과적합（1권 제24강）
4. 생성 디코딩: greedy / temperature（제58~59강）

### 4. 평가가 답해야 할 질문

SFT 평가를 한 장으로 나누면 대략 네 축이다.

| 축 | 질문 | 실패 시 증상 |
|---|---|---|
| 지시 준수 | 요청한 형식·제약을 지키는가 | 무시, 부분 준수 |
| 내용 품질 | 사실·추론·완결성이 충분한가 | 환각, 중도 포기 |
| 일반화 | 본 적 없는 지시에도 버티는가 | 암기, 템플릿 회귀 |
| 안전·거절 | 부적절한 요청을 다루는가 | 과잉 거절/과잉 순응 |

한 숫자로 모든 축을 대체하지 않는다. harness는 **여러 작은 검사**의 묶음이다.

### 5. 실패 사례 1 — Instruction overfitting

#### 5.1 현상

모델이 학습에 나온 **지시 패턴**에만 과도하게 맞춰진다.

예:

- 학습: “다음을 3줄로 요약하세요”
- 평가: “한 문장으로 요약하세요” → 여전히 3줄, 또는 학습셋 문장을 거의 복사

Loss는 낮다. 사용자는 화난다.

#### 5.2 원인（설명）

- 데이터가 좁은 템플릿에 편중
- epoch가 과도 / lr이 큼 / LoRA $r$이 큼
- response 길이가 짧아 **표면 패턴 암기**가 쉬움

#### 5.3 탐지

- **Held-out instruction**: 학습에 없는 동사·형식（표, JSON, 불릿 개수）
- **Paraphrase test**: 같은 의미, 다른 문장
- 학습셋 응답과의 **n-gram 중복**이 비정상적으로 높음

#### 5.4 완화

1. 지시 다양성（형식·언어·길이）을 의도적으로 섞는다.
2. early stopping을 **eval harness** 기준으로 건다（train loss 금지）.
3. LoRA $r$·epoch를 줄이는 ablation.

### 6. 실패 사례 2 — Style collapse

#### 6.1 현상

내용과 무관하게 말투·구조가 한 가지로 붕괴한다.

예:

- 모든 답이 “물론입니다! 아래에 단계별로…”로 시작
- 짧은 질문에 장황한 에세이
- 코드 질문에 불필요한 이모지·면책 문구 도배（데이터에 그 스타일이 많았다면）

#### 6.2 원인（설명）

SFT는 **응답 분포를 학습셋 응답 분포에 끌어당긴다**. 특정 assistant 스타일이 지배적이면, 모델은 “안전한 평균 말투”로 붕괴하기 쉽다.

#### 6.3 탐지

- 시작 문구 빈도 히스토그램
- 평균 응답 길이 / 문장 수
- “짧게/길게/전문가 톤” 지시 준수율

#### 6.4 완화

1. 스타일 다양성 샘플을 데이터에 명시적으로 넣는다.
2. 시스템 프롬프트로 톤을 분리（템플릿 설계, 제72강）.
3. 평가에 **스타일 준수 문항**을 넣는다.

### 7. 실패 사례 3 — Data leakage

#### 7.1 현상

평가 세트（또는 벤치마크）의 문항·답이 학습 데이터에 섞여, 점수가 **가짜로** 높다.

유형:

| 유형 | 예시 |
|---|---|
| Exact leak | 평가 프롬프트가 학습 JSON에 그대로 |
| Near-duplicate | 숫자·고유명사만 바꾼 복제 |
| Benchmark contamination | 공개 벤치 문항이 사전학습/SFT 코퍼스에 존재 |

#### 7.2 탐지

- train/eval 문자열 정규화 후 해시·유사도
- 평가 문항을 **직접 작성**한 private set 유지
- “답을 보지 않고도” 맞출 수 없는 **변환 과제**（새 숫자, 새 이름）

#### 7.3 완화

1. 데이터 파이프라인에서 eval 경로를 물리적으로 분리한다.
2. 공개 벤치 점수는 **참고**로만, 의사결정은 private harness로.
3. 의심되면 문항을 재작성해 재측정한다.

**사실:** 누수가 있으면 자동 점수는 일반화를 과장한다.  
**설명:** “점수가 올랐다 = 모델이 똑똑해졌다”는 해석은 누수 검사 없이는 위험하다.

### 8. 그 밖의 흔한 실패

| 이름 | 증상 | 짧은 처방 |
|---|---|---|
| Format break | JSON/XML 깨짐 | 형식 전용 평가 + 수리 데이터 |
| Catastrophic forgetting | 사전학습 능력이 과도 손상 | epoch↓, lr↓, 혼합 데이터 |
| Verbosity bias | 긴 답이 점수에서 유리 | 길이 정규화·명시적 길이 지시 |
| Sycophancy | 사용자 오류에 동조 | 반례·교정 데이터 |
| Over-refusal | 정상 요청도 거절 | 거절 경계 예시 균형 |
| Mask bug | prompt까지 loss | mask 단위 테스트（제71·76강） |

### 9. Eval harness 아이디어

거창한 프레임워크가 아니어도 된다. **고정된 입출력 계약**이면 harness다.

#### 9.1 최소 구조

```text
evals/
  suite.yaml          # 문항 ID, 프롬프트, 채점 유형
  run_eval.py         # 모델 생성 → 채점 → 리포트
  reports/
    2026-09-16.json
```

`suite.yaml` 개념:

```yaml
items:
  - id: sum_3_bullets
    prompt: "다음 글을 불릿 3개로 요약하세요:\n..."
    scorer: bullet_count
    expect: {min_bullets: 3, max_bullets: 3}

  - id: json_keys
    prompt: "이름과 나이만 JSON으로..."
    scorer: json_keys
    expect: {keys: ["name", "age"]}

  - id: paraphrase_math
    prompt: "사과 3개와 배 5개를 합하면?"
    scorer: contains
    expect: {any: ["8", "여덟"]}
```

#### 9.2 채점기 계층

| 계층 | 예 | 장점 | 한계 |
|---|---|---|---|
| Rule | JSON parse, 불릿 수, 금지어 | 재현·저비용 | 의미 품질 약함 |
| Reference overlap | BLEU/ROUGE 등 | 자동 | 표현 다양성에 취약 |
| LLM-as-judge | Rubric 점수 | 유연 | 편향·비용·비재현 |
| Human | 전문가 라벨 | 기준 진실에 가까움 | 비쌈 |

초보 harness는 **Rule 70% + 소수 인간 샘플**로 시작해도 충분하다.

#### 9.3 실행 루프

```python
def run_suite(model, tokenizer, items, decode_cfg):
    rows = []
    for it in items:
        text = generate(model, tokenizer, it["prompt"], decode_cfg)
        score, detail = SCORERS[it["scorer"]](text, it["expect"])
        rows.append({"id": it["id"], "score": score, "detail": detail, "out": text})
    return rows
```

리포트에는 반드시 남긴다.

- 모델·어댑터 체크포인트 해시
- chat template 버전
- decoding（temperature, top_p, max_new_tokens）
- 데이터 revision

같은 체크포인트라도 decoding이 바뀌면 점수가 흔들린다.

### 10. Before / After 비교 프로토콜

제76강 프로젝트와 맞추기 위한 최소 프로토콜:

1. **동일 프롬프트 10~30개**를 고정한다（학습에 미포함）.
2. Pretrain-only（또는 SFT 전）생성과 SFT 후 생성을 **나란히** 저장한다.
3. 각 문항에 rule score + 한 줄 메모（준수/실패 이유）.
4. “평균 점수만”이 아니라 **실패 유형 태그**를 붙인다.

```text
[prompt] 불릿 2개로 장점만...
[before] 장황한 에세이, 불릿 없음
[after]  - ...
         - ...
[tag] format_ok
```

### 11. 학습 중 모니터 vs 최종 평가

| 신호 | 용도 | 단독 결정? |
|---|---|---|
| Train CE | 버그·발산 확인 | 불가 |
| Val CE | 과적합 조기 신호 | 보조 |
| Harness score | 제품 관점 진행 | 주력 |
| Spot human review | 맹점 보완 | 주기적 |

Val CE가 내려가도 harness가 안 오르면, **마스크 버그·스타일 붕괴·누수**를 의심한다.

### 12. 실전 체크리스트

SFT 실험 종료 전:

- [ ] Train과 eval 문항의 중복 검사
- [ ] Response-only mask 단위 테스트
- [ ] 형식·짧게/길게·JSON·거절 경계 문항 포함
- [ ] Decoding 고정
- [ ] Before/after 샘플 아티팩트 저장
- [ ] LoRA $r$/epoch ablation 적어도 한 번
- [ ] “공개 벤치 점수만으로 성공 선언” 금지

### 13. 핵심 정리

- SFT 성공은 train loss가 아니라 **지시 준수·일반화·형식**으로 정의한다.
- Instruction overfitting, style collapse, data leakage는 서로 다른 처방을 요구한다.
- 작은 rule 기반 harness만으로도 많은 회귀를 잡을 수 있다.
- Before/after를 고정 프로토콜로 남겨야 제76강·이후 실험이 누적된다.

### 14. 핵심 용어

| 용어 | 한 줄 의미 |
|---|---|
| Instruction overfitting | 학습 지시 패턴 암기·과적합 |
| Style collapse | 응답 문체·구조의 단일화 |
| Data leakage | 평가 정보가 학습에 혼입 |
| Eval harness | 고정 문항+채점으로 회귀를 재는 장치 |
| Held-out | 학습에 쓰지 않은 평가 분할 |
| LLM-as-judge | 다른 LLM으로 루브릭 채점 |

### 15. 복습 문제

#### 문제 1（구분）

Train loss↓, held-out 형식 준수율↓ 이면 어떤 실패를 먼저 의하는가?

#### 문제 2（누수）

평가 문항의 숫자만 바꾼 복제가 학습셋에 있을 때, exact hash 매칭만으로 충분한가?

#### 문제 3（harness）

Rule scorer로 잡기 좋은 문항과 나쁜 문항을 한 개씩 쓰시오.

#### 문제 4（스타일）

모든 답이 동일 인사말로 시작하면 어떤 통계를 harness에 추가하겠는가?

#### 문제 5（연결）

제76강 프로젝트에서 before/after에 꼭 넣어야 할 문항 유형 두 가지는?

---

### 정답 및 해설

#### 문제 1

Instruction overfitting（또는 형식 일반화 실패）. 마스크 버그도 점검.

#### 문제 2

충분하지 않다. near-duplicate 유사도 검사나 문항 재작성이 필요하다.

#### 문제 3

좋음: JSON 키 존재, 불릿 개수. 나쁨: “설명의 통찰 깊이” 같은 주관 품질.

#### 문제 4

응답 시작 n-gram 빈도, 또는 고정 prefix 매칭 비율.

#### 문제 5

예: 형식 제약（불릿/JSON）과 paraphrase된 지시（학습 문장과 다른 표현）.

### 16. 다음 강의와 연결

이론은 여기까지다.  
**제76강. 프로젝트 — Mini GPT + SFT**에서는 아주 작은 instruction set으로 SFT를 직접 돌리고, loss mask·before/after 생성·간단한 평가를 한 프로젝트로 닫는다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제74강. QLoRA](74강_QLoRA.md)
- **다음 강:** [제76강. 프로젝트 — Mini GPT + SFT](76강_프로젝트_Mini_GPT_SFT.md)

<!-- /LECTURE_NAV -->
