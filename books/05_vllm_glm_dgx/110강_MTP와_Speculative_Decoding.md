# 110강. MTP와 Speculative Decoding
## 이번 강에서 배우는 내용

- Draft + Verify 파이프라인의 직관
- 수락(acceptance)과 거절 시 어떻게 진행이 이어지는가
- MTP가 “훈련 목표/헤드” 쪽 이야기와 어떻게 맞닿는지
- 언제 TPOT에 도움이 되고, 언제 오버헤드만 남는가
- 제107강 지표로 이득을 정의·측정하는 방법(숫자 날조 없이)

## 왜 중요한가?
제107강에서 긴 답의 체감은 대개 **TPOT / ITL**이다. Continuous Batching·양자화·TP로도 한계에 부딪히면, “스텝당 확정 토큰 수를 늘릴 수 있는가?”가 다음 레버가 된다. 반대로 수락률이 낮으면 **검증 비용만 두 번** 치른다.

## 선수 개념
1. Prefill / Decode · KV (제100~101강)
2. TPOT vs Throughput (제107강)
3. 모델 계열 지도의 MTP 칸 (제108강)
4. (배경) 샘플링·온도 (3권)

## Speculative Decoding — Draft + Verify
### 3.1 역할 분담

| 역할 | 하는 일 | 흔히 기대하는 성질 |
|---|---|---|
| **Draft** | 앞으로 $\gamma$개 토큰 후보를 빠르게 제안 | 저렴·작음·또는 작은 헤드 |
| **Target (Verify)** | 후보를 큰 모델 분포로 검증·수정 | 품질의 기준 모델 |

이상적 한 라운드:

```text
context
  → draft가 토큰 d1..dγ 제안
  → target이 같은 구간을 (가능한 한) 병렬로 점수화/검증
  → 앞쪽부터 수락하다 첫 거절에서 멈추고, 거절 위치는 target 분포로 샘플
  → 수락된 만큼 컨텍스트 확정, KV 갱신
```

### 3.2 왜 “품질을 유지”하려 하는가

올바르게 설계된 speculative 알고리즘은, 최종적으로 나오는 토큰 분포가 **target만으로 샘플한 것과 (거의) 동일**하도록 검증·보정한다. 즉 목표는 “대충 비슷하게 빨리”가 아니라 **분포 동등(또는 명시된 근사) + 기대 가속**이다.  
세부 수락 규칙(예: speculative sampling의 비)는 원논문에 맡긴다. 여기서는 직관만.

### 3.3 수락률이 전부다

$\gamma$개를 제안해도 평균 수락이 $1$에 가까우면 이득이 작다. 수락이 $\gamma$에 가까우면 한 라운드에 여러 토큰이 확정되어 **유효 TPOT**가 내려간다.

Rough 감각(교육용):

$$

\text{유효 토큰/스텝} \approx 1 + \mathbb{E}[\text{수락 개수}]

$$

(정의·구현에 따라 상수 항은 달라질 수 있음 — **공식 암기보다 측정**)

## MTP — Multi-Token Prediction과의 관계
### 4.1 훈련 쪽 이야기

MTP는 모델이 다음 토큰뿐 아니라 **더 먼 위치의 토큰**도 예측하도록 학습 신호·헤드를 두는 계열을 가리키는 데 쓰인다. 세부(손실 가중, 독립 헤드, depth depth 등)는 모델마다 다르다.

### 4.2 추론 쪽 연결

MTP로 단련된 헤드·깊이가 **draft 토큰을 싸게 내는 장치**가 되면, speculative 파이프라인의 draft 단계와 자연스럽게 결합한다.  
반대로, **별도 작은 draft 모델**을 두는 고전적 speculative도 많다. MTP ≠ 유일한 draft 구현.

```text
[별도 draft 모델] ──┐
                    ├─→ speculative verify ← target LM
[MTP/내부 멀티헤드] ─┘
```

**사실:** 체크포인트가 MTP로 학습되었다 해서, 서버가 자동으로 speculative을 켠다는 보장은 없다.  
**해석:** 모델 카드의 훈련 특징과 엔진의 `speculative`/`mtp` 설정을 **따로** 확인한다.

## 직관 그림
```text
시간(이상화, draft 성공 시)

일반 decode:   [T][T][T][T][T][T]   ← 스텝마다 target 1토큰
speculative:   [D D D][V───][D D][V──] ← draft 묶음 + verify로 여러 토큰 확정
                 ↑제안      ↑검증
```

Verify는 “큰 모델을 $\gamma$번 순차 호출”이 아니라, **한 번의 parallel forward로 여러 위치를 본다**는 쪽이 이득의 핵심인 경우가 많다. 구현이 그 병렬을 못 살리면 가속이 사라진다.

## 작은 산수 연습（가상）
가정(**미측정 예시**):

- Target 단독: 스텝당 20 ms, 1토큰 → 50 tok/s (단일 스트림)
- Draft+Verify 라운드: 28 ms, 평균 수락 3토큰 → $3 / 0.028 \approx 107$ tok/s

수락이 평균 1.2토큰으로 떨어지면 $1.2/0.028\approx 43$ tok/s → **단독보다 느릴 수 있음**.

이 숫자는 교재용이다. 실무는 제107강 템플릿으로만 보고한다.

## 언제 도움이 되는가 / 안 되는가
### 도움이 되기 쉬운 조건（직관）

- Decode가 **메모리 바운드**이고, verify forward 한 번에 여러 위치를 싸게 볼 여유가 있을 때
- Draft가 싸고 수락률이 **충분히 높을** 때 (도메인·온도·프롬프트에 민감)
- 단일·저동시성에서 TPOT가 병목일 때

### 이득이 사라지기 쉬운 조건

- 수락률 저조 (고난도, 고온도, draft 부적합)
- Prefill-heavy 워크로드 — 주병목이 TTFT 쪽
- 이미 Continuous Batching으로 GPU를 포화 — 시스템 Throughput 관점에선 미묘
- 엔진이 speculative 경로를 비효율적으로 구현·미지원
- 검증 실패 thrashing

**사실:** 동시성이 매우 높을 때 speculative이 시스템 throughput를 항상 올린다고 단정할 수 없다.  
**해석:** SLO가 TTFT인지 TPOT인지 Throughput인지에 따라 on/off가 갈린다. A/B는 제107·118강 방식으로.

## 서빙 스택에서의 위치
```text
요청 → Scheduler(제106) → 실행 엔진
              │
              ├─ 일반 decode kernel
              └─ speculative/MTP path (옵션)
                     draft → verify → 수락 토큰만큼 KV append
```

관측 메타(가능하면):

- 평균 수락 길이 / 수락률
- Draft 시간 vs Verify 시간
- TPOT_p50/p95 before·after
- 품질 회귀(동일 시드·벤치 — 날조 금지, 내부 평가)

엔진 플래그 이름·기본값은 버전 의존(제112강).

## 코드로 “라운드”만 스케치
```python
from dataclasses import dataclass

@dataclass
class SpecRound:
    draft_tokens: list[int]
    accepted: list[int]
    resampled: int | None

def speculative_round_sketch(
    draft_fn,
    verify_fn,
    context: list[int],
    gamma: int,
) -> SpecRound:
    """교육용: 수락 규칙의 단순 '일치하면 수락' 스케치.
    실제 speculative sampling은 확률적 수락·보정이 들어간다.
    """
    drafts = draft_fn(context, gamma)
    # verify_fn이 각 위치에서 target이 골랐을 토큰(또는 분포)을 돌려준다고 가정
    target_toks = verify_fn(context, drafts)
    accepted: list[int] = []
    resampled = None
    for i, d in enumerate(drafts):
        if d == target_toks[i]:
            accepted.append(d)
        else:
            resampled = target_toks[i]
            break
    else:
        # 전부 수락 시 보너스 토큰 등을 두는 변형도 있음 — 여기선 생략
        pass
    return SpecRound(drafts, accepted, resampled)
```

**경고:** 위 동등성 스케치는 실제 알고리즘의 분포 보존 증명을 대체하지 않는다. 구현·논문의 수락 함수를 따르라.

## 지표와의 연결（템플릿 필드 재강조）
Speculative on/off 비교 시 제107강 필드에 추가:

```text
- method: speculative=<on|off>, draft=<model|mtp|heads>, gamma=<N>,
          TTFT definition=..., workload=...
- model / GPU / quant / context / batch / framework / version  ← 필수 동일
- results: TPOT_*, throughput_*, acceptance_rate=<MEASURED_OR_NA>
```

`acceptance_rate`에 예시 숫자를 꾸며 넣지 말 것.

## 수락 규칙을 과도하게 단순화하지 말 것
교육용으로 “draft와 target 토큰이 같으면 수락”이라고 말했지만, 문헌의 speculative sampling은 **확률비**로 수락·거절·재수량을 정의하는 경우가 많다. 목표는 target 분포와의 **동등성(또는 명시적 근사)** 이다.

학생에게 남길 문장:

> 구현을 베낄 때는 직관이 아니라 **수락 함수와 보정 샘플**이 있는 원전·엔진 코드를 따른다.

이 책의 스케치는 분포 증명을 대체하지 않는다.

## $\gamma$（제안 길이）선택 직관
| $\gamma$ 작음 | $\gamma$ 큼 |
|---|---|
| 오버헤드 적음 | draft·verify 비용 ↑ |
| 수락 실패 시 타격 작음 | 실패 시 낭비 ↑ |
| 가속 상한 낮음 | 수락률 높을 때 상한 ↑ |

도메인·온도·프롬프트가 바뀌면 최적 $\gamma$가 움직인다. **고정 상수 암기 금지.** 제107 템플릿으로 스윕한다.

## 동시성과 speculative
단일 스트림에서는 TPOT 이득이 잘 보인다.  
동시성이 높아 GPU가 이미 포화면:

- speculative이 SM을 더 쓰거나
- 배치 안 시퀀스마다 수락 길이가 달라 **비균일**이 커져
- 시스템 throughput이 기대와 다를 수 있다

**사실:** 고동시성에서 이득이 항상 양의 부호라고 단정할 수 없다.  
**해석:** SLO가 “단일 사용자 타자감”이면 on, “최대 토큰/초 용량”이면 곡선으로 재검증.

## 품질 회귀를 어떻게 감시하는가
속도만 보고 배포하지 말 것.

체크(형식):

1. 동일 시드·프롬프트 세트에서 출력 일치/근접 (알고리즘이 동등성을 약속하는 경우)
2. 내부 벤치(정답률·선호) — **숫자를 책에 기재하지 말고 각자 측정**
3. 수락률 모니터링 알림 — 급락 시 draft 부적합·데이터 시프트 의심

## 스케줄러·KV와의 상호작용
수락된 토큰이 $m$개면 KV에 $m$ 스텝분이 append된다.  
거절 후 재수량 토큰은 한 개라도 상태가 갱신된다.

PagedAttention 관점(제105):

- 한 라운드에 여러 토큰이 확정되면 블록 소비가 한꺼번에 늘 수 있음
- free block 회계는 speculative 라운드 단위로도 맞아야 함

스케줄러(제106)는 speculative 배치의 토큰 예산을 일반 decode와 다르게 잡을 수 있다 — **버전 의존**.

## 실습 — on/off A/B 절차（숫자 금지）
```text
1) 템플릿 8필드 고정
2) method만 speculative=off → 측정 파일 A
3) method만 speculative=on, gamma=<N>, draft=<...> → 파일 B
4) 비교 열: TTFT_p50/p95, TPOT_p50/p95, throughput, acceptance_rate
5) 동시성 c=1과 c=c_high를 분리 표로
6) EXAMPLE 값을 결과에 복사하지 말 것
```

## 의사코드 — 수락률 로깅
```python
def log_spec_metrics(rounds: list) -> dict:
    acc_lens = [len(r.accepted) for r in rounds]
    return {
        "n_rounds": len(rounds),
        "mean_accepted_tokens": "COMPUTE_IN_YOUR_RUN",
        "frac_full_gamma": "COMPUTE_IN_YOUR_RUN",
        # 실측 전 자리표시 — 책에 숫자 고정 금지
    }
```

## 자주 하는 실수
1. Draft 품질을 무시하고 $\gamma$만 키움
2. 분포 보존이 깨진 채 “비슷하면 됐네”로 배포
3. Prefill 병목에 speculative만 반복 튜닝
4. MTP 훈련 사실과 서버 설정을 동일시
5. 시스템 throughput 상승 없이 단일 스트림 tokens/s만 보고 전체 용량을 과대평가
6. 엔진 미지원 아키텍처에 플래그만 켜고 실패를 모델 탓으로 돌림
7. 수락률 모니터링 없이 평균 TPOT만 봄
8. 고동시성 곡선을 생략한 채 용량 계획

## 5 Draft가 틀리는 전형적인 이유
1. Draft가 너무 작아 구문·사실에서 자주 빗나감
2. Target만 domain-adapt/SFT 되었고 draft는 base
3. 온도·top-p가 높아 다음 토큰 엔트로피가 큼
4. 코드/수식처럼 **국소 제약이 강한** 토큰 열
5. 다국어·혼합 프롬프트에서 draft 커버리지 부족

처방전은 “$\gamma$만 키우기”가 아니라 **draft 정렬·도메인 적합성·샘플링**을 먼저 본다.

## 6 Medusa·EAGLE 등 이름과의 관계（지도만）
문헌·엔진에는 Medusa, EAGLE, look-ahead 등 **변형 이름**이 많다.  
공통 분모는 여전히 **값싼 다중 후보 + 큰 모델 검증**에 가깝다.  
차이(트리 주의, 특성 재사용, 헤드 구조)는 원논문·릴리스 노트로 두고, 이 강의는 이름을 암기하지 않는다.

**사실:** 지원 목록은 엔진 버전마다 갱신된다.  
**해석:** 새 이름을 볼 때마다 Draft/Verify/수락률 프레임으로 재분류하면 학습 비용이 줄어든다.

## 7 벽시계 분해 — 라운드 하나
교육용 타이머 구간:

```text
t0 draft_start
t1 draft_end / verify_start
t2 verify_end / accept_done
```

- draft 비율 $(t1-t0)/(t2-t0)$ 이 크면 draft가 병목
- verify 비율이 크고 수락이 짧으면 **실패한 큰 forward**에 가깝다
- 수락 길이 / $(t2-t0)$ 가 유효 토큰율

값을 책에 쓰지 말고, 하네스에 구간을 남기는 습관만 가져간다.

## 8 제108 Family Card에 적을 speculative 칸
```text
prediction: next_token | mtp | speculative_ready
draft_source: none | small_model | mtp_heads | other
engine_spec_flag: <name or none>
gamma_default: <unknown until sweep>
```

모델 카드의 MTP 언급과 `engine_spec_flag`가 동시에 채워져야 “켠다”가 된다.

## 9 요약 시나리오 세 가지
1. **코딩 어시스턴트, 저동시성:** TPOT 민감 → speculative A/B 우선 후보
2. **RAG QA, 긴 컨텍스트:** TTFT·prefill 우선 → speculative 전 입력 경로 정리
3. **멀티테넌트 고동시성:** 시스템 throughput 곡선에서 on/off 재검증

시나리오를 `method`에 명시하지 않은 채 팀 내 “빠르다/느리다” 논쟁을 하지 말 것.

## 10 안전장치 — 언제 끌 것인가
운영 기본값을 on으로 두지 말고, 다음이면 **끄거나 롤백**을 검토한다.

- 수락률이 내부 임계 아래로 지속
- p95 TPOT가 off 대비 악화（실측）
- 검증 경로 예외·타임아웃 증가
- 품질 모니터（동등성·태스크 스코어）회귀
- 엔진 업그레이드 후 speculative 경로 미검증

임계 숫자 자체는 팀 SLO로 정하고 이 책에 고정하지 않는다.

## 11 한 문단 복습
Speculative decoding은 decode가 대역폭에 묶일 때, 값싼 draft로 여러 토큰을 제안하고 target이 한 번에 검증해 **스텝당 확정 토큰**을 늘리려는 기법이다. MTP는 그 draft를 모델 안쪽 다중 예측으로 제공하는 갈래일 수 있다. 이득은 수락률·구현·동시성에 달려 있으며, 제107강 템플릿 없는 속도 주장은 기각한다.

## 12 커리큘럼 앵커 — 104에서 111로
```text
104 vLLM 개요
105 PagedAttention     ← KV 화폐
106 Scheduler          ← 배분
107 TTFT/TPOT/Thru     ← 측정 언어
108 계열 지도          ← 무엇을 올리는가
109 MoE
110 MTP/Speculative    ← 지금 여기
111 GPU 대역폭         ← 왜 decode가 묶이는가
```

110을 건너뛰면 “대역폭 Bound decode” 처방전 중 한 축이 비고, 111만 읽으면 하드웨어 용어만 남는다. 두 강을 한 세트로 보라.

## 13 측정 전에 적을 가설 문장
가설 예:

> “이 워크로드에서 speculative on은 off 대비 TPOT_p50을 개선하고, TTFT_p95는 크게 해치지 않는다. 동시성 c=1에서 먼저 확인하고 c=c_high에서 throughput을 재확인한다.”

가설에 **수치 목표를 책에 박지 말고**, 팀 문서에만 적는다. 실험 후 가설을 유지·기각으로 표시한다.

## LLM에서는 어디에 사용될까?

이번 110강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- Speculative decoding = **값싼 draft 제안 + target verify**로 스텝당 확정 토큰을 늘리려는 기법.
- MTP는 **다중 토큰 예측** 쪽 훈련/헤드 변이이며, draft 구현의 한 갈래와 연결된다.
- 이득의 열쇠는 **수락률**과 verify 병렬 효율이다.
- 보통 노리는 축은 **TPOT**; 만능 가속기가 아니다. 고동시성은 재검증.
- $\gamma$·draft 선택은 워크로드 의존 — 암기 상수 없음.
- 명칭·플래그는 변한다. **Draft/Verify/수락/측정 템플릿**이 남는다.

## 용어 사전
| 용어 | 설명 |
|---|---|
| Speculative Decoding | Draft로 제안하고 target으로 검증하는 가속 골격 |
| Draft model / head | 후보 토큰을 싸게 생성하는 쪽 |
| Verify / Target | 기준 분포로 수락·수정하는 큰 모델 |
| Acceptance rate | 제안 대비 수락되는 비율·길이 |
| $\gamma$ | 라운드당 draft 제안 길이 |
| MTP | Multi-Token Prediction — 다중 미래 토큰 예측 계열 |

## 연습문제
### 문제 1

Speculative decoding에서 draft와 target의 역할을 한 문장씩 쓰시오.

### 문제 2

수락률이 낮을 때 오히려 느려질 수 있는 이유를 쓰시오.

### 문제 3

MTP와 “별도 draft 모델 speculative”의 관계를 한 문장으로.

### 문제 4

긴 프롬프트·짧은 답 워크로드에서 speculative보다 먼저 볼 지표는?

### 문제 5（사실/해석）

“MTP로 학습된 모델은 배포 시 항상 토큰/초가 N배”를 평가하시오.

### 문제 6

제107강 비교 실험에서 speculative on/off에 반드시 같게 유지해야 할 필드 예를 세 개 드시오.

### 문제 7（연결）

이 기법이 주로 완화하려는 하드웨어 병목 감각은 연산(FLOPs)인가, 가중치 메모리 대역폭인가? (전형적인 decode 상황)

### 문제 8

$\gamma$를 과도하게 키울 때의 위험을 쓰시오.

### 문제 9

수락률 급락이 운영 중 보이면 의심할 만한 원인 두 가지를 드시오.

### 문제 10

제104→111 흐름에서 제110강이 연결하는 “구조 축”과 “하드웨어 축”은 각각 무엇인가?

---

## 정답 및 해설
### 문제 1

Draft는 미래 토큰 후보를 싸게 제안하고, target은 그 후보를 기준 분포로 검증·수정한다.

### 문제 2

Verify(및 draft) 비용을 치르고도 확정 토큰이 거의 1개라, 오버헤드만 남기 때문.

### 문제 3

MTP는 다중 토큰 예측 능력/헤드를 주는 훈련·구조 변이일 수 있고, 별도 draft 모델은 speculative의 다른 구현 — 둘 다 draft 역할을 맡을 수 있으나 동일하지 않다.

### 문제 4

TTFT / prefill (제107강 매핑).

### 문제 5

사실이 아님. 엔진 설정·수락률·워크로드 의존의 **과장·미검증 주장**으로 취급.

### 문제 6

예: model, GPU, quantization, context, concurrency, framework version 중 세 개 이상 — 비교 축만 `method(speculative)`로 남긴다.

### 문제 7

전형적으로 **가중치 메모리 대역폭**에 묶인 decode를, 스텝당 더 많은 확정 토큰으로 상쇄하려는 감각.

### 문제 8

draft/verify 비용 증가와 수락 실패 시 낭비가 커져 오히려 느려질 수 있다.

### 문제 9

예: 트래픽 도메인 변화, 온도/샘플링 변화, draft·target 불일치 배포, 프롬프트 분포 시프트.

### 문제 10

구조 축: 제108~109(계열·MoE)와 MTP. 하드웨어 축: 이어지는 제111강(CUDA/SM/대역폭).

## 다음 강의와 연결
왜 decode가 대역폭에 묶이는지, SM·HBM이 무엇인지 뼈대를 알아야 튜닝 우선순위가 선다.  
다음 **제111강. GPU 아키텍처 — CUDA, SM, Memory Bandwidth**에서 하드웨어 좌표를 펼친다.

되돌아보면:

- 제104강 vLLM 개요 → 제105 PagedAttention → 제106 Scheduler → 제107 지표  
- 제108 계열 지도 → 제109 MoE → **제110 MTP/Speculative** → 제111 GPU  

엔진·구조·하드웨어가 한 줄로 이어진다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [109강. MoE](109강_MoE.md)
- **다음 강:** [111강. GPU 아키텍처 — CUDA, SM, Memory Bandwidth](111강_GPU_아키텍처_CUDA_SM_Memory_Bandwidth.md)

<!-- /LECTURE_NAV -->
