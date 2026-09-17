# 107강. LLM Serving 성능 지표 — TTFT, TPOT, Throughput
## 이번 강에서 배우는 내용

- TTFT, TPOT, Throughput(및 tokens/sec)의 정의
- 사용자 체감 지연과 GPU 효율이 왜 한 숫자로 안 접히는가
- 측정 시 반드시 같이 적어야 하는 보고 필드(템플릿)
- 평균만 보고 꼬리 지연·혼합 워크로드를 섞어 버리는 실수

## 왜 중요한가?
서빙 대화에서 흔히 충돌한다.

- “처리량이 높은데 챗봇이 답답하다”
- “첫 토큰은 빠른데 긴 답에서 끊긴다”
- “tokens/s가 높은데 동시 사용자가 적다”

원인은 대개 **지표를 하나로 뭉갠 것**이다. Prefill 지배 구간과 Decode 지배 구간, 단일 스트림과 다중 스트림은 다른 게임이다.

## 선수 개념
1. Prefill vs Decode (제100강)
2. Continuous Batching·스케줄러 큐잉 (제102·106강)
3. 양자화·정밀도가 속도에 미치는 축 (제103강) — 측정 시 반드시 기록
4. (예고) 모델 구조·MoE·스펙큘레이티브가 지표에 주는 영향 (제108~110강)

## 핵심 지표 정의
타임라인을 하나 둔다.

```text
요청 도착 t0
  → (큐 대기)
  → prefill 시작
  → 첫 출력 토큰 방출 t_first
  → 이후 토큰들 … 마지막 토큰 t_last
```

### 3.1 TTFT — Time To First Token

$$

\mathrm{TTFT} = t_{\mathrm{first}} - t_{\mathrm{ref}}

$$

$t_{\mathrm{ref}}$를 무엇으로 잡는지가 핵심이다.

| 선택 | 의미 |
|---|---|
| $t_0$ (도착) | **큐잉+스케줄+prefill** 포함 — 사용자 체감에 가까움 |
| prefill 시작 시각 | 순수 계산에 가깝고 큐 영향 제거 |

**교육용 기본:** 제품 대화에서는 도착 기준 TTFT를 쓰고, 프로파일링에서는 둘을 분리해 적는다. 리포트에 정의를 한 줄로 박제한다.

### 3.2 TPOT — Time Per Output Token

첫 토큰 **이후** 생성 구간의 평균(또는 중앙값) 토큰 간격.

출력 토큰이 $N_{\mathrm{out}}$개이고 $N_{\mathrm{out}}\ge 2$일 때, 흔한 정의:

$$

\mathrm{TPOT}=\frac{t_{\mathrm{last}}-t_{\mathrm{first}}}{N_{\mathrm{out}}-1}

$$

단위는 보통 ms/token. 역수가 **per-request decode tokens/s**에 가깝다.

**주의:** 스펙큘레이티브 디코딩(제110강)처럼 한 스텝에 여러 토큰을 확정하면 “벽시계 간격 / 확정 토큰 수”로 다시 정의해야 한다.

### 3.3 E2E Latency

$$

\mathrm{Latency}_{\mathrm{E2E}} = t_{\mathrm{last}} - t_0

$$

대략 $\mathrm{TTFT} + \mathrm{TPOT}\times(N_{\mathrm{out}}-1)$ 꼴로 분해해 볼 수 있다(정의가 일치할 때).

### 3.4 Throughput — 시스템 처리량

구간 $T$초 동안 서버가 완료한 출력 토큰(또는 요청) 총량:

$$

\mathrm{Throughput}_{\mathrm{tok}}=\frac{\sum N_{\mathrm{out}}}{T}
\quad\text{(tokens/s, system)}

$$

요청 처리량:

$$

\mathrm{Throughput}_{\mathrm{req}}=\frac{\#\text{completed requests}}{T}

$$

**단일 스트림 tokens/s**와 **다중 동시 요청의 시스템 tokens/s**는 다르다. 후자가 Continuous Batching의 본전이다.

### 3.5 그 밖에 자주 같이 보는 것

| 이름 | 용도 |
|---|---|
| ITL (Inter-Token Latency) | 토큰 간 간격 분포 — TPOT의 분포 버전 |
| Goodput | SLO를 만족한 처리량만 인정 |
| 큐 대기 시간 | 스케줄러·부하 분리 |
| GPU util / SM 효율 | 하드웨어 병목 힌트 (제111강) |

## 직관 — 어떤 지표가 누구의 고통인가
| 역할 | 주로 보는 것 |
|---|---|
| 챗 UI 사용자 | TTFT, 이후 스트리밍 끊김(ITL/TPOT) |
| 용량 기획 | 시스템 Throughput, 동시 세션, KV 메모리 |
| 커널·양자화 튜닝 | Prefill tokens/s, Decode tokens/s 분리 |
| SRE | p95/p99 TTFT·E2E, 에러율, 선점·OOM |

한 줄:

> TTFT는 “첫인상”, TPOT는 “말의 속도”, Throughput은 “가게가 하루에 몇 잔 팔았는지”다.

## 측정 방법 — 원칙
1. **워크로드를 적는다** — 입력 길이 분포, 출력 길이, 동시성, 샘플링 파라미터
2. **워밍업** — 컴파일·캐시·GPU 클럭 안정화 후 본측정
3. **통계** — 평균만 금지. 최소 p50/p95(가능하면 p99)
4. **분리** — Prefill-only / Decode-heavy / 혼합을 한 표에 섞지 말 것
5. **정의 고정** — TTFT에 큐잉 포함 여부를 메타데이터에 기록
6. **재현 필드** — 아래 템플릿 필수

클라이언트가 토큰 스트림 타임스탬프를 찍는 방식이 흔하다. 서버 내부 span과 클라이언트 체감은 다를 수 있다. **어느 쪽 시계인지**를 명시한다.

## 보고 템플릿（필수 필드）— 숫자 날조 금지
아래는 **리포트 양식**이다. 값은 모두 자리표시이며, 실측 전까지 채우지 않는다.

```text
### Serving Measurement Report (template)

- model:                 <예: MODEL_NAME_OR_CHECKPOINT — 미기입>
- GPU:                   <예: GPU_NAME × N — 미기입>
- quantization:          <예: FP16 | FP8 | INT4 | … — 미기입>
- context / max length:  <예: in_tokens, out_tokens 또는 max_model_len — 미기입>
- batch / concurrency:   <예: max_num_seqs, 클라이언트 동시 수 — 미기입>
- framework:             <예: vLLM | TensorRT-LLM | SGLang | … — 미기입>
- version:               <예: engine==x.y.z, CUDA, driver — 미기입>
- method:                <예: 측정 하네스, TTFT 정의, 워밍업, 샘플 수 — 미기입>

Results (illustrative placeholders ONLY — NOT measured):
- TTFT_p50_ms:           <EXAMPLE_NOT_MEASURED>
- TTFT_p95_ms:           <EXAMPLE_NOT_MEASURED>
- TPOT_p50_ms_per_tok:   <EXAMPLE_NOT_MEASURED>
- TPOT_p95_ms_per_tok:   <EXAMPLE_NOT_MEASURED>
- throughput_out_tok_s:  <EXAMPLE_NOT_MEASURED>
- notes:                 <워크로드 파일 해시, 시드, 이슈 링크>
```

표 형태 요약:

| 필드 | 기입 예(자리표시) | 왜 필수인가 |
|---|---|---|
| model | `EXAMPLE_MODEL` | 구조·크기·토크나이저 |
| GPU | `EXAMPLE_GPU` | 대역폭·VRAM |
| quant | `EXAMPLE_DTYPE` | 커널·정확도 |
| context | `IN=?, OUT=?` | Prefill/Decode 비중 |
| batch | `CONC=?` | Continuous batch 효과 |
| framework | `EXAMPLE_ENGINE` | 스케줄·커널 |
| version | `v?.?.?` | 재현 |
| method | `정의·하네스` | 비교 가능성 |

**금지:** 위 필드를 비운 채 “우리 엔진이 2배”라고 선언하기. 제118강 리포트 작성법에서 다시 확장한다.

## 작은 숫자로 분해하기（가상 산수）
다음은 **가상의 산수 연습**이지 실측이 아니다.

가정: $t_0=0$, $t_{\mathrm{first}}=200\,\mathrm{ms}$, 이후 토큰 99개를 균등 간격으로 $t_{\mathrm{last}}=200+99\times 20=2180\,\mathrm{ms}$.

- $\mathrm{TTFT}=200\,\mathrm{ms}$（도착 기준 가정）
- $\mathrm{TPOT}=20\,\mathrm{ms/token}$
- $\mathrm{E2E}=2180\,\mathrm{ms}$
- 이 요청의 출력 tokens/s $\approx 1000/20=50$ tokens/s

동시에 같은 패턴 요청이 서버에서 겹치면, **시스템** throughput은 50의 배가 될 수 있다(이상적 스케일). 실제로는 KV·스케줄·커널 한계로 선형이 아니다.

## 무엇을 섞지 말아야 하는가
| 섞으면 안 되는 것 | 이유 |
|---|---|
| 단일 스트림 tokens/s ↔ 다중 동시 throughput | 배치 이득이 후자에만 큼 |
| greedy 짧은 답 ↔ 긴 sampling 답 | Decode 비중·종료 분포 상이 |
| 큐잉 포함 TTFT ↔ 순수 prefill 시간 | 병목 위치가 다름 |
| 다른 모델·양자화·엔진 버전 | 원인 귀속 불가 |
| “요청/s”만으로 토큰 비용 대체 | 출력 길이 무시 |
| 워밍업 구간과 본측정 | 컴파일 스파이크 오염 |

**사실:** 공개 리더보드·블로그 수치는 워크로드가 제각각이다.  
**해석:** 숫자보다 **필드가 채워진 리포트**가 공학이다.

## Prefill/Decode와 지표 매핑
```text
긴 프롬프트, 짧은 답  → TTFT·prefill 최적화 민감
짧은 프롬프트, 긴 답  → TPOT·decode·KV 메모리 민감
높은 동시성           → Throughput·스케줄·PagedAttention 민감
```

튜닝 레버 예고:

- 양자화 → 대개 decode·메모리 (제103)
- 스케줄 예산 → TTFT vs throughput (제106)
- TP/병렬 → 큰 모델 (제113)
- Speculative/MTP → TPOT 쪽 시도 (제110)
- 모델 계열(MoE 등) → 계산·통신 패턴 (제108~109)

## 코드로 “타임스탬프 수집” 스케치
```python
import time
from dataclasses import dataclass

@dataclass
class StreamMetrics:
    t_arrive: float
    t_first: float | None = None
    t_tokens: list[float] | None = None

    def __post_init__(self) -> None:
        self.t_tokens = []

    def on_token(self) -> None:
        now = time.perf_counter()
        if self.t_first is None:
            self.t_first = now
        self.t_tokens.append(now)

    def ttft_ms(self) -> float:
        assert self.t_first is not None
        return (self.t_first - self.t_arrive) * 1000

    def tpot_ms(self) -> float | None:
        if len(self.t_tokens) < 2:
            return None
        return (self.t_tokens[-1] - self.t_tokens[0]) * 1000 / (len(self.t_tokens) - 1)
```

프로덕션에서는 시계 동기화, 취소된 스트림, 버퍼링된 소켓 플러시를 고려한다.

## 분포 · 꼬리 · SLO
평균은 요약이지 계약이 아니다.

| 통계 | 쓰는 이유 |
|---|---|
| p50 | “전형적인” 체감 |
| p95 | 많은 사용자가 가끔 겪는 나쁨 |
| p99 | 장애·용량 계획·SLO 위반에 민감 |
| max | 이상치·타임아웃과 함께 볼 때 |

SLO 예(형식만, 수치 날조 금지):

```text
TTFT_p95_ms <= <YOUR_BUDGET>   # 정의: 도착 기준 / 큐잉 포함
TPOT_p95_ms_per_tok <= <...>
error_rate <= <...>
# 모두 method·workload와 함께 기록
```

**Goodput:** SLO를 깨는 요청은 처리량에서 빼거나 별도 집계한다. “많이 뱉었지만 대부분 지각”은 성공이 아니다.

## Prefill tokens/s 와 Decode tokens/s 분리
한 숫자 `tokens/s`만 있으면 국면이 안 보인다.

```text
prefill_tokens_s  ≈  (합산 입력 토큰) / (prefill에 쓴 벽시계)
decode_tokens_s   ≈  (합산 출력 토큰) / (decode에 쓴 벽시계)
system_out_tok_s  ≈  (합산 출력 토큰) / (전체 구간 T)   ← 제4.4 Throughput
```

주의:

- 구간 정의가 겹치면 이중 계산된다
- Continuous Batching에서는 한 스텝에 prefill·decode가 섞일 수 있어 **서버 내부 단계 타이머**가 필요할 수 있다
- 클라이언트만의 타임스탬프는 국면 분리에 한계가 있다

## 동시성과 곡선 — 한 점으로 말하지 말 것
동시성 $c=1,2,4,\ldots$를 올려 가며 TTFT·TPOT·Throughput을 같이 그린다(실측 시).

```text
c ↑  →  Throughput ↑ (어느 구간)  →  포화
     →  TTFT·TPOT 악화 시작점 = 용량 한계 힌트
```

**사실:** 포화점은 모델·GPU·양자화·스케줄 설정에 따라 다르다.  
**해석:** “우리 GPU는 동시 32” 같은 문장은 **곡선 한 장** 없이는 주장으로 쓰지 않는다.

## 워크로드 명세 — 길이 분포가 절반이다
고정할 것:

1. 입력 길이: 고정 / 히스토그램 / 실로그 리플레이
2. 출력 길이: `max_tokens`·EOS 분포
3. 샘플링: temperature, top_p, seed 정책
4. 도착 과정: closed-loop(동시 c 유지) vs open-loop(포아송 등)
5. 캐시: cold vs warm prefix

같은 평균 길이라도 **분산**이 크면 p99 TTFT가 달라진다. 리포트에 분포 요약(예: p50/p90 입력 길이)을 남긴다. 값 자체는 실측·데이터셋에만.

## 클라이언트 vs 서버 시계
| 관점 | 잡는 것 | 놓치기 쉬운 것 |
|---|---|---|
| 클라이언트 | 체감 E2E, 네트워크 | 서버 큐 내부 분해 |
| 서버 span | prefill/decode/큐 | 사용자 단말·LB 지연 |
| GPU 프로파일러 | 커널·대역폭 | 요청 단위 SLO |

비교 실험에서는 **어느 시계로 TTFT를 정의했는지**를 `method`에 적는다. 서로 다른 시계의 숫자를 같은 열에 넣지 말 것.

## 실습 — 측정 하네스 체크리스트（숫자 없이）
1. 워밍업 $N_w$ 요청 후 본측정 $N$ 요청
2. 각 스트림에 `t_arrive`, `t_first`, `t_last`, `n_out` 기록
3. TTFT/TPOT/E2E 백분위 계산
4. 템플릿 8필드 기입 — 값 없으면 `NOT_RECORDED`
5. `EXAMPLE_NOT_MEASURED`를 결과에 남기지 말 것(양식 연습과 실측 파일을 분리)

```python
def summarize(stream_metrics):
    ttfts = [m.ttft_ms() for m in stream_metrics]
    tpots = [m.tpot_ms() for m in stream_metrics if m.tpot_ms() is not None]
    # percentile 함수는 표준 라이브러리/numpy 등 — 여기선 자리만
    return {
        "n": len(stream_metrics),
        "ttft_p50": "COMPUTE_IN_YOUR_RUN",
        "ttft_p95": "COMPUTE_IN_YOUR_RUN",
        "tpot_p50": "COMPUTE_IN_YOUR_RUN",
        "tpot_p95": "COMPUTE_IN_YOUR_RUN",
    }
```

## 잘못된 비교 사례（교육용 안티패턴）
| 안티패턴 | 문제 |
|---|---|
| A엔진 FP16 vs B엔진 INT4 | 양자화·엔진이 동시에 변함 |
| 동시 1의 tokens/s로 용량 계획 | 배치 이득 무시 |
| 평균만으로 “2배 빠름” | 꼬리·워크로드 미상 |
| 블로그 숫자 전재 | 필드 공백 |
| 워밍업 포함 | 컴파일 스파이크 |

올바른 자세: **한 번에 하나의 축만** 바꾼다.

## 제106·105·103과의 진단 매핑
| 증상 | 먼저 볼 곳 |
|---|---|
| TTFT p99↑, GPU util 낮음 | 큐·admission·waiting (106) |
| TTFT↑, prefill 길음 | 입력 길이·커널·양자화 |
| TPOT↑, 동시성↑ | KV 블록 압박·선점 (105·106) |
| Throughput↓ after quant | 커널 미지원·폴백 의심 (103) |
| 첫토큰 OK, 이후 지터 | ITL 분포·선점 thrashing |

지표는 진단 언어이고, 원인 모듈은 이전 강의다.

## 자주 하는 실수
1. 평균 TTFT만 보고 p99 사용자 불만을 놓침
2. Throughput을 올려 놓고 TTFT SLO 위반을 “별문제 없음”으로 처리
3. 출력 길이 분포를 고정하지 않은 A/B
4. 클라이언트 재시도·타임아웃을 성공 throughput에 포함
5. 예시 숫자를 실측처럼 인용 — **이 책의 `EXAMPLE_NOT_MEASURED`를 실수로 옮기지 말 것**
6. Prefill/Decode tokens/s와 system throughput을 같은 셀에 섞음
7. 동시성 곡선 없이 용량을 단정
8. 서버·클라이언트 시계를 구분하지 않음

## 5 토큰/초 표기 — 분자·분모를 말하라
`tokens/s`만 적으면 다음이 모호하다.

| 표기 | 분자 | 분모 |
|---|---|---|
| 출력 토큰 처리량 | $\sum N_{\mathrm{out}}$ | 벽시계 $T$ |
| 입력+출력 | $\sum(N_{\mathrm{in}}+N_{\mathrm{out}})$ | $T$ |
| Prefill only | $\sum N_{\mathrm{in}}$ | prefill 시간 |
| Per-stream decode | $N_{\mathrm{out}}-1$ | $t_{\mathrm{last}}-t_{\mathrm{first}}$ |

리포트 `method`에 **어느 정의인지** 한 줄을 강제한다. 팀 내에서 표기가 갈리면 제118강 양식으로 통일한다.

## 6 에러·취소·타임아웃
처리량 집계에서:

- 타임아웃으로 끊긴 스트림의 부분 토큰을 성공에 넣을지
- 클라이언트 abort를 분모 $T$에 남길지
- 재시도가 이중 계산인지

를 미리 정한다. 정하지 않으면 Throughput이 **낙관적으로 부풀거나** 반대로 과도하게 낮아진다. SLO 대시보드와 벤치 하네스의 정의를 맞출 것.

## LLM에서는 어디에 사용될까?

이번 107강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- **TTFT** = 첫 토큰까지의 시간(정의·큐잉 명시).
- **TPOT** = 첫 토큰 이후 출력 토큰당 시간.
- **Throughput** = 단위 시간당 시스템 산출(토큰 또는 요청).
- Prefill/Decode tokens/s와 시스템 throughput을 분리한다.
- 사용자 체감과 용량 지표를 **한 점수로 합치지 않는다.**
- 동시성 곡선·백분위·워크로드 분포가 “한 점 숫자”를 이긴다.
- 모든 비교는 **model / GPU / quant / context / batch / framework / version / method** 템플릿과 함께.

## 용어 사전
| 용어 | 설명 |
|---|---|
| TTFT | Time To First Token |
| TPOT | Time Per Output Token |
| ITL | 토큰 간 지연(분포로 볼 때) |
| Throughput | 시스템 단위 시간당 처리량 |
| Goodput | SLO 충족분만 인정한 처리량 |
| Closed-loop | 동시 요청 수를 유지하며 부하 |
| Open-loop | 도착 과정으로 부하를 가함 |

## 연습문제
### 문제 1

$t_0=0$, $t_{\mathrm{first}}=0.3\,\mathrm{s}$, $t_{\mathrm{last}}=1.3\,\mathrm{s}$, 출력 토큰 51개일 때 TPOT(초/토큰)는? (첫 토큰 이후 50간격)

### 문제 2

시스템 throughput(tokens/s)과 단일 요청 $1/\mathrm{TPOT}$가 다를 수 있는 이유를 쓰시오.

### 문제 3

TTFT 정의에서 큐잉 포함 여부를 밝히지 않으면 어떤 오해가 생기는가?

### 문제 4

성능 리포트에 반드시 넣어야 할 8개 필드 묶음을 나열하시오.

### 문제 5

긴 컨텍스트 QA(긴 입력·짧은 출력)에서 우선 개선할 지표 축은?

### 문제 6（사실/해석）

블로그에 “300 tokens/s”만 있고 GPU·동시성·버전이 없다. 이 숫자를 사실적 비교 기준으로 써도 되는가?

### 문제 7

Goodput이 Throughput과 다른 점은?

### 문제 8

동시성만 바꾸고 나머지 템플릿 필드를 고정한 실험의 목적은?

### 문제 9

클라이언트 TTFT와 서버 prefill span을 같은 열에 두면 안 되는 이유는?

### 문제 10

E2E latency를 TTFT·TPOT로 대략 분해하는 식을 쓰시오($N_{\mathrm{out}}\ge 2$, 정의가 일치할 때).

---

## 정답 및 해설
### 문제 1

$(1.3-0.3)/50=0.02$ 초/토큰 $=20\,\mathrm{ms/token}$.

### 문제 2

다중 요청 Continuous Batching으로 시스템 합산 토큰율이 단일 스트림보다 커질 수 있고, 반대로 혼잡·선점으로 작아질 수도 있다.

### 문제 3

부하(큐) 문제를 모델·커널 문제로 오인하거나 그 반대를 한다.

### 문제 4

model, GPU, quantization, context, batch/concurrency, framework, version, method.

### 문제 5

TTFT / prefill 경로(및 이에 영향을 주는 스케줄·커널).

### 문제 6

안 된다. 재현 필드가 비어 있으면 **비교 불가능한 주장**이다.

### 문제 7

Goodput은 SLO를 만족한 산출만 인정한다. 지각·실패를 성공 처리량에 넣지 않는다.

### 문제 8

용량·포화·지연 악화 시작점을 보기 위한 **단일 축 비교**.

### 문제 9

네트워크·LB·직렬화가 포함 여부가 달라 **병목 위치가 다른 시계**이기 때문.

### 문제 10

$\mathrm{E2E}\approx\mathrm{TTFT}+\mathrm{TPOT}\times(N_{\mathrm{out}}-1)$.

## 다음 강의와 연결
같은 엔진·GPU라도 **모델 구조**가 바뀌면 KV 크기·연산 패턴·배치 효율이 달라진다.  
다음 **제108강. 모델 알고리즘 계열별 모델 구조**에서 Dense decoder-only, MoE, hybrid/SSM 등 계열 지도를 그리고 서빙 함의를 연결한다.

지표는 “점수”가 아니라 **진단 언어**다. 제106강의 큐 상태와 이 강의의 TTFT/TPOT를 같이 읽어야 병목이 보인다. 제118강에서 리포트 작성법을 다시 확장한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [106강. vLLM Scheduler](106강_vLLM_Scheduler.md)
- **다음 강:** [108강. 모델 알고리즘 계열별 모델 구조](108강_모델_알고리즘_계열별_모델_구조.md)

<!-- /LECTURE_NAV -->
