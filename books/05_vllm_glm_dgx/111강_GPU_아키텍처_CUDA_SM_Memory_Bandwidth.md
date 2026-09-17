# 제111강. GPU 아키텍처 — CUDA, SM, Memory Bandwidth

> **학습 목표**
> - CUDA 실행 모델에서 SM(Streaming Multiprocessor), warp, thread block의 역할을 구분한다.
> - HBM(또는 해당 플랫폼 메모리) 대역폭(bandwidth) 이 LLM decode에 왜 자주 병목이 되는지 설명한다.
> - compute-bound 와 memory-bound 를 직관과 간단한 roofline 스케치로 구분한다.
> - 벤더 스펙 시트의 “TFLOPS / GB/s”를 어떻게 읽는지 알고, 임의 성능 수치를 지어내지 않는다.
> - 제112~116강(엔진 선택·TP·NCCL·DGX Spark·서빙 프로젝트)이 왜 이 기초 위에 올라가는지 연결한다.

---
## 1. 왜 이것을 배우는가

LLM 서빙에서 자주 나오는 문장들이다.

```text
“GPU utilization이 낮다”
“HBM이 꽉 찼다”
“decode가 느리다”
“TP를 늘리면 빨라질 것이다”
```

이 문장들은 **같은 계층의 주장**이 아니다.

| 관찰 | 실제로 묻는 질문 |
|---|---|
| utilization 낮음 | SM이 일을 못 받는가, 아니면 메모리 대기인가? |
| 메모리 꽉 참 | 용량(capacity) 문제인가, 대역폭(bandwidth) 문제인가? |
| decode 느림 | 가중치·KV를 읽는 비용이 지배적인가? |
| TP 확대 | 통신 비용이 계산 절약을 덮지 않는가? |

제100~101강에서 prefill/decode와 KV Cache를 배웠다. 그 개념을 GPU에 대응시키면:

```text
Prefill  : 긴 시퀀스 × 큰 matmul → 상대적으로 compute에 가깝게 갈 수 있음
Decode   : 토큰 1개 × 전체 가중치·KV 읽기 → 상대적으로 memory에 가깝게 갈 수 있음
```

설명: “항상”이 아니다. 배치 크기, 양자화, 커널, 모델 폭에 따라 경계가 움직인다. 다만 **decode가 memory-bound 쪽에 기울기 쉽다**는 직관은 서빙 설계의 출발점이다.

## 2. 먼저 알아야 할 개념

1. **Training vs Inference** — 제99강
2. **Prefill / Decode** — 제100강
3. **KV Cache** — 제101강
4. **Quantization이 바이트를 줄이는 이유** — 제103강
5. **TTFT / TPOT / Throughput** — 제107강

하드웨어 세대·칩 이름은 빠르게 바뀐다. 이 강의는 **CUDA 실행 모델의 공통 골격**과 **대역폭 직관**에 초점을 둔다. 특정 SKU의 최신 숫자는 벤더 문서를 직접 확인하는 방법을 제115강과 함께 익힌다.

## 3. CUDA란 무엇인가

### 3.1 용어

| 영어 | 한국어 | 쉬운 정의 | 왜 필요한가 |
|---|---|---|---|
| **CUDA** | 쿠다 | NVIDIA GPU에서 병렬 커널을 실행하기 위한 플랫폼·프로그래밍 모델 | LLM 커널·추론 엔진이 GPU를 구동하는 기본 계층 |
| **Kernel** | 커널 | GPU에서 실행되는 함수(수천~수백만 스레드로 펼쳐짐) | matmul, attention, softmax 등이 커널로 돌아감 |
| **Host / Device** | 호스트 / 디바이스 | CPU 쪽 / GPU 쪽 | 데이터 이동과 실행 위치를 구분 |

사실:

- CUDA는 NVIDIA의 병렬 컴퓨팅 플랫폼이다.
- 애플리케이션(PyTorch, vLLM 등)은 CUDA 런타임·드라이버 위에 올라간다.

설명:

- “CUDA를 안다”는 것이 반드시 CUDA C++를 매일 작성한다는 뜻은 아니다. 서빙 엔지니어에게는 **실행 단위·메모리 계층·병목 분류**를 읽는 능력이 더 자주 쓰인다.

### 3.2 실행이 펼쳐지는 방식 (개요)

```text
호스트(CPU) 코드
    │
    ├─ 커널 launch
    ▼
디바이스(GPU)
    ├─ Grid (여러 block)
    │     └─ Block (여러 thread)
    │           └─ Warp (보통 32 threads가 한 묶음으로 스케줄)
    └─ 각 thread가 인덱스로 자신의 일을 고름
```

LLM 관점: “한 번의 forward”는 사실 **수많은 커널의 연쇄**다. 엔진(제112강)은 이 연쇄를 배치·그래프·캐시로 재구성한다.

## 4. SM — Streaming Multiprocessor

### 4.1 용어

**SM(Streaming Multiprocessor)** 은 GPU 안의 **연산 공장 단위**다. 여러 SM이 칩에 모여 있고, 각 SM이 워프·블록을 받아 실행한다.

비유:

```text
GPU 칩 = 공장 단지
SM     = 공장 1동
Warp   = 동시에 같은 동작을 하는 작업반
```

### 4.2 SM이 가진 것 (개념 수준)

세대마다 세부 구성은 다르다. 공통적으로 이야기하는 요소:

| 요소 | 역할 (설명) |
|---|---|
| CUDA Core / 연산 유닛 | 일반 산술 |
| Tensor Core (지원 시) | 행렬·저정밀 가속 |
| Register file | 스레드별 초고속 임시 저장 |
| Shared memory / L1 | 블록 내 공유·캐시 |
| 스케줄러 | 워프를 이슈 |

사실 vs 설명:

- **사실**: SM 개수·코어 구성·Tensor Core 세대는 **칩/제품 스펙**에 따른다.
- **설명**: “SM이 많다 = 항상 서빙이 빠르다”는 성립하지 않는다. 메모리가 막히면 SM은 기다려야 한다.

### 4.3 Occupancy 직관

**Occupancy** 는 “SM이 동시에 얼마나 많은 워프를 붙들고 있는가”에 가까운 지표 계열이다(정확한 정의는 도구·문서에 따름).

설명:

- occupancy가 높다고 항상 빠르지 않다.
- 다만 occupancy가 극단적으로 낮으면, 메모리 latency를 숨길 워프가 부족해 **대기**가 드러날 수 있다.

서빙에서 더 자주 보는 신호는 occupancy 숫자 자체보다:

```text
GPU SM Active / utilization
메모리 대역폭 활용
커널 시간 분포 (profiler)
```

이다.

## 5. Warp — 스케줄의 기본 묶음

### 5.1 용어

**Warp** 는 CUDA에서 **함께 스케줄되는 스레드 묶음**이다. 전통적으로 **32 threads** 가 한 warp다(아키텍처 문서의 기본 가정; 세부 변형은 세대 문서를 본다).

```text
1 warp ≈ 32 threads
같은 instruction을 lockstep에 가깝게 진행 (분기 시 divergence)
```

### 5.2 Warp divergence

같은 warp 안에서 `if`로 서로 다른 경로를 타면, 일부 스레드는 쉬고 일부가 실행되는 식의 **분기 패널티**가 생길 수 있다.

LLM 커널 설계에서:

- attention mask, variable sequence length, MoE 라우팅 등은 divergence·부하 불균형을 유발할 수 있다.
- 서빙 엔진·커널이 이를 완화하는 방식이 성능 차이의 한 축이다(제112강 연결).

### 5.3 왜 서빙 사람이 warp를 알아야 하는가

직접 커널을 안 짜더라도:

1. “배치를 키우면 왜 효율이 오르는가” — SM·워프에 일이 채워짐
2. “너무 작은 decode batch” — 계산 밀도가 낮아지고 memory 대기가 드러남
3. “특수 샘플링·구조화 출력” — 커널 경로가 갈라져 그래프/캐시 이득이 줄 수 있음

## 6. 메모리 계층과 Bandwidth

### 6.1 계층 스케치

개념적 피라미드(구체 용량·속도는 칩마다 다름):

```text
빠름·작음
  Registers
  Shared memory / L1
  L2 cache
  HBM / device memory  (또는 통합 메모리 플랫폼의 DRAM)
  Host memory (PCIe/NVLink 등으로 연결)
느림·큼
```

**Memory Bandwidth(메모리 대역폭)** 는 단위 시간당 메모리에서 읽고 쓸 수 있는 **바이트 양**이다. 단위는 보통 GB/s 또는 TB/s.

### 6.2 HBM이란

**HBM(High Bandwidth Memory)** 은 GPU에 가까이 붙어 높은 대역폭을 제공하도록 설계된 메모리 기술이다. 많은 데이터센터 GPU가 HBM을 쓴다.

주의 (플랫폼 차이):

- 모든 “AI 박스”가 동일한 HBM 구성을 갖는 것은 아니다.
- 예를 들어 **DGX Spark / GB10** 계열은 공개 문서상 **통합(LPDDR 계열) 메모리**와 명시된 대역폭을 쓴다(제115강). “HBM”이라는 단어를 모든 장치에 습관적으로 붙이지 말 것.

이 강의에서 “HBM bandwidth”라고 부를 때는:

```text
(설명) “디바이스 메모리 대역폭”의 대표적 사례로 HBM을 말한다
(사실) 실제 제품의 메모리 종류·GB/s는 스펙 시트를 본다
```

### 6.3 용량(capacity) vs 대역폭(bandwidth)

| | Capacity | Bandwidth |
|---|---|---|
| 질문 | 몇 GB가 들어가나? | 초당 몇 GB를 움직이나? |
| OOM | 용량 초과 | (직접 원인은 아님) |
| 느린 decode | 용량이 남아 있어도 발생 가능 | 가중치·KV 읽기가 지배적일 때 |
| 대응 예 | 양자화, TP로 샤딩, 짧은 context | 양자화, 커널 융합, 배치, 캐시 재사용 |

제101강 KV Cache는 **용량** 압박의 대표다. 제103강 양자화는 **용량과 대역폭 모두**에 도움을 줄 수 있다(바이트↓ → 같은 대역폭으로 더 많은 “유효 파라미터”를 읽음 — 설명).

## 7. Compute-bound vs Memory-bound

### 7.1 정의 (직관)

| 구분 | 직관 | LLM에서 자주 보이는 장면 |
|---|---|---|
| **Compute-bound** | 연산기(ALU/Tensor Core)가 바쁘고, 메모리는 충분히 먹여 줌 | 큰 배치 prefill, 넓은 GEMM |
| **Memory-bound** | 연산기는 놀고, 데이터를 기다리느라 시간이 감 | 작은 배치 decode, 거대 가중치 스트리밍 |

### 7.2 산술강도 (Arithmetic Intensity)

간단한 정의:

$$

\text{Arithmetic Intensity} \approx \frac{\text{연산 횟수 (FLOPs)}}{\text{이동한 바이트 (Bytes)}}

$$

- intensity가 높으면 → compute 쪽으로 기울기 쉽다.
- intensity가 낮으면 → memory 쪽으로 기울기 쉽다.

### 7.3 Decode가 memory에 기울기 쉬운 이유 (스케치)

한 토큰 decode에서(단순화):

```text
해야 할 일:
  - 거의 모든 가중치를 한 번씩 읽음 (배치=1이면 재사용 적음)
  - KV cache에서 과거 키·값을 읽음
  - 상대적으로 “작은” 출력 로짓 계산
```

배치를 $B$로 키우면 가중치 읽기를 $B$개 요청이 공유하므로, **바이트당 연산**이 올라갈 수 있다. 이것이 continuous batching(제102강)이 처리량에 중요한 이유 중 하나다.

사실: 실제 intensity는 커널 구현·융합·캐시·양자화에 크게 의존한다.  
설명: 위 스케치는 **왜 배치·양자화·캐시가 성능 레버인지**를 위한 지도다.

## 8. Roofline — 가볍게

### 8.1 그림으로 보는 천장

Roofline 모델은 성능을 두 천장으로 본다.

```text
성능 (FLOP/s)
    ^
    |          /¯¯¯¯¯¯¯¯¯¯ compute roof (Peak FLOP/s)
    |         /
    |        /
    |       /
    |______/  memory roof = Peak Bandwidth × Intensity
    +-----------------------------> Arithmetic Intensity
```

- 왼쪽(낮은 intensity): **memory roof**에 걸림
- 오른쪽(높은 intensity): **compute roof**에 걸림

### 8.2 서빙에서의 사용법

Roofline을 “정확한 숫자 예측기”로 쓰지 않는다. 대신:

1. 지금 구간이 왼쪽인가 오른쪽인가?
2. 최적화 A(양자화)는 intensity를 올리는가, 바이트를 줄이는가?
3. 최적화 B(TP)는 계산을 나누지만 통신을 추가하는가?

제117강 GPU 최적화 실험에서 이 질문을 측정 템플릿과 연결한다.

### 8.3 숫자 약속

이 책은 **임의로 Peak TFLOPS·실측 tok/s를 지어내지 않는다**.

읽는 법:

1. 벤더 스펙에서 Peak FLOP/s, Memory Bandwidth를 확인한다.
2. profiler로 커널 시간·메모리 처리량을 본다.
3. “이론 상한”과 “실측” 사이를 **효율**로 해석한다.

스펙의 Peak는 **이상적 조건의 상한**인 경우가 많다(정밀도, sparsity, 문제 크기 전제). 각주를 읽는다.

## 9. LLM 연산과 GPU 매핑

### 9.1 주요 연산 블록

| 블록 | 대략적 성격 (설명) | 메모리 관점 |
|---|---|---|
| QKV / O projection, MLP GEMM | 큰 행렬곱 | 가중치 읽기 + activation |
| Attention score / context | seq 길이에 민감 | QKV·KV cache 트래픽 |
| Softmax / norm / act | 상대적으로 가벼울 수 있음 | 커널 launch·대역폭 낭비 주의 |
| Sampling | 보통 작음 | 그래프 깨짐·동기화에 민감할 수 있음 |

### 9.2 Prefill vs Decode 다시 보기

```text
Prefill
  입력 토큰 많음 → 행렬이 “두꺼워짐” → compute에 가까워질 여지
  TTFT에 직접 영향 (제107강)

Decode
  토큰 1개씩 → 행렬이 “얇아짐” → memory에 가까워질 여지
  TPOT에 직접 영향
```

스펙ulative decoding(제110강)은 “검증·초안”으로 **유효 토큰당 메모리 트래픽**을 바꾸려는 시도로도 읽을 수 있다(성공 여부는 수용률·오버헤드에 달림).

## 10. 대역폭 예산 — 사고 실험 (숫자 예시)

**예시(설명용 가정, 특정 제품 실측 아님):**

가정:

- 모델 가중치 읽기량 $W$ 바이트/토큰(배치 1, 단순화)
- 디바이스 메모리 대역폭 상한 $BW$ 바이트/초(스펙)
- 다른 오버헤드 무시

거친 상한:

$$

\text{토큰/초} \lesssim \frac{BW}{W}

$$

의미:

- $W$를 줄이면(양자화, 불필요 재읽기 감소) 상한이 올라갈 **수 있다**.
- $BW$를 키우면(더 빠른 메모리 subsystem) 상한이 올라갈 **수 있다**.
- 실제는 커널 비효율·동기화·샘플러·스케줄러가 공간을 먹는다.

이 식을 **측정값처럼 인용하지 말 것**. “병목 후보를 좁히는 사고 도구”다.

## 11. 코드로 감각 잡기 (스케치)

아래는 **대역폭 병목을 흉내 내는 사고 코드**다. 벤치마크 결과가 아니다.

```python
# sketch: 바이트와 연산의 비율을 말해 보는 장난감
# 실제 GPU 커널·성능과 무관한 설명용 스케치

def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved

# 예: 배치1 decode에서 거대 가중치를 읽고 상대적으로 적은 FLOP
w_bytes = 14e9          # 설명용 가정: 14GB 가중치를 읽는다
flops = 7e9             # 설명용 가정
print("intensity ~", arithmetic_intensity(flops, w_bytes))

# 예: 배치를 키워 가중치 재사용
batch = 16
print("intensity ~", arithmetic_intensity(flops * batch, w_bytes))
```

해석 포인트:

- 배치가 커지면 intensity가 올라갈 **수 있다**(가정이 맞을 때).
- 반대로 KV가 길어지면 읽기 바이트가 늘어 intensity가 내려갈 **수 있다**.

## 12. nvidia-smi · 프로파일러로 무엇을 볼까

### 12.1 운영 중 자주 보는 신호

| 신호 | 가능한 해석 (설명) |
|---|---|
| 메모리 사용량 ≈ 한계 | capacity 압박, OOM 위험 |
| utilization 높은데 TPOT 나쁨 | 바쁜데 비효율, 또는 통신 대기 포함 |
| utilization 낮은데 TPOT 나쁨 | 입력 부족, 스케줄 공백, 동기화, CPU 병목 |
| SM 대비 메모리 throughput 높음 | memory-bound 후보 |

단정하지 말 것. **가설 → 측정 → 재현** 순서를 지킨다(제118강 리포트).

### 12.2 벤더 문서 읽기 체크리스트

1. 제품 페이지와 **User Guide / Data Sheet**를 구분한다.
2. Peak FLOPS의 **정밀도(FP4/FP8/BF16…)** 와 sparsity 전제를 읽는다.
3. Memory bandwidth 숫자의 **메모리 종류**를 확인한다.
4. TDP·쿨링·전원 제한이 성능을 깎는지 확인한다.
5. 날짜·버전을 기록한다. SKU는 개정된다.

제115강에서 DGX Spark 공개 스펙을 이 방법으로 읽는다.

## 13. 실습

### 실습 A — 병목 분류 연습

다음 시나리오를 compute / memory / capacity / 통신 중 **주 후보**로 분류하고 이유를 한 줄로 쓰시오.

1. 배치 1, 긴 생성, GPU 메모리는 50%만 사용, TPOT가 목표 미달
2. 컨텍스트를 키우자마자 OOM
3. 단일 GPU에서는 괜찮다가 TP=2로 나누니 토큰/초가 기대보다 낮음(제113~114강 복선)
4. Prefill TTFT만 길고, 이후 TPOT는 수용 가능

### 실습 B — 스펙 시트 독서

사용 가능한 GPU(또는 DGX Spark 문서)에서 다음을 표로 옮기시오.

| 항목 | 값 | 출처 URL·날짜 |
|---|---|---|
| 메모리 종류·용량 | | |
| 메모리 대역폭 | | |
| Peak 연산 (정밀도별로) | | |
| 비고(sparsity 등) | | |

값을 외우지 말고 **출처를 남긴다**.

## 14. 자주 하는 실수

1. **Utilization = 성능** 으로 단정한다.
2. **OOM만 메모리 문제**로 보고, bandwidth 병목을 놓친다.
3. 블로그의 tok/s를 자기 환경 수치로 인용한다.
4. Prefill 최적화와 Decode 최적화를 같은 레버로 취급한다.
5. “더 많은 SM / 더 큰 칩”이면 서빙이 무조건 나아진다고 본다.
6. 통합 메모리 플랫폼에 HBM 가정을 그대로 이식한다.

## 15. 핵심 정리

- CUDA 실행은 Grid → Block → Warp → Thread로 펼쳐지고, **SM**이 그 공장이다.
- LLM decode는 종종 **memory-bound** 쪽에 기울며, 배치·양자화·캐시가 산술강도를 바꾼다.
- **Capacity**와 **Bandwidth**를 분리해 생각해야 KV·양자화·TP 결정을 덜 헷갈린다.
- Roofline은 예측기가 아니라 **병목 방향 나침반**이다.
- 성능 숫자는 발명하지 말고, 스펙·프로파일·재현 가능한 측정으로 말한다.

## 16. 핵심 용어

| 용어 | 의미 |
|---|---|
| CUDA | NVIDIA GPU 병렬 컴퓨팅 플랫폼·모델 |
| SM | Streaming Multiprocessor, GPU 내 연산 단위 |
| Warp | 함께 스케줄되는 스레드 묶음(보통 32) |
| Kernel | GPU에서 실행되는 함수 |
| HBM | 고대역폭 메모리 기술(제품에 따라 사용) |
| Memory Bandwidth | 초당 이동 가능한 메모리 바이트 |
| Arithmetic Intensity | FLOPs / Bytes |
| Compute-bound | 연산 천장이 지배 |
| Memory-bound | 대역폭 천장이 지배 |
| Roofline | intensity에 따른 성능 상한 모델 |

## 17. 연습 문제
### 문제 1 (구분)

다음 중 **용량(capacity)** 문제에 더 가까운 것은?  
(a) KV cache가 늘어 OOM  
(b) 배치 1 decode에서 가중치 읽기가 시간을 지배

### 문제 2 (개념)

Warp divergence가 성능에 영향을 줄 수 있는 이유를 한 문장으로 쓰시오.

### 문제 3 (직관)

Continuous batching이 decode의 산술강도를 올릴 수 있는 메커니즘을 설명하시오.

### 문제 4 (사실/설명)

“이 GPU는 1 PFLOP이므로 어떤 모델이든 초당 N 토큰이다” 문장의 문제점을 두 가지 쓰시오.

### 문제 5 (연결)

제103강 양자화가 memory-bound decode에 도움이 될 **수 있는** 이유를 대역폭 관점에서 쓰시오.

### 문제 6 (설계)

TTFT만 나쁠 때와 TPOT만 나쁠 때, 각각 우선 의볼 GPU/알고리즘 레버를 하나씩 제시하시오.

---

## 정답 및 해설

### 문제 1

(a). OOM은 용량. (b)는 대역폭·memory-bound 후보.

### 문제 2

한 warp 안 스레드가 서로 다른 제어 경로를 타면, 일부는 대기하고 일부가 실행되어 유효 연산 효율이 떨어질 수 있다.

### 문제 3

여러 요청이 같은 가중치 로드를 공유하면, 이동 바이트당 처리하는 토큰·연산이 늘어 intensity가 올라갈 수 있다.

### 문제 4

예: (1) Peak FLOP은 특정 정밀도·조건의 상한이며 실측 토큰속도와 직접 등치되지 않음 (2) decode는 memory-bound일 수 있어 FLOP만으로 결정되지 않음.

### 문제 5

가중치·활성화 바이트가 줄면 같은 대역폭으로 더 많은 유효 계산을 밀어 넣을 여지가 생긴다(품질·커널 지원은 별개).

### 문제 6

예: TTFT → prefill 커널·최대 배치 토큰·프롬프트 길이 / TPOT → 배치 효율·양자화·KV 트래픽·샘플러 오버헤드. (환경에 따라 다름)

## 18. 다음 강의와 연결

GPU의 천장과 병목 종류를 보았다. 다음 질문은 **그 위에서 도는 소프트웨어 엔진을 무엇을 고를까**다.

이전 강의: **제110강. MTP와 Speculative Decoding**  
다음 강의: **제112강. Inference Engine 비교 — vLLM · TensorRT-LLM · SGLang**

엔진마다 PagedAttention, CUDA Graph, Radix Cache 같은 무기가 다르다. “항상 승자”는 없고, 워크로드·팀·배포 제약에 맞는 선택을 제112강에서 비교한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [110강. MTP와 Speculative Decoding](110강_MTP와_Speculative_Decoding.md)
- **다음 강:** [112강. Inference Engine 비교 — vLLM · TensorRT-LLM · SGLang](112강_Inference_Engine_비교_vLLM_TensorRT_LLM_SGLang.md)

<!-- /LECTURE_NAV -->
