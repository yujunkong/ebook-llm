# 114강. NCCL과 RoCE
## 이번 강에서 배우는 내용

- NCCL이 무엇인지, 어떤 collective 를 LLM 서빙 TP가 쓰는지 개요로 말한다.
- 단일 노드(NVLink 등)와 멀티 노드(이더넷/RoCE/InfiniBand)에서 통신 경로가 어떻게 달라지는지 구분한다.
- RoCE(RDMA over Converged Ethernet)의 위치를 개요 수준으로 설명한다.
- NCCL 디버깅에 쓰는 환경변수·체크 순서를 처방전 남발 없이 정리한다.
- 제115강 2× DGX Spark의 ConnectX·QSFP 링크를 이해할 준비를 마친다.

## 왜 중요한가?
현장의 대표 증상:

```text
“TP=2로 켰더니 한 GPU만 100%이고 멈춘다”
“단일 노드보다 듀얼 노드가 더 느리다”
“대역폭 스펙은 200GbE인데 체감이 아니다”
```

이 증상들의 상당수는 모델 코드가 아니라:

1. 잘못된 NIC 선택 (느린 관리용 10GbE로 NCCL이 흐름)  
2. RDMA/RoCE 경로 미사용 (TCP 폴백)  
3. MTU·케이블·포트 매핑 실수  
4. 방화벽·권한·컨테이너 네트워크 네임스페이스  

에 있다. 제113강의 알고리즘 비용을 **물리·라이브러리 비용**으로 번역하는 강의가 이번이다.

## 선수 개념
1. Tensor Parallel과 collectives — 제113강  
2. Memory vs compute vs communication bound — 제111강  
3. 서빙 엔진의 distributed backend — 제112강  
4. (다음) DGX Spark ConnectX-7 — 제115강  

## NCCL이란
### 3.1 용어

| 영어 | 한국어 | 쉬운 정의 | 왜 필요한가 |
|---|---|---|---|
| **NCCL** | 엔씨씨엘 | NVIDIA Collective Communications Library. 멀티 GPU/노드 집합 통신 라이브러리 | TP/PP 등에서 all-reduce 등을 고속으로 수행 |
| **Rank** | 랭크 | 통신 그룹 안의 프로세스/GPU 번호 | 누가 어떤 샤드를 도는지 식별 |
| **Communicator** | 커뮤니케이터 | 참여 순위들의 통신 세계 | 월드·TP 그룹 등 |

사실: NCCL은 NVIDIA가 제공하는 집합 통신 라이브러리다. PyTorch Distributed, 많은 LLM 엔진이 이를 호출한다.

설명: NCCL을 직접 API로 부르지 않아도, `tensor-parallel-size>1` 이면 **경로 어딘가에서 NCCL이 도는 경우가 많다**.

### 3.2 하는 일 / 안 하는 일

```text
NCCL이 하는 일:
  GPU 메모리 상의 텐서를 참여 장치들과 collective로 주고받음

NCCL이 안 하는 일:
  모델 샤딩 전략 결정 (엔진·프레임워크 몫)
  물리 케이블을 마법처럼 빠르게 만듦 (하위 네트워킹에 의존)
```

## 주요 Collectives 개요
| Collective | 직관 | TP에서 자주 보이는 역할 |
|---|---|---|
| **All-reduce** | 모두 기여 → 환원(예:합) → 모두 동일 결과 | 부분 합 결합 |
| **All-gather** | 각자 조각 → 모두 전체 확보 | 샤드 이어 붙이기 |
| **Reduce-scatter** | 환원과 분산을 결합 | 일부 최적 패턴 |
| **Broadcast** | 하나 → 모두 | 초기화·메타데이터 |
| **Send/Recv** | 점대점 | PP 경계 |

그림 (all-reduce):

```text
GPU0: a0      GPU1: a1
   \          /
    \        /
     combine (sum)
    /        \
GPU0: a0+a1  GPU1: a0+a1
```

교육용 경고: 알고리즘(ring, tree 등)은 NCCL·토폴로지가 고른다. 사용자가 매 호출마다 고르는 경우는 드물다.

## 토폴로지 — 데이터가 지나가는 길
### 5.1 단일 노드 멀티 GPU

가능한 경로(플랫폼 의존):

```text
GPU ↔ GPU  (NVLink / NVSwitch 등)
GPU ↔ CPU ↔ GPU  (상대적으로 비선호될 수 있음)
```

사실: 구체 링크 종류·대역폭은 서버 모델마다 다르다.  
설명: “같은 노드”라도 토폴로지가 대칭이 아닐 수 있다.

### 5.2 멀티 노드

```text
GPU → (로컬 인터커넥트) → NIC → 네트워크 → NIC → GPU
```

여기서 네트워크가:

- **InfiniBand**
- **RoCE** (이더넷 위에서 RDMA)
- 일반 **TCP/IP 이더넷**

중 무엇이냐에 따라 latency·CPU 개입·튜닝 포인트가 달라진다.

## RoCE 개요
### 6.1 용어

**RoCE (RDMA over Converged Ethernet)** 는 이더넷 네트워크에서 **RDMA(Remote Direct Memory Access)** 를 쓰도록 하는 기술이다.

| 용어 | 쉬운 정의 |
|---|---|
| **RDMA** | CPU를 거의 거치지 않고 메모리 영역을 원격으로 읽고 쓰는 접근 방식 |
| **RoCE** | 그 RDMA를 이더넷(Converged Ethernet) 위에서 운반 |
| **IB verbs** | RDMA를 다루는 API·경로의 한 계열(명칭은 환경에 따름) |

### 6.2 왜 LLM 멀티노드에 나오나

TP의 all-reduce는 **작고 잦은** 동기화에 민감할 수 있다.  
RDMA/RoCE 경로는 (올바르게 구성되면) TCP 소켓 경로보다 **지연·CPU 오버헤드** 면에서 유리한 **후보**가 된다.

설명: “RoCE를 꽂으면 무조건 빠르다”가 아니다. 설정이 잘못되면 NCCL이 **TCP로 폴백**하거나, 링크는 살아 있어도 collective 효율이 낮을 수 있다.

### 6.3 RoCEv1 / v2 메모

버전·라우팅 차이는 네트워크 문서에 맡긴다. 서빙 엔지니어가 먼저 할 일:

```text
1) 케이블·포트가 의도한 NIC에 올라왔는가
2) RDMA 디바이스 이름이 보이는가
3) NCCL이 그 디바이스를 쓰는가
4) 단순 대역폭 테스트와 NCCL 테스트가 둘 다 통과하는가
```

## NCCL이 고르는 경로
개념적 우선순위(단순화·설명용):

```text
가능하면: GPU Direct / 빠른 로컬 링크 / RDMA(RoCE·IB)
차선:     공유 메모리, TCP 등
```

사용자가 개입하는 지점 예:

| 환경변수 (대표) | 의도 |
|---|---|
| `NCCL_DEBUG` | 로그로 실제 경로·문제 확인 |
| `NCCL_SOCKET_IFNAME` | TCP/소켓용 인터페이스 선택 |
| `NCCL_IB_HCA` | 사용할 RDMA HCA 지정 |
| `NCCL_IB_DISABLE` | IB/RoCE 경로 on/off (잘못 끄면 TCP) |

사실: 변수 이름·기본값은 NCCL 버전에 따라 다를 수 있다. **공식 NCCL 문서**를 기준으로 한다.  
설명: 블로그에 복사한 export 묶음을 이해 없이 붙이면, 다른 머신의 인터페이스 이름 때문에 더 느려질 수 있다.

## “스펙 대역폭” vs “NCCL 실효”
제115강 DGX Spark 공개 자료에는 ConnectX-7과 QSFP, 최대 200Gb/s급 링크 이야기가 나온다. 그와 별개로 기억할 층위:

```text
1) 물리 링크 속도 (예: 200GbE급)
2) 점대점 RDMA 측정
3) NCCL collective 실효 대역폭
4) 실제 vLLM/엔진 토큰 속도
```

설명: 상위 층으로 갈수록 **프로토콜·동기화·메시지 패턴** 때문에 숫자가 달라질 수 있다.  
이 책은 타인의 블로그 실측치를 **정답 스펙처럼 재인용하지 않는다**. 자신의 환경에서 1→4를 재라는 절차만 강조한다.

## 디버깅 순서 (실무 체크리스트)
멀티노드 TP가 이상할 때:

### 9.1 L0 — 물리·링크

- [ ] 케이블이 의도한 포트에 꽂혔는가  
- [ ] `ip link` / 벤더 도구로 링크 Up인가  
- [ ] 관리용 NIC와 고속 NIC를 혼동하지 않았는가  

### 9.2 L1 — 주소·SSH

- [ ] 고속 인터페이스에 IP가 있는가  
- [ ] 노드 간 ping/SSH가 **그 IP**로 되는가  
- [ ] 호스트명 해석이 잘못된 NIC로 가지 않는가  

### 9.3 L2 — RDMA 디바이스

- [ ] RDMA 디바이스 목록이 보이는가 (`ibv_devinfo` 등, 환경에 따라)  
- [ ] 컨테이너라면 RDMA·디바이스 마운트 권한이 있는가  

### 9.4 L3 — NCCL

- [ ] `NCCL_DEBUG=INFO` 로 사용 중 수송 경로 확인  
- [ ] HCA/IFNAME이 고속 장치를 가리키는가  
- [ ] `nccl-tests`(또는 배포판 제공 테스트)로 all-reduce 확인  

### 9.5 L4 — 엔진

- [ ] TP size, 노드 수, executor backend 일치  
- [ ] 양쪽 노드의 모델·이미지·CUDA/NCCL 버전 정합  
- [ ] hang 시 한 쪽만 busy인지 확인  

이 순서를 건너뛰고 모델 파라미터부터 바꾸지 말 것.

## 코드·명령 스케치
### 10.1 PyTorch가 NCCL을 쓰는지 확인 (스케치)

```python
import torch

print("cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
# 분산 초기화는 런처(torchrun 등)와 함께 사용하는 것이 일반적
# backend="nccl" 은 GPU 집단 통신의 대표 선택
```

### 10.2 환경변수 기록 템플릿

측정·장애 티켓에 남길 최소 세트:

```bash
# 값과 함께 날짜·호스트·인터페이스 이름을 기록할 것
env | grep -E 'NCCL|CUDA_VISIBLE|GLOO|UCX' | sort
```

### 10.3 절대 하지 말 것

- 출처 불명의 “최적 NCCL 매직 넘버”를 프로덕션에 즉시 적용  
- `NCCL_IB_DISABLE=1` 을 성능 특효약으로 오해 (대개 RDMA 끄기)  
- 한 번 성공한 export를 인터페이스 이름이 다른 머신에 무비판 복제  

## 보안·운영 메모
1. 고속 클러스터망과 사무실 LAN을 **평면으로 섞지** 않는 설계가 안전하다.  
2. RoCE 망의 혼잡 제어·PFC 설정은 네트워크 담당과 함께한다. 잘못되면 pause storm 등(용어만 인지) 이슈가 될 수 있다.  
3. 컨테이너 오케스트레이션에서 **호스트 네트워크 vs CNI** 선택이 NCCL에 큰 영향을 준다.  
4. 버전 핀: CUDA, NCCL, 드라이버, 엔진 이미지를 세트로 기록한다(제118~119강).

## 실습
### 실습 A — 계층 라벨링

다음 문장을 물리 / NCCL / 엔진 중 어디에 가까운지 분류하시오.

1. QSFP 케이블이 빠져 링크 Down  
2. all-reduce가 TCP로 폴백했다는 로그  
3. `tensor-parallel-size` 와 실제 GPU 수 불일치  

### 실습 B — 질문지 작성

동료가 “200GbE인데 느려요”라고 한다. 되물을 질문 10개를 작성하시오. (숫자 단정 없이)

## 자주 하는 실수
1. 10GbE 관리 포트로 NCCL 소켓을 흘린다.  
2. 점대점 iperf만 보고 collective·엔진 성능을 보장됐다고 본다.  
3. 한 노드 성공 경험을 멀티노드에 그대로 이식한다.  
4. NCCL 로그를 안 켠 채 애플리케이션만 재시작한다.  
5. RoCE·IB·TCP를 동의어로 말한다.

## 수식 보강 — AllReduce 부피

데이터 크기 $B_{\mathrm{bytes}}$, 참여 GPU $N$이면 ring allreduce 통신량은 대략

$$
\sim 2\frac{N-1}{N}B_{\mathrm{bytes}}
$$

차수입니다. 대역폭·지연이 TP 스케일을 제한합니다.


<!-- enrich-block-114 -->
## 통신 비용 스케치

올리듀스 볼륨(대략):

$$
V_{\mathrm{allreduce}}\approx 2\cdot\frac{P-1}{P}\cdot |g|
$$

대역폭 한계 시간:

$$
t_{\mathrm{comm}}\approx\frac{V}{B_{\mathrm{eff}}}
$$

RoCE/NCCL에서 $B_{\mathrm{eff}}$는 이론 PCIe/IB보다 낮게 잡는 것이 안전합니다.

### 텐서 병렬 통신

$$
t_{\mathrm{step}}\approx t_{\mathrm{compute}}+t_{\mathrm{allreduce}}+t_{\mathrm{sync}}
$$


<!-- enrich-extra-114 -->
## 실습 — all-reduce 시간 추정

```python
def t_allreduce_sec(bytes_msg: float, gbps_eff: float, P: int) -> float:
    # 대략 ring: 2*(P-1)/P * size / bandwidth
    vol = 2 * (P - 1) / P * bytes_msg
    return vol / (gbps_eff * 1e9 / 8)

print("ms", 1e3 * t_allreduce_sec(2e8, gbps_eff=50, P=8))
```

$$
t\approx \frac{2(P-1)}{P}\cdot\frac{|g|}{B_{\mathrm{eff}}}
$$

## All-reduce 부피·시간 스케치
메시지（환원할 텐서）바이트를 $M$, 참여 GPU 수를 $P$, 실효 대역폭을 $B_{\mathrm{eff}}$라 하자.  
Ring all-reduce의 교육용 근사（상수·구현 세부 무시）:

$$

T_{\mathrm{AR}} \;\gtrsim\;
\frac{2(P-1)}{P}\cdot\frac{M}{B_{\mathrm{eff}}}
\;+\;
T_{\mathrm{latency}}

$$

해석:

- $M$↑ → 통신 시간↑（큰 activation/부분합）
- $B_{\mathrm{eff}}$↓（잘못된 NIC, TCP 폴백）→ 같은 $M$도 느림
- Decode처럼 **자주·작은** collective는 latency 항이 두드러질 수 있음

**사실:** 실제 NCCL은 알고리즘·토폴로지를 고른다. 위 식은 직관용.  
**비주장:** 특정 GB/s 실측치.

### 링크 스펙 vs 실효

$$

B_{\mathrm{eff}} \le B_{\mathrm{link}}
$$

등호는 거의 성립하지 않는다. 프로토콜·동기화·메시지 크기·혼잡이 깎는다.

```text
B_link (예: 200GbE급 상한)
  → RDMA 점대점
    → NCCL collective 실효
      → 엔진 토큰 속도
```

층마다 따로 측정한다（제115·118강）.

### TP와 통신 빈도（정성）

레이어마다 all-reduce/all-gather가 붙으면, 출력 토큰 하나당

$$

T_{\mathrm{token}} \approx T_{\mathrm{compute}} + \sum_{\ell} T_{\mathrm{collective}}^{(\ell)}
$$

잘못된 수송 경로는 $\sum T_{\mathrm{collective}}$를 키워 TPOT를 망가뜨린다.


<!-- enrich-batch4-114 -->
## NCCL 집단 통신 패턴

| 패턴 | 학습/추론 |
|---|---|
| all-reduce | DP grad |
| all-gather | TP |
| reduce-scatter | FSDP류 |

```python
def ring_volume_bytes(numel, dtype_bytes, P):
    # ring all-reduce 대략량
    return 2*(P-1)/P * numel * dtype_bytes
print(ring_volume_bytes(1e9, 2, 8)/1e9, "GB")
```

### RoCE 체크

$$
B_{\mathrm{eff}}=B_{\mathrm{link}}\cdot \eta_{\mathrm{nic}}\cdot \eta_{\mathrm{pcIe}}
$$

케이블·스위치·PFC/ECN 설정이 $\eta$를 좌우합니다.

## LLM에서는 어디에 사용될까?
- `tensor-parallel-size>1` 서빙의 침묵 hang
- 듀얼 노드가 단일보다 느린 역설
- 컨테이너에서 RDMA 디바이스 누락
- 온콜 티켓에 `NCCL_DEBUG` 로그 첨부 규율

## 실습 C — 부피 사고실험
$M$이 2배가 되면 ring 근사에서 통신 시간이 대략 어떻게 되는지 쓰시오（$B_{\mathrm{eff}}$ 고정）.

## 실습 D — 폴백 탐지
TCP 폴백을 의심할 때 확인할 로그·장치 증거 세 가지를 쓰시오.


## 작은 숫자로 보는 부피（가정）
교육용: $M=64\,\mathrm{MiB}$, $P=2$, $B_{\mathrm{eff}}=10\,\mathrm{GB/s}$（가정값 — 실측 아님）.

Ring 근사에서 $\frac{2(P-1)}{P}=1$ 이므로

$$

T_{\mathrm{AR}} \gtrsim \frac{64\times 2^{20}}{10\times 10^9}\,\mathrm{s}
\approx 6.7\,\mathrm{ms}
$$

여기에 latency·커널 오버헤드가 더해진다. Decode 토큰마다 이 항이 여러 층 쌓이면 TPOT에 가시화된다.  
**같은 $M$이라도** $B_{\mathrm{eff}}$가 TCP 폴백으로 1/10이 되면 시간이 대략 10배로 늘어날 **여지**가 있다.

### 체크리스트에 붙일 한 줄
```text
측정 전: NCCL이 고른 장치 이름을 티켓에 붙여라.
측정 후: B_link / RDMA / NCCL / tok/s 를 층별로 기록하라.
```

## GPU Direct·NUMA 메모（개요）
데이터가 CPU를 우회할수록 latency에 유리한 **후보**가 된다. NUMA 노드와 NIC 친화도가 어긋나면 같은 케이블도 느릴 수 있다. 플랫폼 문서를 보고, 추측으로 `export`를 복사하지 말 것.


## 핵심 요약
- NCCL은 멀티 GPU 집합 통신의 사실상 표준 경로다.
- TP의 비용은 수식만이 아니라 **NCCL이 고른 수송 경로**에서 결정된다.
- RoCE는 이더넷 위 RDMA로, 멀티노드 LLM 통신의 중요 후보 기술이다.
- 스펙 링크 속도 ≠ NCCL 실효 ≠ 토큰 속도. 층위별로 측정한다.
- 다음 강의에서 이 지식을 **2× DGX Spark** 에 적용한다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| NCCL | NVIDIA 집합 통신 라이브러리 |
| Collective | 다수 순위가 참여하는 통신 연산 |
| All-reduce | 환원 후 결과 공유 |
| RDMA | 원격 직접 메모리 접근 |
| RoCE | 이더넷 위 RDMA |
| HCA | Host Channel Adapter (RDMA NIC 쪽) |
| TCP fallback | RDMA 경로 실패·비활성 시 소켓 경로 |

## 연습문제
### 문제 1 (개념)

NCCL과 Tensor Parallel의 관계를 한 문장으로 쓰시오.

### 문제 2 (구분)

RoCE와 “일반 TCP 이더넷으로 all-reduce”의 차이를 사용자 관점 한 줄로 쓰시오.

### 문제 3 (디버깅)

TP hang 시 `NCCL_DEBUG=INFO`를 켜는 이유를 쓰시오.

### 문제 4 (설계)

왜 점대점 대역폭 테스트만으로 서빙 TP 준비를 끝낼 수 없는가?

### 문제 5 (연결)

제113강에서 “decode에서 통신 latency가 드러난다”는 말이 NCCL 토폴로지 선택과 어떻게 연결되는가?

### 문제 6 (사실/설명)

제품 페이지의 “200Gb/s”를 곧바로 “우리 TPOT가 X ms”로 환산하면 안 되는 이유를 두 가지 쓰시오.

---

## 정답 및 해설
### 문제 1

TP가 층 내부에서 필요로 하는 all-reduce/all-gather 등을 NCCL이 (주로) 수행한다.

### 문제 2

RoCE는 RDMA 경로를 목표로 하고, TCP 경로는 소켓·CPU 개입이 커질 수 있는 다른 수송이다. (구성에 따라 실측은 달라짐)

### 문제 3

실제 사용 중인 인터페이스·수송·에러를 로그로 드러내 물리/설정 문제를 애플리케이션 문제와 분리하기 위함.

### 문제 4

서빙 TP는 collective 패턴·메시지 크기·동기화 빈도·엔진 오버헤드가 점대점 bulk 전송과 다르기 때문이다.

### 문제 5

토큰마다 다층 collective가 쌓이므로, latency가 큰 수송(잘못된 NIC/TCP 폴백)은 TPOT에 직접 가산된다.

### 문제 6

예: (1) 링크 속도는 상한·조건이 있음 (2) NCCL·엔진 효율이 별개 층 (3) 워크로드·정밀도 미반영. 두 가지 이상.

## 다음 강의와 연결
이론과 체크리스트를 **책상 위 2대 클러스터**에 옮긴다.

이전 강의: **제113강. Tensor Parallel**  
다음 강의: **제115강. 2× DGX Spark 환경 구성**

제115강에서는 GB10 / DGX Spark의 공개 하드웨어 맥락, QSFP·ConnectX-7 토폴로지, 듀얼 노드에서 멀티 GPU(노드) 서빙을 올리는 서사를 다룬다.

<!-- enrich-114-depth -->
## 집합통신 시간과 대역폭 추정

메시지 크기 $S$바이트, 실효 대역폭 $B_{\mathrm{eff}}$일 때 단순 모형:

$$
T_{\mathrm{comm}}
\approx
T_0 + \frac{S}{B_{\mathrm{eff}}}
$$

AllReduce（링, 대략）:

$$
T_{\mathrm{AR}}
\approx
2\frac{n-1}{n}\Big(T_0+\frac{S}{B_{\mathrm{eff}}}\Big)
$$

텐서병렬 한 층의 통신량이 커지면 decode 토큰 시간이

$$
t_{\mathrm{tok}}
\approx
t_{\mathrm{compute}}+t_{\mathrm{comm}}
$$

로 분해된다. $t_{\mathrm{comm}}$이 지배하면 GPU FLOPs를 더 사도 안 빨라진다.

### RoCE 체크（정성）

- PFC/ECN·손실 설정이 맞는가
- nic·numa 배치가 크로스 트래픽을 키우는가
- NCCL 알고요/버전이 벤치와 동일한가

$$
B_{\mathrm{eff}}=\frac{S}{T_{\mathrm{meas}}-T_0}
$$

로 실측 대역폭을 남겨 이론 링크 속도와 비교한다.


<!-- enrich-114-extra -->
## 한 줄 복습

링크 속도 ≠ $B_{\mathrm{eff}}$. 측정·토폴로지·메시지 크기를 같이 적는다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [113강. Tensor Parallel](113강_Tensor_Parallel.md)
- **다음 강:** [115강. 2× DGX Spark 환경 구성](115강_2x_DGX_Spark_환경_구성.md)

<!-- /LECTURE_NAV -->
