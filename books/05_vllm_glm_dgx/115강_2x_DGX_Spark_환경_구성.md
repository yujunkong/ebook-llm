# 115강. 2× DGX Spark 환경 구성
## 이번 강에서 배우는 내용

- GB10 Grace Blackwell Superchip 과 DGX Spark의 공개된 구조·스펙을 출처와 함께 읽는다.
- 단일 Spark와 2× Spark 토폴로지(특히 ConnectX-7 / QSFP)를 그림으로 설명한다.
- 네트워크·SSH·NCCL 검증 순서를 체크리스트로 수행 계획에 넣는다.
- 멀티 노드에서 서빙 엔진을 올릴 때의 서사(물리 → 망 → 통신 → 엔진)를 따른다.
- 불확실한 최신 세부 사양은 단정하지 않고 벤더 문서 확인법으로 대체한다.

## 왜 중요한가?
구 커리큘럼에서는 “GB10과 DGX Spark 구조”가 엔진 비교 자리(112강)에 있었다. 지금은 역할이 분리된다.

| 강 | 역할 |
|---|---|
| 제112강 | 소프트웨어 엔진 비교 |
| **제115강** | **하드웨어·랩 환경·2대 구성** |
| 제116강 | 그 위에서의 실제 서빙 프로젝트 |

이유: 하드웨어 스펙은 제품 주기로 바뀌고, 엔진 API도 바뀐다. **환경 구성 서사**를 한곳에 모아 두면 실습자가 링크를 따라가며 갱신할 수 있다.

## 선수 개념
1. CUDA / SM / bandwidth — 제111강  
2. Inference engine 선택 — 제112강  
3. Tensor Parallel — 제113강  
4. NCCL · RoCE — 제114강  

## DGX Spark와 GB10 — 공개 맥락
### 3.1 제품 위치 (설명)

**NVIDIA DGX Spark** 는 NVIDIA가 제시하는 **데스크탑형 개인 AI 슈퍼컴퓨터** 포지션의 제품이다. 클라우드 랙스터 전체가 아니라, 로컬에서 프로토타입·에이전트·중대형 모델 실험을 돌리기 위한 **완제품 플랫폼**에 가깝다.

**GB10** 은 그 안의 **Grace Blackwell Superchip**(SoC) 명칭으로 공개 자료에 등장한다. CPU(Grace 계열 Arm)와 Blackwell GPU가 **한 패키지/플랫폼**으로 묶인 형태다.

### 3.2 사실 표 — 공개 스펙에서 옮긴 항목

아래 값은 NVIDIA 제품 페이지 및 DGX Spark User Guide Hardware Overview 등 **공개 문서에 기재된 내용**을 교육용으로 요약한 것이다.  
**구매·배포 시점의 정확한 구성(스토리지 SKU 등)은 반드시 최신 문서를 재확인**할 것.

| 항목 | 공개 문서에 등장하는 내용 (요약) | 출처 유형 |
|---|---|---|
| 아키텍처 | NVIDIA Grace Blackwell | 제품 스펙 표 |
| GPU | Blackwell, 5th Gen Tensor Cores, 4th Gen RT Cores | 스펙 표 / User Guide |
| CPU | 20-core Arm (10× Cortex-X925 + 10× Cortex-A725) | 스펙 표 / User Guide |
| CUDA Cores | User Guide에 6,144 기재 | User Guide |
| 시스템 메모리 | 128 GB LPDDR5x **coherent unified** system memory | 스펙 표 |
| 메모리 인터페이스 | 256-bit | 스펙 표 |
| 메모리 대역폭 | **273 GB/s** | 스펙 표 / User Guide |
| 스토리지 | NVMe M.2 self-encryption, **1 TB 또는 4 TB** 등 SKU 차이 가능 | User Guide 주의 |
| 네트워크 | 10 GbE RJ-45, **ConnectX-7**, Wi-Fi 7, BT 5.4 | 스펙 / User Guide |
| ConnectX | QSFP 포트, 문서상 포트당 최대 **200 Gb/s**급 (케이블에 의존) | Clustering 문서 |
| 전원 | 240 W 어댑터, GB10 SoC TDP **140 W** 등 | User Guide |
| 크기·무게 | 약 150×150×50.5 mm, 1.2 kg | 스펙 표 |
| OS | NVIDIA DGX OS | 스펙 표 |
| 모델 규모(마케팅·가이드) | 단일에서 최대 ~200B, dual-Spark에서 더 큰 구성 언급 | User Guide 문장 — **조건·정밀도 전제 확인** |

### 3.3 성능 관련 문구를 읽는 법

공개 자료에는 예를 들어 다음류의 문구가 있다.

- “Up to 1 PFLOP FP4” (제품 페이지 스펙 표의 Tensor Performance 계열)
- User Guide의 “up to 1,000 TOPS … / up to 1 PFLOP at FP4 … with sparsity” 류 서술

**사실**: 벤더가 명시한 **상한·조건**이다.  
**설명**: 정밀도(FP4), sparsity 여부, 워크로드가 전제다. 이를 곧바로 “우리 모델의 tok/s”로 환산하지 말 것(제111강).

이 책은 위 상한을 **재측정한 것처럼 재인용하거나 임의 tok/s로 변환하지 않는다**.

### 3.4 통합 메모리 — HBM 가정 금지

제111강에서 경고한 대로, DGX Spark 공개 스펙의 메모리는 **LPDDR5x unified / coherent** 로 기술된다. 데이터센터 GPU의 HBM 대역폭 숫자를 습관적으로 대입하면 사고 실험이 틀린다.

실습 시 스스로 적을 문장:

```text
우리 장치의 메모리 종류: ________
대역폭(문서): ________
출처 URL·날짜: ________
```

## 단일 DGX Spark 논리 구조
```text
┌─────────────────────────────────────────────┐
│                 DGX Spark                   │
│  ┌───────────────────────────────────────┐  │
│  │           GB10 Superchip              │  │
│  │   Arm CPU  ↔  coherent memory  ↔ GPU  │  │
│  └───────────────────────────────────────┘  │
│           │                │                │
│        NVMe SSD      ConnectX-7 NIC         │
│                           │                 │
│              QSFP QSFP   10GbE   Wi-Fi       │
└─────────────────────────────────────────────┘
```

User Guide에 따르면 ConnectX-7은 외부 QSFP와 연결되고, SoC와는 PCIe 링크로 이어지는 식으로 기술된다. **세부 레인·펌웨어는 문서 개정**을 따른다.

교육 포인트:

1. **CPU–GPU 통합 메모리** 감각이 기존 “복사 많은 이산 GPU”와 다를 수 있다.  
2. 고속 클러스터링은 **10GbE가 아니라 QSFP/ConnectX-7 쪽**을 쓰는 것이 일반적 실습 경로다.  
3. 디스플레이·Wi-Fi는 개발 편의용이며 TP 데이터플레인으로 쓰지 말 것.

## 2× DGX Spark 토폴로지
### 5.1 목표

두 대를 묶어:

- 더 큰 모델을 **TP 또는 PP** 로 분할 서빙하거나  
- 분산 런타임(예: Ray 기반 executor)으로 한 논리 서버를 만들거나  
- (별도) 레플리카로 가용성을 실험  

하는 것이 실습 목표 후보이다.  
공개 User Guide는 dual-Spark에서 더 큰 파라미터 규모를 언급한다. **실제 가능 여부는 정밀도·KV·엔진·네트워크 실측**에 달렸으므로 마케팅 문장을 SLA로 쓰지 말 것.

### 5.2 물리 연결 (개념)

NVIDIA DGX Spark clustering 문서·플레이북의 요지:

```text
Spark A  QSFP  ←—— QSFP 케이블 ——→  QSFP  Spark B
```

요점:

- 각 장치 후면의 **QSFP(ConnectX-7) 포트**를 사용한다.  
- 포트당 최대 속도는 문서상 200 Gb/s급이나 **케이블·구성에 의존**.  
- QSFP 포트는 서로 교환 가능해 보이더라도, **공식 플레이북이 가정하는 좌/우 포트**를 따르는 편이 안전하다.  
- 한 물리 포트가 Linux에서 **여러 이더넷/RoCE 인터페이스 이름**으로 보일 수 있다(문서 설명). 이름 규칙을 로그에 남긴다.

```text
잘못된 기본값:  양쪽 10GbE RJ-45만 연결하고 TP=2
올바른 기본 습관: QSFP로 데이터플레인, 10GbE는 관리/인터넷
```

### 5.3 논리 네트워크

플레이북류에서 흔히 하는 일:

1. 고속 인터페이스에 주소 부여 (link-local 또는 수동 `/24` 등)  
2. 동일 사용자명·비밀번호 없는 SSH  
3. Cluster Assistant 또는 스크립트로 검증 (지원 범위는 문서: 직접 케이블 소수는 소수 노드, 스위치 경유는 별도)

이 책에서는 **특정 IP 대역을 표준처럼 강제하지 않는다**. 팀 규약을 따르되, **NCCL이 그 인터페이스를 쓰는지**를 제114강 체크리스트로 확인한다.

## 소프트웨어 스택 스케치
전형적인 층:

```text
DGX OS
  ├─ NVIDIA 드라이버 / CUDA
  ├─ 컨테이너 런타임 + NVIDIA Container Toolkit
  ├─ (선택) Ray / 기타 분산 런타임
  └─ vLLM 또는 TensorRT-LLM 등 엔진 이미지
```

실습 원칙:

1. 두 노드의 **이미지 태그·엔진 버전·모델 리비전**을 맞춘다.  
2. 모델 가중치는 양쪽에 동일 경로/캐시로 둔다(또는 공유 스토리지 정책 명확화).  
3. 먼저 **단일 노드 서빙 성공**, 그다음 듀얼.  

## 구성 서사 — 단계별 체크리스트
아래는 **실행 순서 템플릿**이다. 명령 플래그는 배포 시점에 공식 플레이북으로 치환한다.

### 7.1 Phase 0 — 문서 고정

- [ ] 제품 페이지 URL, User Guide Hardware, Clustering/ConnectX 문서 URL과 **열람 날짜**를 노트에 기록  
- [ ] 스토리지 SKU(1TB vs 4TB 등) 확인  
- [ ] 사용 케이블이 QSFP·속도 등급에 맞는지 확인  

### 7.2 Phase 1 — 단일 노드 건강

각 노드에서:

- [ ] 부팅·온도·전원(공식 240W 어댑터 사용 권고 — User Guide)  
- [ ] GPU/통합 디바이스 인식  
- [ ] 작은 모델로 엔진 1회 서빙 성공  
- [ ] `nvidia-smi` 또는 플랫폼 동등 도구로 메모·유틸 관찰 방법 확보  

### 7.3 Phase 2 — 물리 클러스터링

- [ ] QSFP 케이블 연결  
- [ ] 링크 Up 확인  
- [ ] 인터페이스 이름 기록 (`ip link`, 문서 권장 도구)  
- [ ] 고속 IP 부여  

### 7.4 Phase 3 — SSH·파일

- [ ] 고속 IP로 SSH  
- [ ] 동일 UID/사용자 가정 충족 여부  
- [ ] 모델 디렉터리·허깅페이스 토큰(필요 시) 동기화  

### 7.5 Phase 4 — NCCL·RoCE

- [ ] RDMA 디바이스 노출 확인  
- [ ] `NCCL_DEBUG`로 수송 경로 확인  
- [ ] 공식/배포 nccl 테스트 또는 플레이북 검증  
- [ ] 관리용 NIC로 폴백되지 않았는지 확인  

### 7.6 Phase 5 — 분산 서빙

- [ ] head / worker(또는 동등 역할) 기동  
- [ ] 클러스터 status가 두 노드를 보는지 확인  
- [ ] `tensor-parallel-size 2` 또는 `pipeline-parallel-size 2` 중 **실험 계획에 맞는 것**부터  
- [ ] 짧은 프롬프트로 smoke test  
- [ ] 제116강 측정 템플릿으로 TTFT/TPOT 기록  

## TP vs PP on 2× Spark — 선택 가이드
| 전략 | 직관 | 통신 패턴 | 언제 후보 |
|---|---|---|---|
| TP=2 | 층 내 분할 | 층마다 collective | 엔진·헤드 수가 지원, NCCL이 건강 |
| PP=2 | 층 구간 분할 | stage 경계 | collective 부담을 줄이는 실험 |
| Replica 2 | 모델 복제 | 거의 없음 | 단일 노드에 모델이 들어갈 때 처리량 |

제113~114강 복습: **네트워크가 약하면 TP가 특히 아프다.**  
커뮤니티·포럼에는 TP hang 사례와 NCCL 환경변수 조언이 있으나, **인터페이스 이름은 머신마다 다르다.** 남의 export를 그대로 붙이지 말고 제114강 절차로 맞춘다.

## 명령 스케치 (예시 · 비보장)
아래는 **학습용 스케치**다. 이미지명·스크립트명은 NVIDIA playbook / 팀 표준으로 교체한다.

```bash
# --- 스케치: 인터페이스 관찰 ---
ip -br link
# 플랫폼에 따라 RDMA 매핑 도구 사용 (문서 확인)

# --- 스케치: NCCL 디버그 켠 채 짧은 테스트 ---
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=...   # 고속 IF 이름
# export NCCL_IB_HCA=...          # 문서·ib 도구로 확인한 이름
# (테스트 바이너리 또는 플레이북 단계 실행)

# --- 스케치: 분산 서빙 개념 ---
# node0: Ray head (또는 동등) + 고속 IP 바인딩
# node1: worker join
# head: vllm serve <MODEL> --tensor-parallel-size 2 --distributed-executor-backend ray
```

**주의 문구를 책에 남긴다:** 위 플래그는 버전·플랫폼에 따라 거부되거나 권장 백엔드가 바뀔 수 있다. 실행 전 `vllm serve -h` 및 공식 Spark+vLLM 플레이북을 본다.

## 관측 — 무엇을 로그에 남길까
환경 구성 노트 템플릿:

```text
날짜:
Spark A 시리얼/호스트명:
Spark B 시리얼/호스트명:
케이블 종류:
고속 IF 이름 A/B:
고속 IP A/B:
NCCL 수송 (로그 요약):
엔진 이미지 태그:
모델 ID·리비전·dtype:
병렬 설정 (TP/PP/DP):
Smoke test 결과:
이슈·해결:
문서 URL·버전:
```

이 노트가 제116~118강 리포트의 **환경 섹션**이 된다.

## 벤더 문서 갱신에 대처하기
스펙·플레이북은 업데이트된다. 불확실하면:

1. `docs.nvidia.com` 의 DGX Spark User Guide에서 Hardware / Networking / Clustering을 연다.  
2. 제품 마케팅 페이지와 **User Guide 표가 다르면**, 실습 기준으로는 **User Guide·릴리즈 노트를 우선** 검토한다.  
3. 스토리지·와이파이·Bluetooth 등 서빙 비핵심은 과감히 생략한다.  
4. 책에 없는 최신 숫자라도 **자신의 노트에 출처와 날짜를 남기면** 제118강 리포트 품질이 올라간다.

불확실한 것을 아는 척하지 않는 것이 이 강의의 규범이다.

## 실습
### 실습 A — 스펙 독서

공식 Hardware Overview를 열고, 다음만 손필기/노트로 옮기시오. (암기 시험 아님)

1. 메모리 용량·대역폭  
2. ConnectX 관련 문장 한 줄  
3. 전원/TDP 관련 주의 한 줄  
4. 열람 날짜  

### 실습 B — 토폴로지 도면

2× Spark를 상단에서 본 도면을 그리고, **데이터플레인**과 **관리플레인**을 다른 색으로 표시하시오.

### 실습 C — 실패 주입 계획

다음 중 하나를 의도적으로 가정하고, 제114강 어느 층(L0~L4)에서 걸릴지 쓰시오.

1. QSFP 대신 10GbE만 연결  
2. SSH는 되지만 NCCL_IB_DISABLE=1  
3. 모델 리비전이 노드마다 다름  

## 자주 하는 실수
1. 112강에서 하드웨어를 찾을 때 — 이제 **115강**이다.  
2. Wi-Fi로 두 대를 묶어 TP를 시도한다.  
3. 단일 노드 성공 없이 멀티노드로 직행한다.  
4. 공식 240W 어댑터 주의를 무시한다(User Guide 경고).  
5. dual-Spark “더 큰 모델” 문구를 측정 없이 SLA에 넣는다.  
6. 커뮤니티의 인터페이스 이름을 그대로 복사한다.

## 수식 보강 — 다중 노드 메모리

모델 병렬로 샤딩하면 랭크당 가중치 메모리는 대략 $1/N$이 됩니다. KV는 요청 길이·동시성에 따라 따로 늘어납니다.


<!-- enrich-block-115 -->
## 멀티 노드 메모리·배치

모델 파라미터 메모리:

$$
\mathrm{Mem}_W = N_{\mathrm{params}}\cdot b_{\mathrm{bytes}}\cdot f_{\mathrm{overhead}}
$$

데이터 병렬 시 글로벌 배치:

$$
B_{\mathrm{global}}=B_{\mathrm{local}}\cdot N_{\mathrm{gpu}}
$$

유효 학습률/노이즈는 $B_{\mathrm{global}}$에 따라 달라지므로 LR 재스케일을 검토합니다.


<!-- enrich-extra-115 -->
## 구성 체크 — 2노드

```python
# 환경 변수 스케치 (교육용)
import os
cfg = {
    "NNODES": 2,
    "NPROC_PER_NODE": 8,
    "MASTER_ADDR": "10.0.0.1",
    "MASTER_PORT": "29500",
}
world = int(cfg["NNODES"]) * int(cfg["NPROC_PER_NODE"])
print("world_size", world)
```

$$
B_{\mathrm{global}}=B_{\mathrm{micro}}\cdot N_{\mathrm{accum}}\cdot N_{\mathrm{gpu}}
$$

## 수식·용량으로 보는 2× 구성
### 통합 메모리·KV 예산（개념）

가중치 바이트 $W$, KV 캐시 $M_{\mathrm{KV}}$, 기타 오버헤드 $O$에 대해 대략

$$

W + M_{\mathrm{KV}} + O \;\le\; M_{\mathrm{avail}}
$$

$M_{\mathrm{KV}}$는 동시 요청·컨텍스트에 비례（제101·105강）:

$$

M_{\mathrm{KV}} \propto B_{\mathrm{inflight}}\cdot L\cdot(\text{층}\cdot\text{헤드차원}\cdot\text{바이트})
$$

**사실:** DGX Spark는 통합 메모리 등 플랫폼 특성이 있다. HBM-only 가정으로 환산표를 만들지 말 것.  
**해석:** 2대 연결의 목적은 “메모리를 이어 붙인다”만이 아니라 **병렬·통신 경로를 연다**는 데 있다.

### TP=2일 때 통신

노드 간 TP를 켠다면 제114강의

$$

T_{\mathrm{token}} \approx T_{\mathrm{compute}} + \sum T_{\mathrm{collective}}
$$

가 그대로 적용된다. 케이블·RoCE가 준비되기 전에 TP size만 올리면 hang·역설적 감속이 난다.

### 단일 vs 듀얼 — 언제이득인가（정성）

| 상황 | 경향 |
|---|---|
| 단일 GPU에 모델이 여유 | 듀얼은 통신 비용만 살 수 있음 |
| 가중치·KV가 단일 한계 | TP/샤딩 후보 |
| 처리량 수평 확장 | 복제(replica) vs 샤딩을 구분 |

복제와 TP를 혼동하지 말 것. 복제는 요청 라우팅, TP는 한 모델의 쪼갬이다.


<!-- enrich-batch4-115 -->
## 2× 노드 토폴로지 스케치

$$
N_{\mathrm{gpu}}=N_{\mathrm{node}}\cdot G_{\mathrm{per\ node}}
$$

```python
nodes, gpus = 2, 8
print("world", nodes*gpus)
# MASTER_ADDR는 노드0, 방화벽/포트 허용 필요
```

### 헬스 체크

1. `nvidia-smi` 전 GPU 가시성
2. NCCL test bandwidth
3. 시계 동기(로그 상관)

$$
t_{\mathrm{step}}=t_{\mathrm{comp}}+t_{\mathrm{comm}}+t_{\mathrm{idle}}
$$

## LLM에서는 어디에 사용될까?
- 랩에서 2대 Spark로 멀티노드 서빙 PoC
- NCCL 경로를 제품 문서의 QSFP/ConnectX와 대조
- 제116강 Serving 프로젝트의 하드웨어 전제
- 제117강 최적화 실험의 “GPU count” 축

## 실습 D — 메모리 부등식
가상으로 $W$가 $M_{\mathrm{avail}}$의 70%일 때, 동시성 $B$를 키우면 어떤 항이 먼저 한계에 닿는지 쓰시오.

## 실습 E — 토폴로지 라벨
물리 케이블 / IP / NCCL / 엔진 TP 설정을 한 장의 레이어 그림으로 그리시오.


## 핵심 요약
- DGX Spark는 GB10 Grace Blackwell 기반 데스크탑 AI 플랫폼이며, 공개 스펙은 **출처와 날짜를 붙여** 읽는다.
- 메모리는 공개상 **128GB LPDDR5x unified, 273 GB/s** 등으로 기술되며 HBM과 동일시하지 않는다.
- 2× 구성의 핵심은 **QSFP/ConnectX-7 데이터플레인 + NCCL 검증**이다.
- 구성 순서는 물리 → 주소/SSH → NCCL → 엔진이다.
- 다음 강의는 이 환경(또는 동등 GPU)에서 **실제 서빙 프로젝트**를 돌린다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| DGX Spark | NVIDIA 데스크탑형 AI 시스템 제품 |
| GB10 | Grace Blackwell Superchip (공개 명칭) |
| Unified memory | CPU·GPU가 공유하는 일관 메모리 (문서 표현) |
| ConnectX-7 | 고속 Smart NIC |
| QSFP | 고속 광/동축 트랜시버·포트 폼팩터 |
| Data plane | TP/NCCL이 흐르는 고속 경로 |
| Management plane | SSH·패키지·10GbE 등 관리 경로 |

## 연습문제
### 문제 1 (사실)

공개 스펙 기준으로 DGX Spark 시스템 메모리 용량과 대역폭으로 문서에 나오는 대표 숫자를 쓰시오. (암기가 아니라 문서 확인 습관 — 틀리면 문서를 다시 연다)

### 문제 2 (설계)

왜 2× Spark TP 실습에서 10GbE RJ-45만 연결하는 것이 위험한가?

### 문제 3 (절차)

단일 노드 서빙 성공 전에 멀티노드 Ray/vLLM을 올리면 디버깅이 어려워지는 이유를 쓰시오.

### 문제 4 (연결)

제114강의 “스펙 200GbE ≠ NCCL 실효”를 2× Spark에 적용해 한 문장으로 쓰시오.

### 문제 5 (문서)

스토리지가 1TB인지 4TB인지가 갈릴 때 마케팅 페이지만 믿지 말고 무엇을 확인하는가?

### 문제 6 (판단)

모델이 한 Spark에 들어가고 동시성만 필요하다면 2× TP보다 먼저 검토할 대안은?

---

## 정답 및 해설
### 문제 1

공개 표·가이드 기준 예: 128 GB, 273 GB/s. (열람 시점 문서가 다르면 문서 값을 정답으로 하고 출처를 남긴다.)

### 문제 2

TP collectives가 느린 관리망으로 흘러 latency가 커지고, “듀얼이 더 느린” 결과가 나기 쉽기 때문이다.

### 문제 3

실패 원인이 모델·엔진·네트워크·분산 런타임 중 어디인지 분리되지 않기 때문이다.

### 문제 4

QSFP 링크가 Up이어도 NCCL·엔진 층에서 실효 대역폭·지연이 달라질 수 있으므로 층별 측정이 필요하다.

### 문제 5

User Guide/주문 SKU/실제 `lsblk` 등 장치 실측으로 확인한다.

### 문제 6

동일 모델 레플리카 + 로드밸런싱(처리량 확장)을 우선 검토한다.

## 다음 강의와 연결
환경 서사가 끝났다. 이제 **서빙을 제품처럼** 올리고 측정한다.

이전 강의: **제114강. NCCL과 RoCE**  
다음 강의: **제116강. 프로젝트 — 실제 LLM Serving**

제116강은 vLLM(또는 스케치)으로 모델을 띄우고, TTFT/TPOT 템플릿으로 병목을 해석한다. 제117강은 그 측정을 GPU 최적화 실험으로 확장한다.

<!-- enrich-115-depth -->
## 2노드 구성을 용량 식으로

노드당 GPU 메모리 $M$, 모델 가중치 $W$, KV 여유 $K$라 하면 대략

$$
W + K_{\mathrm{reserve}} + M_{\mathrm{runtime}}
\le
M
$$

텐서병렬 차수 $t_p$로 가중치를 나누면

$$
W_{\mathrm{perGPU}}
\approx
\frac{W}{t_p}
$$

대신 통신 $t_{\mathrm{comm}}$이 늘어난다.

### 네트워크·스토리지 체크

```text
1) 노드 내 NVLink/NVSwitch 경로
2) 노드 간 RoCE/IB 대역·지연
3) 체크포인트 로딩 대역（콜드스타트 TTFT에 영향）
4) 시각·NTP·컨테이너 런타임 일치
```

동시성 $C$에서 KV:

$$
K(C)
\approx
c\cdot \bar{T}_{\mathrm{in+out}}\cdot C
$$

$K(C)+W>M$이면 OOM 또는 preempt 폭증이 난다. 제116강 서빙 프로젝트의 용량 계획과 맞춘다.

### 스모크 테스트 최소셋

1. 단노드 1요청 TTFT/TPOT
2. 단노드 동시성 스위프
3. 2노드 TP 스모크（동일 프롬프트）
4. NCCL 대역 마이크로벤치 기록

통과 전에 “클러스터 준비 완료”라고 쓰지 않는다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [114강. NCCL과 RoCE](114강_NCCL과_RoCE.md)
- **다음 강:** [116강. 프로젝트 — 실제 LLM Serving](116강_프로젝트_실제_LLM_Serving.md)

<!-- /LECTURE_NAV -->
