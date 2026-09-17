# 113강. Tensor Parallel
## 이번 강에서 배우는 내용

- 가중치·활성화를 여러 GPU에 샤딩한다는 말이 층(layer) 안에서 무엇을 뜻하는지 설명한다.
- TP에 필요한 집합 통신(collective) 이 왜 생기는지 그림으로 말한다.
- TP가 도움이 되는 조건과, 오히려 느려질 수 있는 조건을 구분한다.
- Pipeline Parallel·Data Parallel과 역할이 다름을 표로 정리한다.
- 제114~115강의 NCCL·멀티노드 환경이 TP의 토대임을 연결한다.

## 왜 중요한가?
서빙 현장에서 TP를 켜는 이유는 보통 둘 중 하나다.

```text
(1) Capacity: 모델(+KV)이 GPU 한 장에 안 들어간다
(2) Latency/Throughput: 한 장보다 여러 장에 나눠 계산하고 싶다
```

그러나 TP는 공짜가 아니다.

```text
이득: 장당 메모리·계산 부담 ↓
비용: 매 forward마다 GPU 간 통신 ↑
```

제111강의 언어로 말하면, TP는 **디바이스 메모리 용량 문제를 통신·동기화 문제로 교환**하는 기술에 가깝다. 네트워크가 느리면(제114~115강) 교환이 손해로 끝난다.

## 선수 개념
1. Transformer block의 Attention + MLP — 2권, 제108강  
2. Prefill / Decode 메모리 특성 — 제100~101, 111강  
3. 서빙 엔진이 멀티 GPU를 붙이는 방식 — 제104, 112강  
4. (예고) NCCL collectives — 제114강  

## 용어
| 영어 | 한국어 | 쉬운 정의 |
|---|---|---|
| **Tensor Parallel (TP)** | 텐서 병렬 | 한 층의 가중치·연산을 여러 장치에 쪼개 동시에 계산 |
| **Shard** | 샤드 | 쪼갠 조각(가중치의 일부 열/행 등) |
| **Collective** | 집합 통신 | all-reduce, all-gather 등 여러 순위가 참여하는 통신 |
| **TP size / world** | TP 크기 | 한 텐서 병렬 그룹에 속한 GPU 수 (예: 2, 4, 8) |

## 병렬화 지도 — TP만 있는 것이 아니다
| 전략 | 무엇을 나누나 | 전형적인 통신 | 서빙에서 |
|---|---|---|---|
| **Data Parallel (DP)** | 요청/배치 | 학습에선 gradient, 서빙에선 복제본 간 거의 없음 | 레플리카 스케일아웃 |
| **Tensor Parallel (TP)** | 층 내부 텐서 | 잦은 all-reduce/all-gather | 거대 모델 단일 복제본 |
| **Pipeline Parallel (PP)** | 층 범위(앞/뒤) | stage 경계 send/recv | 깊이를 노드에 분할 |
| **Expert Parallel (EP)** | MoE 전문가 | all-to-all 등 | MoE 서빙 (제109강) |

설명: 실제 시스템은 위 전략을 **조합**한다 (예: DP×TP, TP×PP). 이 강의는 TP 코어만 깊게 본다.

```text
DP:  모델 복제 × 요청 분산
TP:  모델 1개 × 가중치 분할
PP:  모델 1개 × 층 구간 분할
```

## 직관 — 행렬을 세로로 자르기
간단한 GEMM을 생각한다.

$$

Y = X W

$$

$W$의 열을 GPU 두 장에 나눈다.

```text
W = [ W0 | W1 ]   (열 방향 분할 예시)

GPU0: Y0 = X W0
GPU1: Y1 = X W1
Y  = [ Y0 | Y1 ]  (필요 시 gather)
```

Attention/MLP에서는 이런 분할이 **연속된 투영**과 맞물리도록** 설계된다. 유명한 패턴이 Megatron-LM 스타일 분할이다(원 논문·구현 디테일은 문헌 참고; 여기선 직관만).

핵심 메시지:

> 각 GPU는 **전체 $X$의 일부 결과**만 갖고, 다음 연산에 필요하면 **통신으로 합치거나 동기화**한다.

## Transformer 층에서의 TP 스케치
### 6.1 MLP 예 (개념)

MLP가 대략 $W_1$, $W_2$ 두 행렬이라 하자(활성화 생략).

전형적인 아이디어(설명용):

```text
W1 을 열 분할 → 각 GPU가 부분 활성화 계산
W2 을 행 분할 → 부분 결과를 더해 최종으로 만듦
마지막에 all-reduce 로 합치기
```

결과는 “층 하나 끝날 때마다 **한 번 이상**의 collective가 필요할 수 있다”는 것이다.

### 6.2 Attention 예 (개념)

Q/K/V 투영을 헤드 또는 헤드 그룹 단위로 나눈다.

```text
GPU0: head 0..h/2-1
GPU1: head h/2..h-1
```

주의:

- **GQA/MQA**(제108강 계열)에서는 헤드 수·KV 헤드 수가 TP size와 **나누어떨어지지 않으면** 설정이 거절되거나 비효율이 난다.
- 엔진마다 제약·자동 조정이 다르다. 문서를 확인한다.

### 6.3 그림

```text
        X (활성화, 필요 시 broadcast/복제)
        │
   ┌────┴────┐
   ▼         ▼
  GPU0      GPU1     ← 각자 W의 샤드
   │         │
   └────┬────┘
        ▼
   collective (예: all-reduce)
        │
        ▼
     다음 층
```

## 통신이 생기는 이유
단일 GPU에서는 층 출력이 **같은 메모리**에 있다. TP에서는 출력이 **장치에 흩어져** 있다.

| 상황 | 필요한 일 |
|---|---|
| 다음 연산이 전체 벡터를 요구 | all-gather 또는 등가물 |
| 부분 합이 최종 값 | all-reduce (합) |
| 다른 병렬 모드 | send/recv, all-to-all |

통신 비용의 대략적 감각(정량 단정 아님):

```text
비용 ∝ 메시지 크기 × 왕복 횟수 / 유효 대역폭 + 지연(latency) 항
```

Decode처럼 **작은 활성화**라도 층마다 동기화하면 **지연 항**이 누적된다. 그래서 네트워크 latency가 나쁜 멀티노드 TP는 특히 아프다(제114~115강).

## 언제 TP가 도움이 되는가
### 8.1 도움이 되는 후보 조건

1. **메모리가 부족**해 양자화·압축만으로 목표 모델을 못 올린다.  
2. TP size로 나눈 뒤에도 **각 샤드의 GEMM이 충분히 큼** (너무 잘게 쪼개 SM이 굶지 않음).  
3. GPU 간 연결이 **충분히 빠름** (동일 노드 NVLink급, 또는 검증된 고속 인터커넥트).  
4. 엔진이 해당 모델·TP size 조합을 지원한다.

### 8.2 도움이 약하거나 해로운 후보 조건

1. 모델이 이미 한 장에 여유 있게 들어가고, 병목이 **요청 수**라면 → **복제(DP/레플리카)** 가 나을 수 있다.  
2. TP size가 과도해 행렬이 너무 얇아진다.  
3. 노드 간 링크가 느리고 NCCL이 TCP로 폴백한다.  
4. MoE에서 TP만 생각하고 EP/토폴로지를 무시한다.  
5. Prefill만 보고 좋아졌다고 착각하고, decode TPOT는 통신에 잡아먹힌다.

### 8.3 결정 트리 (실무용 스케치)

```text
모델+KV가 한 GPU에 들어가나?
  ├─ No → 양자화/더 짧은 ctx/TP·PP 검토
  └─ Yes
       └─ 목표 미달 지표가 무엇인가?
            ├─ 처리량(동시 사용자) → 레플리카 우선 검토
            └─ 단일 요청 지연 → 커널·배치·양자화·(여력 시) TP 실험
```

단정 금지: 트리의 각 가지는 **측정으로 검증**.

## TP size 고르기
실무 팁(설명):

| 규칙 | 이유 |
|---|---|
| 모델·헤드 수가 TP로 **나누어떨어지는지** 확인 | 구현 제약 |
| 2 → 4 → …처럼 **점진적** 실험 | 통신 비용 관찰 |
| 동일 워크로드로 TP=1과 비교 | 회귀 감지 |
| 메모리만 보지 말고 TTFT/TPOT 함께 | 제107강 |

vLLM 등에서의 개념적 설정(스케치, 옵션명은 버전 확인):

```bash
# 예시 스케치 — 환경·모델·버전을 채워 사용
vllm serve <MODEL> \
  --tensor-parallel-size 2
```

멀티노드면 executor backend·클러스터 런타임(Ray 등)이 추가된다(제115~116강).

## Prefill / Decode와 TP
| 단계 | TP 효과 직관 |
|---|---|
| Prefill | 큰 GEMM을 나누면 계산 시간 감소 여지. 통신도 커질 수 있음 |
| Decode | 계산은 작아지고, **층마다 통신 latency**가 더 잘 드러남 |

제111강 roofline과 연결:

- Prefill: compute roof 쪽으로 갈 때 TP의 계산 분할이 빛날 수 있음  
- Decode: memory+comm 쪽으로 가면 TP가 “용량 해결”에는 성공해도 “속도”는 실패할 수 있음

## 수치 없는 사고 실험
가정(설명용):

- 층마다 all-reduce 1회
- 층 수 $L$
- 링크 유효 지연이 커지면 decode 토큰당 비용에 $\propto L$ 항이 더해짐

교훈:

> TP=2가 메모리를 해결해도, $L$이 큰 모델에서는 **통신 latency 예산**을 먼저 재라.

측정은 `nccl-tests`·엔진 로그·실제 TPOT로(제114, 116강). **여기 숫자 발명 금지**.

## 구현 스케치 — 분할 GEMM (교육용)
실제 엔진 커널이 아니다. **개념 확인용** CPU 코드다.

```python
import torch

def column_parallel_linear(x, w_shards):
    """x: [B, In], w_shards: list of [In, Out_i] on 'logical' GPUs"""
    partials = [x @ w for w in w_shards]
    return torch.cat(partials, dim=-1)

def row_parallel_linear(x_shards, w_shards):
    """각 샤드가 부분 곱을 만든 뒤 합산 (all-reduce 흉내)"""
    partials = [xs @ ws for xs, ws in zip(x_shards, w_shards)]
    y = partials[0]
    for p in partials[1:]:
        y = y + p  # all-reduce SUM의 장난감 버전
    return y

# 데모
B, In, Out = 2, 8, 6
x = torch.randn(B, In)
w = torch.randn(In, Out)
w0, w1 = w[:, : Out // 2], w[:, Out // 2 :]

y_ref = x @ w
y_tp = column_parallel_linear(x, [w0, w1])
assert torch.allclose(y_ref, y_tp, atol=1e-5)
print("column-parallel match OK")
```

## 서빙 엔진에서의 TP
엔진(제112강)은 TP를 다음처럼 감싼다.

```text
사용자: TP size = 2
엔진:  샤드 로딩 → 디바이스 배치 → collective 삽입 → 스케줄러와 공존
```

확인할 것:

1. **샤드 로딩**이 올바른가 (같은 리비전 가중치).  
2. **GPU affinity / visible devices**가 의도한 장치인가.  
3. **NCCL 환경변수**가 올바른 NIC를 가리키는가 (제114강).  
4. 로그에 TP 초기화·타임아웃·hang이 없는가.

## TP와 레플리카 — 혼동 금지
```text
잘못된 생각: GPU가 2장이면 무조건 TP=2
올바른 질문: 2장으로 “한 모델 분할”이 필요한가, “두 복제본”이 필요한가?
```

| 목표 | 선호 후보 |
|---|---|
| 더 큰 모델 | TP/PP/양자화 |
| 더 많은 동시 사용자 | 레플리카 + 로드밸런서 |
| 둘 다 | 충분 메모리 위에서 DP×TP 조합 |

## 실습
### 실습 A — 헤드 수 나누기

어떤 모델이 query heads = 32, kv heads = 8이다.

1. TP=2, 4, 8이 가능한지(나누어떨어짐) 표로 쓰시오.  
2. TP=3이 까다로운 이유를 한 줄로 쓰시오.

### 실습 B — 전략 선택

다음 각 상황에 TP / PP / Replica 중 **우선 후보**를 고르고 이유를 쓰시오.

1. 7B급이 한 장에 넉넉, 동시 사용자 폭증  
2. 거대 dense 모델이 한 장 OOM, 빠른 NVLink 노드 1대에 GPU 8장  
3. 2× 데스크탑급 노드, 층 수를 반으로 나누는 실험 (제115강 복선)

## 자주 하는 실수
1. 메모리 OOM만 보고 TP를 키운 뒤, 통신 hang을 엔진 버그로만 단정한다.  
2. TP=2와 replica 2를 같은 설정으로 본다.  
3. GQA 헤드 수와 TP size 불일치를 무시한다.  
4. 단일 요청 벤치만으로 TP 이득을 선언하고 동시성을 안 본다.  
5. NCCL이 쓰는 NIC를 확인하지 않은 채 “TP는 원래 느리다”고 결론낸다.

## 수식 보강 — Tensor Parallel 분할

선형층 $Y=XW$에서 $W$를 열 방향으로 $N$개로 나누면

$$
W=[W_1\,|\,\cdots\,|\,W_N],\quad
Y_i=X W_i,\quad
Y=[Y_1\,|\,\cdots\,|\,Y_N]
$$

입니다. 행 분할이면 all-reduce가 필요합니다. Attention의 QKV/출력 투영도 같은 방식으로 샤딩합니다.

## 수학적으로 이해하기 — 샤드 행렬곱

### 5b.1 Column-parallel

$W\in\mathbb{R}^{d_{\mathrm{in}}\times d_{\mathrm{out}}}$를 TP size $N$으로 열 분할한다.

$$

W=\big[W^{(0)}\,\big|\,W^{(1)}\,\big|\,\cdots\,\big|\,W^{(N-1)}\big],
\quad
W^{(r)}\in\mathbb{R}^{d_{\mathrm{in}}\times(d_{\mathrm{out}}/N)}

$$

각 순위 $r$에서

$$

Y^{(r)}=X W^{(r)}

$$

필요 시

$$

Y=\mathrm{AllGather}\big(Y^{(0)},\ldots,Y^{(N-1)}\big)
$$

또는 다음 층이 행 분할이면 gather 없이 이어질 수 있다(구현 패턴 의존).

### 5b.2 Row-parallel

입력을 분할해 두었다면

$$

X=\big[X^{(0)}\,\big|\,\cdots\,\big|\,X^{(N-1)}\big],
\quad
W=\begin{bmatrix}W^{(0)}\\ \vdots \\ W^{(N-1)}\end{bmatrix}

$$

$$

Y^{(r)}=X^{(r)} W^{(r)},\qquad
Y=\mathrm{AllReduce}\sum_{r} Y^{(r)}

$$

Megatron 스타일 MLP는 대략 **column-parallel $W_1$ + row-parallel $W_2$** 로, 층 끝에 all-reduce가 한 번 들어가게 맞춘다(세부는 문헌·엔진).

### 5b.3 Attention 헤드 샤딩

쿼리 헤드 $H_q$를 $N$으로 나눈다.

$$

H_q \bmod N = 0
\quad\text{(이상적 조건)}

$$

GQA에서 KV 헤드 $H_{kv}$도 마찬가지로 나누어떨어지는지 확인한다.

$$

H_{kv} \bmod N = 0
$$

만족하지 않으면 엔진이 거절하거나 패딩·비효율이 생긴다. **설정 전 산수**.

## 정량 스케치 — 통신 vs 계산

### 6b.1 All-reduce 메시지 크기

은닉 크기 $d$, 마이크로배치 토큰 수 $M$, 원소당 $b$ 바이트일 때, 층 출력 all-reduce 페이로드 감각:

$$

\mathrm{Bytes}_{\mathrm{msg}} \approx M\cdot d\cdot b

$$

층마다 1회라면 디코더 $L$층에서

$$

\mathrm{Bytes}_{\mathrm{per\,token\,step}} \propto L\cdot d\cdot b
$$

(실제는 알고리즘·퓨전·overlapp에 따라 달라짐. **비례 감각**만.)

### 6b.2 Latency 항

유효 대역폭 $B_w$, 지연 $\alpha$라 두면 매우 거친 모형:

$$

T_{\mathrm{comm}} \approx \alpha + \frac{\mathrm{Bytes}_{\mathrm{msg}}}{B_w}

$$

Decode에서 $M$이 작으면 $\mathrm{Bytes}/B_w$보다 **$\alpha\cdot L$** 누적이 두드러질 수 있다.  
이것이 “NVLink에선 괜찮은 TP가 노드 간에선 아프다”는 이야기의 골격이다.

### 6b.3 계산 분할 이득（이상화）

단일 GPU GEMM 시간 $T_{\mathrm{compute}}$가 대략 행렬 크기에 비례한다면, 이상적 $N$분할은

$$

T_{\mathrm{compute}}^{(N)} \approx \frac{T_{\mathrm{compute}}^{(1)}}{N}

$$

총 시간 스케치:

$$

T_{\mathrm{total}}^{(N)} \approx \frac{T_{\mathrm{compute}}^{(1)}}{N} + T_{\mathrm{comm}}(N)
$$

$T_{\mathrm{comm}}$이 $N$과 함께 늘거나 줄어드는 형태는 토폴로지에 달렸다.  
**교차점**은 측정으로만 확정한다. 여기서 숫자를 발명하지 않는다.

### 6b.4 손계산 — 나누어떨어짐

| $H_q$ | $H_{kv}$ | TP=2 | TP=4 | TP=8 |
|---:|---:|---|---|---|
| 32 | 8 | OK | OK | OK |
| 32 | 4 | OK | OK | 불가($4\nmid 8$은 아님; $H_{kv}=4$, TP=8이면 $4\bmod 8\neq0$) |
| 40 | 8 | OK | 불가 | 불가 |

표는 **산수 연습**이다. 실제 모델 헤드 수는 문서 확인.

## Prefill·Decode 수치 사고실험（가정）

가정(설명용, 벤치 아님):

- $L=40$ 층
- 층당 all-reduce 1회
- 링크 왕복 지연 항이 decode 토큰당 $\alpha_{\mathrm{eff}}$

그러면 통신만의 하한 감각:

$$

T_{\mathrm{comm}}^{\mathrm{decode}} \gtrsim L\cdot \alpha_{\mathrm{eff}}
$$

$\alpha_{\mathrm{eff}}$가 커지면 목표 TPOT 예산을 통신이 먼저 잠식한다.  
Prefill은 $M$이 커 $T_{\mathrm{compute}}$ 비중이 커서 TP 이득이 **상대적으로** 보이기 쉽다.

## DP×TP 메모리 감각

레플리카(DP) $R$개, 각 복제본이 TP size $N$이면 GPU 수는 $R\cdot N$이다.

| 목표 | 노브 |
|---|---|
| 모델이 안 들어감 | $N$↑ (또는 양자화·PP) |
| 동시 요청 | $R$↑ |
| 둘 다 | 메모리 여유를 측정하며 $R,N$ 격자 탐색 |

잘못된 기본값: “GPU가 8장이니 TP=8”.  
올바른 질문: “한 복제본을 몇 장에 쪼갤 것인가?”

## 구현 — row-parallel 수치 검증

```python
import torch

def row_parallel_demo():
    B, In, Out, N = 2, 8, 4, 2
    assert In % N == 0
    x = torch.randn(B, In)
    w = torch.randn(In, Out)
    # 행 분할: 입력을 In/N 조각, W도 행 분할
    x_shards = x.chunk(N, dim=-1)
    w_shards = w.chunk(N, dim=0)
    partials = [xs @ ws for xs, ws in zip(x_shards, w_shards)]
    y_tp = sum(partials)  # all-reduce SUM
    y_ref = x @ w
    assert torch.allclose(y_tp, y_ref, atol=1e-5)
    print("row-parallel match OK")

row_parallel_demo()
```

column/row를 **둘 다** 손으로 맞춰 보면, 엔진 로그의 collective 위치가 덜 무섭다.

## 실패 모드 카탈로그（확장）

| 증상 | 후보 원인 | 다음 행동 |
|---|---|---|
| 기동 시 hang | NCCL/토폴로지 | 제114강 체크리스트 |
| OOM 해소·TPOT 악화 | 통신 latency | TP↓ 또는 복제 재설계 |
| 이상한 수치 출력 | 샤드 로딩 불일치 | 가중치 리비전·맵핑 확인 |
| 특정 TP size만 실패 | 헤드 수 비분할 | $H_q,H_{kv}$ 산수 |
| Prefill만 개선 | decode 통신 지배 | 국면별 지표 분리 |

## FAQ（TP）

**Q. TP=2가 TP=1보다 항상 빠른가?**  
A. 아니다. 용량이 필요할 때는 필수일 수 있으나, 속도는 통신·행렬 두께에 달렸다.

**Q. PP와 동시에 쓰면?**  
A. 층 구간(PP) + 층 내부(TP). 통신 패턴이  alike다. 격자 탐색이 필요하다.

**Q. MoE의 EP와 TP 중 무엇을 먼저?**  
A. 모델·엔진 권장 토폴로지를 따른다. 일반 규칙은 “항상 TP 먼저”가 아니다.

## 미니 실측 프로토콜（숫자 기입란만）

```text
model:
TP_size: 1 | 2 | 4
GPU_topology: single-node NVLink | multi-node
workload: prefill_tokens=  decode_tokens=  concurrency=
TTFT_p50/p95:
TPOT_p50/p95:
mem_per_gpu:
notes_nccl:
```

칸을 채우는 값은 **해당 환경 실측**이다. 강의 본문에 예시 벤치 숫자를 넣지 않는다.


## LLM에서는 어디에 사용될까?

이번 113강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- Tensor Parallel은 **층 내부 텐서를 GPU들에 샤딩**하고, 그 대가로 **collectives**를 낸다.
- 주 목적 후보는 **용량 해결**과 (조건부의) **계산 분할**이다.
- Decode에서는 통신 latency가 두드러질 수 있어, 빠른 인터커넥트와 올바른 NCCL 설정이 전제다.
- DP/PP/EP와 역할을 섞어 말하지 말 것.
- 다음 강의는 그 통신을 실제로 수행하는 **NCCL과 RoCE**다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Tensor Parallel | 층 내 가중치·연산 분할 |
| Shard | 분할된 파라미터/텐서 조각 |
| All-reduce | 모든 순위의 값을 환원(예:합) 후 공유 |
| All-gather | 각 순위 조각을 모아 전체로 |
| TP size | 텐서 병렬 그룹 크기 |
| Pipeline Parallel | 층을 스테이지로 분할 |
| Replica / DP | 모델 복제 후 요청 분산 |

## 연습문제
### 문제 1 (개념)

TP와 PP의 차이를 “무엇을 나누는가” 기준으로 한 문장씩 쓰시오.

### 문제 2 (통신)

왜 TP는 학습의 DP보다 서빙 forward에서 통신이 **더 자주** 보일 수 있는가?

### 문제 3 (판단)

한 GPU에 모델이 들어가고 동시성만 문제일 때 TP=2를 기본값으로 두는 것이 위험한 이유는?

### 문제 4 (연결)

제111강 memory-bound decode와 TP 통신 latency가 겹치면 TPOT에 어떤 일이 생길 수 있는가?

### 문제 5 (실무)

`tensor-parallel-size`를 올린 뒤 바로 봐야 할 신호 세 가지를 쓰시오.

### 문제 6 (설계)

2노드×각 1GPU 환경에서 TP=2를 쓸 때, 제114~115강으로 넘어가야 하는 이유를 쓰시오.

---

## 정답 및 해설
### 문제 1

TP: 한 층 안의 텐서를 장치들에 분할. PP: 연속된 층 구간을 스테이지·장치에 분할.

### 문제 2

TP는 층 내부 연산 완성을 위해 forward마다 collectives가 필요할 수 있는 반면, 서빙 DP 복제본은 일반적으로 요청 경로에서 모델 가중치를 동기화하지 않는다.

### 문제 3

불필요한 통신·동기화를 넣어 단일 복제본 지연을 악화시키고, 동시에 처리량 확장(레플리카) 기회를 놓칠 수 있다.

### 문제 4

이미 메모리 대기가 있는 구간에 통신 대기까지 더해져 TPOT가 목표를 못 맞출 수 있다(이득은 계산 분할·용량에 한정될 수 있음).

### 문제 5

예: 메모리 footprint, TTFT/TPOT, NCCL/엔진 로그의 타임아웃·느린 링크 경고.

### 문제 6

장치 간 통신이 노드를 건너 네트워크(RoCE/NCCL)를 타므로, 케이블·NIC·NCCL 설정이 TP 성패를 가른다.

## 다음 강의와 연결
TP의 “비용” 칸을 채울 차례다.

이전 강의: **제112강. Inference Engine 비교 — vLLM · TensorRT-LLM · SGLang**  
다음 강의: **제114강. NCCL과 RoCE**

제114강에서는 all-reduce가 실제로 어떤 라이브러리로 도는지, RoCE가 멀티노드에서 어떤 위치인지 개관한다. 제115강에서 그 지식을 2× DGX Spark 토폴로지에 적용한다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [112강. Inference Engine 비교 — vLLM · TensorRT-LLM · SGLang](112강_Inference_Engine_비교_vLLM_TensorRT_LLM_SGLang.md)
- **다음 강:** [114강. NCCL과 RoCE](114강_NCCL과_RoCE.md)

<!-- /LECTURE_NAV -->
