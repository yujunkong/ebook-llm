# 53강. 최신 Transformer 변형 개요
## 이번 강에서 배우는 내용

- GQA / MQA가 무엇을 바꾸는지(K/V 헤드 공유) 한 문장으로 말한다.
- FlashAttention류가 겨냥하는 문제($T\times T$ materialize·메모리 이동)를 개요로 안다.
- MoE가 “모든 토큰이 모든 FFN을 쓰지 않을 수 있다”는 아이디어임을 안다.
- 사실(구조/동기)과 설명(대략적 이득)과 시시각각 변하는 세부를 구분한다.

## 왜 중요한가?
논문을 펼치면 약어가 쏟아진다. 전부 외울 필요는 없다. 대신 제52강의 압력 지도를 기억한다.

```text
길이 T ↑  →  Attention 비용·메모리 압력
생성 서비스 →  KV cache 메모리 압력
모델 규모 ↑ →  연산·통신 압력
```

변형들은 대개 이 압력 중 하나를 완화하려는 **공학적 응답**이다. 오늘의 목표는 이름과 동기이며, “지금 당장 프로덕션 커널을 짜는 것”이 아니다.

## 읽기 규칙 — 사실 / 설명 / 변동
이 강의의 문장은 세 층으로 읽는다.

| 층 | 의미 | 예 |
|---|---|---|
| **사실(구조)** | 정의·연결 관계 | MQA는 K/V 헤드를 하나로 공유한다 |
| **설명(동기)** | 왜 쓰는지에 대한 합리적 이야기 | KV cache 크기를 줄이려는 동기 |
| **변동(세부)** | 구현·수치·채택은 시점에 따라 바뀜 | 특정 속도비, 특정 모델 기본값 |

**이 강의에서 하지 않는 것:** 허구의 벤치마크 표, “X가 Y보다 항상 N% 빠르다” 같은 단정.

세부 알고리즘·API는 해마다 바뀐다. 개념 좌표만 고정하자.

## 복습 — 표준 Multi-Head의 KV
표준 MHA(제41강):

- Query / Key / Value 각각 헤드 수 $H$
- 생성 시 KV cache는 (대략) 층마다 $K,V$를 $H$개 헤드분 저장

변형의 출발점:

> Q는 풍부하게 두고, K/V 표현은 공유해 **캐시와 대역폭**을 줄일 수 있지 않을까?

## MQA — Multi-Query Attention
**사실(구조):**  
Multi-Query Attention에서는 **Query는 여러 헤드**, **Key/Value는 헤드 하나(또는 동등한 공유)**를 쓰는 형태가 일반적 정의다.

**설명(동기):**  
생성 시 KV cache와 메모리 대역폭 부담을 줄이려는 동기. Q 헤드가 많아도 같은 K/V를 바라보게 한다.

**변동:**  
어떤 모델 세대가 기본으로 채택했는지, 품질 트레이드오프의 크기는 시점·과제·학습 설정에 따라 달라진다. 여기서 수치를 고정하지 않는다.

직관 그림:

```text
표준 MHA:  Q1 Q2 Q3 Q4   K1 K2 K3 K4   V1 V2 V3 V4
MQA:       Q1 Q2 Q3 Q4   K_shared        V_shared
```

## GQA — Grouped-Query Attention
**사실(구조):**  
Grouped-Query Attention은 MHA와 MQA **사이**다. Query 헤드를 그룹으로 나누고, 그룹마다 K/V를 공유한다.

```text
예: H_q=8, H_kv=2
  (Q1,Q2,Q3,Q4) → KV_group1
  (Q5,Q6,Q7,Q8) → KV_group2
```

**설명(동기):**  
MQA만큼 극단적으로 공유하지 않으면서도 KV 메모리를 줄이려는 절충.

**변동:**  
그룹 수 선택, 학습 안정성, 서빙 이득의 크기는 구현·하드웨어에 의존한다. “항상 최적”이라고 말하지 않는다.

Mini Transformer(제49강)는 표준 MHA로 충분하다. GQA/MQA는 **읽을 줄 아는 약어**로 남긴다.

## FlashAttention — 아이디어만
제52강에서 $T\times T$ 행렬 materialize가 메모리·대역폭을 압박한다고 했다.

**사실(동기·방향):**  
FlashAttention류 기법은 **exact Attention**(수학적으로 같은 softmax attention)을 목표로 하면서, 큰 $T\times T$를 덜 명시적으로 저장하고 GPU SRAM 계층을 활용하는 **IO-aware** 구현으로 알려져 있다.

**설명:**  
병목을 단순 연산 FLOPs가 아니라 **메모리 이동(IO)**으로 보는 관점과 맞닿아 있다.

**변동·주의:**

- 버전(FlashAttention 1/2/…)、프레임워크 통합、하드웨어 제약은 계속 바뀐다.
- “켜기만 하면 항상 N배” 같은 숫자는 이 책에 적지 않는다.
- 근사 Attention(희소·선형)과 혼동하지 말 것 — FlashAttention류는 보통 **exact**를 표방하는 구현 계열이다(세부 가정은 논문·릴리즈 노트를 따를 것).

이 강의에서 커널 코드를 작성하지 않는다. 제52강의 “$T^2$ 통증”에 대한 **응답 중 하나**로만 기억한다.

## MoE — Mixture of Experts 티저
표준 FFN은 모든 토큰이 같은 MLP를 지난다.

**사실(아이디어):**  
Mixture of Experts(MoE)는 여러 **전문가(FFN)** 후보를 두고, 토큰(또는 토큰 그룹)마다 라우터가 **일부 전문가만** 활성화하는 계열의 구조다.

```text
토큰 → Router → 전문가 중 top-k만 실행 → 결합
```

**설명(동기):**  
파라미터 수(용량)는 키우면서, 토큰당 계산은 모든 파라미터를 다 쓰지 않게 하려는 동기.

**변동:**  
라우팅 알고리즘, 부하 균형, 통신 비용, 학습 불안정성 대응은 활발히 연구·제품화되는 영역이다. 2권 범위 밖이며, 필요 시 이후 권에서 선택적으로 다룬다.

지금은 “FFN 자리가 MoE로 바뀔 수 있다”는 **지도의 핀**만 꽂는다.

### MoE 라우터 수식 (개요)

토큰 히든 $x\in\mathbb{R}^{C}$에 대해 라우터 로짓:

$$

\mathbf{g} = x W_r \in \mathbb{R}^{E}
\quad(W_r\in\mathbb{R}^{C\times E})

$$

top-$k$ 인덱스 집합 $\mathcal{S}(|\mathcal{S}|=k)$를 고르고, 게이트 가중치:

$$

\pi_i = \frac{e^{g_i}}{\sum_{j\in\mathcal{S}} e^{g_j}}
\quad(i\in\mathcal{S}),\qquad
\pi_i=0\ (i\notin\mathcal{S})

$$

출력:

$$

y = \sum_{i=1}^{E} \pi_i\, \mathrm{FFN}_i(x)

$$

**작은 예:** $E=4$, $k=2$, $\mathbf{g}=(0.1, 2.0, 0.5, -1.0)$.  
top-2는 전문가 2와 3($g=2.0,0.5$). Softmax 후 $\pi_2\approx 0.82$, $\pi_3\approx 0.18$ 수준.  
$y \approx 0.82\,\mathrm{FFN}_2(x) + 0.18\,\mathrm{FFN}_3(x)$.

부하 균형 손실·보조 손실은 구현마다 다르므로 여기서 고정하지 않는다.

## 그 밖에 이름만 알아둘 것들
깊이 설명하지 않고 목록만 둔다(검색 키워드).

| 이름 | 한 줄 힌트 |
|---|---|
| Sliding Window Attention | 모든 과거 대신 창 안만 보는 형태 |
| ALiBi / RoPE 스케일링 | 길이 확장과 관련된 위치 편향·보간 이야기 |
| Parallel Transformer 블록 | Attn과 FFN 배치를 병렬화하는 변형(구현 의존) |
| Speculative decoding | 생성 가속 계열(모델 구조 변형과는 별 축) |

이것들도 **세부·수치는 변동**이다.

## Sliding Window — $T^2$를 창으로 자르기
모든 과거를 보는 Causal Attention 대신, 각 위치가 최근 $w$개만 보게 하면 score 연결 수는 대략:

$$

O(T\cdot w)
\quad\text{instead of}\quad
O(T^2)
$$

$w\ll T$이면 점근적으로 이득이다. 먼 과거는 직접 연결이 없어지므로, 층이 쌓이며 수용 야전(receptive field)이 넓어지는 효과에 기댄다.  
**사실:** 창 밖은 마스크로 $-\infty$ 처리하는 구조가 가능하다.  
**설명:** 장거리 품질과의 트레이드오프는 과제·층 수에 의존한다(단정 금지).

## RoPE 스케일링과의 연결 (제43강 복습)
긴 컨텍스트에서 학습 때 본 위치 범위를 넘기면 회전각이 분포 밖으로 나간다.  
NTK-aware·YaRN 등은 **주파수 $\theta_i$를 재스케일**해 보간/외삽을 안정화하려는 기법이다.

$$

\theta_i' = f(\theta_i;\, T_{\mathrm{train}}, T_{\mathrm{target}})
$$

정확한 $f$는 논문·구현마다 다르다. 이 강의에서는 “위치 모듈도 길이 압력의 응답 지점”이라는 좌표만 기억한다.

## Mini Transformer와의 관계
| 주제 | Mini (49~50) | 현대 LLM |
|---|---|---|
| Attention | 표준 MHA | MHA/GQA/MQA 등 |
| 위치 | learned PE | RoPE 등이 흔함 |
| FFN | 단일 MLP | 단일 또는 MoE |
| Attention 커널 | 순진한 matmul | FlashAttention류 가능 |
| 목표 | 학습·이해 | 규모·서빙 효율 |

미니로 원리를 익힌 뒤, 약어가 나와도 “어느 압력을 줄이려는가?”로 분류하면 길을 잃지 않는다.

## GQA/MQA — KV 메모리 수식 스케치
제52강의 KV 메모리:

$$

\mathrm{Mem}_{\mathrm{KV}}
\propto
N \cdot H_{\mathrm{kv}} \cdot T \cdot d

$$

표준 MHA에서는 $H_{\mathrm{kv}} = H_q = H$다.

- **MQA:** $H_{\mathrm{kv}} = 1$ → 이상적으로 KV 메모리가 약 $H$배 축소(다른 항 무시).
- **GQA:** $H_{\mathrm{kv}} = G$ (그룹 수), $1 \le G \le H$, 보통 $H$가 $G$의 배수.

비율만 적으면:

$$

\frac{\mathrm{Mem}_{\mathrm{KV}}^{\mathrm{(GQA)}}}{\mathrm{Mem}_{\mathrm{KV}}^{\mathrm{(MHA)}}}
=
\frac{H_{\mathrm{kv}}}{H_q}

$$

**작은 예:** $H_q=32$, $H_{\mathrm{kv}}=4$ (GQA)이면 비율 $4/32 = 1/8$.  
$H_{\mathrm{kv}}=1$ (MQA)이면 $1/32$.

주의(설명):

- Query 쪽 계산·표현력은 $H_q$를 유지한다.
- 품질·학습 안정성은 $G$ 선택에 좌우되며, “항상 MQA가 최고”가 아니다.
- 캐시 절감 ≠ 프롬프트 prefill 구간의 모든 FLOPs가 $1/H$로 줄어듦. Prefill은 여전히 $T^2$ 항이 있다.

### Attention score shape

한 스텝 생성에서 새 query 길이 1, 과거 키 길이 $t$:

| | score shape (개념) |
|---|---|
| MHA | `[H_q, 1, t]` — 헤드마다 다른 $K$ |
| GQA | `[H_q, 1, t]` — 그룹 내 Q가 같은 $K$를 공유 |
| MQA | `[H_q, 1, t]` — 모든 Q가 하나의 $K$ |

출력 가중합도 같은 패턴으로 $V$를 공유한다. **수학적 Softmax Attention 식은 동일**하고, $K/V$ 텐서의 **복제·공유 구조**만 다르다.

## FlashAttention — IO 관점의 정량 스케치
순진한 Attention은 대략 다음을 HBM(대규모 GPU 메모리)에 쓴다.

1. $S = QK^\top$ 를 $T\times T$로 **명시 저장**
2. Softmax로 $A$를 만들어 또 저장
3. $AV$ 계산

$T$가 크면 이 행렬들의 **읽기/쓰기 대역폭**이 FLOPs보다 먼저 한계가 된다.

설명용 비교(정확한 배수 단정 금지):

- 연산량(대략): $O(T^2 d)$ — 줄이기 어렵다(exact 유지 시).
- 명시적 $T\times T$ 저장: $O(T^2)$ 원소 — **타일링으로 working set를 SRAM에 유지**하면 HBM 왕복을 줄일 수 있다.

FlashAttention류의 핵심 문장:

> Exact softmax attention을 유지하면서, 큰 $S,A$를 통째로 HBM에 materialize하지 않도록 **블록 단위로** 계산한다.

온라인 Softmax(블록 최댓값·합 재조정) 아이디어가 들어가면, 타일만 보고도 최종 Softmax와 같은 결과를 만들 수 있다. 이 강의에서는 알고리즘 세부 증명 대신 **“$T^2$ IO 통증에 대한 exact 구현 응답”**으로 고정한다.

근사 Attention과의 구분:

| | Exact (FlashAttention류) | Approximate (희소·선형 등) |
|---|---|---|
| Softmax attention 값 | 수학적으로 동일(가정 충족 시) | 다름(근사) |
| 복잡도 | 보통 여전히 $O(T^2 d)$ 연산 | $O(T)$~$O(T\log T)$ 등을 노리기도 |
| 동기 | IO·실측 속도 | 점근 복잡도 자체 |

## MoE — 용량과 토큰당 연산
표준 FFN: 모든 토큰이 동일 MLP를 통과 → 토큰당 비용 $O(C\cdot C_{\mathrm{ff}})$.

MoE(개요): 전문가 $E$개 중 토큰마다 top-$k$만 실행.

$$

\mathrm{Cost}_{\mathrm{token}}
\approx
O\!\big(k\cdot C\cdot C_{\mathrm{ff}}^{\mathrm{(expert)}}\big)
\;+\;
O(\text{router})

$$

파라미터(용량)는 대략:

$$

\#\mathrm{params}_{\mathrm{FFN-MoE}}
\approx
E \cdot \#\mathrm{params}_{\mathrm{(one\ expert)}}

$$

**사실:** 활성화되는 전문가 수 $k$와 전체 전문가 수 $E$를 분리할 수 있다.  
**설명:** “파라미터는 큰데 토큰당 FLOPs는 상대적으로 작게” 가져가려는 설계 동기.  
**변동:** 라우팅 손실, 부하 균형, 올-투-올 통신은 구현 난이도의 핵심이며 여기선 티저만.

작은 숫자 스케치: $E=8$, $k=2$이면 토큰당 전문가 계산은 8개 전부가 아니라 2개분.  
용량(저장)은 8개분을 가질 수 있다(공유 임베딩 등은 별도).

## 압력 → 변형 매핑표
| 압력 (제52강) | 응답 변형 | 건드리는 항 |
|---|---|---|
| KV cache $\propto H_{\mathrm{kv}} T$ | MQA / GQA | $H_{\mathrm{kv}}$ |
| $T\times T$ HBM IO | FlashAttention류 | materialize·대역폭 |
| FFN 파라미터·FLOPs | MoE | 토큰당 활성 전문가 |
| 긴 $T$의 $T^2$ | Sliding window / 희소 등 | score 연결성 |
| 길이 확장 | RoPE scaling / YaRN 등 | 위치 주파수 |

새 약어를 만나면 표를 채워 “어느 열인가?”만 물어보면 길을 잃지 않는다.

## 작은 설계 퀴즈 (손계산)
설정: $H_q=16$, $d=64$, $N=24$, $T=8192$, FP16.

1. MHA($H_{\mathrm{kv}}=16$)의 KV 바이트 비례값: $2\cdot24\cdot16\cdot8192\cdot64\cdot2$
2. GQA($H_{\mathrm{kv}}=4$)로 바꾸면 비율은?
3. Prefill 한 번의 Attention $T^2 C$ 항이 GQA로 **연산량이 $1/4$이 되나?**

힌트:

1. 약 $1.03\times 10^{9}$ bytes $\approx 1\,\mathrm{GB}$ 수준(설명용).
2. $4/16 = 1/4$로 KV 메모리 항만 축소.
3. **아니요.** Prefill의 $QK^\top$는 여전히 모든 query 위치×key 위치. GQA는 주로 **저장·대역폭(캐시)** 쪽 이득이 분명하고, prefill FLOPs 이야기는 더 미묘하다(공유여도 score는 $H_q$개).

## 수식 보강 — Attention 복잡도 · 근사

표준 self-attention의 점수 계산은 시퀀스 길이 $T$에 대해

$$
\mathrm{FLOPs}\sim O(T^2 d)
$$

입니다. KV cache 메모리(대략)는

$$
\mathrm{Mem}_{\mathrm{KV}} \approx 2 \cdot L \cdot T \cdot d \cdot b
$$

($L$층, $b$는 바이트/원소, multi-head면 $d$ 대신 $h\cdot d_h$).

선형 Attention·SSM 계열은 점수 $T^2$를 $O(T)$에 가깝게 줄이는 것이 목표입니다. 정확한 상수는 구현·하드웨어에 따라 달라 **임의 벤치마크 숫자는 적지 않습니다**.

## LLM에서는 어디에 사용될까?

이번 53강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- 최신 변형은 대개 $T^2$·KV 메모리·규모 비용에 대한 응답이다.
- MQA/GQA는 K/V 공유로 캐시 쪽 부담을 줄이려는 구조 변형이다. $\mathrm{Mem}_{\mathrm{KV}}$ 비율은 $H_{\mathrm{kv}}/H_q$.
- FlashAttention류는 exact attention의 IO-aware 구현 방향이다(세부 변동).
- MoE는 전문가 $E$개·활성 $k$개로 **용량과 토큰당 연산**을 분리하는 티저다.
- 벤치 수치·채택 현황은 시점에 따라 변하므로 개요 이상으로 단정하지 않는다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| MQA | Query 다중, K/V 공유가 강한 Attention |
| GQA | Query 그룹마다 K/V를 공유하는 절충형 |
| KV cache pressure | 생성 시 저장·대역폭 부담 |
| FlashAttention (idea) | $T\times T$ IO를 줄이는 exact attention 구현 계열 |
| MoE | 라우팅으로 전문가 FFN 일부만 활성화 |
| Exact vs approximate | 수학적 동일성 유지 여부 |

## 연습문제
### 문제 1 (사실)

MQA와 표준 MHA의 구조적 차이를 K/V 관점에서 쓰시오.

### 문제 2 (설명)

GQA가 MQA와 MHA의 “사이”라고 불리는 이유를 쓰시오.

### 문제 3 (구분)

“FlashAttention은 항상 2배 빠르다”가 이 강의 규칙상 부적절한 이유를 쓰시오.

### 문제 4 (연결)

제52강 KV memory 비례식과 GQA의 연결 고리를 한 문장으로 쓰시오.

### 문제 5 (지도)

MoE가 바꾸는 블록이 Attention인지 FFN인지 고르고, 이유를 한 줄로 쓰시오.

### 문제 6 (수식)

$H_q=32$, $H_{\mathrm{kv}}=8$일 때 GQA의 KV 메모리 비율(대 MHA)은?

### 문제 7 (MoE)

$E=8$, $k=2$일 때 토큰당 활성 전문가 비율과, 파라미터(전문가 저장) 배율을 각각 한 줄로 쓰시오.

---

## 정답 및 해설
### 문제 1

MHA는 헤드마다 K/V가 있고, MQA는 K/V를 공유(보통 한 세트)한다. Q 헤드는 여러 개일 수 있다.

### 문제 2

K/V 헤드 수를 1(MQA)과 $H$(MHA) 사이의 그룹 수로 두어 공유 강도를 조절하기 때문이다.

### 문제 3

이득은 길이·하드웨어·구현·버전에 의존하는 **변동 세부**이며, 개요 강의에서 고정 배수를 단정하면 안 된다.

### 문제 4

KV 메모리가 헤드 수에 비례하므로 K/V 헤드를 줄이는 GQA/MQA는 캐시 크기를 줄이려는 방향과 맞닿는다.

### 문제 5

FFN(전문가 MLP). 라우터가 토큰별로 어느 MLP를 실행할지 고른다.

### 문제 6

$8/32 = 1/4$.

### 문제 7

활성 비율 $2/8=1/4$. 저장(전문가 가중치)은 이상적으로 약 8배(공유 비전문가 모듈 제외).

## 다음 강의와 연결
변형의 지도까지 보았다. **제54강. 2권 총정리 — GPT로 가는 길**에서 Tokenizer부터 Mini Transformer까지를 한 줄로 다시 잇고, 3권 제55강(GPT란 무엇인가)으로 넘어갈 체크리스트를 만든다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [52강. 계산 복잡도와 메모리](52강_계산_복잡도와_메모리.md)
- **다음 강:** [54강. 2권 총정리 — GPT로 가는 길](54강_2권_총정리_GPT로_가는_길.md)

<!-- /LECTURE_NAV -->
