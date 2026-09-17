# 101강. KV Cache
## 이번 강에서 배우는 내용

- KV Cache가 무엇을 저장하고 왜 필요한지
- Prefill이 캐시를 채우고 Decode가 캐시를 읽고 덧붙이는 흐름
- 메모리 양을 기호로 세는 식（층·헤드·차원·길이·dtype·요청 수）
- 컨텍스트 길이·동시 요청이 늘 때 무엇이 먼저 터지는지（OOM 감각）
- 정적 텐서 캐시와 페이지 단위 관리（PagedAttention, 제105강） 가 왜 이어지는지
- 사실（정의·식의 구조）과 설명（경향）과 해석（설계 선택）을 구분

## 왜 중요한가?
서빙에서 가중치만 보면 반은 맞다. 나머지 반은 종종 **살아 있는 대화의 컨텍스트**다.

```text
가중치: 모든 요청이 공유（보통）
KV:     요청(또는 시퀀스)마다 따로 성장
```

동시 사용자가 늘거나 컨텍스트가 길어지면, 가중치는 그대로인데 KV만으로 메모리가 가득 찰 수 있다.  
Continuous Batching（제102강）·vLLM（제104강）·PagedAttention（제105강）은 모두 **이 성장하는 상태**를 전제로 한다.

**해석:** KV를 모르면 “모델 크기만 줄이면 동시성이 선형으로 오른다”는 환상이 생긴다.

## 선수 개념
1. **Q, K, V** — 제36강. Attention은 $Q$가 $K$와 유사도를 내고 $V$를 가중합한다.
2. **Causal Self-Attention** — 제40·48강. 위치 $t$는 $1..t$만 본다.
3. **Prefill / Decode** — 제100강.
4. **복잡도** — 제52강. 순진한 재계산의 $O(T^3)$ 감각.
5. **dtype 바이트** — 제64강. FP16/BF16≈2B, FP32≈4B 등.

## 핵심 개념
### 3.1 KV Cache란?

**KV Cache**는 Autoregressive 생성 중 각 Transformer 층·헤드에서 이미 계산한 **Key와 Value 텐서**를 보관해, 이후 스텝에서 재사용하는 메모리 구조다.

```text
매 토큰마다:
  새 x_t → (Q_t, K_t, V_t)
  K_cache ← concat(K_cache, K_t)
  V_cache ← concat(V_cache, V_t)
  Attn ← softmax(Q_t K_cache^T / √d) V_cache
```

**Query는 보통 캐시하지 않는다.**  
이유는 직관적이다: 새 토큰의 $Q$만 있으면 되고, 과거 위치의 $Q$는 과거 출력에 이미 쓰였다.

### 3.2 왜 K와 V만인가

위치 $i$의 출력은

$$

o_i=\sum_{j\le i}\alpha_{ij}\,v_j,\quad
\alpha_{ij}\propto \exp(q_i^\top k_j/\sqrt{d})

$$

에 의존한다. 미래 토큰 $t>i$가 와도 **과거 $k_j,v_j$는 바뀌지 않는다**（표준 Self-Attention, 고정 가중치, 같은 RoPE 적용이 이미 반영된 표현을 저장하는 경우）.

**설명:** RoPE를 언제 적용해 저장하는지는 구현마다 디테일이 다르다. “K/V에 위치를 어떻게 굽는지”는 엔진·모델 코드를 따라야 한다. 이 강의는 **재사용 가능성**에 초점을 둔다.

### 3.3 Prefill과의 관계

| 국면 | KV에 하는 일 |
|---|---|
| Prefill | 프롬프트 $L$토큰의 K/V를 한 번에 채워 넣음 |
| Decode | 길이 1（또는 소수）K/V를 append |

Prefill 없이 Decode만 있는 시스템은 없다（프롬프트 길이 0은 예외적）.  
**사실:** 빈 프롬프트·특수 시작 토큰만 있는 API도 있지만, 개념적으로는 짧은 Prefill이다.

### 3.4 무엇이 “캐시 히트”인가

전통 CPU 캐시와 비유하면:

- **히트:** 과거 $K,V$를 메모리에서 읽어 재사용
- **미스에 해당하는 낭비:** 과거 토큰을 다시 embed→층 전체 재계산

다만 LLM KV는 CPU L1 같은 하드웨어 캐시가 아니라, **명시적으로 할당하는 텐서 버퍼**에 가깝다.

## 직관적으로 이해하기
칠판 비유（**설명**）:

- 매 문장마다 앞 문장 전체를 다시 받아 적지 않는다.
- 이미 받아 적은 **밑줄（K）과 요약 카드（V）** 를 책상 위에 남겨 둔다.
- 새 문장은 새 밑줄·카드만 추가한다.

책상이 가득 차면（VRAM 한도）:

- 새 손님（요청）을 받지 못하거나
- 오래된 대화를 치우거나（evict）
- 책상 배치를 조각내지 않게 정리한다（페이지 관리 → 제105강）

## 수학적으로 이해하기
### 5.1 단일 요청·단일 층·단일 헤드

길이 $t$, 헤드 차원 $d$일 때

$$

K,V \in \mathbb{R}^{t \times d}

$$

저장량（원소 수）은 $2td$.

### 5.2 Multi-Head · Multi-Layer

헤드 수 $H$, 층 수 $L_{\mathrm{layer}}$（기호 충돌을 피해 층은 $n_{\ell}$로도 쓴다）.  
흔히 층당

$$

K,V \in \mathbb{R}^{H \times t \times d}

$$

또는 동등하게 $t \times (H d)$로 합쳐 저장한다.

전체 원소 수 감각:

$$

N_{\mathrm{elem}} \approx 2 \cdot n_{\ell} \cdot H \cdot t \cdot d

$$

### 5.3 GQA / MQA（미리 보기）

일부 모델은 Key/Value 헤드 수가 Query 헤드보다 적다.

- **MHA:** $H_k=H_q$
- **GQA(Grouped Query Attention):** $H_k < H_q$（그룹）
- **MQA(Multi-Query Attention):** $H_k=1$에 가까움

**사실:** GQA/MQA는 모델 아키텍처 선택이다.  
**설명:** KV 헤드가 줄어들면 **같은 $t$에서 KV 메모리·대역폭**이 줄어들 수 있다. 품질·속도 트레이드오프는 모델별로 평가한다（제108강 계열 이야기와 연결 가능）.

식에서는 $H$ 대신 $H_{kv}$를 쓴다.

$$

N_{\mathrm{elem}} \approx 2 \cdot n_{\ell} \cdot H_{kv} \cdot t \cdot d

$$

### 5.4 바이트로 환산

원소당 바이트를 $b$라 하면（예: FP16/BF16이면 가정 $b=2$）

$$

\mathrm{Bytes}_{\mathrm{KV}} \approx N_{\mathrm{elem}} \cdot b

$$

동시 시퀀스（요청）가 $B$개이고 길이가 각각 $t_i$면

$$

\mathrm{Bytes}_{\mathrm{KV}} \approx \sum_{i=1}^{B} 2\, n_{\ell}\, H_{kv}\, t_i\, d\, b

$$

**사실:** 위는 **이상화된 하한 감각**에다. 실제 엔진은 정렬·블록·여유·워크스페이스를 더 둔다.

### 5.5 컨텍스트가 2배면

모든 $t_i$가 2배이고 다른 값이 고정이면, 이상적 KV 바이트도 약 2배.  
**설명:** “컨텍스트 윈도우 광고”는 가중치뿐 아니라 **KV 예산**과 함께 읽어야 한다.

## 작은 숫자로 직접 계산하기
> 전부 **가정**이다. 특정 공개 모델의 공식 스펙 인용처럼 읽지 말 것.

가정:

- $n_{\ell}=32$ 층
- $H_{kv}=8$ （GQA를 가정）
- $d=128$
- $t=4096$
- $b=2$ （FP16 가정）
- 요청 $B=1$

원소 수:

$$

N=2\cdot 32\cdot 8\cdot 4096\cdot 128

$$

단계적으로:

$$

8\cdot 128=1024

$$

$$

1024\cdot 4096=4{,}194{,}304

$$

$$

32\cdot 4{,}194{,}304=134{,}217{,}728

$$

$$

N=2\cdot 134{,}217{,}728=268{,}435{,}456

$$

바이트:

$$

268{,}435{,}456 \times 2 \approx 5.12\times 10^8 \ \text{B} \approx 0.51\ \text{GiB 감각}

$$

（$1024^3$으로 나누는 GiB 환산은 관례에 따라 표기가 달라질 수 있음. **정확한 제품 메모리가 아님**.）

이제 $B=8$이고 모두 $t=4096$이면 이상적으로 약 8배.  
가중치가 예를 들어 “수십 GB”인 모델（가정）이어도, **동시 긴 문맥**이 겹치면 KV가 무시 못 할 수 있다.

**실습 확장:** $t=8192$로만 바꾸면 위 단일 요청 KV는 약 2배. 식을 다시 풀어 확인할 것.

## 코드로 구현하기
### 7.1 NumPy로 “한 층” KV append

```python
import numpy as np

rng = np.random.default_rng(0)
d = 4
t_prompt = 3

# Prefill: 프롬프트 K,V
K = rng.normal(size=(t_prompt, d))
V = rng.normal(size=(t_prompt, d))

def decode_append(K, V, k_new, v_new):
    K = np.concatenate([K, k_new[None, :]], axis=0)
    V = np.concatenate([V, v_new[None, :]], axis=0)
    return K, V

def attn_one(q, K, V):
    # q: (d,), K,V: (t,d)
    scores = K @ q / np.sqrt(d)          # (t,)
    scores = scores - scores.max()       # 안정화
    a = np.exp(scores)
    a = a / a.sum()
    return a @ V                         # (d,)

# Decode 두 스텝
for step in range(2):
    q = rng.normal(size=(d,))
    k = rng.normal(size=(d,))
    v = rng.normal(size=(d,))
    K, V = decode_append(K, V, k, v)
    out = attn_one(q, K, V)
    print(f"step {step}: cache_len={len(K)}, out_norm={np.linalg.norm(out):.3f}")
```

관찰:

- `cache_len`이 4, 5로 늘어난다
- 매 스텝 `K,V` 전체를 재샘플하지 않는다

### 7.2 “순진 재계산”과 결과 일치 확인（작은 스케일）

같은 Q/K/V 스트림을 두고, 캐시 방식과 전체 재계산 방식의 출력이 같은지 본다.

```python
def attn_full(Q, K, V):
    # Q,K,V: (t,d) — 마지막 위치 출력만
    q = Q[-1]
    scores = K @ q / np.sqrt(d)
    scores = scores - scores.max()
    a = np.exp(scores); a = a / a.sum()
    return a @ V

# 미리 만들어 둔 전체 시퀀스
t = 5
Q = rng.normal(size=(t, d))
K_all = rng.normal(size=(t, d))
V_all = rng.normal(size=(t, d))

# 캐시로 t까지 쌓기
K_c = K_all[:1].copy()
V_c = V_all[:1].copy()
for i in range(1, t):
    K_c, V_c = decode_append(K_c, V_c, K_all[i], V_all[i])

out_cache = attn_one(Q[-1], K_c, V_c)
out_full = attn_full(Q, K_all, V_all)
print(np.allclose(out_cache, out_full))
```

**사실:** 수치 오차 범위에서 같아야 한다（이 구현 기준）.  
이것이 KV Cache의 **정확성 계약**이다: 빨라야 할 뿐 아니라, 이상적으로는 **동일 결과**여야 한다.

### 7.3 PyTorch 모듈 스케치

```python
import torch
import torch.nn as nn

class OneHeadAttnWithCache(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.d = d

    def forward(self, x, k_cache=None, v_cache=None):
        # x: [B, T, D]  Prefill이면 T=L, Decode면 T=1
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        # 점수: [B, T, T_total] — 교육용 dense
        scale = self.d ** -0.5
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        # causal mask (간단: T==T_total인 prefills만 정확히 맞음)
        T, Tt = att.size(-2), att.size(-1)
        if T == Tt:
            mask = torch.triu(torch.ones(T, Tt, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float("-inf"))
        w = torch.softmax(att, dim=-1)
        out = torch.matmul(w, v)
        return out, k, v
```

**설명:** Decode($T=1$)에서는 “새 행만” 있으면 되고, 과거 쿼리에 대한 마스크는 이미 과거 출력에 반영되었다.  
위 마스크 분기는 교육용이다. 프로덕션 커널은 fused attention으로 처리한다.

## LLM Serving에서의 KV
### 8.1 요청당 상태

서빙 엔진은 대략 다음을 붙잡고 산다.

```text
Request
  ├─ input tokens / arriving chunks
  ├─ sampling params (temperature, max_tokens, …)
  └─ kv_handle  → 층별 K/V 블록들
```

요청이 끝나면 `kv_handle`을 반환（free）해야 다음 요청이 산다.

### 8.2 정적 contiguous 버퍼의 한계

단순 구현:

```text
미리 [B_max, n_ℓ, H, T_max, d] 를 큼직하게 할당
```

문제:

- 짧은 요청이 $T_{\max}$를 예약하면 **내부 조각 낭비**
- 요청이 끝나도 배치 슬롯이 고정이면 **구멍**이 생긴다
- 길이 천차가 static batch와 상성이 나쁘다

이 한계가 Continuous Batching + **PagedAttention**으로 이어진다（제102·105강）.

### 8.3 메모리 예산 사고방식

운영자가 묻는 질문:

> “이 GPU에 가중치를 올린 뒤, **동시에** 길이 약 $T$짜리 대화를 몇 개 올릴 수 있나?”

이상화:

$$

B_{\text{rough}} \approx \frac{\mathrm{VRAM} - \mathrm{Weights} - \mathrm{Workspace}}{\mathrm{KV\_per\_seq}(T)}

$$

**사실:** Workspace·단편화·CUDA 컨텍스트 때문에 이 식은 **상한이 아닌 거친 감각**이다.  
측정 없이 슬롯 수를 약속하지 말 것.

### 8.4 정밀도

KV를 FP8/INT8로 줄이는 기법도 연구·제품에 존재한다.  
**설명:** 가중치 양자화（제103강）와 별개로, **KV 양자화**는 길이·동시성에 직결된다.  
이 강의에서는 “가능하다” 수준만 표시하고, 수치 이득을 날조하지 않는다.

## 실습
### 실습 A — 기호 채우기

자신의 노트에 식을 적는다.

$$

\mathrm{Bytes} \approx 2 \cdot n_{\ell} \cdot H_{kv} \cdot t \cdot d \cdot b \cdot B

$$

각 기호가 커질 때 Prefill/Decode/동시성에 주는 압력을 한 줄씩 쓴다.

### 실습 B — 가정 계산

가정: $n_{\ell}=40$, $H_{kv}=8$, $d=128$, $b=2$, $t=2048$, $B=4$.  
$\mathrm{Bytes}$ 감각을 계산한다. $t$만 4096으로 바꾸면 몇 배가 되는지 확인한다.

### 실습 C — NumPy 일치 테스트

§8.2의 `allclose`를 실행한다.  
고의로 `K_c`에 잘못된 값을 넣으면 False가 됨을 보고, 캐시 버그가 **속도가 아니라 정답성**을 깨짐을 체감한다.

### 실습 D — 낭비 시나리오

$T_{\max}=8192$ 버퍼를 요청마다 예약하는데 실제 평균 길이가 512인 상황을 가정한다.  
슬롯당 낭비 비율을

$$

1 - \frac{512}{8192}

$$

로 적고, 왜 페이지 단위 할당이 필요한지 한 문장으로 연결한다（제105강）.

### 실습 E — 용어 구분

다음을 구분하는 표를 만든다: Activation（학습 시）, Optimizer state, Weight, KV Cache.

## 자주 하는 실수
1. **Q도 캐시해야 한다고 생각**  
   새 토큰의 Q만 있으면 되는 것이 기본이다.

2. **KV 크기를 가중치와 혼동**  
   파라미터 수 공식과 KV 공식은 다르다.

3. **$t$를 “최대 컨텍스트”로만 보고 평균을 무시**  
   동시 요청의 **실제 상주 길이 합**이 예산을 가른다.

4. **concat마다 새 거대 할당**  
   Python/NumPy 예제는 교육용이다. 엔진은 미리 확보한 블록에 쓴다.

5. **캐시 dtype을 무조건 FP32**  
   보통 추론 가중치와 맞춘 낮은 정밀도를 쓴다（구현 의존）.

6. **벤치 “KV cache로 N배”** 암기  
   $L$,$N$,하드웨어에 따라 달라진다.

7. **요청 종료 후 KV 미반환**  
   서빙에서는 누수 = 곧 OOM.

## 수식 보강 — KV Cache 용량

층 $L$, 헤드 구성상 키/값 채널 합이 $d$에 해당하고, 정밀도 $b$바이트, 시퀀스 $T$, 배치 $B$이면 대략

$$
\mathrm{Mem}_{\mathrm{KV}} \approx 2 \cdot B \cdot L \cdot T \cdot d \cdot b
$$

입니다(구현·GQA에 따라 계수는 달라짐). $T$가 길수록 메모리가 선형으로 늘고, 이게 긴 컨텍스트 서빙의 병목이 됩니다.

## LLM에서는 어디에 사용될까?

이번 101강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- KV Cache는 과거 Key/Value를 저장해 Decode 재계산을 피한다.
- Prefill이 채우고 Decode가 append한다. Q는 보통 캐시하지 않는다.
- 메모리는 $n_{\ell}, H_{kv}, t, d, b, B$에 비례하는 감각이다.
- 동시 긴 문맥에서 KV가 가중치 못지않은 주인공이 될 수 있다.
- 정적 거대 버퍼는 낭비를 낳고, 페이지형 관리로 이어진다.

## 용어 사전
| 용어 | 한 줄 의미 |
|---|---|
| KV Cache | 층별 Key/Value 재사용 버퍼 |
| Prefill fill | 프롬프트로 캐시 초기 구간 기록 |
| Append / grow | Decode마다 K/V 연장 |
| $H_{kv}$ | KV 헤드 수（GQA/MQA에서 감소 가능） |
| Contiguous allocation | 길이 최대로 미리 크게 잡는 방식 |
| Fragmentation | 빈 구멍이 나 재사용이 어려운 상태 |
| PagedAttention | KV를 고정 블록으로 나눠 관리（제105강） |
| KV quantization | K/V를 낮은 비트로 저장하는 기법（개요만） |

## 연습문제
### 문제 1（정의）

KV Cache에 들어가는 것과 들어가지 않는 것（기본）을 쓰시오.

### 문제 2（국면）

Prefill과 Decode가 캐시에 하는 일을 한 줄씩.

### 문제 3（식）

$\mathrm{Bytes}\approx 2 n_{\ell} H_{kv} t d b B$에서 $t$와 $B$가 동시에 2배가 되면 이상적으로 몇 배인가?

### 문제 4（가정 계산）

§7 가정에서 $t=2048$이면 바이트 감각은 약 얼마의 비율로 줄어드는가?

### 문제 5（정확성）

캐시 구현이 빨라졌는데 답이 달라졌다. 가장 먼저 의볼 계약은?

### 문제 6（서빙）

정적 `[T_max]` 예약이 Continuous Batching과 충돌하는 이유를 한 문장으로.

### 문제 7（다리）

제102강 제목을 쓰고, KV가 “요청마다 다른 속도로 커진다”는 사실이 배치에 주는 압력을 한 줄로.

---

## 정답 및 해설
### 문제 1

들어감: K, V. 기본으로 Q는 안 넣음（새 토큰 Q만 계산）.

### 문제 2

Prefill: 프롬프트 구간 K/V를 채움. Decode: 새 토큰 K/V를 append하며 읽기.

### 문제 3

약 4배.

### 문제 4

$t$가 4096→2048이면 이상적으로 약 1/2.

### 문제 5

동일 입력에서 캐시 경로와 풀 재계산 경로의 수치 일치（정확성 계약）.

### 문제 6

요청 길이가 제각각인데 최대 길이로 고정 예약하면 낭비·구멍·동시성 저하가 난다.

### 문제 7

제목: Continuous Batching.  
압력: 배치 구성원이 서로 다른 KV 점유·남은 수명을 가져 static 패딩 배치가 비효율이 된다.

## 다음 강의와 연결
캐시가 **요청마다·시각마다 다르게 자란다**는 것이 핵심이다.  
다음 **제102강. Continuous Batching**에서는 요청이 끝나는 즉시 자리를 비우고, 새 요청을 빈 GPU 시간에 끼워 넣는 **연속 배치**를 다룬다. KV 예산을 빼앗지 않으면서 처리량을 올리는 스케줄의 입구다.

> 메모리가 토큰을 기억한다면, 스케줄러는 그 기억의 자리를 빌려 준다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [100강. Prefill과 Decode](100강_Prefill과_Decode.md)
- **다음 강:** [102강. Continuous Batching](102강_Continuous_Batching.md)

<!-- /LECTURE_NAV -->
