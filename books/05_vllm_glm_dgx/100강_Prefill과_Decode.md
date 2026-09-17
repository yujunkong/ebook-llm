# 100강. Prefill과 Decode
## 이번 강에서 배우는 내용

- Prefill과 Decode의 정의·입력·출력·병목 감각
- 왜 프롬프트 처리와 토큰 생성이 같은 forward라도 시스템적으로 다른지
- TTFT가 Prefill과 어떻게 연결되는지（정의 수준; 수치 단정 없음）
- TPOT / 토큰 간 지연이 Decode와 어떻게 연결되는지
- KV Cache가 Decode를 가능하게 하는 위치（제101강 예고）
- Continuous Batching이 두 국면을 섞어 스케줄하는 이유（제102강 예고）

## 왜 중요한가?
사용자 체감은 대개 두 문장으로 갈린다.

```text
“답이 언제 처음 보이지?”     → 첫 토큰까지의 시간
“그 다음부터 글이 얼마나 매끄럽게 이어지지?” → 토큰이 이어지는 속도
```

이 두 감각을 하나의 “모델이 느리다”로 뭉개면 최적화가 헛방향으로 간다.

| 증상（설명） | 먼저 의볼 국면 |
|---|---|
| 긴 문서 붙여넣기 후 한참 침묵 | Prefill |
| 첫 글자는 빨리 나오는데 이후가 끊김 | Decode / 스케줄 / 동시성 |
| 동시 사용자가 늘수록 둘 다 악화 | 배치·KV 메모리·스케줄러 |

**해석:** Prefill/Decode를 모르면 vLLM·PagedAttention·양자화 이야기를 “마법”으로 듣게 된다. 국면이 보여야 도구가 보인다.

## 선수 개념
1. **Autoregressive generation** — 제58강. $y_t \sim \pi(\cdot\mid x,y_{<t})$.
2. **Self-Attention / QKV** — 제36~41강. 새 토큰의 Q가 과거 K,V와 만난다.
3. **Causal mask** — 제40강. 미래 토큰을 보지 않는다.
4. **Training vs Inference** — 제99강. 추론은 보통 forward-only.
5. **복잡도 감각** — 제52강. 순진한 재계산 vs 캐시.

아직 Continuous Batching·PagedAttention 구현은 열지 않는다.

## 핵심 개념
### 3.1 Prefill이란?

**Prefill(프리필)**은 요청의 **프롬프트（입력） 토큰 전체**를 모델에 넣어, 필요한 중간 상태（특히 각 층의 Key/Value）를 준비하고 **첫 생성 토큰의 logits**까지 만드는 국면이다다.

```text
입력:  prompt token ids  [B, L]  (B는 이 순간 함께 처리하는 요청 수)
처리:  L개 토큰을 (보통) 한 번의 forward로 병렬 처리
출력:  다음 토큰 logits + (캐시 사용 시) KV cache 초기 구간
```

교사강제 학습의 “한 번에 길이 $L$ forward”와 **계산 모양**은 닮았다.  
차이는 목표가 loss가 아니라 **생성 준비**라는 점이다.

### 3.2 Decode란?

**Decode(디코드)**는 Prefill 이후, **새 토큰을 하나씩（또는 speculative로 묶음）** 이어 붙이며 logits → 샘플/argmax → append를 반복하는 국면이다다.

```text
입력:  직전 토큰 (보통 길이 1) + 과거 KV
처리:  새 토큰의 Q,K,V 계산 + 과거 KV와 Attention
출력:  다음 토큰 logits, 갱신된 KV
```

**사실:** 스펙큘러티브 디코딩 등은 “한 스텝에 후보 여러 개”를 다루지만, 기본 서빙 이야기에서는 **토큰 단위 연장**이 중심이다（제110강에서 확장）.

### 3.3 한 요청의 수명

```text
요청 도착
  │
  ▼
[Prefill]  프롬프트 L 토큰 처리 → KV[0..L) 준비 → 첫 logits
  │
  ▼
첫 토큰 방출   ←── 사용자가 “응답 시작”을 느끼는 지점（TTFT와 관련）
  │
  ▼
[Decode]  토큰 1개씩 반복 … EOS 또는 max_new_tokens
  │
  ▼
요청 종료 · KV 해제（엔진이 관리）
```

### 3.4 용어를 헷갈리지 않기

| 이름 | 이 강의에서의 뜻 | 혼동 주의 |
|---|---|---|
| Decode | 생성 국면（토큰 연장） | “디코더 전용 Transformer”의 decoder와 동음이의 |
| Prefill | 프롬프트 일괄 처리 | “데이터를 디스크에 미리 채움” 일반 IT 용어와 구분 |
| Prompt processing | Prefill과 거의 동의로 쓰이기도 함 | 문서마다 표기 차이 가능 |

**사실:** 엔진·논문 문서에 `prompt phase` / `generation phase` 등으로 쓰이기도 한다. 개념은 같다.

## 직관적으로 이해하기
도서관 비유（**설명**）:

- **Prefill:** 참고문헌 더미（긴 프롬프트）를 한 번에 읽고, 색인 카드（KV）를 만들어 책상에 펼친다.
- **Decode:** 이미 펼친 색인을 보면서 **한 문장씩** 답을 써 나간다. 새 문장마다 색인에 카드 한 장만 추가한다.

색인 없이 매 문장마다 참고문헌 전체를 다시 읽으면 Decode가 폭발한다 → KV Cache（제101강）.

식당 비유:

- Prefill: 주문지（긴 주문서）를 주방에 전달·준비
- Decode: 접시를 **한 장씩** 내는 코스 요리

손님이 “첫 접시가 언제 나오나”와 “이후 접시 간격”을 따로 느낀다.

## 수학적으로 이해하기
### 5.1 Prefill의 Attention 모양

프롬프트 길이 $L$, 헤드 차원 $d$라 하자. 한 헤드에서

$$

Q,K,V \in \mathbb{R}^{L \times d}

$$

$$

\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}+M\right)V

$$

$QK^\top$는 $L\times L$이다.  
**사실:** 표준 dense Attention은 프롬프트 길이에 대해 제곱 항을 갖는다.  
**설명:** 긴 프롬프트일수록 Prefill 비용·메모리가 커지기 쉽다.

### 5.2 Decode 한 스텝

이미 $t$개 토큰（프롬프트+생성）의 $K_{1:t}, V_{1:t}$가 있다고 하자. 새 토큰 하나로

$$

q_{t+1}, k_{t+1}, v_{t+1} \in \mathbb{R}^{d}

$$

을 만든 뒤

$$

\mathrm{Attn}=\mathrm{softmax}\left(\frac{q_{t+1} K_{1:t+1}^\top}{\sqrt{d}}\right)V_{1:t+1}

$$

를 계산한다.  
점수 벡터 길이는 $t+1$이다. **새 행/열 전체를 $L\times L$로 다시 만들 필요가 없다** — 과거 $K,V$를 재사용할 때.

### 5.3 국면별 FLOPs 감각（단정 금지）

대략:

| 국면 | Attention 관련 감각 |
|---|---|
| Prefill | $O(L^2 d)$ 규모가 한 번에 |
| Decode 1 step | $O(t\, d)$ 규모（과거 길이에 선형） |

층·헤드·MLP를 곱하면 전체가 된다.  
**설명:** Prefill은 토큰 병렬도가 커 **compute-bound에 가까운 이야기**가 나오기 쉽고, Decode는 토큰 1개라 **메모리에서 가중치·KV를 읽는 비용**이 두드러지는 이야기가 나오기 쉽다.  
**사실:** 실제 바운드는지정 모델·GPU·커널·동시성에 따라 측정해야 한다. 이 책은 “항상 Prefill=compute, Decode=memory”를 만능 법칙으로 주장하지 않는다.

## 작은 숫자로 직접 계산하기
> 숫자는 **가정**이다.

가정: $L=8$ 프롬프트, 생성 $N=4$ 토큰, KV 사용.

| 단계 | 처리하는 “새” 토큰 수 | Attention이 보는 과거 길이（끝 시점） |
|---|---|---|
| Prefill | 8 | 8（한 번에） |
| Decode 1 | 1 | 9 |
| Decode 2 | 1 | 10 |
| Decode 3 | 1 | 11 |
| Decode 4 | 1 | 12 |

순진한 재계산이면 Decode $k$번째에 길이 $8+k$ 전체를 다시 넣는다.  
KV를 쓰면 Decode 입력이 보통 길이 1이다.

**연습:** $L=128$, $N=64$일 때 순진한 방식에서 forward가 보는 길이의 합

$$

\sum_{k=1}^{64}(128+k)

$$

을 계산해 보고, Prefill 1회 + Decode 64회（길이 1）와 대조한다.  
합만으로 속도를 예측하지 말 것.

## 코드로 구현하기
### 7.1 국면을 드러내는 의사코드

```python
def prefill(model, prompt_ids):
    """프롬프트 전체 → logits(마지막 위치) + KV."""
    logits, kv = model.forward_all(prompt_ids)  # 교육용 API
    return logits[:, -1, :], kv

def decode_step(model, token_id, kv):
    """토큰 1개 + KV → 다음 logits + 갱신 KV."""
    logits, kv = model.forward_one(token_id, kv)
    return logits[:, -1, :], kv

def generate(model, prompt_ids, max_new=32, eos_id=None):
    logits, kv = prefill(model, prompt_ids)
    out = []
    for _ in range(max_new):
        next_id = int(logits.argmax(dim=-1))  # greedy 예시
        out.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        logits, kv = decode_step(model, next_id, kv)
    return out
```

### 7.2 PyTorch — 캐시 없이 Prefill/Decode를 “시간으로만” 구분

캐시 API 없이, **호출 패턴**만 분리해 본다.

```python
import torch
import torch.nn as nn

class TinyCausal(nn.Module):
    """교육용: 임베딩 + 선형. Attention/KV는 생략."""
    def __init__(self, vocab=64, d=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.lm = nn.Linear(d, vocab)

    def forward(self, ids):
        # ids: [B, T]
        return self.lm(self.emb(ids))

def greedy_generate(model, prompt, max_new=8):
    model.eval()
    ids = prompt.clone()
    with torch.no_grad():
        # --- Prefill에 해당하는 첫 forward ---
        logits = model(ids)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)

        # --- Decode: 매 스텝 전체 ids를 다시 넣음(순진) ---
        for _ in range(max_new - 1):
            logits = model(ids)  # 재계산
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if int(next_id) == 0:  # 가정: 0 = EOS
                break
    return ids

model = TinyCausal()
prompt = torch.randint(1, 64, (1, 12))
print(greedy_generate(model, prompt).shape)
```

이 코드는 KV를 쓰지 않지만, **첫 forward（프롬프트）**와 **이후 루프**를 주석으로 나눈다.  
제101강에서 `past_key_values` 형태로 연결한다.

### 7.3 로깅으로 국면 표시

```python
import time

def timed_generate(model, prompt_ids, max_new=16):
    times = {}
    t0 = time.perf_counter()
    logits, kv = prefill(model, prompt_ids)
    times["prefill_s"] = time.perf_counter() - t0

    decode_dt = []
    token = sample(logits)
    for _ in range(max_new - 1):
        t1 = time.perf_counter()
        logits, kv = decode_step(model, token, kv)
        token = sample(logits)
        decode_dt.append(time.perf_counter() - t1)
        if is_eos(token):
            break

    times["decode_steps"] = decode_dt
    # 평균·합은 이 환경의 관찰일 뿐, 책에 일반화하지 않는다.
    return times
```

**주의:** `time.perf_counter()` 결과는 머신·부하에 의존한다. 공유·비교용 벤치마크 숫자로 제시하지 말 것.

## 수식 보강 — Prefill / Decode 비용 감각

입력 길이 $T_{\mathrm{in}}$, 출력 길이 $T_{\mathrm{out}}$일 때

- Prefill: 대략 $O(T_{\mathrm{in}}^2 d)$ 성격의 Attention + MLP (한 번에 병렬)
- Decode: 스텝마다 $O(T_{\mathrm{ctx}} d)$ (KV cache 사용 시), $T_{\mathrm{out}}$번 반복

총 decode 비용 스케치:

$$
\mathrm{Cost}_{\mathrm{decode}} \sim T_{\mathrm{out}}\cdot O(T_{\mathrm{ctx}} d)
$$

$T_{\mathrm{ctx}}$는 해당 스텝의 문맥 길이입니다.

## 정량 스케치 — 국면별 비용

Prefill Attention: $\mathrm{FLOPs}\propto L H S^2 d$ ($S^2$ 감각).

Decode 스텝(캐시 길이 $t$): $\propto L H t d$.

KV: $\mathrm{Bytes}_{\mathrm{KV}}(t)\approx 2 L H_{kv} d t b$.

$$

\mathrm{TTFT}\approx T_{\mathrm{queue}}+T_{\mathrm{prefill}},\quad
\mathrm{TPOT}\approx\mathrm{mean}(T_{\mathrm{decode}})
$$

총지연 감각 $\approx\mathrm{TTFT}+(N_{\mathrm{out}}-1)\mathrm{TPOT}$.

| 국면 | 전형 감각 | 힌트 |
|---|---|---|
| Prefill | compute 여지 | 커널·양자화·병렬 |
| Decode | memory/bandwidth | KV·배치 |

가정 숫자로 용량 계획하지 말 것 — 비례 연습만.


## 스케줄러가 국면을 나누는 이유（정량 감각）

한 iteration에 prefill 토큰 $S_{\mathrm{sum}}$과 decode 시퀀스 $B_{\mathrm{dec}}$가 섞이면, 작업 프로필이 이질적이다.

```text
Prefill-heavy chunk → SM compute·큰 GEMM
Decode-heavy chunk → KV bandwidth·작은 GEMM
```

지표를 국면 없이 평균하면 TTFT/TPOT 진단이 흐려진다. 로그에 `phase=prefill|decode|mixed`를 남긴다.

### 길이 레버

| 늘리는 것 | 먼저 맞는 지표 |
|---|---|
| 프롬프트 $S$ | TTFT |
| 출력 $N_{\mathrm{out}}$ | 총지연·TPOT×길이 |
| 동시성 $B$ | KV 메모리·스케줄 대기 |


## LLM에서는 어디에 사용될까?
### 8.1 TTFT와 Prefill

**TTFT(Time To First Token, 첫 토큰 시간)**는 요청이 도착한 뒤 **첫 출력 토큰**이 나올 때까지의 지연이다.

**설명:** 많은 시스템에서 TTFT의 큰 몫이 Prefill（+ 큐 대기）과 관련된다.  
**사실:** 정확한 정의（클라이언트 수신 시각 vs 서버 enqueue 시각 등）는 제품·측정 도구마다 다를 수 있다. 제107강에서 지표를 정리한다.

### 8.2 TPOT / 토큰 간 지연과 Decode

**TPOT(Time Per Output Token)**류 지표는 출력 토큰이 이어지는 속도를 본다.  
**설명:** Decode 루프·동시 요청 수·스케줄 정책이 여기에 強く 영향을 준다.

### 8.3 왜 스케줄러가 두 국면을 구분하는가

Continuous Batching（제102강）에서는 같은 GPU iteration에

- 어떤 요청은 Prefill 중
- 어떤 요청은 Decode 중

이 섞일 수 있다.  
국면마다 연산 모양이 달라 **한 방에 같은 커널만** 돌리기 어렵다. 엔진이 예약을 나누는 이유다.

### 8.4 프롬프트가 길수록

| 변화 | Prefill | Decode |
|---|---|---|
| $L$ ↑ | 비용·TTFT 압력 ↑（경향） | 시작 KV가 커짐 → 이후 Decode도 과거 길이 ↑ |
| $N$ ↑ | （직접）작음 | 스텝 수 ↑ |

긴 시스템 프롬프트·RAG 문서는 Prefill을 키운다.  
긴 CoT 답변은 Decode를 키운다（제98강 매핑）.

## 실습
### 실습 A — 수명 그리기

자신의 말로 시퀀스 다이어그램을 그린다.

```text
prompt → Prefill → first token → Decode loop → EOS
```

각 화살표 아래에 “KV가 생기는지/늘어나는지”를 표시한다.

### 실습 B — 합 계산（가정）

$L=256$, $N=128$일 때

$$

S=\sum_{k=1}^{N}(L+k)

$$

를 계산한다. KV 사용 시 “Prefill 토큰 수 + Decode 스텝 수”와 비교하는 한 문장을 쓴다.  
$S$를 벤치마크라고 부르지 말 것.

### 실습 C — 코드에 국면 주석 달기

제58강식 generate 루프를 가져와 `# PREFILL` / `# DECODE` 주석을 강제한다.  
캐시가 없으면 Decode가 사실상 “전체 재Prefill에 가깝다”는 문장을 코드 주석에 남긴다.

### 실습 D — 증상 분류

다음 증상을 Prefill / Decode / Queue / 기타로 분류한다（정답은 단정이 아니라 **가설**）.

1. 시스템 프롬프트 10배 연장 후 “첫 응답이 늦다”
2. 짧은 질문인데 답이 매우 긴 추론만 느리다
3. 사용자를 8명으로 늘리니 모두가 일제히 느리다

## 자주 하는 실수
1. **Prefill과 Decode를 하나의 “forward 시간”으로만 측정**  
   최적화가 어디에 필요한지 안 보인다.

2. **항상 Prefill이 더 무겁다 / 항상 Decode가 더 무겁다**고 단정  
   $L$, $N$, 동시성, 커널에 따라 주인공이 바뀐다.

3. **Decoder-only 모델의 “Decoder”와 Decode 국면을 동일시**  
   전자는 아키텍처 이름, 후자는 서빙 국면이다다.

4. **TTFT만 보고 제품이 빠르다고 선언**  
   긴 답변에서는 Decode가 체감을 지배할 수 있다.

5. **패딩된 긴 배치를 Prefill 효율로 착각**  
   짧은 요청에 긴 패딩을 붙이면 GPU는 바쁘지만 유용 토큰은 적다（제102강）.

6. **벤치 숫자 암기**  
   공개 블로그의 “2× faster”를 맥락 없이 인용하지 말 것.

## 핵심 요약
- Prefill은 프롬프트를 일괄 처리해 KV·첫 logits를 준비한다.
- Decode는 토큰을 이어 붙이며 KV를 갱신한다.
- 사용자 체감의 “첫 응답”과 “이후 속도”는 국면이 다르다.
- 순진한 재계산 Decode는 Prefill을 반복하는 셈이다.
- 서빙 최적화·스케줄·지표는 이 이분법 위에 쌓인다.

## 용어 사전
| 용어 | 한 줄 의미 |
|---|---|
| Prefill | 프롬프트 일괄 처리 국면 |
| Decode | 토큰 단위（또는 묶음）생성 국면 |
| Prompt tokens | 모델에 먼저 넣는 입력 토큰 |
| Output / completion tokens | Prefill 이후 생성되는 토큰 |
| TTFT | 첫 토큰까지의 시간 |
| TPOT | 출력 토큰당 시간（정의는 측정 체계에 따름） |
| KV Cache | Decode가 과거 K,V를 재쓰는 저장소 |
| Compute-bound / Memory-bound | 연산량 vs 메모리 대역이 지배하는 상태（측정 필요） |

## 연습문제
### 문제 1（정의）

Prefill과 Decode를 입력·출력 관점에서 한 줄씩 정의하시오.

### 문제 2（수명）

한 요청 수명에서 KV가 처음 크게 만들어지는 국면과, 토큰마다 늘어나는 국면을 구분하시오.

### 문제 3（복잡도）

Decode 한 스텝에서 과거 길이가 $t$일 때, Attention 점수 벡터의 길이 감각을 쓰시오.

### 문제 4（체감）

“첫 글자가 안 나온다”와 “글자가 끊긴다”를 각각 어느 국면·지표와 연결할지 쓰시오.

### 문제 5（순진한 구현）

KV 없이 `model(전체 ids)`를 매 스텝 호출하면 Decode가 사실상 무엇을 반복하는가?

### 문제 6（사실/설명）

“Decode는 항상 memory-bound다”를 사실로 쓸 수 있는지 판단하고, 이유를 쓰시오.

### 문제 7（다리）

제101강 제목을 쓰고, Prefill이 남기는 것이 무엇인지 한 단어군으로 답하시오.

---

## 정답 및 해설
### 문제 1

Prefill: 프롬프트 전체 → 첫 생성용 logits +（보통）KV 초기화.  
Decode: 새 토큰(+KV) → 다음 logits + KV 갱신.

### 문제 2

처음 크게: Prefill. 토큰마다 증가: Decode.

### 문제 3

대략 길이 $t$（또는 $t+1$, 새 키 포함 정의에 따름）.

### 문제 4

첫 글자: Prefill·큐와 연관된 TTFT 쪽.  
끊김: Decode·동시성·TPOT 쪽（가설 수준으로 서술）.

### 문제 5

늘어난 컨텍스트 전체에 대한 forward（Prefill에 가까운 재계산）를 반복한다.

### 문제 6

단정할 수 없다. 하드웨어·모델·동시성·구현에 따라 측정해야 하며, 강의의 “경향 설명”일 뿐이다.

### 문제 7

제목: KV Cache. Prefill이 남기는 것: 각 층의 Key/Value（KV）상태.

## 다음 강의와 연결
국면이 보였다.  
다음 **제101강. KV Cache**에서는 Decode가 다시 읽어야 하는 **Key/Value를 어디에·얼마나·어떤 수식으로** 저장하는지 파고든다. Prefill이 “캐시를 채우는 일”이고 Decode가 “캐시를 쓰며 한 칸씩 늘리는 일”임을 정식화한다.

> 두 국면을 나눈 이유는, 그 사이에 놓인 메모리 구조가 서빙의 주인공이기 때문이다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [99강. Training과 Inference의 차이](99강_Training과_Inference의_차이.md)
- **다음 강:** [101강. KV Cache](101강_KV_Cache.md)

<!-- /LECTURE_NAV -->
