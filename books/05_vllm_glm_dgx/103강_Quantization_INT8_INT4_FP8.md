# 103강. Quantization — INT8, INT4, FP8
## 이번 강에서 배우는 내용

- Quantization의 정의와 Training Mixed Precision（제64강）과의 차이
- INT8 / INT4 / FP8이 각각 무엇을 뜻하는지（비트·표현 감각）
- Weight-only vs Weight+Activation 양자화
- 스케일·제로포인트 등 기본 수식
- 메모리 절감의 이상적 비율과 실제 이득이 어긋날 수 있는 이유（설명）
- 품질·하드웨어 지원·엔진 호환을 함께 보는 체크리스트

## 왜 중요한가?
동일 VRAM에서:

```text
가중치 바이트 ↓  →  더 큰 모델 또는 더 많은 KV/동시성 여유
대역폭 압력 ↓   →  Decode처럼 메모리에 민감한 국면에 도움 여지
```

그러나:

```text
표현력 ↓  →  품질 하락·이상 출력 위험
커널 미지원 →  오히려 느려지거나 로드 실패
교정 부족 →  특정 층만 붕괴
```

**해석:** 양자화는 “공짜 가속”이 아니라 **정확도·호환성·측정을 동반하는 압축**이다.

## 선수 개념
1. **FP32 / FP16 / BF16** — 제64강
2. **파라미터 메모리 감각** — 파라미터 수 × 바이트
3. **KV Cache 예산** — 제101강. 가중치가 줄면 동시성 여유가 생길 **수 있음**
4. **QLoRA / 4-bit** — 제74강（학습 시 저비트 가중치）
5. Prefill/Decode — 제100강. 어느 국면이 대역폭에 민감한지는 **측정** 문제

## 핵심 개념
### 3.1 Quantization이란?

**Quantization(양자화)**은 연속적（또는 높은 비트）값을 **더 적은 비트의 이산 표현**으로 근사하는 것이다.

고수준:

$$

\hat{x} = \mathrm{Dequant}\big(\mathrm{Quant}(x;\theta_{\mathrm{q}})\big)

$$

$\theta_{\mathrm{q}}$는 스케일, 제로포인트, 코드북 등이다.

서빙에서 줄이는 대상의 예:

| 대상 | 동기 |
|---|---|
| Weights | 모델 로드 크기·추론 중 가중치 읽기 |
| Activations | 층간 텐서 대역폭·연산 |
| KV Cache | 긴 문맥·동시성（별 기법으로 다루기도 함） |

### 3.2 Mixed Precision Training과의 차이

| | Mixed Precision（학습, 64강） | Quantization（서빙 중심） |
|---|---|---|
| 목적 | 학습 안정·속도·메모리 | 배포 메모리·지연·비용 |
| 전형 | FP16/BF16 autocast + FP32 마스터 | INT8/INT4/FP8 가중치 등 |
| 역전파 | 있음 | 기본 추론은 없음 |
| 오차 | loss scaling 등으로 관리 | calibration / QAT / 평가로 관리 |

**사실:** 둘 다 “비트 수”를 다루지만 파이프라인·실패 모드가 다르다.  
**설명:** “BF16으로 학습했으니 INT4 서빙도 안전”은 성립하지 않는다.

### 3.3 INT8

**INT8**은 8비트 정수 표현이다. 부호 있으면 대략 $-128..127$ 구간.

가중치 $w$를 구간마다 선형 양자화하는 대표 형태:

$$

w \approx s\cdot(q - z)

$$

- $q$: 정수 코드（INT8）
- $s$: scale（스케일）
- $z$: zero-point（제로포인트, asymmetric일 때）

**설명:** symmetric이면 $z=0$으로 두는 경우가 많다.

### 3.4 INT4

**INT4**는 4비트 정수（16수준）다. 압축률이 더 크지만 표현이 거칠다.

**설명:** 서빙에서 “4-bit LLM”은 종종 **weight-only**（가중치만 4비트, 연산은 높은 정밀도로 복원 후 수행）다.  
**사실:** 구체 포맷（GPTQ, AWQ, GGUF variants 등）은 알고리즘·툴체인이 다르다. 이 강의는 포맷 전쟁 대신 **공통 원리**를 고정한다.

### 3.5 FP8

**FP8**은 8비트 부동소수점 계열이다. 정수 INT8과 달리 **지수·가수**를 나눠 동적 범위를 노린다.

대표적으로 논의되는 형식 예:

- E4M3（지수 4, 가수 3）
- E5M2（지수 5, 가수 2）

**사실:** 비트 필드 정의는 표준·벤더 문서에 따른다.  
**설명:** 학습·추론 모두에서 FP8을 쓰는 스택이 늘어나는 추세가 있으나, **지원 GPU·커널·프레임워크**를 확인해야 한다. 이 책은 특정 칩의 TFLOPS를 숫자로 단언하지 않는다.

### 3.6 Weight-only vs Activation Quantization

| 방식 | 가중치 | 활성화 | 직관 |
|---|---|---|---|
| Weight-only | 저비트 저장 | 높은 정밀도 | 로드·저장 이득, 연산 전 dequant |
| W8A8 등 | 저비트 | 저비트 | 저비트 GEMM 기회（커널 필요） |

**설명:** Weight-only INT4는 구현이 상대적으로 흔하고, 활성화까지 깎으면 교정이 더 까다로워질 수 있다.

### 3.7 PTQ와 QAT

- **PTQ(Post-Training Quantization):** 학습된 FP 모델을 데이터 일부로 교정해 저비트로 변환
- **QAT(Quantization-Aware Training):** 학습 중 양자화 오차를 시뮬레이션해 적응

**사실:** 이름이 가리키는 단계는 위와 같다.  
**설명:** 큰 LLM 서빙에서는 PTQ 계열이 실무에서 자주 논의된다. 항상 최적이라고 단정하지 않는다.

## 직관적으로 이해하기
사진 압축 비유（**설명**）:

- FP16 가중치 ≈ 원본에 가까운 색 깊이
- INT8 ≈ 색을 256단계로
- INT4 ≈ 16단계 — 평탄한 하늘은 버티지만 세부 질감이 뭉개질 수 있음

지도 축척 비유:

- Scale $s$는 “한 칸이 실제 몇 미터인가”
- 층·채널·그룹마다 축척을 따로 두면（finer granularity） 오차를 줄일 **여지**가 있다

Decode 국면 비유:

- 매 토큰마다 **거대한 가중치 행렬을 메모리에서 읽어야** 한다면, 바이트가 줄어든 것이 곧 시간 절감으로 이어질 **수 있다**
- 그러나 dequant 오버헤드·비효율 커널이면 이득이 사라질 수 있다

## 수학적으로 이해하기
### 5.1 Uniform Affine Quantization

실수 벡터 $x$의 최솟값·최댓값（또는 percentile）을 $x_{\min}, x_{\max}$라 하고, $n$-bit 정수 범위 $q_{\min}, q_{\max}$로 보낼 때:

$$

s = \frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\quad
z = \mathrm{round}(q_{\min} - \frac{x_{\min}}{s})

$$

$$

q = \mathrm{clamp}(\mathrm{round}\big(\frac{x}{s}+z\big), q_{\min}, q_{\max})

$$

복원:

$$

\hat{x}=s\,(q-z)

$$

### 5.2 오차

원소별 오차 $x-\hat{x}$의 크기·편향이 층 출력으로 전파된다.  
LLM에서는 **특정 층·outlier channel**이 전체 품질을 좌우한다는 관찰이 문헌에 자주 나온다（**설명:** 모델 의존）.

### 5.3 이상적 메모리 비율

파라미터 수 $P$일 때（옵티마이저 없음, 추론）:

| 저장 dtype | 이상적 가중치 바이트 |
|---|---|
| FP16 / BF16 | $\approx 2P$ |
| INT8 | $\approx 1P$ + 스케일 오버헤드 |
| INT4 | $\approx 0.5P$ + 오버헤드 |

FP16 대비 INT4는 **이상적으로 약 1/4**（2P → 0.5P）.  
**사실:** 스케일·코드북·패딩·토크나이저·KV·워크스페이스는 포함되지 않는다.  
**해석:** “모델이 4배 작아졌다”고 말하려면 **무엇을 측정했는지**를 명시해야 한다.

## 작은 숫자로 직접 계산하기
### 6.1 1D 양자화 손계산

가정: 값 $x\in\{-1.0, -0.2, 0.5, 1.5\}$, INT8 symmetric, $q\in[-127,127]$, $z=0$.

$$

s = \frac{\max|x|}{127} = \frac{1.5}{127}\approx 0.01181

$$

$0.5$ → $q=\mathrm{round}(0.5/s)\approx \mathrm{round}(42.3)=42$ → $\hat{x}=42s\approx 0.496$

오차 $\approx 0.004$（이 예시에 한함）.

### 6.2 INT4로 같은 범위

$q\in[-7,7]$ 가정이면（부호 있는 4비트의 한 가지 잡기）

$$

s=\frac{1.5}{7}\approx 0.214

$$

$0.5$ → $q=\mathrm{round}(0.5/0.214)\approx 2$ → $\hat{x}=0.428$  
오차가 INT8보다 큼을 **이 장난감에서** 확인한다.

### 6.3 모델 크기 감각（가정）

가정: $P=7\times 10^9$ 파라미터.

- FP16: $2P \approx 14\times 10^9$ B $\approx 14$ GB 감각
- INT4 weight-only: $0.5P \approx 3.5$ GB 감각 + 오버헤드

**주의:** 실제 체크포인트·샤드·토크나이저·런타임 버퍼는 별도다. 이 숫자를 제품 사양처럼 인용하지 말 것.

## 코드로 구현하기
### 7.1 NumPy 대칭 양자화

```python
import numpy as np

def quantize_symmetric(x, n_bits=8):
    qmax = 2 ** (n_bits - 1) - 1
    m = np.max(np.abs(x)) + 1e-12
    s = m / qmax
    q = np.clip(np.round(x / s), -qmax, qmax).astype(np.int32)
    x_hat = q * s
    return q, s, x_hat

rng = np.random.default_rng(0)
w = rng.normal(size=(64, 64)).astype(np.float32)

for bits in (8, 4):
    q, s, w_hat = quantize_symmetric(w, bits)
    mse = np.mean((w - w_hat) ** 2)
    print(f"bits={bits}, scale={s:.6f}, mse={mse:.6e}")
```

관찰: 같은 분포에서 bits↓ → mse↑ 경향（이 실험에 한함）.

### 7.2 Per-channel scale

```python
def quantize_per_channel(w, n_bits=8, axis=0):
    # w: [out, in], axis=0 → out채널별 scale
    qmax = 2 ** (n_bits - 1) - 1
    m = np.max(np.abs(w), axis=1, keepdims=True) + 1e-12
    s = m / qmax
    q = np.clip(np.round(w / s), -qmax, qmax)
    return q, s, q * s

q, s, w_hat = quantize_per_channel(w, 4)
print(s.shape, np.mean((w - w_hat) ** 2))
```

**설명:** granularity를 잘게 가져가면 오차가 줄어들 **여지**가 있다. 대신 스케일 저장·커널 복잡도가 는다.

### 7.3 “Weight-only matmul” 스케치

```python
def matmul_weight_only(x, q_w, s):
    # x: [B, in], q_w: [out, in] int, s: [out,1] or scalar
    w_hat = q_w.astype(np.float32) * s
    return x @ w_hat.T
```

실제 INT4 커널은 패킹（2개가 1바이트）·fused dequant를 쓴다. 위는 **논리 등가**다.

### 7.4 PyTorch — fake quant 시뮬레이션

```python
import torch

def fake_quant_symmetric(x, n_bits=8):
    qmax = 2 ** (n_bits - 1) - 1
    m = x.detach().abs().max().clamp_min(1e-12)
    s = m / qmax
    q = torch.clamp(torch.round(x / s), -qmax, qmax)
    return q * s

x = torch.randn(2, 16)
y = fake_quant_symmetric(x, 4)
print((x - y).abs().mean().item())
```

학습 그래프에 넣을 때는 STE（straight-through estimator）등이 필요하나, 서빙 PTQ 이야기에서는 “오차를 시뮬”하는 용도로 충분하다.

## 수식 보강 — 양자화 스케일

균등 양자화 스케치:

$$
q = \mathrm{clip}(\mathrm{round}\big(\frac{x}{s}\big)+z),\quad
\hat{x}=s(q-z)
$$

$s$는 스케일, $z$는 zero-point입니다. INT8/INT4/FP8은 표현 범위·오차 트레이드오프가 다릅니다.

## LLM에서는 어디에 사용될까?
### 8.1 엔진에서의 위치

vLLM 등 엔진은 모델 로드 시

```text
checkpoint (FP16/BF16/양자화 포맷)
  → 내부 표현으로 적재
  → 커널이 지원하는 방식으로 GEMM / attention
```

을 수행한다.  
**사실:** 지원 양자화 방식 목록은 엔진 버전·백엔드 문서에 따른다.  
이 책에서 “모든 INT4가 모든 엔진에서 동일”하다고 쓰지 않는다.

### 8.2 Continuous Batching과의 결합

제102강 복습: 동시성은 KV 예산에 묶인다.  
가중치가 줄면

$$

\mathrm{Budget}_{KV} \approx \mathrm{VRAM} - \mathrm{Weights} - \mathrm{Workspace}

$$

가 커질 **수 있다**.  
**해석:** 양자화의 이득은 “커널 ms”뿐 아니라 **슬롯 수**로 나타날 수도 있다. 반드시 측정.

### 8.3 품질 평가 습관

양자화 전후를 비교할 때:

- 동일 프롬프트 세트
- 동일 디코딩 파라미터（temperature 등）
- task metric + 수동 샘플 검수
- （가능하면）장기 실행 안정성

**설명:** perplexity만으로 충분하다고 단정하지 않는다（제67강 취지와 동일）.

### 8.4 FP8을 고르는 상황（설명 수준）

- 하드웨어·프레임워크가 FP8 텐서 경로를 지원할 때
- INT4보다 품질 여유가 필요할 때
- 학습과 추론을 같은 저비트 계열로 맞추려 할 때

반대로 지원이 없으면 FP8 “설정”만으로는 가속이 없다.

## 실습
### 실습 A — 손계산

§7.1의 $x=1.5,0.5$에 대해 INT8 symmetric 복원값을 다시 계산한다.

### 실습 B — MSE 곡선

§8.1 코드로 bits = 8,6,4,3 에 대해 MSE를 표로 남긴다.  
**이 표는 가우시안 장난감 결과**임을 명시한다.

### 실습 C — 이상적 GB

$P=13\times 10^9$ 가정에서 FP16 vs INT4 이상적 가중치 GB 감각을 계산한다. KV·오버헤드를 빠뜨리면 안 된다는 주의 문장을 붙인다.

### 실습 D — 용어 매칭

PTQ / QAT / weight-only / W8A8 / scale / zero-point를 한 줄 정의로 카드화한다.

### 실습 E — 문장 교정

원문: “INT4로 바꾸면 어떤 모델이든 정확도 손실 없이 4배 빠르다.”  
사실 검증 가능하게 고친다.

### 실습 F — QLoRA 연결

제74강의 4-bit이 **학습 시 adapter**와 어떻게 다른지, 서빙 INT4와 표로 비교한다.

## 자주 하는 실수
1. **이상적 1/4 메모리를 체감 속도 4×로 환산**  
   병목이 다르면 이득이 다르다.

2. **교정 데이터 없이 PTQ**  
   outlier·분포 이동을 놓친다.

3. **학습 BF16과 서빙 INT4를 동일시**  
   목적·오차 특성이 다르다.

4. **엔진 미지원 포맷을 강제**  
   로드 실패·느린 fallback.

5. **샘플 1~2개만 보고 “품질 동일” 선언**  
   회귀는 긴 꼬리에서 난다.

6. **KV·워크스페이스를 잊고 VRAM 계획**  
   가중치만 줄인 계획서는 틀린다.

7. **공개 블로그의 % 숫자를 재현 없이 책의 사실로 전재**  
   금지.

## 핵심 요약
- Quantization은 값을 저비트로 근사하는 압축이며, 서빙의 메모리·대역폭 도구다.
- INT8·INT4는 정수 격자, FP8은 8비트 부동소수점 계열이다.
- Weight-only와 활성화 양자화는 난이도·커널 요구가 다르다.
- 이상적 바이트 비율과 실제 속도·품질은 별개로 측정한다.
- Continuous Batching 동시성·KV 예산과 맞물려 이득이 나타날 수 있다.

## 용어 사전
| 용어 | 한 줄 의미 |
|---|---|
| Quantization | 저비트 근사 표현 |
| INT8 / INT4 | 8·4비트 정수 양자화 |
| FP8 | 8비트 부동소수 형식 계열 |
| Scale / Zero-point | 선형 복원 파라미터 |
| Weight-only | 가중치만 저비트 |
| PTQ / QAT | 학습 후 양자화 / 인식 학습 |
| Calibration | PTQ용 분포 추정 절차 |
| Dequantization | 정수·FP8 코드를 연산용 정밀도로 복원 |
| Granularity | per-tensor / per-channel / per-group 등 |

## 연습문제
### 문제 1（정의）

Quantization을 한 문장으로 정의하고, Mixed Precision Training과 목적 차이를 한 줄로 쓰시오.

### 문제 2（수식）

$w\approx s(q-z)$에서 각 기호의 역할을 쓰시오.

### 문제 3（계산）

§7.1 가정에서 $x=-1.0$의 INT8 복원 근사 과정을 스케치하시오.

### 문제 4（비율）

FP16→INT4 weight-only의 이상적 바이트 비율은? 오버헤드를 왜 빼면 안 되나?

### 문제 5（방식）

Weight-only와 W8A8의 차이를 한 줄로.

### 문제 6（서빙）

양자화가 Continuous Batching 동시성에 간접적으로 도움이 될 수 있는 경로를 쓰시오.

### 문제 7（다리）

제104강 제목을 쓰고, 엔진이 양자화와 만나는 지점（로드·커널）을 한 줄로.

---

## 정답 및 해설
### 문제 1

저비트 근사. 학습 mixed precision은 학습 안정·메모리, 서빙 양자화는 배포 메모리·지연·비용이 중심.

### 문제 2

$q$ 정수코드, $s$ 스케일, $z$ 제로포인트.

### 문제 3

$s\approx0.01181$, $q=\mathrm{round}(-1/s)\approx-85$, $\hat{x}\approx -85s\approx -1.004$（반올림에 따라 인접 정수 가능）.

### 문제 4

약 1/4. 스케일·패딩·기타 버퍼가 있어 체감 용량·속도는 달라짐.

### 문제 5

전자는 가중치만 저비트（활성은 높은 정밀도）, 후자는 활성도 저비트로 두어 저비트 연산을 노림.

### 문제 6

가중치 VRAM 감소 → KV 예산 증가 여지 → 입학 가능 요청 수 증가 여지.

### 문제 7

제목: vLLM 개요와 구조. 지점: 체크포인트 로드 시 포맷 해석 + 실행 시 지원 커널 경로.

## 다음 강의와 연결
부품（Prefill/Decode·KV·Batching·Quantization）이 모였다.  
다음 **제104강. vLLM 개요와 구조**에서는 이 부품들을 한 런타임으로 묶는 **Inference Engine**의 공개 개념 지도 — API·엔진·워커·스케줄·캐시 — 를 그린다. 내부 미공개 구현을 사실처럼 단정하지 않고, **역할 다이어그램**으로 이해한다.

> 비트를 줄이는 법을 알았다면, 이제 그 가중치를 요청 바다 위에 올리는 엔진을 본다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [102강. Continuous Batching](102강_Continuous_Batching.md)
- **다음 강:** [104강. vLLM 개요와 구조](104강_vLLM_개요와_구조.md)

<!-- /LECTURE_NAV -->
