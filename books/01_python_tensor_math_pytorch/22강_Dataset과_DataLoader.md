# 22강. Dataset과 DataLoader
## 이번 강에서 배우는 내용

- `Dataset`이 샘플 하나를 정의하는 인터페이스임을 설명한다.
- `DataLoader`가 배치·셔플·병렬 로딩을 담당함을 이해한다.
- `batch`, `shuffle`, `collate_fn`의 역할을 구분한다.
- LLM 학습에서 “배치로 토큰을 묶는 이유”를 연결한다.

## 왜 중요한가?
작은 예제에서는 이렇게 해도 된다.

```python
x = torch.randn(100, 4)
y = torch.randint(0, 2, (100,))
loss = criterion(model(x), y)
```

그러나 데이터가 커지면 문제가 생긴다.

- 전부 RAM에 올리지 못할 수 있다.
- 매 epoch마다 순서를 섞어야 한다.
- 샘플 길이가 달라 바로 `stack`이 안 된다.
- GPU를 바쁘게 유지하려면 다음 배치를 미리 준비해야 한다.

**Dataset**과 **DataLoader**는 이 문제를 나누어 푼다.

- Dataset: “샘플 하나란 무엇인가?”
- DataLoader: “샘플들을 어떻게 묶고, 어떤 순서로, 얼마나 빠르게 가져올 것인가?”

LLM Pretraining에서는 수조 토큰을 다루므로, 이 분리 없이는 학습 파이프라인이 성립하지 않는다.

## 선수 개념
1. **배치(Batch)** — 한 번에 모델에 넣는 샘플들의 묶음
2. **에폭(Epoch)** — 학습 데이터 전체를 한 번 순회하는 단위
3. **텐서 shape** — 보통 `(batch, ...)`가 앞축
4. **지도 학습** — 입력 $x$와 정답 $y$가 짝을 이룸 (제13강 Loss와 연결)

토큰화 세부 규칙은 2권에서 다룬다. 지금은 “샘플 → 배치 → 모델” 파이프만 고정한다.

## 핵심 개념
### 3.1 Dataset — 샘플의 정의

**`Dataset`(데이터셋)**은 PyTorch에서 **인덱스로 샘플 하나를 꺼낼 수 있는 객체**이다.

직접 만들 때는 보통 `torch.utils.data.Dataset`을 상속하고 두 메서드를 구현한다.

| 메서드 | 의미 |
|---|---|
| `__len__` | 데이터 개수 |
| `__getitem__(idx)` | `idx`번째 샘플 반환 |

```python
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
```

`__getitem__`이 반환하는 형태는 자유다. 텐서 쌍, 딕셔너리, 문자열 경로 등. 다만 **DataLoader가 배치로 묶을 수 있는 형태**를 미리 설계하는 것이 중요하다.

### 3.2 DataLoader — 배치 공급기

**`DataLoader`(데이터로더)**는 Dataset을 받아 **미니배치 스트림**을 만들어 주는 유틸리티이다.

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)
for xb, yb in loader:
    ...
```

자주 쓰는 인자:

| 인자 | 역할 |
|---|---|
| `batch_size` | 한 배치의 샘플 수 |
| `shuffle` | 매 epoch 시작 시 순서 섞기 |
| `num_workers` | 데이터 로딩 서브프로세스 수 |
| `drop_last` | 마지막 불완전 배치 버릴지 여부 |
| `collate_fn` | 샘플 리스트를 배치 텐서로 묶는 함수 |

### 3.3 Batch — 왜 묶는가

**배치 학습(mini-batch training)**은 샘플 전체가 아니라 일부만 모아 Gradient를 추정하는 방법이다.

이유:

1. **메모리** — 전체를 한 번에 올리지 않는다.
2. **속도** — GPU는 큰 행렬 연산에 유리하다. 샘플 1개보다 32·64개가 처리량(throughput)이 높은 경우가 많다.
3. **기울기 추정** — 배치 평균 Loss의 Gradient는 전체 Gradient의 확률적 추정이다. (제12강 Gradient Descent의 SGD 확장)

배치가 너무 작으면 기울기 잡음이 크고, 너무 크면 메모리 부족·일반화 저하가 생길 수 있다. LLM에서는 토큰 수 기준 배치(global batch tokens)로 이야기하는 경우가 많다.

### 3.4 Shuffle — 왜 섞는가

**Shuffle(셔플)**은 학습 샘플 순서를 무작위로 바꾸는 것이다.

같은 순서로만 반복하면 모델이 “데이터의 순서 패턴”에 과하게 적응할 수 있다. 또한 비슷한 샘플이 한곳에 몰려 있으면, 어떤 배치는 쉬운 예만, 어떤 배치는 어려운 예만 담겨 학습이 불안정해진다.

검증/테스트 로더는 보통 `shuffle=False`로 두어 재현 가능한 평가를 한다.

### 3.5 collate_fn — 길이가 다를 때

**`collate_fn`(콜레이트 함수)**은 DataLoader가 뽑아 온 **샘플 리스트를 하나의 배치로 결합**하는 함수이다.

기본 `collate_fn`은 텐서를 `stack`한다. 모든 샘플 shape가 같을 때 잘 동작한다.

그러나 텍스트는 길이가 다르다.

```text
샘플1: 12 토큰
샘플2: 40 토큰
샘플3: 7 토큰
```

이때는 패딩(padding)하거나, 비슷한 길이끼리 묶거나(packing), 고정 길이로 잘라야 한다. 그 로직이 `collate_fn`에 들어간다.

```python
def collate_pad(batch):
    # batch: List[Tuple[tensor, label]]
    xs, ys = zip(*batch)
    # 실제 LLM에서는 pad token으로 길이를 맞춘다
    xb = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    yb = torch.tensor(ys)
    return xb, yb
```

2권·3권에서 Tokenizer와 Dataset Packing을 배울 때 이 개념이 다시 등장한다. 지금은 “배치 결합 규칙을 커스터마이즈할 수 있다”는 점만 기억한다.

## 직관적으로 이해하기
식당에 비유하자.

- **Dataset**은 냉장고에 있는 재료 목록이다. `dataset[3]`은 네 번째 재료.
- **DataLoader**는 요리사에게 접시를 나르는 웨이터다. 한 접시 = 한 배치.
- **shuffle**은 재료를 무작위로 집어 접시에 담는 것.
- **collate_fn**은 접시에 올리기 전 “모양을 맞추는” 손질이다. 길이가 다른 생선을 같은 접시에 올리려면 손질이 필요하다.

모델(요리사)은 개별 재료의 창고 구조를 몰라도 된다. 접시만 계속 받으면 된다.

## 수학적으로 이해하기
데이터셋 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$이 있을 때, 배치 $B$에 대한 평균 손실은

$$

L_B(\theta) = \frac{1}{|B|} \sum_{i \in B} \ell(f_\theta(x_i), y_i)

$$

전체 손실의 Gradient $\nabla L_{\mathcal{D}}$ 대신 $\nabla L_B$를 사용해 파라미터를 갱신한다. 이것이 Mini-batch SGD의 핵심이다.

배치 크기 $|B|$가 커질수록 $\nabla L_B$는 $\nabla L_{\mathcal{D}}$에 가까워지는 경향이 있지만, 계산·메모리 비용도 커진다.

## 작은 숫자로 직접 계산하기
샘플이 5개, `batch_size=2`, `drop_last=False`라고 하자.

```text
인덱스: 0 1 2 3 4
```

한 epoch에서 배치는 대략 3개:

```text
배치1: 2개
배치2: 2개
배치3: 1개  (마지막 남은 샘플)
```

`drop_last=True`이면 마지막 1개짜리 배치는 버려져 배치가 2개만 된다.

`shuffle=True`이면 매 epoch마다 인덱스 순열이 달라진다. 예:

```text
epoch1: [2,0] [4,1] [3]
epoch2: [1,3] [0,4] [2]
```

이 작은 그림만 머리에서 그릴 수 있어도, 학습 로그의 `steps/epoch` 숫자를 해석할 수 있다.

## 코드로 구현하기 — 커스텀 Dataset
```python
# toy_loader.py
import torch
from torch.utils.data import Dataset, DataLoader

class RegressionDataset(Dataset):
    """y = 2x0 - x1 + noise 형태의 장난감 회귀 데이터."""

    def __init__(self, n: int = 100, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 2, generator=g)
        noise = 0.05 * torch.randn(n, 1, generator=g)
        self.y = (2 * self.x[:, 0] - self.x[:, 1]).unsqueeze(1) + noise

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def main():
    ds = RegressionDataset(n=10, seed=0)
    print("len:", len(ds))
    x0, y0 = ds[0]
    print("sample0 x:", x0, "y:", y0)

    loader = DataLoader(ds, batch_size=4, shuffle=True)
    for step, (xb, yb) in enumerate(loader, start=1):
        print(f"step {step}: xb={tuple(xb.shape)} yb={tuple(yb.shape)}")

if __name__ == "__main__":
    main()
```

예상 출력 형태:

```text
len: 10
sample0 x: tensor([...]) y: tensor([...])
step 1: xb=(4, 2) yb=(4, 1)
step 2: xb=(4, 2) yb=(4, 1)
step 3: xb=(2, 2) yb=(2, 1)
```

`n=10`, `batch_size=4`이면 배치 개수는 3이다. (`ceil(10/4)=3`)

## PyTorch로 구현하기 — TensorDataset과 딕셔너리 배치
단순한 텐서 쌍은 `TensorDataset`으로 충분하다.

```python
from torch.utils.data import TensorDataset, DataLoader

x = torch.randn(100, 4)
y = torch.randint(0, 3, (100,))
ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=16, shuffle=True)
```

실무·LLM 쪽에서는 샘플을 딕셔너리로 두는 경우가 많다.

```python
class DictDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
        }
```

기본 collate는 같은 키끼리 리스트/텐서로 묶는다. 텍스트 문자열은 텐서가 아니므로, 학습 전에 토큰 ID 텐서로 바꾸거나 커스텀 `collate_fn`이 필요하다.

## 수식 보강 — 배치 · 에폭 · 스텝 수

데이터셋 크기 $N$, 배치 크기 $B$이면 한 에폭의 업데이트 횟수는

$$
\mathrm{steps/epoch} = \left\lceil \frac{N}{B} \right\rceil
$$

입니다. $E$ 에폭 학습하면 대략

$$
\mathrm{total\ steps} \approx E \cdot \left\lceil \frac{N}{B} \right\rceil
$$

입니다.

미니배치 평균 손실은 샘플 손실의 평균입니다.

$$
L_{\mathcal{B}}(\theta)
=
\frac{1}{|\mathcal{B}|}
\sum_{i\in\mathcal{B}} \ell(f_\theta(x_i), y_i)
$$

LLM의 언어모델링에서는 $(x_i,y_i)$가 “문맥 토큰 → 다음 토큰” 쌍이며, 패킹되면 한 배치 Shape가 `(B, T)` 토큰 ID가 됩니다.

> **핵심**
>
> DataLoader는 수학적으로 “인덱스 집합 $\mathcal{B}$를 샘플링해 $L_{\mathcal{B}}$를 만드는 장치”입니다.


<!-- enrich-batch2-22 -->
## DataLoader와 미니배치 수식

$$
\mathcal{B}_k=\{x_{(k-1)B+1},\ldots,x_{kB}\}
$$

한 스텝 손실:

$$
L_k=\frac{1}{|\mathcal{B}_k|}\sum_{x\in\mathcal{B}_k}\ell(f_\theta(x),y)
$$

```python
from torch.utils.data import DataLoader, TensorDataset
import torch
X = torch.randn(100, 8)
y = torch.randint(0, 3, (100,))
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)  # (16,8), (16,)
```

드롭 라스트:

$$
N_{\mathrm{steps}}=\lfloor N/B\rfloor\ \text{(drop_last)}
$$

## LLM에서는 어디에 사용될까?
LLM 학습 데이터는 대략 다음 파이프라인을 따른다.

```text
원문 텍스트
  → Tokenizer로 token id 서열
    → 고정 길이로 자르거나 packing
      → input_ids / labels 텐서
        → DataLoader가 배치 공급
          → 모델 forward → next-token loss
```

배치가 중요한 이유:

1. **GPU 효율** — 행렬곱은 큰 배치에서 장치 점유율이 올라간다.
2. **안정적 학습** — 토큰 단위로 보면 배치가 Gradient 분산을 줄인다.
3. **분산 학습** — 여러 GPU가 micro-batch를 처리한 뒤 Gradient를 합친다. (5권·3권에서 확장)

또한 LLM은 문장 길이가 제각각이라 `collate_fn` 또는 사전 packing이 필수다. 3권의 “Tokenization Pipeline과 Dataset Packing”이 이 강의의 연장선이다.

지금은 숫자 행렬 toy dataset으로도, “샘플 정의와 배치 공급의 분리”를 몸에 익히는 것이 목표다.

## 실습
### 실습 1 — epoch당 step 수 계산

**목표:** `len(dataset)`, `batch_size`, `drop_last` 관계를 확인한다.

1. 샘플 100개 Dataset을 만든다.
2. `batch_size=16`, `drop_last=False`일 때 step 수를 예측하고 실행으로 확인한다.
3. `drop_last=True`로 바꿔 다시 센다.

**정답 힌트:** `False`면 7 step (`6*16 + 4`), `True`면 6 step.

### 실습 2 — shuffle 재현성

**목표:** 난수 시드와 로더 동작의 관계를 본다.

1. `torch.manual_seed(0)` 후 `shuffle=True` 로더로 첫 배치 인덱스를 간접 확인한다. (값을 출력)
2. 시드 없이 여러 번 실행해 순서가 바뀌는지 본다.
3. 검증용 로더는 `shuffle=False`로 두고 결과가 고정되는지 확인한다.

### 실습 3 — 최소 collate_fn

**목표:** 길이가 다른 1D 텐서를 패딩으로 묶는다.

```python
# 길이 3, 5, 2 인 텐서 세 개를 배치로 묶고
# shape이 (3, 5)가 되는지 확인한다.
```

`torch.nn.utils.rnn.pad_sequence(..., batch_first=True)`를 사용한다.

## 자주 하는 실수
1. **Dataset에서 너무 무거운 전처리를 매 `__getitem__`마다 반복한다**  
   필요하면 캐시하거나, 오프라인으로 전처리한 뒤 가벼운 로딩만 남긴다.

2. **검증 로더까지 `shuffle=True`로 둔다**  
   평가 재현성이 깨진다. 학습만 섞는다.

3. **`batch_size`만 키우고 shape 오류를 무시한다**  
   모델이 `(batch, feature)`를 기대하는데 collate가 리스트를 그대로 주면 실패한다.

4. **`num_workers > 0`에서 디버깅이 어려워진다**  
   먼저 `num_workers=0`으로 로직을 검증한 뒤 늘린다.

5. **LLM 텍스트를 바로 `TensorDataset`에 넣으려 한다**  
   문자열은 텐서가 아니다. 토큰 ID로 수치화한 뒤 배치를 만든다.

## 미니 예제 보강 — 배치·스텝·토큰 수

### 배치 개수 손계산

$N=10$, $B=4$이면

$$
\left\lceil \frac{10}{4} \right\rceil = 3
$$

배치이며, 마지막 배치는 크기 $10 \bmod 4 = 2$입니다. `drop_last=True`이면 마지막을 버려 스텝 수는

$$
\left\lfloor \frac{N}{B} \right\rfloor = 2
$$

가 됩니다.

### 에폭 전체의 샘플 방문

셔플을 켠 한 에폭에서 각 샘플은 (대략) 한 번씩 나옵니다. $E$ 에폭이면 샘플 방문 총횟수는

$$
N_{\text{visits}} = E \cdot N
$$

이고, 업데이트 횟수는 (drop_last 없을 때)

$$
N_{\text{update}} = E \cdot \left\lceil \frac{N}{B} \right\rceil
$$

입니다.

### 토큰 배치 (LLM 감각)

시퀀스 길이 $T$인 샘플 $B$개를 묶으면 배치 텐서 shape는 대개 $(B,T)$이고, 토큰 수는

$$
N_{\text{tok}} = B \cdot T
$$

입니다. (패딩을 넣으면 “실제 유효 토큰”은 이보다 작을 수 있습니다.) DataLoader는 이 $(B,T)$ 묶음을 반복해 공급하는 장치입니다.

### 복잡도 감각

한 에폭에서 Dataset `__getitem__`은 대략 $N$번 호출됩니다. collate·복사 비용을 $C$라 하면 데이터 로딩은 $O(N\cdot C)$ 스케일입니다. `num_workers>0`은 이 비용을 **겹쳐 실행**해 GPU 대기 시간을 줄이려는 장치입니다. (이득은 환경에 따라 다릅니다.)

### 연습용 손계산 — 세 에폭

$N=100$, $B=16$, $E=3$, `drop_last=False`이면

$$
\left\lceil \frac{100}{16} \right\rceil = 7,\qquad
N_{\text{update}} = 3\times 7 = 21
$$

입니다. `drop_last=True`이면 $\lfloor 100/16\rfloor=6$이므로 업데이트는 $18$회입니다. DataLoader 설정을 바꾸기 전에 이렇게 손계산해 두면, 로그에 찍힌 step 수와 대조할 수 있습니다.

### LLM 연결 — 패딩 마스크와의 만남

길이가 다른 시퀀스를 $T_{\max}$로 패딩하면 배치 텐서는 직사각형이 되지만, Loss에는 pad 위치를 빼야 합니다. 마스크 $m_{b,t}\in\{0,1\}$에 대해 유효 토큰만 평균하는 형태는

$$
L = \frac{\sum_{b,t} m_{b,t}\, \ell_{b,t}}{\sum_{b,t} m_{b,t}}
$$

처럼 쓸 수 있습니다. collate_fn이 pad와 mask를 함께 만드는 이유가 여기에 있습니다. (구현 세부는 이후 토큰·학습 강의에서 이어집니다.)

## 핵심 요약
- `Dataset`은 `__len__`과 `__getitem__`으로 샘플을 정의한다.
- `DataLoader`는 배치 크기, 셔플, 병렬 로딩, collate를 담당한다.
- 배치는 메모리·속도·확률적 Gradient 추정을 위한 핵심 단위다.
- 길이가 다른 샘플은 `collate_fn`으로 패딩·패킹한다.
- LLM 학습도 같은 공급 구조를 쓰며, 토큰 배치가 GPU 효율과 학습 안정성을 좌우한다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Dataset | 인덱스로 샘플을 제공하는 데이터 추상화 |
| DataLoader | Dataset에서 미니배치 스트림을 만드는 유틸리티 |
| Batch | 한 번에 모델에 넣는 샘플 묶음 |
| Epoch | 전체 학습 데이터를 한 번 순회하는 단위 |
| Shuffle | 샘플 순서를 무작위로 섞기 |
| `collate_fn` | 샘플 리스트를 배치로 결합하는 함수 |
| Padding | 길이를 맞추기 위해 채움 토큰/값을 넣는 기법 |
| `TensorDataset` | 텐서들을 병렬로 인덱싱하는 기본 Dataset |

## 연습문제
### 문제 1 (개념)

Dataset과 DataLoader의 책임을 한 문장씩으로 구분하시오.

### 문제 2 (계산)

샘플 1000개, `batch_size=64`, `drop_last=False`일 때 한 epoch의 step 수는?

### 문제 3 (개념)

검증 데이터에서 `shuffle=False`를 권하는 이유를 쓰시오.

### 문제 4 (코드)

다음 `__getitem__`이 DataLoader 기본 collate와 잘 맞는지 판단하고, 문제치면 이유를 쓰시오.

```python
def __getitem__(self, idx):
    return [1, 2, 3], "cat"  # 리스트와 문자열
```

### 문제 5 (연결)

LLM Pretraining에서 배치를 크게 가져가려는 이유를 GPU 관점과 Gradient 관점에서 각각 한 줄로 쓰시오.

---

## 정답 및 해설
### 문제 1

Dataset은 개별 샘플을 정의·반환한다. DataLoader는 그 샘플들을 배치로 묶어 순회 가능한 형태로 공급한다.

### 문제 2

$$

\lceil 1000 / 64 \rceil = \lceil 15.625 \rceil = 16

$$

### 문제 3

평가 결과를 재현 가능하게 유지하고, 배치 구성 변화가 메트릭 변동으로 해석되는 일을 줄이기 위해서이다.

### 문제 4

기본 collate는 숫자 리스트를 텐서로 묶으려 할 수 있지만, 문자열 라벨은 바로 텐서가 되지 않는다. 라벨을 정수 ID로 바꾸거나 커스텀 `collate_fn`이 필요하다.

### 문제 5

GPU 관점: 큰 행렬 연산으로 장치 활용률·처리량을 높인다. Gradient 관점: 더 많은 토큰 평균으로 기울기 추정의 분산을 줄여 학습을 안정화한다.

## 다음 강의와 연결
데이터 접시가 준비되었다. 이제 요리사가 실제로 **맛을 보고 조미료(파라미터)를 조절**하는 절차가 필요하다.

다음 **제23강. 학습 루프와 최적화기**에서는 `zero_grad → forward → loss → backward → step`의 표준 루프와 SGD/Adam, 학습률을 다룬다. Module·DataLoader·Autograd가 비로소 하나의 학습 엔진으로 연결된다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [21강. nn.Module로 모델 만들기](21강_nn_Module로_모델_만들기.md)
- **다음 강:** [23강. 학습 루프와 최적화기](23강_학습_루프와_최적화기.md)

<!-- /LECTURE_NAV -->
