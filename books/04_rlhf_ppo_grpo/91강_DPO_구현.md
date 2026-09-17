# 제91강. DPO 구현

> **학습 목표**
> - 시퀀스 로그확률을 logits에서 모으는 방법
> - chosen/rejected · policy/ref 네 갈래 로그확률 정리
> - `logsigmoid`로 $\mathcal{L}_{\mathrm{DPO}}$ 구현
> - $\beta$·마스크·no_grad 참조 forward
> - 최소 트레이닝 스케치와 로깅 항목
> - 구현에서 자주 터지는 버그

---
## 1. 왜 이것을 배우는가

DPO는 수식은 짧지만 구현에서 실수 포인트가 많다.

```text
배치 (x, y_w, y_l)
  → π_θ forward ×2 (또는 packed)
  → π_ref forward ×2 (no grad)
  → logp 합산 (response mask)
  → loss = -log σ(β[(lp_w^θ-lp_w^ref)-(lp_l^θ-lp_l^ref)])
  → backward (θ만)
```

한 군데만 뒤집혀도 “학습은 되는데 정렬이 반대로” 갈 수 있다. 장난감으로 **부호·마스크·참조 고정**을 몸에 익힌다.

## 2. 먼저 알아야 할 개념

- DPO 손실 (90강)
- `F.log_softmax`, `gather`, `masked sum`
- `torch.no_grad()` / `requires_grad`
- SFT 학습 루프 감각 (3권 71강)
- PPO 구현에서 봤던 logprob 유틸 (88강) — 있으면 재사용

## 3. 손실을 코드 변수로 번역

$$

\mathcal{L}=-\log\sigma\Big(
\beta\big(\Delta_w-\Delta_l\big)
\Big),\quad
\Delta=\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x)

$$

| 수학 | 코드 변수 예 |
|---|---|
| $\log\pi_\theta(y_w\mid x)$ | `lp_w_pi` |
| $\log\pi_{\mathrm{ref}}(y_w\mid x)$ | `lp_w_ref` |
| $\log\pi_\theta(y_l\mid x)$ | `lp_l_pi` |
| $\log\pi_{\mathrm{ref}}(y_l\mid x)$ | `lp_l_ref` |
| $\beta$ | `beta` |
| $\mathcal{L}$ | `loss` |

안정 구현:

```python
import torch
import torch.nn.functional as F

logits = beta * ((lp_w_pi - lp_w_ref) - (lp_l_pi - lp_l_ref))
loss = -F.logsigmoid(logits).mean()
```

`logsigmoid`는 $\log\sigma(z)$를 수치적으로 더 안정적으로 계산한다. `-log(sigmoid(z))`를 직접 쓰면 극단 $z$에서 언더플로가 나기 쉽다.

## 4. 시퀀스 로그확률 유틸

Causal LM logits는 보통 “다음 토큰” 예측이므로, label을 한 칸 shift한다.

```python
def shift_logits_and_labels(logits, labels):
    # logits: [B, T, V], labels: [B, T]
    return logits[:, :-1, :], labels[:, 1:]

def sequence_logprobs_from_logits(logits, labels, mask):
    """
    logits, labels, mask 는 이미 shift된 텐서.
    mask: 응답 토큰=1, 프롬프트/패딩=0
    return: [B] 시퀀스 로그확률 합
    """
    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)
```

실무 팁:

- 프롬프트와 응답을 concat한 뒤 **응답 구간만** mask=1
- 길이 정규화(`sum` vs `mean`)는 팀 규약에 따른다. 원형 식은 보통 **합(joint logprob)**
- pad label은 gather 전에 0 등으로 바꿔도 되지만, mask로 반드시 제외

## 5. 장난감 배치로 손·코드 일치

어휘가  Tiny하고, 이미 시퀀스 로그확률을 알고 있다고 가정한다(모델 forward를 생략한 단위 테스트).

검산용으로 **시퀀스 로그확률을 직접 넣은** 단위 테스트를 쓴다(모델 forward 생략).

```python
import torch
import torch.nn.functional as F

beta = 0.1
lp_w_pi  = torch.tensor([0.0], requires_grad=True)
lp_l_pi  = torch.tensor([0.0], requires_grad=True)
lp_w_ref = torch.tensor([-0.5])  # no grad
lp_l_ref = torch.tensor([0.4])

# Δw = 0.0 - (-0.5) = 0.5
# Δl = 0.0 - 0.4     = -0.4
# margin = 0.5 - (-0.4) = 0.9
margin = (lp_w_pi - lp_w_ref) - (lp_l_pi - lp_l_ref)
z = beta * margin                 # 0.09
loss = -F.logsigmoid(z)
loss.backward()
# σ(0.09)≈0.5225, -log≈0.6491
print(float(loss), float(lp_w_pi.grad), float(lp_l_pi.grad))
```

기대:

- `loss ≈ 0.649`
- `lp_w_pi.grad < 0` → loss를 줄이려면 $lp_w$를 **키우는** 방향(∂loss/∂var)
- `lp_l_pi.grad > 0` → $lp_l$을 **줄이는** 방향

배치 크기 2로 확장한 예:

```python
beta = 0.1
lp_w_pi  = torch.tensor([-2.0, -3.0], requires_grad=True)
lp_l_pi  = torch.tensor([-2.5, -2.0], requires_grad=True)
lp_w_ref = torch.tensor([-2.2, -2.5])
lp_l_ref = torch.tensor([-1.5, -2.8])
# sample0: Δw=0.2, Δl=-1.0, margin=1.2
# sample1: Δw=-0.5, Δl=0.8, margin=-1.3
logits = beta * ((lp_w_pi - lp_w_ref) - (lp_l_pi - lp_l_ref))
loss = -F.logsigmoid(logits).mean()
loss.backward()
```

한 번 실행해 부호·평균 손실을 확인하는 것이 이 강의의 최소 합격선이다.

## 6. 모델 forward까지 붙인 최소 스케치

실제로는 policy/ref 두 모델이 있다. 참조는 freeze.

```python
def dpo_loss_on_batch(policy, ref, batch, beta=0.1):
    """
    batch keys:
      input_ids_w, attention_mask_w, labels_w, loss_mask_w
      input_ids_l, attention_mask_l, labels_l, loss_mask_l
    labels / loss_mask 는 이미 shift 규칙에 맞게 준비했다고 가정.
    """
    # --- policy ---
    out_w = policy(input_ids=batch["input_ids_w"],
                   attention_mask=batch["attention_mask_w"])
    out_l = policy(input_ids=batch["input_ids_l"],
                   attention_mask=batch["attention_mask_l"])

    logits_w, lab_w = shift_logits_and_labels(out_w.logits, batch["labels_w"])
    logits_l, lab_l = shift_logits_and_labels(out_l.logits, batch["labels_l"])
    mask_w = batch["loss_mask_w"][:, 1:]
    mask_l = batch["loss_mask_l"][:, 1:]

    lp_w_pi = sequence_logprobs_from_logits(logits_w, lab_w, mask_w)
    lp_l_pi = sequence_logprobs_from_logits(logits_l, lab_l, mask_l)

    # --- reference (frozen) ---
    with torch.no_grad():
        ref_w = ref(input_ids=batch["input_ids_w"],
                    attention_mask=batch["attention_mask_w"])
        ref_l = ref(input_ids=batch["input_ids_l"],
                    attention_mask=batch["attention_mask_l"])
        r_logits_w, _ = shift_logits_and_labels(ref_w.logits, batch["labels_w"])
        r_logits_l, _ = shift_logits_and_labels(ref_l.logits, batch["labels_l"])
        lp_w_ref = sequence_logprobs_from_logits(r_logits_w, lab_w, mask_w)
        lp_l_ref = sequence_logprobs_from_logits(r_logits_l, lab_l, mask_l)

    logits = beta * ((lp_w_pi - lp_w_ref) - (lp_l_pi - lp_l_ref))
    loss = -F.logsigmoid(logits).mean()

    stats = {
        "loss": loss.detach(),
        "margin": ((lp_w_pi - lp_w_ref) - (lp_l_pi - lp_l_ref)).detach().mean(),
        "acc": (logits.detach() > 0).float().mean(),  # 암묵 보상이 chosen 승
    }
    return loss, stats
```

학습 루프:

```python
opt = torch.optim.AdamW(policy.parameters(), lr=5e-7)  # LLM에선 더 보수적으로

for step, batch in enumerate(loader):
    opt.zero_grad(set_to_none=True)
    loss, stats = dpo_loss_on_batch(policy, ref, batch, beta=0.1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()

    if step % 10 == 0:
        print(step, {k: float(v) for k, v in stats.items()})
```

스케치일 뿐이다. 실제 LLM에서는 bf16, gradient checkpointing, packing, LoRA 등이 붙는다(3권·88강 인프라를 재사용).

## 7. 학습 스케치 — 체크리스트

```text
1. SFT 체크포인트 로드 → policy
2. 동일 가중치로 ref 복사 후 requires_grad=False
3. Preference collator: (x,y_w), (x,y_l) 각각 mask
4. β, lr, batch, max_length 설정
5. step마다 loss / margin / pair accuracy / chosen·rejected Δ 로그
6. 주기적으로 정성 generate (같은 프롬프트 before/after)
7. 과도한 KL 이탈(평균 Δ)이면 β↑ 또는 lr↓ 검토
```

초기에는 `acc`(암묵 보상이 chosen을 이기는 비율)가 0.5 근처에서 올라가는지를 본다. loss만 보면 스케일 착시가 있다.

## 8. 구현 변형과 “정본” 고지

오픈소스(TRL 등)는 다음을 추가로 넣기도 한다.

- average logprob(길이 정규화)
- label smoothing
- IPO/hinge 등 대체 손실
- reference-free 변형
- chosen/rejected를 한 forward로 packing

이 강의 코드는 **교육용 원형 DPO**다. 라이브러리 기본값과 숫자가 다를 수 있다. 프로덕션에서는 해당 라이브러리 문서의 식을 기준으로 맞춘다.

## 9. 흔히 하는 실수

1. **ref에 그래디언트가 흐름**  
   `no_grad` 누락 또는 ref가 policy와 파라미터 공유.

2. **mask에 프롬프트 포함**  
   프롬프트 토큰까지 올리면 신호가 희석·왜곡된다.

3. **shift 불일치**  
   logits와 labels의 time index가 어긋나면 로그확률이 쓰레기다.

4. **chosen/rejected 부호 반대**  
   loss가 내려가도 선호가 역전된다. 단위 테스트로 margin 부호를 고정한다.

5. **β 스케일을 로그10과 혼동**  
   논문·코드는 자연로그 기반이 기본이다.

6. **배치 평균 위치**  
   토큰 평균과 시퀀스 평균을 섞지 않는다.

## 10. 작은 통합 예제 — 가짜 logits

모델 없이 “gather 경로”만 검증하는 패턴이다.

```python
torch.manual_seed(0)
B, T, V = 2, 4, 5
# fake policy logits (이미 shift된 길이 T)
logits_pi = torch.randn(B, T, V, requires_grad=True)
labels = torch.randint(0, V, (B, T))
mask = torch.ones(B, T)
# rejected는 간단히 labels를 한 칸 바꾼 척
labels_l = (labels + 1) % V

lp_w_pi = sequence_logprobs_from_logits(logits_pi, labels, mask)
# 별도 rejected logits라고 가정
logits_pi_l = logits_pi + 0.1
lp_l_pi = sequence_logprobs_from_logits(logits_pi_l, labels_l, mask)

with torch.no_grad():
    logits_ref = torch.randn(B, T, V)
    lp_w_ref = sequence_logprobs_from_logits(logits_ref, labels, mask)
    lp_l_ref = sequence_logprobs_from_logits(logits_ref, labels_l, mask)

beta = 0.5
loss = -F.logsigmoid(
    beta * ((lp_w_pi - lp_w_ref) - (lp_l_pi - lp_l_ref))
).mean()
loss.backward()
assert logits_pi.grad is not None
```

이 테스트가 통과하면 “로그확률 파이프 + 손실 연결”은 산 것이다.

## 11. LLM 파이프라인에서의 위치

```text
88 PPO 구현 (online, RM, value)
  → 89 KL 역할
    → 90 DPO 유도
      → ★ 91 DPO 구현 (offline preference)
        → 92 GRPO (그룹 상대, 샘플 기반)
```

91강까지가 “선호 쌍 → 정책”의 최소 완결이다. 95강 프로젝트에서 작은 데이터로 다시 묶는다.

## 12. Collator 설계 메모

선호 배치를 만들 때 최소 필드:

```text
prompt tokens
chosen response tokens
rejected response tokens
loss_mask_w / loss_mask_l   # 응답만 1
attention_mask_*
```

권장 전처리:

1. chat template을 SFT와 **동일**하게 적용(72강)
2. 최대 길이 초과 시 프롬프트 truncate 정책을 명시
3. chosen/rejected 길이 통계를 데이터 카드에 기록
4. 동일 응답이 양쪽에 들어간 중복 쌍을 제거

잘못된 template은 DPO보다 먼저 성능을 망친다. “손실은 맞는데 생성이 이상”하면 template부터 의심한다.

## 13. LoRA로 DPO 돌리기

메모리가 빡세면 policy에 LoRA를 얹고 ref는 베이스(+필요 시 병합된 SFT)를 freeze한다(3권 73~74강).

체크:

- ref에 LoRA가 켜져 있으면 안 된다(또는 ref는 merge 후 freeze).
- 학습 가능 파라미터가 adapter뿐인지 `requires_grad`로 확인.
- 저장 시 adapter와 tokenizer/template 메타를 함께 남긴다.

LoRA rank가 과하면 오프라인 선호에 과적합하기 쉽다. 작은 $r$로 시작해 margin·holdout을 본다.

## 14. 디버그 시나리오 네 가지

| 증상 | 가능한 원인 | 첫 확인 |
|---|---|---|
| loss≈0.693 고정 | lr=0, mask 전부 0, ref=policy 공유 | mask sum, param grad |
| loss↓ but 선호 역전 | chosen/rejected 부호 반대 | 단위 테스트 margin |
| pair acc→1 즉시 | 데이터 너무 쉬움 / β·lr 과다 | holdout generate |
| OOM | ref+policy 동시 full forward | LoRA, checkpointing, batch |

단위 테스트 한 줄 철학:

> 가짜 로그확률로 loss 숫자와 grad 부호를 **손계산과 일치**시킨 뒤에야 큰 모델을 붙인다.

## 15. 레퍼런스 로그확률 캐시

ref는 고정이므로, 데이터셋 전부에 대해 `lp_w_ref`, `lp_l_ref`를 **미리 계산해 디스크에 저장**하는 최적화가 가능하다.  
장점: 매 스텝 ref forward 제거.  
단점: tokenizer/template/모델 버전이 바뀌면 캐시 무효. 캐시 키에 커밋 해시를 넣는다.

교육 코드에서는 가독성을 위해 매 스텝 계산을 유지한다.

## 16b. End-to-end 가짜 배치 워크스루

텍스트로 한 바퀴만 더 돈다.

```text
batch size = 1
y_w tokens logp_π = -10.0, logp_ref = -11.0  → Δw = +1.0
y_l tokens logp_π = -12.0, logp_ref = -10.0  → Δl = -2.0
margin = 3.0, β = 0.1 → z = 0.3
loss = -log σ(0.3) ≈ 0.554
```

backward 후:

- $y_w$ 쪽 토큰 가능도 ↑ 압력
- $y_l$ 쪽 토큰 가능도 ↓ 압력
- ref logits에는 grad 없음

이 숫자를 노트북에 재현해 두면 91강의 단위 테스트가 완성된다.

## 16c. 91강 종료 체크

- [ ] 손계산 loss와 `logsigmoid` 결과가 일치
- [ ] chosen grad 부호 / rejected grad 부호가 기대와 일치
- [ ] ref 파라미터 `requires_grad=False`
- [ ] response mask에 프롬프트가 섞이지 않음
- [ ] chat template이 SFT와 동일

## 16. 핵심 정리

- DPO 구현의 핵은 네 로그확률과 `-logsigmoid(β·margin)`이다.
- 참조 모델은 반드시 freeze + `no_grad`.
- 응답 마스크와 shift를 단위 테스트로 고정한다.
- 로깅은 loss뿐 아니라 margin·pair acc·Δlogp를 본다.
- 라이브러리 변형이 많으니, 원형 식으로 먼저 검증한 뒤 옵션을 켠다.
- Collator·chat template·LoRA·ref 캐시는 실무에서 성능을 좌우한다.

## 17. 수치 안정과 mixed precision

bf16/fp16에서 log_softmax·logsigmoid는 대체로 안정적이지만, 다음을 지킨다.

1. 손실은 fp32로 누적하는 편이 안전하다(`loss.float()`).
2. `exp(logp_theta - logp_old)` 형태의 ratio는 PPO/GRPO에서 더 민감하고, DPO는 log-space margin이라 상대적으로 낫다.
3. Grad scaler를 쓰면 DPO에서도 overflow 로그를 본다.

## 18. 미니 학습 일지 템플릿

```text
step | loss | margin | pair_acc | chosen_Δ | rejected_Δ | lr | notes
---- | ---- | ------ | -------- | -------- | ----------- | -- | -----
0    | 0.69 | 0.00   | 0.50     | 0.00     | 0.00        | .. | start
...
```

`chosen_Δ = mean(logπ_θ(y_w)-logπ_ref(y_w))`  
`rejected_Δ`도 같이 보면 “승자만 올리는지, 패자만 내리는지”가 보인다.

## 19. 핵심 용어

| 용어 | 한 줄 의미 |
|---|---|
| sequence logprob | 응답 토큰 $\sum\log p$ |
| loss mask | 손실/로그확률에 포함할 위치 |
| margin | $\Delta_w-\Delta_l$ |
| pair accuracy | margin(또는 β margin)>0 비율 |
| reference freeze | $\pi_{\mathrm{ref}}$ 가중치 고정 |
| logsigmoid | $\log\sigma$의 안정 구현 |
| ref cache | 고정 참조 로그확률 사전계산 |
| collator | 선호 쌍을 텐서 배치로 묶는 전처리 |

## 20. 연습 문제
### 문제 1（코드）

`loss = -F.logsigmoid(beta * ((lp_w_pi - lp_w_ref) - (lp_l_pi - lp_l_ref))).mean()`에서 rejected 로그비를 **더하는** 쪽으로 괄호를 잘못 치면 학습이 어떻게  degenerates 하는가?

### 문제 2（계산）

$\Delta_w-\Delta_l=0$, $\beta=0.1$일 때 샘플 loss $-\log\sigma(0)$는?

### 문제 3（구현）

`ref` forward에서 `torch.no_grad()`를 빼면 메모리/학습에 생기는 문제 두 가지는?

### 문제 4（설계）

평균 응답 길이가 chosen이 항상 2배인 데이터로 sum-logprob DPO를 돌릴 때 어떤 편향이 우려되는가? 완화 아이디어 하나는?

### 문제 5（연결）

제92강 GRPO는 DPO처럼 “상대 비교”를 쓴다. DPO의 비교 단위와 GRPO의 비교 단위는 어떻게 다르다고 예상하는가?

### 문제 6（디버그）

학습 시작부터 pair accuracy가 1.0인데 holdout 생성 품질이 나빠졌다. 가능한 원인 두 가지는?

### 문제 7（LoRA）

policy만 LoRA 학습하고 ref는 베이스 SFT를 freeze한다. ref에도 같은 LoRA를 실수로 켜면 어떤 신호가 약해지는가?

### 문제 8（로그）

`chosen_Δ`만 크게 오르고 `rejected_Δ`는 그대로인 것과, 그 반대인 것은 각각 무엇을 시사하는가?

---

## 정답 및 해설

### 문제 1

rejected를 올리거나 chosen을 내리는 쪽으로 기울기가 반전되어, 선호와 **반대 정책**을 학습한다.

### 문제 2

$\sigma(0)=1/2$, $-\log(1/2)=\log 2\approx 0.693$.

### 문제 3

(1) 참조 쪽 활성화가 그래프에 남아 메모리가 커진다. (2) 실수로 ref 파라미터가 갱신되거나, 공유 가중치면 닻이 움직인다.

### 문제 4

긴 시퀀스의 joint logprob 스케일 차이로 길이 편향이 날 수 있다. 길이 정규화·길이 매칭 샘플링·길이 페널티 등을 검토한다(선택과 효과는 데이터 종속).

### 문제 5

DPO는 **고정된 선호 쌍** $(y_w,y_l)$의 상대 로그비다. GRPO는 같은 프롬프트에서 **새로 샘플한 그룹** 안에서 보상(또는 점수)의 상대적 우위로 advantage를 만든다.

### 문제 6

데이터가 너무 쉽거나 β/lr이 과다해 암묵 보상만 과적합. 또는 평가 프롬프트 분포가 학습과 다름.

### 문제 7

$\log(\pi/\pi_{\mathrm{ref}})$의 참조 쪽도 같이 움직여 KL 닻·상대 로그비 신호가 왜곡·약화된다.

### 문제 8

전자는 chosen 강화 위주, 후자는 rejected 억제 위주. margin은 둘 다 커질 수 있으나 생성 품질 부작용이 다를 수 있어 정성 평가가 필요하다.

## 21. 다음 강의와 연결

오프라인 선호 학습의 손이 끝났다.  
다음 **제92강. GRPO**에서는 PPO식 클리핑·그룹 샘플·**비평가(value) 없이** 상대 이득으로 정책을 올리는 최근 흐름을 **설명 관점**으로 정리한다. 세부 하이퍼파라미터는 논문·구현마다 다르다는 전제를 명시한다.

> 쌍(pair)에서 그룹(group)으로, 비교의 단위가 확장된다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제90강. DPO Preference를 직접 학습하기](90강_DPO_Preference를_직접_학습하기.md)
- **다음 강:** [제92강. GRPO](92강_GRPO.md)

<!-- /LECTURE_NAV -->
