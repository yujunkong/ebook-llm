# 65강. Checkpoint 관리
## 이번 강에서 배우는 내용

- 저장해야 할 구성 요소(model, optimizer, scheduler, step 등)를 목록화한다.
- Resume(재개) 시 순서를 설명한다.
- `best`와 `last` 체크포인트의 역할을 구분한다.
- `torch.save`와 safetensors를 형식 옵션으로 비교한다(설명 수준).
- 손상·부분 저장·버전 불일치 같은 실패 모드를 예방한다.

## 왜 중요한가?
체크포인트가 없으면:

- 인스턴스가 죽었을 때 **처음부터**다.
- Validation이 좋아진 지점을 복원할 수 없다.
- 실험을 분기(더 학습 / 평가만 / SFT로 이관)할 수 없다.

```text
학습 중 상태 = 가중치만이 아니다
  가중치
  + Optimizer 모멘트
  + Scheduler 위치
  + global_step / tokens_seen
  + (선택) GradScaler, RNG, 데이터 이터레이터 위치
```

제66강 Validation은 “언제 best를 갱신할지”의 신호를 준다. 이번 강의는 **저장 형식과 재개 절차**다.

## 선수 개념
1. **`state_dict`** — 제21강 `nn.Module`
2. **AdamW state / scheduler** — 제63강
3. **GradScaler / AMP** — 제64강
4. **global_step / token budget** — 제62강

## 핵심 개념 — Checkpoint란
**Checkpoint**는 특정 시점의 학습 상태를 디스크에 직렬화한 스냅샷이다.

두 가지 목적이 섞여 있다.

| 목적 | 필요한 것 | 예 |
|---|---|---|
| **학습 재개 (resume)** | 가중치 + optim + sched + step + … | 장애 복구, 이어 학습 |
| **추론·이관 (export)** | 대개 가중치(+설정) | 생성 테스트, SFT 초기화 |

같은 폴더에 둘 다 둘 수 있지만, **파일 역할 이름**을 분리하는 편이 안전하다.

## 무엇을 저장할 것인가
### 4.1 최소 resume 세트

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "global_step": global_step,
    "tokens_seen": tokens_seen,
    "config": config_dict,          # 구조 재현용
    "rng": {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    },
}
# FP16 scaler를 쓰면:
# checkpoint["scaler"] = scaler.state_dict()
```

### 4.2 항목별 이유

| 키 | 없으면 |
|---|---|
| `model` | 재개 자체가 무의미 |
| `optimizer` | Adam 모멘트 리셋 → 재개 직후 거동 변화 |
| `scheduler` | lr이 warmup/cosine 위치를 잊음 |
| `global_step` / `tokens_seen` | 로그·예산·스케줄 축 붕괴 |
| `scaler` | FP16 scale이 리셋되어 초반 불안정 가능 |
| `config` | 구조가 다르면 `load_state_dict` 실패 |
| RNG | 완전 재현은 어려워도, 셔플 재개에 도움 |

데이터 로더 위치(몇 번째 샤드)까지 저장하는 시스템은 더 복잡하다. 최소 교육 범위에서는 step/tokens와 가중치·optim 일관성을 우선한다.

## 저장 코드 패턴
```python
from pathlib import Path
import torch

def save_checkpoint(path, model, optimizer, scheduler, meta: dict, scaler=None):
    """
    meta 예: global_step, tokens_seen, best_val_loss, config
    원자적 저장을 위해 임시 파일에 쓴 뒤 rename 하는 패턴을 권장.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        **meta,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # 같은 파일시스템에서 원자적에 가까운 교체
```

설명: 학습 도중 프로세스가 죽으면 **쓰던 파일이 반쯤 써진 채** 남을 수 있다. tmp+rename은 그 위험을 줄이려는 관행이다. 완벽한 원자성을 모든 OS/파일시스템에서 보장한다고 단정하지는 않는다.

## 재개(Resume) 절차
```python
def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    global_step = ckpt.get("global_step", 0)
    tokens_seen = ckpt.get("tokens_seen", 0)
    return global_step, tokens_seen, ckpt
```

권장 순서:

```text
1. config로 모델·optim·sched 객체를 "빈 상태"로 생성
2. checkpoint 로드
3. state_dict들 주입
4. global_step / tokens_seen으로 루프 시작점 설정
5. device로 이동 (로드 시점 정책에 따라 전후가 바뀔 수 있음)
6. model.train()으로 학습 재개
```

함정:

- GPU에서 저장한 텐서를 CPU-only 환경에서 로드할 때 `map_location` 필요.
- `strict=True`(기본)면 키 불일치 시 실패한다. 구조를 바꿨다면 의도적 `strict=False`와 누락 키 로그가 필요하다.
- DataParallel/`module.` 접두어 불일치.

## Best vs Last
### 7.1 Last checkpoint

**Last**는 가장 최근 step의 스냅샷이다. 장애 복구의 기본이다.

```text
checkpoints/
  last.pt
  step_0001000.pt
  step_0002000.pt
```

주기 저장 + `last` 갱신:

```python
if global_step % ckpt_every == 0:
    save_checkpoint(out_dir / f"step_{global_step:07d}.pt", ...)
    save_checkpoint(out_dir / "last.pt", ...)
```

### 7.2 Best checkpoint

**Best**는 선택 기준(보통 held-out validation loss 최소)이 갱신될 때만 저장한다.

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_checkpoint(
        out_dir / "best.pt",
        model, optimizer, scheduler,
        meta={
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
            "config": config_dict,
        },
        scaler=scaler,
    )
```

| | Last | Best |
|---|---|---|
| 갱신 조건 | 주기 / 매 저장 시점 | 지표 개선 시 |
| 주 용도 | 재개 | 최종 후보·조기 선택 |
| 위험 | 말기 과적합·붕괴 상태일 수 있음 | 지표가 목표와 불일치하면 잘못된 “최고” |

설명: Pretraining에서 val loss best가 곧 “생성 품질 best”는 아니다. 제66~67강에서 평가 축을 분리한다. 그래도 val loss는 **조기 발산·과적합의 운영 신호**로 유용하다.

## 저장 주기와 디스크
너무 잦으면:

- tok/s 하락
- 디스크 가득 참

너무 드물면:

- 장애 시 손실 step이 큼
- best 구간을 놓침

실무 패턴(설명):

- 고정 step 간격으로 `last` + 회전(rotation, 최근 K개만 유지)
- eval 시점에 best 갱신
- 학습 종료 시 final export

숫자 간격은 예산·체크포인트 크기·SLA에 따라 정한다. 이 강의는 특정 “매 N step”을 표준처럼 제시하지 않는다.

## 형식 옵션 — `torch.save`와 safetensors
### 9.1 `torch.save` / `torch.load`

PyTorch 기본 직렬화. 임의 Python 객체(딕셔너리)를 넣기 쉽다. resume용으로 optim state 등을 한 파일에 묶기 편하다.

주의(설명): pickle 기반 로드는 **신뢰할 수 없는 파일**을 열 때 보안 위험이 있다고 알려져 있다. 출처가 불명확한 `.pt`를 함부로 `torch.load`하지 말 것.

### 9.2 safetensors

**safetensors**는 텐서 가중치를 저장·로드하기 위한 형식 중 하나로, 많은 가중치 배포 파이프라인에서 선택지로 쓰인다.

설명 수준에서 알아둘 점:

- 주 용도는 **텐서 맵(가중치)** 저장에 가깝다.
- 임의의 Python 객체·Optimizer state 전체를 한 파일에 넣는 용도로는 `torch.save` 번들보다 역할이 다르다.
- “더 안전하다/더 빠르다”는 주장은 구현·환경·측정에 의존하므로, 이 책에서 벤치마크 수치를 붙여 단정하지 않는다.

실무적 분리 제안:

```text
resume 번들:  last.pt / step_XXXX.pt   (model+optim+sched+meta)
가중치 export: model.safetensors 또는 model.pt  (추론·공유용)
```

```python
# 개념 예 — safetensors가 설치된 환경
# from safetensors.torch import save_file, load_file
# save_file(model.state_dict(), "model.safetensors")
# state = load_file("model.safetensors")
# model.load_state_dict(state)
```

제68강 Mini GPT Pretraining, 제76강 SFT 프로젝트에서 “재개용”과 “내보내기용”을 폴더 규약으로 나누면 혼란이 줄어든다.

## Export만 할 때
추론·다음 단계(SFT) 초기화에는 종종 가중치만 필요하다.

```python
def export_weights(path, model):
    torch.save({"model": model.state_dict(), "config": config_dict}, path)
```

또는 safetensors로 가중치만 저장.  
Optimizer를 빼면 파일이 작아지고, 실수로 “resume 가능한 줄 앎” 사고도 줄어든다. 파일명에 `weights_only` / `export`를 명시하라.

## 분산·래퍼 관련 메모
단일 GPU 교육 코드와 달리, 래퍼가 있으면 `state_dict` 키가 달라진다.

```python
# 개념: DDP라면
state = model.module.state_dict()
```

이 책 3권 본문 루프는 단일 장치 중심이지만, 체크포인트 규약을 짤 때 **래퍼 유무를 한 줄로 기록**해 둔다. 나중에 분산으로 확할 때 가장 먼저 터지는 지점이다.

## 검증 — 저장이 진짜인지 확인
저장 직후 권장 스모크 테스트:

1. 새 프로세스에서 모델 생성 → `load_state_dict`
2. 동일 배치로 train mode loss 또는 eval loss 비교(허용 오차 내)
3. `global_step`이 메타와 일치하는지 출력
4. (resume) 한 step 더 돌려 NaN이 아닌지 확인

“파일이 생겼다” ≠ “로드 가능한 완전한 체크포인트”.

## 디렉터리 규약 예시
```text
runs/mini_gpt_exp01/
  config.yaml          # 또는 config.json
  train.log
  checkpoints/
    last.pt
    best.pt
    step_0000500.pt
    step_0001000.pt
  export/
    model.safetensors  # 또는 weights_only.pt
    config.json
```

규약을 미리 정하면 제68강 프로젝트에서 “파일이 어디 있지?”로 시간을 잃지 않는다.

## 회전(Rotation) — 디스크 지키기
모든 step 체크포인트를 영구 보관하면 디스크가 먼저 죽는다.

```python
from collections import deque

kept = deque(maxlen=3)  # 최근 3개 step ckpt만

def save_rotating(step, path_fn, save_fn):
    path = path_fn(step)
    save_fn(path)
    kept.append(path)
    # deque가 오래된 경로를 밀어내면 파일 삭제
    # (maxlen 초과 시 왼쪽부터 사라지므로, 삭제 목록을 따로 관리해도 됨)
```

`last.pt`와 `best.pt`는 회전에서 **제외**하는 것이 일반적이다.

## 메타데이터에 넣을 것
재현에 도움이 되는 메타:

```python
meta = {
    "global_step": global_step,
    "tokens_seen": tokens_seen,
    "best_val_loss": best_val_loss,
    "config": config_dict,
    "code_git_sha": git_sha_or_none,   # 있으면
    "torch_version": torch.__version__,
    "notes": "amp=bf16 accum=4",
}
```

설명: git SHA까지 넣으면 “어느 코드로 뽑힌 가중치인지” 추적이 쉬워진다. 필수는 아니지만 실험 노트의 품질이 올라간다.

## 부분 로드와 전이학습 예고
SFT(제71강)나 LoRA(제73강)로 넘어갈 때는 종종 **가중치만** 로드한다.

```python
ckpt = torch.load("export/weights_only.pt", map_location="cpu")
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
print("missing", missing)
print("unexpected", unexpected)
```

`strict=False`는 강력하지만 위험하다. 누락 키를 **반드시 출력**한다. 침묵 속 부분 로드는 버그의 온상이다.

## 흔한 버그
1. **가중치만 저장하고 lr·step을 잊음** — 재개 시 cosine이 처음부터.
2. **best를 last 경로에 덮어씀** — 복구 포인트 소실.
3. **저장 중 예외 후 빈 파일** — tmp+rename 미사용.
4. **config 미저장** — 몇 주 뒤 hidden size를 기억 못함.
5. **eval 모드 가중치를 저장한 뒤 train 버퍼를 오해** — Dropout 자체는 대개 가중치에 없으나, 버퍼(BatchNorm 등)가 있으면 모드 유의. LLM LayerNorm은 다른 이슈이나 “모드와 저장”을 한 묶음으로 점검.
6. **신뢰 불가 경로의 `torch.load`** — 보안 이슈.
7. **장치 불일치** — CUDA 텐서가 박힌 ckpt를 CPU에서 map_location 없이 로드.

## 수식 보강 — Validation 손실

$$
L_{\mathrm{val}}=-\frac{1}{|\mathcal{T}_{\mathrm{val}}|}\sum\log p_\theta(x_t\mid x_{<t})
$$

Early stopping은 $L_{\mathrm{val}}$이 개선되지 않으면 학습을 멈춥니다.


## 수학적으로 이해하기 — 체크포인트 크기 스케치

가중치만 float32로 저장하면

$$

\mathrm{Size}_{\mathrm{weights}}
\approx
4\cdot\#\theta
\quad\text{bytes}

$$

입니다. AdamW를 함께 저장하면 1차·2차 모멘트가 가중치와 같은 형태의 텐서를 추가로 갖습니다（대략）.

$$

\mathrm{Size}_{\mathrm{resume}}
\approx
4\cdot\#\theta
\;+\;
4\cdot\#\theta
\;+\;
4\cdot\#\theta
\;=\;
12\cdot\#\theta

$$

（model + m + v, 매우 거친 상한 감각; bias·버퍼·scaler·RNG 제외）.  
예: $\#\theta=50\times 10^6$ → weights ≈ 200MB, resume 상한 감각 ≈ 600MB.

bf16/fp16 가중치만 저장하면 계수 4 대신 2에 가깝습니다. **정확한 MB를 벤치처럼 주장하지 말고**, “optim 포함 시 수 배” 감각만 잡습니다.

### best vs last

Validation 손실 $L_{\mathrm{val}}(t)$에 대해

$$

t_{\mathrm{best}} = \arg\min_{t\in\mathcal{S}} L_{\mathrm{val}}(t)
$$

（$\mathcal{S}$는 평가한 step 집합）.  
`last`는 재개 편의, `best`는 일반화 후보입니다. 둘을 같은 파일로 덮어쓰지 마세요.

## 작은 숫자 워크드 — 원자적 저장

```text
write  ckpt.pt.tmp   (완전 기록)
fsync  （가능하면）
rename ckpt.pt.tmp → ckpt.pt
```

중간에 죽으면 옛 `ckpt.pt`가 남거나, tmp만 남습니다. **반쯤 쓰인 ckpt.pt**를 읽어 생기는 오류를 줄이려는 패턴입니다.

## 직관적으로 이해하기 — 세이브 포인트

게임의 세이브와 같습니다.

- `last`: 방금 그 자리로 복귀
- `best`: 가장 높았던 스테이지 점수
- export: 친구에게 캐릭터만 공유（장비=optim 제외）

SFT 이관은 보통 export에 가깝습니다.

## 부록 A. load 실패 체크리스트

1. `config`와 현재 모델 폭/층 불일치
2. `DataParallel`/`module.` prefix
3. dtype/device map
4. LoRA 키와 base 키 혼재
5. PyTorch 버전 불일치로 scaler state 거부

## 부록 B. 메타데이터 최소 필드

```python
meta = {
  "global_step": int,
  "tokens_seen": int,
  "best_val_loss": float,
  "git_commit": str,
  "config": dict,
}
```

숫자가 없어도 “어느 실험의 자식인지”를 잃습니다.

## 부록 C. 수식 카드

$$

\begin{aligned}
&\text{export}&& \theta\\
&\text{resume}&& \theta,\ m,\ v,\ \eta(\cdot),\ t,\ \text{scaler}\\
&\text{best}&& \theta(t_{\mathrm{best}})
\end{aligned}

$$


<!-- enrich-batch2-65 -->
## Checkpoint 내용

저장 최소셋:

$$
\mathrm{ckpt}=(\theta,\omega_{\mathrm{opt}},t,\eta,\mathrm{rng})
$$

```python
import torch
def save_ckpt(path, model, opt, step):
    # 재개에 필요한 상태만
    torch.save({
        "model": model.state_dict() if hasattr(model, "state_dict") else {},
        "opt": opt.state_dict() if hasattr(opt, "state_dict") else {},
        "step": step,
    }, path)
print("ok")
```

원자적 저장: `*.tmp` → rename.

## LLM에서는 어디에 사용될까?

이번 65강에서 배운 개념은 이후 Transformer · GPT · 서빙 강의에서 반복해서 등장합니다. 각 수식·코드 블록을 “실제 모델의 어느 단계인가”와 연결해 다시 읽어 보세요.

## 핵심 요약
- Resume 체크포인트는 model + optimizer + scheduler + step(+ scaler)을 묶는다.
- Last는 복구, Best는 지표 기준 선택이다. 역할을 파일명으로 분리한다.
- 원자적 저장(tmp+rename)과 로드 스모크 테스트가 운영 품질을 가른다.
- safetensors는 가중치 배포·로드의 **형식 옵션**으로 이해한다. resume 번들과 용도를 섞지 말 것.
- 제66강 val 지표가 best 갱신 트리거가 된다.

## 용어 사전
| 용어 | 의미 |
|---|---|
| Checkpoint | 학습 상태 스냅샷 |
| `state_dict` | 파라미터/버퍼 이름→텐서 맵 |
| Resume | 스냅샷에서 학습 재개 |
| Last / Best | 최근본 / 지표 최고본 |
| Export | 추론·이관용 가중치 내보내기 |
| Atomic save | 임시 기록 후 교체하는 저장 관행 |
| safetensors | 텐서 가중치 저장 형식 옵션 |
| `map_location` | 로드 시 장치 배치 |
| Rotation | 오래된 ckpt 삭제·유지 정책 |
| `strict=False` | 부분 키 로드 허용(위험 동반) |

## 연습문제
### 문제 1 (목록)

학습을 동일 lr 스케줄에서 이어가려면 최소 어떤 상태들이 필요한가?

### 문제 2 (개념)

Best와 Last를 모두 남기는 이유를 장애 복구와 모델 선택 관점에서 쓰시오.

### 문제 3 (절차)

다음 재개 코드의 문제를 지적하시오.

```python
model = GPT(config)
ckpt = torch.load("last.pt")
model.load_state_dict(ckpt["model"])
# optimizer/scheduler 새로 만든 채 학습 계속
```

### 문제 4 (형식)

Resume 번들과 safetensors export를 한 파일로 섞지 말라고 한 이유를 쓰시오.

### 문제 5 (연결)

제64강 `GradScaler`를 쓰는 FP16 학습에서 체크포인트에 scaler를 빼면 어떤 현상이 생길 수 있는가?

### 문제 6 (운영)

`best.pt`만 남기고 `last.pt`를 안 남기면 장애 복구에서 무엇이 불편해지는가?

---

## 정답 및 해설
### 문제 1

모델 가중치, Optimizer state, Scheduler state, global_step(또는 동등한 스케줄 축). 토큰 예산·scaler 등 사용 중이면 함께.

### 문제 2

Last는 언제 죽어도 가까운 지점에서 재개하기 위함. Best는 validation 등 기준으로 “지금까지 가장 나은” 가중치를 남겨 최종 선택·조기 사용에 쓰기 위함.

### 문제 3

Optimizer/Scheduler state와 step을 복원하지 않아 모멘트·lr 위치가 리셋된다. 가중치만 같은 “새 학습”에 가깝다.

### 문제 4

역할이 다르다. safetensors는 주로 가중치 맵용이고, resume는 optim 등 비텐서/부가 상태가 필요하다. 섞으면 로더·보안·크기·의도가 모호해진다.

### 문제 5

loss scale이 초기값으로 돌아가 재개 직후 overflow·step skip·불안정이 날 수 있다.

### 문제 6

best는 예전의 좋은 지점일 수 있어, 장애 직전 최신 상태(모멘트·step 포함)로 이어가기 어렵다. last가 복구의 기본이다.

## 다음 강의와 연결
Best를 고르려면 **기준 지표**가 필요하다.

다음 **제66강. Validation과 Evaluation**에서는 held-out validation loss, 평가 시점, `train`/`eval` 모드, 발산의 조기 신호를 다룬다. 그 지표가 이 강의의 `best.pt` 갱신 조건이 된다.

이전 강의: **제64강. Mixed Precision과 Gradient Accumulation**  
다음 강의: **제66강. Validation과 Evaluation**

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [64강. Mixed Precision과 Gradient Accumulation](64강_Mixed_Precision과_Gradient_Accumulation.md)
- **다음 강:** [66강. Validation과 Evaluation](66강_Validation과_Evaluation.md)

<!-- /LECTURE_NAV -->
