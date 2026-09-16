# 5권. vLLM · GLM · DGX Spark

## 제105강. PagedAttention

### 1. 이번 강의에서 배울 것

제101강에서 KV Cache를 “이미 본 토큰의 Key·Value를 GPU 메모리에 남겨 두고 decode를 싸게 만드는 장치”로 배웠다. 제104강에서는 vLLM이 그 KV를 **엔진 핵심**으로 다룬다는 지도를 잡았다. 이번 강의는 한 걸음 더 들어가, vLLM 논문·엔진이 강조한 **PagedAttention**의 직관을 고정한다.

이 강의를 마치면 다음을 말할 수 있어야 한다.

- 왜 요청마다 연속 큰 KV 버퍼를 잡으면 **내부 단편화(internal fragmentation)** 가 생기는가
- OS의 가상 메모리·페이지 테이블과 PagedAttention의 **블록 테이블(block table)** 이 어떻게 닮았는가
- Attention이 “논리적으로 연속인 KV”를 **물리적으로 흩어진 블록**에서 어떻게 읽는지
- Continuous Batching(제102·106강)과 결합할 때 메모리가 왜 “더 많은 동시 요청”으로 바뀌는가

**사실:** PagedAttention은 vLLM(UC Berkeley 등)이 제안·구현한 KV 관리 기법으로 널리 알려졌다.  
**해석·주의:** 블록 크기·스왑·접두사 공유(prefix caching) 세부는 엔진 버전·설정에 따라 달라진다. 이 강의는 **불변의 API 스펙이 아니라 교육용 직관**이다. 운영 시에는 해당 엔진 문서·릴리스 노트를 우선한다.

### 2. 왜 이것을 배우는가

서빙에서 GPU HBM은 대개 다음으로 갈라진다.

```text
가중치(weights)  +  KV cache  +  활성화·임시 버퍼
```

가중치는 모델이 정해지면 비교적 고정이다. 반면 KV는 **동시 요청 수 × 컨텍스트 길이 × 층 × 헤드 × dtype**에 비례해 늘어난다. 메모리가 부족하면 배치를 줄이거나 요청을 거절해야 한다.

문제는 “총 KV 바이트가 GPU에 들어가는가”만이 아니다. **할당 방식**이 나쁘면, 이론상으로는 여유가 있는데도 새 요청을 못 넣는 **단편화**가 생긴다. PagedAttention은 그 낭비를 OS가 쓰는 페이지 기법과 비슷한 방식으로 줄이려 한다.

### 3. 먼저 알아야 할 개념

1. Prefill / Decode와 KV 성장 (제100~101강)
2. Continuous Batching — 요청이 서로 다른 길이로 동시에 decode되는 그림 (제102강)
3. vLLM이 “엔진 = 스케줄러 + 실행 엔진 + 메모리 관리”라는 개요 (제104강)
4. (비유) OS 가상 메모리: 가상 주소 → 페이지 테이블 → 물리 프레임

### 4. 고전적 KV 할당의 병 — 예약과 낭비

단순 구현을 상상하자. 요청마다 최대 길이 `max_len`에 맞춰 **한 덩어리의 연속 버퍼**를 미리 잡는다.

```text
요청 A: max_len=2048 → KV[2048] 예약
실제로 쓴 길이: 180
남는 슬롯: 2048−180 = 1868  ← 다른 요청이 못 씀 (내부 단편화)
```

더 나쁜 경우:

- 생성 중에 길이가 늘어날 것을 대비해 **과대 예약**
- 요청이 끝나면 구멍이 여기저기 남아 **외부 단편화**에 가까운 상황
- Continuous Batching으로 요청이 자주 들어오고 나가면, “큰 연속 빈 공간”을 찾기 어려움

**사실:** 연속 할당 + 최대 길이 예약은 구현이 단순하지만, 가변 길이 트래픽에서는 메모리 이용률이 떨어지기 쉽다.  
**해석:** “GPU가 80GB인데도 동시 세션이 적다”는 증상 중 상당수는 **가중치 때문이 아니라 KV 할당 정책**에서 온다.

### 5. 가상 메모리 비유

운영체제는 프로세스에게 **거대한 연속 가상 주소 공간**을 보여 주고, 실제로는 물리 메모리를 **페이지(page)** 단위로 나눠 붙인다.

| OS | PagedAttention 비유 |
|---|---|
| 가상 주소 공간 | 요청의 **논리적** KV 시퀀스 (토큰 0…t) |
| 페이지 | 고정 길이 **KV 블록(block)** |
| 물리 프레임 | GPU HBM 위의 실제 블록 슬롯 |
| 페이지 테이블 | **블록 테이블(block table)** |
| page fault / 할당 | 새 토큰이 블록을 가득 채우면 **새 블록 할당** |

핵심 한 줄:

> Attention 커널은 “논리 위치 i의 K/V”를 읽되, 그 위치가 어느 **물리 블록·오프셋**인지는 블록 테이블로 찾는다.

그래서 논리적으로는 한 줄로 이어진 KV가, 물리적으로는 여기저기 흩어져 있어도 된다.

### 6. 블록과 블록 테이블

#### 6.1 블록

KV를 토큰 단위가 아니라 **블록 크기 $B_s$ 토큰**씩 묶는다. 예: $B_s=16$이면 토큰 0~15가 블록 0, 16~31이 블록 1.

각 물리 블록에는 해당 구간의 Key·Value(층·헤드 포함 설계에 따라)가 저장된다. 정확한 레이아웃은 엔진 구현에 맡긴다.

#### 6.2 블록 테이블

요청 $r$에 대해:

$$
\text{BlockTable}[r][j] = \text{물리 블록 ID}
$$

논리 블록 인덱스 $j$가 가리키는 물리 위치를 저장한다. Decode로 길이가 늘면 마지막 블록이 가득 찰 때 **새 물리 블록을 free list에서 받아** 테이블에 append한다.

```text
논리:  [tok0..15] [tok16..31] [tok32..40]
물리:    blk 7       blk 3       blk 19     ← 연속일 필요 없음
테이블:  7, 3, 19
```

#### 6.3 Attention 읽기

스텝 $t$에서 query는 과거 위치 $0..t$의 K/V와 연산한다. 구현 직관:

1. 논리 인덱스 → (블록 번호, 블록 내 오프셋)
2. 블록 테이블로 물리 블록 조회
3. 해당 물리 주소에서 K/V 로드
4. Softmax attention 계산

**사실:** “페이지처럼 나누면 attention 수식이 바뀐다”가 아니다. **저장·주소 변환**이 바뀔 뿐이다.  
**해석:** 커널이 gather에 가깝게 동작하므로, 블록이 너무 작으면 간접 참조 오버헤드가, 너무 크면 다시 내부 단편화가 커질 수 있다. 블록 크기는 트레이드오프다.

### 7. 작은 숫자로 보기 — 단편화 이득

가정(설명용, **벤치마크 주장 아님**):

- GPU에 올 수 있는 물리 KV 블록: 10개
- 블록 크기 $B_s=4$ 토큰
- 요청 최대 예약이 예전 방식이라면 요청당 16토큰(=4블록) 연속 예약

요청 세 개가 실제로는 각각 5, 6, 3 토큰만 쓰는 중이라고 하자.

**연속 최대 예약:**

- 요청당 4블록 × 3 = 12블록 필요 → 물리 10개로는 **동시 3개 불가**

**페이지(블록) 할당:**

- 5토큰 → 2블록, 6토큰 → 2블록, 3토큰 → 1블록 → 합 5블록  
- 남는 5블록으로 **추가 요청**을 받을 여지가 생김

이 숫자 자체는 교재용 미니어처다. 실측 GB·처리량은 제107·118강 방식의 **보고 템플릿**으로만 적는다.

### 8. 직관적으로 이해하기

도서관 비유:

- 고전 방식: 손님마다 “최대 빌릴 수 있는 책장 한 줄”을 통째로 예약. 책은 세 권인데 책장 전체가 잠김.
- PagedAttention: 책을 **칸(블록)** 단위로만 빌려 줌. 손님이 더 빌리면 빈 칸을 추가로 붙임. 논리적으로는 “내 대출 목록”이 이어지지만, 물리 책장은 흩어져 있어도 대출 장부(블록 테이블)로 찾는다.

Continuous Batching과 만나면:

- 어떤 요청은 방금 끝나 블록을 반납하고
- 어떤 요청은 한 토큰 더 써서 블록 하나를 새로 잡고
- 스케줄러는 “지금 free 블록이 몇인가”로 새 prefill을 받을지 결정한다 (제106강)

### 9. 코드로 스케치하기 (교육용)

아래는 **엔진 코드가 아니다.** 블록 테이블 아이디어만 담은 의사코드다.

```python
from dataclasses import dataclass, field

BLOCK_SIZE = 4


@dataclass
class BlockAllocator:
    num_blocks: int
    free: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.free = list(range(self.num_blocks))

    def alloc(self) -> int:
        if not self.free:
            raise RuntimeError("OOM: no free KV blocks")
        return self.free.pop()

    def free_block(self, bid: int) -> None:
        self.free.append(bid)


@dataclass
class Sequence:
    seq_id: str
    block_table: list[int] = field(default_factory=list)
    num_tokens: int = 0

    def append_token(self, allocator: BlockAllocator) -> None:
        # 새 토큰이 새 블록을 필요로 하면 할당
        need_new = (self.num_tokens % BLOCK_SIZE == 0)
        if need_new:
            self.block_table.append(allocator.alloc())
        self.num_tokens += 1

    def logical_to_physical(self, logical_idx: int) -> tuple[int, int]:
        blk = logical_idx // BLOCK_SIZE
        off = logical_idx % BLOCK_SIZE
        return self.block_table[blk], off
```

Attention 쪽은 “모든 논리 위치에 대해 `logical_to_physical`로 gather”라고 생각하면 된다. 실제 커널은 이 루프를 융합·벡터화한다.

### 10. 실제 LLM Serving에서의 위치

PagedAttention이 풀어 주는 것:

| 이득 | 설명 |
|---|---|
| 내부 단편화 감소 | 최대 길이 전체를 미리 연속 예약하지 않음 |
| 가변 길이 친화 | 요청마다 다른 생성 길이에 맞춰 블록만 증가 |
| 높은 동시성 | 같은 HBM으로 더 많은 시퀀스를 running에 유지하기 쉬움 |
| 스케줄러 연동 | free block 수가 admission·preemption 기준이 됨 |

여전히 남는 것:

- 가중치 자체 크기, 통신(TP), 커널 효율
- 초장문에서 attention 계산량 $O(T)$ per step의 누적
- 양자화(제103강)·스펙큘레이티브(제110강) 등 다른 축

**사실 vs 마케팅:** “PagedAttention = 무조건 N배 빨라짐”은 과장이다. 이득의 상당 부분은 **메모리 이용률 → 더 큰 유효 배치 → 처리량** 경로로 온다. 단일 요청 latency만 보면 주소 변환 오버헤드가 보일 수도 있다. 측정은 제107강 지표로 분리한다.

### 11. Prefix 공유와 Copy-on-Write 직관

시스템 프롬프트·도구 명세·RAG 헤더처럼 **여러 요청이 같은 접두사**를 공유하면, 논리적으로는 같은 KV를 두 번 쓸 이유가 없다.

페이지/블록 세계에서는 대략 이렇게 확장된다.

```text
요청 A: [sys][userA...]   블록: S0 S1 | A2 A3
요청 B: [sys][userB...]   블록: S0 S1 | B2 B3
                 ↑ 공유        ↑ 분기 이후 전용
```

공유 구간은 참조 카운트를 올리고, 분기 이후부터 새 블록을 붙인다. OS의 copy-on-write와 같은 냄새다.

**사실:** prefix caching / automatic prefix caching 등은 엔진·버전 기능이다. 켜짐·해시 키·무효화 규칙은 문서에 따른다.  
**해석:** PagedAttention의 블록 테이블이 있어서 “물리 블록을 여러 논리 시퀀스가 가리키기”가 자연스럽다. 연속 거대 버퍼만 있으면 접두사 공유 구현이 훨씬 아프다.

서빙 함의:

- 동일 시스템 프롬프트 트래픽 → prefill 비용·KV 메모리 절감 여지
- 캐시 히트율은 워크로드 의존 — 숫자를 책에 고정하지 말 것
- 캐시 키에 템플릿·토크나이저·LoRA 어댑터 ID가 들어가면 “같아 보이는 문자열”도 미스날 수 있다

### 12. 블록 크기 $B_s$ 트레이드오프를 표로

| $B_s$ 작음 | $B_s$ 큼 |
|---|---|
| 내부 단편화 ↓ | 내부 단편화 ↑ (마지막 블록 빈칸) |
| 테이블·간접 참조 ↑ | 주소 변환 횟수 ↓ |
| 할당/해제 빈번 | 관리 단순 |
| 짧은 요청에 유리한 경향 | 긴 요청·관리 오버헤드 민감 |

실무에서는 엔진 기본값을 출발점으로 삼고, **제107강 템플릿**으로 A/B한다. “항상 16이 최적” 같은 문장은 쓰지 않는다.

### 13. 외부 단편화 vs 내부 단편화 — 한 번 더

- **내부:** 할당받은 공간 안의 빈칸 (최대 길이 예약, 또는 블록 내 미사용 슬롯)
- **외부:** 총 빈 용량은 있는데 **원하는 모양(큰 연속 구간)** 이 없어 할당 실패

PagedAttention이 특히 겨냥하는 것은, 가변 길이 트래픽에서 연속 할당이 만들던 **실질적 낭비·할당 실패**다. 블록 풀은 “연속 거대 구멍”을 요구하지 않으므로, 외부 단편화에 가까운 실패 모드가 줄어든다.

여전히 남는 실패:

```text
free_blocks == 0  →  새 토큰·새 요청 불가 → 스케줄러 preemption/거절 (제106강)
```

### 14. Attention 수식은 그대로 — 주소만 바뀐다

위치 $t$의 쿼리 $q_t$에 대해 과거 $i\le t$의 키·값과 연산하는 정의는 동일하다.

$$
\alpha_{ti}=\mathrm{softmax}_i\left(\frac{q_t^\top k_i}{\sqrt{d}}\right),\quad
o_t=\sum_i \alpha_{ti} v_i
$$

구현만:

$$
(k_i,v_i)=\mathrm{Load}\big(\mathrm{BlockTable}[\lfloor i/B_s\rfloor],\ i\bmod B_s\big)
$$

**사실:** 수학적 attention과 페이지화된 저장은 직교 관심사다.  
**해석:** “PagedAttention 논문 = 새로운 attention 변형”으로 오해하면 FlashAttention류 커널 이야기와 섞인다. 커널 융합·IO 최적화는 **별 축**이다.

### 15. Continuous Batching과 결합하는 메모리 시계열

한 스텝의 블록 회계(교육용):

```text
t=0  free=100
  admit req1 (prompt 로 블록 5 소비)  free=95
  admit req2 (블록 3)                 free=92
t=1  decode: req1 새 블록 필요 없음   free=92
t=2  decode: req1 블록 +1             free=91
t=3  req2 종료, 블록 3 반납           free=94
t=4  admit req3 ...
```

스케줄러는 `free`를 **입장 티켓**으로 본다. PagedAttention이 없으면 “연속 구멍 찾기”가 티켓 검사보다 복잡해진다.

### 16. 스왑·오프로드 — 있을 수도 있는 확장

일부 엔진은 GPU 블록이 부족할 때 KV를 CPU 메모리로 옮겼다가 되돌리는 **스왑**을 옵션으로 둔다.

| | GPU 블록만 | + CPU 스왑 |
|---|---|---|
| 동시성 | HBM 한계 | 논리 동시성 ↑ 여지 |
| 지연 | 상대적으로 안정 | PCIe 왕복 비용·지터 |
| 복잡도 | 낮음 | 스케줄·핀닝·단편화 관리 ↑ |

**사실:** 지원 여부·기본값은 버전 의존.  
**해석:** 스왑은 PagedAttention의 필수 정의가 아니라 **운영 확장**이다. “페이징 = 디스크 스왑”으로 학생에게 가르치지 말 것.

### 17. PyTorch로 미니 블록 풀 + gather 스케치

아래는 **학습용 미니어처**다. 실제 multi-head·다층·dtype·커널은 생략한다.

```python
import torch

class MiniPagedKV:
    def __init__(self, num_blocks: int, block_size: int, dim: int):
        self.block_size = block_size
        # 물리 풀: [num_blocks, block_size, dim]
        self.K = torch.zeros(num_blocks, block_size, dim)
        self.V = torch.zeros(num_blocks, block_size, dim)
        self.free = list(range(num_blocks))

    def alloc_block(self) -> int:
        if not self.free:
            raise MemoryError("no free KV blocks")
        return self.free.pop()

    def free_blocks(self, ids: list[int]) -> None:
        self.free.extend(ids)

    def append(self, table: list[int], k: torch.Tensor, v: torch.Tensor, n: int) -> list[int]:
        """n: 현재 시퀀스 토큰 수(append 전). k,v: [dim]."""
        if n % self.block_size == 0:
            table = table + [self.alloc_block()]
        bid = table[n // self.block_size]
        off = n % self.block_size
        self.K[bid, off] = k
        self.V[bid, off] = v
        return table

    def gather(self, table: list[int], length: int) -> tuple[torch.Tensor, torch.Tensor]:
        ks, vs = [], []
        for i in range(length):
            bid = table[i // self.block_size]
            off = i % self.block_size
            ks.append(self.K[bid, off])
            vs.append(self.V[bid, off])
        return torch.stack(ks, 0), torch.stack(vs, 0)


def tiny_attend(q, K, V):
    # q: [d], K,V: [T,d]
    scores = torch.matmul(K, q) / (q.numel() ** 0.5)
    w = torch.softmax(scores, dim=0)
    return torch.matmul(w, V)


pool = MiniPagedKV(num_blocks=8, block_size=4, dim=8)
table: list[int] = []
for t in range(10):
    k = torch.randn(8)
    v = torch.randn(8)
    table = pool.append(table, k, v, n=t)
q = torch.randn(8)
K, V = pool.gather(table, length=10)
out = tiny_attend(q, K, V)
print("blocks_used", table, "out_norm", float(out.norm()))
```

실습 관찰 포인트:

1. `table`이 비연속 정수여도 `gather` 결과가 논리 순서대로인지
2. `num_blocks`를 줄여 `MemoryError`를 재현 → 제106강 preemption 동기
3. `block_size`를 키워 마지막 블록 낭비(내부 단편화)를 손으로 세기

### 18. 실습 — 단편화 시뮬레이션 체크리스트

벤치마크 숫자를 만들지 말고, **구조 실험**만 한다.

1. 요청 길이 분포를 두 가지 둔다: (a) 거의 최대 길이 (b) 평균≪최대
2. 연속 예약 모델: 요청당 `max_len` 슬롯 점유
3. 블록 모델: `ceil(len / B_s)` 블록 점유
4. 같은 총 슬롯 예산에서 **동시 수용 가능 요청 수**를 비교
5. 결과를 표로 남기되, GPU 실측이라고 쓰지 말 것

보고 형식 예:

```text
sim_only: true
allocator: contiguous_max | paged_blocks
max_len: <N>
block_size: <Bs>
workload: <file or generator seed>
admitted_seqs: <count>
# 실측 GPU 지표 금지 — 제107 템플릿은 실측용
```

### 19. 실제 LLM Serving 시나리오 세 가지

#### 19.1 챗 동시성

짧은 질문·중간 길이 답이 섞인다. 연속 예약은 빈 칸이 많고, 블록 할당은 동시 세션을 더 받기 쉽다 → **Throughput·큐 대기**에 영향 (측정은 템플릿).

#### 19.2 장문 요약

프롬프트가 길어 prefill KV 블록을 한꺼번에 소모한다. Admission이 까다로워지고 TTFT 꼬리가 길어진다. Paging이 있어도 **총량**은 커진다.

#### 19.3 공유 시스템 프롬프트

Prefix 공유가 켜져 있으면 블록 재사용으로 입장 비용이 줄어들 여지. 꺼져 있으면 매 요청이 같은 접두사 KV를 중복 보유한다.

### 20. 자주 하는 실수

1. **KV를 없앤다고 오해** — 페이지화는 관리 기법이지, 저장량을 마법처럼 0으로 만들지 않는다.
2. **블록 크기 무시** — 마지막 블록의 빈 슬롯은 작은 내부 단편화로 남는다. $B_s$가 크면 낭비가 다시 커진다.
3. **OS 스왑과 동일시** — 일부 엔진은 CPU 오프로드/스왑을 지원하지만, 기본 직관은 GPU 내 블록 풀이다. 디스크 스왑과 혼동하지 말 것.
4. **수식 변경으로 착각** — Softmax attention 정의는 같고, 메모리 레이아웃이 다르다.
5. **버전 고착** — prefix caching, 블록 크기 기본값, 멀티모달 KV 등은 릴리스마다 확장된다. 문서의 해당 버전을 인용하라.
6. **단편화 이득 = FLOPs 이득**으로 혼동 — 대개 동시성·메모리 이용률 경로다.
7. **실측 없이 “N배”** — 제107강 필드 없이 속도 주장을 쓰지 말 것.
8. **스케줄러와 분리해서만 이해** — free block은 스케줄 화폐다.

### 21. 핵심 정리

- 서빙 메모리는 가중치뿐 아니라 **KV 할당 정책**에 좌우된다.
- PagedAttention은 KV를 **고정 크기 블록**으로 나누고 **블록 테이블**로 논리→물리를 연결한다.
- OS 가상 메모리와 같은 이유로 **단편화 손실을 줄이고** Continuous Batching과 잘 맞는다.
- 이득의 본체는 종종 “같은 GPU에서 **동시 시퀀스를 더**”이다.
- Prefix 공유·스왑은 확장축이다. 기본 직관과 섞어 단정하지 말 것.
- 엔진 세부는 진화한다. **직관은 유지, 숫자는 실측·문서**.

### 22. 핵심 용어

| 용어 | 설명 |
|---|---|
| PagedAttention | 블록 단위 KV + 블록 테이블로 attention이 KV를 참조하는 기법 |
| KV block | 고정 토큰 수만큼의 K/V 저장 단위 |
| Block table | 논리 블록 → 물리 블록 ID 매핑 |
| Internal fragmentation | 예약한 공간 중 실제로 안 쓰는 낭비 |
| External fragmentation | 총 여유는 있으나 원하는 연속 구간이 없어 실패 |
| Free block pool | 할당 가능한 물리 블록 집합 |
| Prefix caching | 공통 접두사 KV 블록 재사용(엔진 기능) |
| Copy-on-write (비유) | 공유 블록을 가리키다 분기 후 전용 블록을 붙이는 감각 |

### 23. 복습 문제

#### 문제 1（개념）

PagedAttention이 OS 가상 메모리와 닮은 점 두 가지를 쓰시오.

#### 문제 2（계산·미니）

$B_s=16$, 현재 토큰 수 50일 때 논리 블록은 몇 개인가? 51번째 토큰을 추가하면 블록 수는?

#### 문제 3（단편화）

`max_len=1024`를 연속 예약한 요청이 실제 80토큰만 썼다. 내부 단편화 관점에서 무엇이 낭비인가?

#### 문제 4（사실/해석）

“PagedAttention을 쓰면 attention FLOPs가 줄어든다”는 문장은 사실인가, 해석 오류인가? 한 줄로 근거를 대시오.

#### 문제 5（연결）

Continuous Batching에서 요청이 끝날 때 스케줄러가 가장 먼저 돌려받고 싶어 하는 자원은 무엇인가? (PagedAttention 용어로)

#### 문제 6（트레이드오프）

블록 크기 $B_s$를 매우 크게 잡으면 어떤 낭비가 다시 커지는가?

#### 문제 7（주소）

논리 인덱스 $i=37$, $B_s=16$일 때 블록 번호와 블록 내 오프셋은?

#### 문제 8（prefix）

두 요청이 시스템 프롬프트 블록을 공유하다가 사용자 구간에서 갈라질 때, 블록 테이블 관점에서 무엇이 갈라지는가?

#### 문제 9（실습）

미니 시뮬에서 contiguous와 paged의 `admitted_seqs`만 비교하고 GPU tokens/s를 같은 표에 섞으면 안 되는 이유는?

#### 문제 10（사실/해석）

“페이징이 있으니 KV가 CPU 디스크로 항상 스왑된다”를 평가하시오.

---

### 정답 및 해설

#### 문제 1

(예시) ① 논리적으로 연속인 공간을 물리적으로 흩어진 단위로 뒷받침한다. ② 테이블(페이지/블록 테이블)로 주소 변환한다.

#### 문제 2

$\lceil 50/16\rceil=4$개. 51은 새 블록이 필요하므로 5개.

#### 문제 3

예약 1024 중 사용 80 → 나머지 슬롯(및 그에 묶인 KV 용량)이 다른 요청에 재사용되지 못하는 낭비.

#### 문제 4

해석 오류(또는 과장). 주소 지정·레이아웃이 바뀔 뿐, 토큰 수에 대한 attention 연산량 자체는 같은 알고리즘이면 본질적으로 같다.

#### 문제 5

해당 시퀀스의 **물리 KV 블록**(free pool로 반납).

#### 문제 6

블록 단위 내부 단편화(마지막·과대 블록의 빈 칸).

#### 문제 7

블록 $\lfloor 37/16\rfloor=2$, 오프셋 $37\bmod 16=5$.

#### 문제 8

공유 접두사 블록 ID는 공통으로 남고, 분기 이후 논리 위치에 새 물리 블록 ID가 붙는다.

#### 문제 9

전자는 할당 정책 시뮬이고 후자는 하드웨어·커널 실측이라 **원인 축이 다름**. 제107 템플릿 없이 섞으면 허위 비교가 된다.

#### 문제 10

사실이 아님. 기본 직관은 GPU 블록 풀이며, CPU/디스크 스왑은 **옵션·버전 의존 확장**이다.

### 24. 다음 강의와 연결

메모리를 블록으로 잘게 쓸 수 있게 되면, 다음은 **누구를 언제 GPU에 올릴지**다.  
다음 **제106강. vLLM Scheduler**에서는 waiting/running 큐, preemption, Continuous Batching 스케줄러의 직관을 다룬다.

> 블록이 화폐라면, 스케줄러는 그 화폐를 언제 주고받을지 정하는 중앙은행이다.

이전 **제104강. vLLM 개요와 구조**에서 엔진 지도를 펼쳤다면, 이번 강은 그 지도의 **메모리 층**을 채운 셈이다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [제104강. vLLM 개요와 구조](104강_vLLM_개요와_구조.md)
- **다음 강:** [제106강. vLLM Scheduler](106강_vLLM_Scheduler.md)

<!-- /LECTURE_NAV -->
