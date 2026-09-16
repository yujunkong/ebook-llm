# 5권. vLLM · GLM · DGX Spark

## 제116강. 프로젝트 — 실제 LLM Serving

### 1. 이번 강의에서 배울 것

제99~115강에서 Inference 기초, vLLM 개념, 지표, 모델 계열, GPU·엔진·TP·NCCL·DGX Spark 환경까지 쌓았다. 이번 강의는 **한 바퀴 완주 프로젝트**다. 모델을 실제로(또는 재현 가능한 스케치로) 서빙하고, **TTFT·TPOT**를 템플릿으로 측정하며, 병목을 해석한다.

이 강의를 마치면 다음을 할 수 있어야 한다.

- 서빙 올리기 **체크리스트**(환경 → 모델 → 엔진 → API → 관측)를 순서대로 수행한다.
- OpenAI 호환 엔드포인트로 요청을 보내고 스트리밍/비스트리밍을 구분한다.
- TTFT / TPOT / 처리량 측정 **템플릿**을 채워 해석한다.
- 실패·느림의 원인을 capacity / bandwidth / 스케줄 / 네트워크 후보로 분류한다.
- 제117강 GPU 최적화 실험으로 넘길 **가설 목록**을 남긴다.

목표가 아니다: 리더보드 SOTA, 멀티테넌트 상용 플랫폼 완성.

목표가 맞다: **“띄운다 → 잰다 → 해석한다 → 다음 실험을 정한다”** 루프의 증거물(`artifacts/`)을 남기는 것.

### 2. 왜 이것을 배우는가

문서만 읽으면 서빙은 한 줄이다다.

```bash
vllm serve <MODEL>
```

현장에서는 그 한 줄 앞뒤에 다음이 붙는다.

| 숨은 작업 | 빠지면 생기는 일 |
|---|---|
| 드라이버·CUDA·이미지 정합 | import/시작 실패 |
| 모델 라이선스·토큰 | 다운로드 403 |
| dtype·양자화·max length | OOM 또는 품질 이상 |
| 워밍업 | 첫 측정만 느려 오판 |
| 동시성 시나리오 | 단일 요청 환상 |
| 메트릭 정의 통일 | 팀 간 숫자 논쟁 |

제95강 Preference 프로젝트가 “선호 루프 한 바퀴”였듯, 이번은 **서빙 루프 한 바퀴**다.

### 3. 프로젝트 목표와 성공 기준

#### 3.1 목표

1. (경로 A) GPU가 있는 환경에서 **vLLM으로 소형~중형 모델 서빙**  
2. (경로 B) GPU가 없으면 **클라이언트·측정·가짜 서버 스케치**로 동일 산출물 형식 완성  
3. TTFT/TPOT 측정 스크립트와 `artifacts/report.md` 작성  
4. 병목 가설 3개와 제117강용 실험안 작성  

#### 3.2 성공 기준

- [ ] `GET /v1/models` 또는 동등 헬스 확인이 된다 (경로 B는 mock)  
- [ ] 동일 프롬프트 세트로 N회 측정 로그가 남는다  
- [ ] TTFT·TPOT 정의가 report에 명시된다 (제107강과 일치)  
- [ ] 환경 섹션에 하드웨어·엔진 버전·모델 리비전이 있다  
- [ ] “다음 실험”이 숫자 없는 희망이 아니라 **조작 변수**로 적혀 있다  

경로 A를 권장한다. 경로 B만 완료해도 **측정 규율**은 통과다. 단 report에 GPU 미사용을 명시한다.

### 4. 권장 디렉터리

```text
ch116_llm_serving/
├── README.md
├── checklist.md           # 본문 §6을 실행하며 체크
├── configs/
│   └── serve_env.example  # 환경변수 템플릿
├── prompts/
│   ├── short.jsonl
│   ├── long_prefix.jsonl
│   └── mixed.jsonl
├── scripts/
│   ├── smoke_chat.py
│   ├── measure_ttft_tpot.py
│   ├── summarize_metrics.py
│   └── mock_openai_server.py   # 경로 B
├── artifacts/
│   ├── env_snapshot.txt
│   ├── raw_timings.jsonl
│   ├── metrics_summary.json
│   └── report.md
└── notes/
    └── bottlenecks.md
```

### 5. 두 경로

```text
경로 A (권장): 실 GPU + vLLM (또는 팀 표준 엔진) + 측정
경로 B (대체): mock 서버 + 동일 측정 스크립트 + “실측 시 절차” 문서화
```

| | 경로 A | 경로 B |
|---|---|---|
| 필요 | GPU·드라이버·엔진 | Python만 |
| 산출물 | 실측 숫자 | 파이프라인·정의 검증 |
| 한계 | 환경 의존 | 성능 결론 금지 |

### 6. 서빙 체크리스트 (Step-by-step)

실행하며 `checklist.md`에 날짜·결과·로그 경로를 적는다.

#### Step 1 — 환경 스냅샷

```bash
date -Is | tee artifacts/env_snapshot.txt
echo "=== GPU ===" | tee -a artifacts/env_snapshot.txt
nvidia-smi >> artifacts/env_snapshot.txt 2>&1 || echo "no nvidia-smi" >> artifacts/env_snapshot.txt
echo "=== Python ===" | tee -a artifacts/env_snapshot.txt
python -V >> artifacts/env_snapshot.txt
pip show vllm 2>/dev/null | tee -a artifacts/env_snapshot.txt || true
```

2× DGX Spark라면 제115강 환경 노트(고속 IF, NCCL 요약)를 여기에 붙인다.

#### Step 2 — 모델 선택 원칙

초보 프로젝트 권장:

1. **라이선스가 명확**한 소형 Instruct 모델  
2. 단일 GPU(또는 단일 Spark)에 **여유 있게** 들어가는 크기  
3. 토크나이저·chat template이 엔진과 맞음  

거대 모델·TP=2는 **단일 노드 smoke 성공 후** 확장(제113~115강).

#### Step 3 — 서버 기동 (경로 A 스케치)

옵션명은 버전별로 다르다. 실행 전 헬프를 본다.

```bash
# 스케치 — MODEL_ID, 포트를 채울 것
export MODEL_ID="REPLACE_ME"
export PORT=8000

vllm serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 4096
  # 필요 시: --dtype, --quantization, --tensor-parallel-size, ...
```

기동 로그에서 확인할 것:

- 모델 로드 완료  
- KV/캐시 관련 초기화 메시지  
- 바인딩 주소·포트  

#### Step 4 — 헬스·모델 목록

```bash
curl -s "http://127.0.0.1:${PORT}/v1/models" | head
```

실패 시: 방화벽, 바인드 주소, 아직 로딩 중, 잘못된 포트.

#### Step 5 — Smoke chat

```bash
python scripts/smoke_chat.py --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL_ID"
```

#### Step 6 — 워밍업

측정 전 **동일 유형 요청**을 수회 버려 콜드스타트·컴파일·그래프 캡처를 분리한다.  
report에 `warmup_runs: N` 을 명시한다.

#### Step 7 — 본측정

```bash
python scripts/measure_ttft_tpot.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL_ID" \
  --prompts prompts/mixed.jsonl \
  --out artifacts/raw_timings.jsonl \
  --concurrency 1 \
  --repeats 20
```

이후 동시성 4, 8 등으로 **별도 파일**을 남긴다. 한 파일에 섞지 말 것.

#### Step 8 — 요약·리포트

```bash
python scripts/summarize_metrics.py \
  --in artifacts/raw_timings.jsonl \
  --out artifacts/metrics_summary.json

# report.md 는 템플릿(§10)을 채워 커밋
```

### 7. 측정 정의 — 다시 고정

제107강과 맞춘다. report 상단에 그대로 붙인다.

| 지표 | 정의 (이 프로젝트) |
|---|---|
| **TTFT** | 요청 송신 시각 → **첫 토큰(또는 첫 chunk) 수신** 시각 |
| **TPOT** | 첫 토큰 이후 생성 토큰들에 대한 평균 간격 (스트림 기준) |
| **E2E** | 요청 → 완료 |
| **Throughput** | 단위 시간당 완료 요청 또는 생성 토큰 (시나리오에 명시) |

주의:

1. 비스트리밍만 쓰면 TTFT를 **근사**만 할 수 있다. 가능하면 스트림.  
2. 토크나이저 왕복·네트워크 지연이 클라이언트에 포함된다. **측정 위치가 클라이언트**임을 명시.  
3. 서버 내부 메트릭이 있으면 나란히 두되, 정의를 섞지 말 것.

### 8. 실행 가능 스케치 코드

#### 8.1 `scripts/smoke_chat.py`

```python
#!/usr/bin/env python3
"""OpenAI-compatible chat smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def post_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="Say hello in one short sentence.")
    args = p.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
    }
    out = post_json(args.base_url.rstrip("/") + "/chat/completions", payload)
    try:
        text = out["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print("Unexpected response:", json.dumps(out, ensure_ascii=False)[:1000])
        raise SystemExit(2) from e
    print("SMOKE_OK:", text.replace("\n", " ")[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

#### 8.2 `scripts/measure_ttft_tpot.py` (스트림 기준)

```python
#!/usr/bin/env python3
"""
Measure TTFT / TPOT against an OpenAI-compatible streaming chat API.

Definitions (client-side):
  TTFT = t_first_text_chunk - t_request_start
  TPOT = mean interval between subsequent text chunks that contain tokens
         (approximation if API lacks token timestamps)

This is a teaching sketch — adapt to your server's SSE format.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any, Iterator


def iter_sse_chat(url: str, payload: dict, timeout: float = 300.0) -> Iterator[dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            yield json.loads(body)


def extract_text_delta(event: dict[str, Any]) -> str:
    try:
        delta = event["choices"][0].get("delta") or {}
        content = delta.get("content")
        return content or ""
    except (KeyError, IndexError, TypeError):
        return ""


def load_prompts(path: str) -> list[dict[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def measure_one(base_url: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    t0 = time.perf_counter()
    t_first = None
    chunk_times: list[float] = []
    pieces: list[str] = []

    for ev in iter_sse_chat(url, payload):
        text = extract_text_delta(ev)
        if not text:
            continue
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        else:
            chunk_times.append(now)
        pieces.append(text)

    t_end = time.perf_counter()
    if t_first is None:
        return {
            "ok": False,
            "error": "no_text_chunks",
            "ttft_ms": None,
            "tpot_ms": None,
            "e2e_ms": (t_end - t0) * 1000,
        }

    # TPOT approximation: gaps between successive non-empty chunks after the first
    gaps = []
    prev = t_first
    for t in chunk_times:
        gaps.append(t - prev)
        prev = t
    tpot_ms = (sum(gaps) / len(gaps) * 1000) if gaps else None

    return {
        "ok": True,
        "ttft_ms": (t_first - t0) * 1000,
        "tpot_ms": tpot_ms,
        "e2e_ms": (t_end - t0) * 1000,
        "output_chars": sum(len(x) for x in pieces),
        "num_chunks_after_first": len(gaps),
        "note": "TPOT is inter-chunk mean; for token-accurate TPOT use server token timestamps if available",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=1, help="documented only in v1 sketch")
    args = ap.parse_args()

    if args.concurrency != 1:
        print("This sketch runs sequential requests; for concurrency use a load tool or extend the script.")

    prompts = load_prompts(args.prompts)
    with open(args.out, "w", encoding="utf-8") as out:
        for r in range(args.repeats):
            row = prompts[r % len(prompts)]
            prompt = row.get("prompt") or row.get("content") or ""
            result = measure_one(args.base_url, args.model, prompt, args.max_tokens)
            rec = {
                "repeat": r,
                "prompt_id": row.get("id", f"idx-{r % len(prompts)}"),
                "concurrency": args.concurrency,
                **result,
                "ts": time.time(),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps({k: rec[k] for k in ("repeat", "ok", "ttft_ms", "tpot_ms", "e2e_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

설명: chunk ≠ token 일 수 있다. report에 **근사임**을 적는다. 엔진이 토큰 사용량·타임스탬프를 주면 그쪽으로 고친다.

#### 8.3 `scripts/summarize_metrics.py`

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if r.get("ok")]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None]
    e2e = [r["e2e_ms"] for r in ok if r.get("e2e_ms") is not None]

    def pack(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": stats.fmean(xs),
            "p50": percentile(xs, 50),
            "p95": percentile(xs, 95),
            "max": max(xs),
            "min": min(xs),
        }

    summary = {
        "num_rows": len(rows),
        "num_ok": len(ok),
        "ttft_ms": pack(ttft),
        "tpot_ms": pack(tpot),
        "e2e_ms": pack(e2e),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

#### 8.4 경로 B — `scripts/mock_openai_server.py`

```python
#!/usr/bin/env python3
"""Minimal streaming mock for measuring client scripts without a GPU."""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path.startswith("/v1/models"):
            body = json.dumps({"data": [{"id": "mock-model", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        req = json.loads(raw.decode("utf-8"))
        stream = bool(req.get("stream"))
        tokens = ["Hello", " from", " mock", " server", "."]

        if not stream:
            body = json.dumps(
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": "".join(tokens)}}
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.05)  # fake TTFT
        for t in tokens:
            time.sleep(0.02)  # fake TPOT-ish
            ev = {"choices": [{"delta": {"content": t}}]}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
```

경로 B report에는 반드시 적는다: **“수치는 mock이며 하드웨어 성능이 아니다.”**

### 9. 프롬프트 세트

`prompts/short.jsonl` 예:

```json
{"id": "s1", "prompt": "한국어로 한 문장 인사해 줘."}
{"id": "s2", "prompt": "2+2는?"}
{"id": "s3", "prompt": "HTTP가 무엇인지 한 문장으로."}
```

`prompts/long_prefix.jsonl` 예 아이디어:

- 동일 긴 시스템 설명 + 짧은 질문 여러 개 → **prefix cache** 가설용 (제112강)
- 서로 다른 긴 본문 → 캐시 미스 대조

실서비스 데이터를 넣지 말 것. PII·비밀키 금지.

### 10. `artifacts/report.md` 템플릿

```markdown
# Ch116 Serving Report

## 1. 환경
- 날짜:
- 하드웨어: (예: 단일 GPU / 1× Spark / 2× Spark)
- 엔진·버전:
- 모델 ID·리비전·dtype/quant:
- 병렬: TP= / PP= / replica=
- 네트워크(해당 시): 고속 IF, NCCL 요약
- 경로: A / B

## 2. 지표 정의
- TTFT:
- TPOT:
- 측정 위치: client streaming sketch
- warmup_runs:

## 3. 시나리오
| 이름 | 프롬프트 세트 | concurrency | max_tokens | n |
|---|---|---|---|---|

## 4. 결과 (요약 숫자만 표로 — 발명 금지, 실측만)
| 시나리오 | TTFT p50/p95 | TPOT p50/p95 | E2E p50 | 비고 |
|---|---|---|---|---|

## 5. 해석
- 가장 의 병목 후보:
- 근거 (로그·utilization·메모리·NCCL):
- 기각한 가설:

## 6. 제117강으로 넘길 실험
| 가설 | 조작 변수 | 고정 변수 | 성공 기준 |
|---|---|---|---|
```

### 11. 병목 해석 가이드

| 관찰 | 후보 원인 | 다음 레버 (제117 연결) |
|---|---|---|
| TTFT만 큼 | prefill, 긴 프롬프트, 콜드캐시 | max batched tokens, prefix cache, 양자화 |
| TPOT만 큼 | decode memory-bound, 작은 배치 | 동시성, quant, 커널/그래프 |
| 동시성↑ 시 급격 악화 | 스케줄·메모리 압박 | gpu memory util, 길이 제한 |
| TP=2가 더 느림 | NCCL/네트워크 | 제114~115 재검증, PP 실험 |
| OOM | capacity | 양자화, 짧은 ctx, TP/PP |
| utilization 낮은데 느림 | CPU·토크나이저·클라이언트·락 | 프로파일, 배치 유입 |

단정하지 말고 **가설 → 한 변수 변경 → 재측정** (제118강 리포트 규율).

### 12. 도전 과제 (Challenges)

필수 1개 + 선택:

1. **필수**: concurrency 1 vs 4 비교 표  
2. **선택**: 짧은 프롬프트 vs 긴 prefix 공유 세트  
3. **선택**: dtype/quant 변경 A/B (지원될 때)  
4. **선택**: 2× Spark에서 TP=2 vs PP=2 smoke (제115강 완료자)  
5. **선택**: 동일 측정을 다른 엔진(제112강)으로 반복 — 공정 비교 체크리스트 첨부  

각 도전은 report §6 표에 한 줄로 남긴다.

### 13. 자주 하는 실수

1. 워밍업 없이 첫 요청만 보고 “느린 엔진” 결론  
2. 스트림 없이 TTFT를 자신 있게 발표  
3. 동시성 테스트 없이 프로덕션 용량 계획  
4. mock 숫자를 하드웨어 성능처럼 공유  
5. 환경 스냅샷 없이 옵션만 잔뜩 변경  
6. TP 멀티노드를 체크리스트 Step 1에 배치  

### 14. 핵심 정리

- 서빙 프로젝트의 산출물은 데모 문장이 아니라 **체크리스트·측정·해석**이다.
- TTFT/TPOT 정의를 고정하고, 클라이언트 근사 한계를 명시한다.
- 병목은 capacity / bandwidth / schedule / network 후보로 분류한다.
- 경로 B는 파이프라인 연습이며 성능 결론을 내지 않는다.
- 다음 강의는 여기서 만든 가설을 **GPU 최적화 실험**으로 설계한다.

### 15. 핵심 용어

| 용어 | 의미 |
|---|---|
| Smoke test | 최소 성공 확인 |
| Warmup | 본측정 전 콜드 효과 분리 |
| TTFT | 첫 토큰(청크) 지연 |
| TPOT | 이후 토큰(청크) 간격 |
| OpenAI-compatible API | 널리 쓰이는 HTTP 표면 |
| Artifact | 재현·감사 가능한 산출물 |

### 16. 복습 문제

#### 문제 1 (절차)

서빙 체크리스트에서 멀티노드 TP보다 **먼저** 해야 할 두 단계를 쓰시오.

#### 문제 2 (지표)

비스트리밍 응답만으로 TTFT를 말할 때의 한계를 쓰시오.

#### 문제 3 (해석)

TTFT p95만 크고 TPOT p50은 목표 안일 때, 먼저 볼 후보 두 가지를 쓰시오.

#### 문제 4 (규율)

경로 B mock의 `ttft_ms`를 팀 슬랙에 “Spark 성능”으로 올려도 되는가? 이유와 함께.

#### 문제 5 (연결)

제112강 prefix cache 이득을 이 프로젝트에서 검증하려면 프롬프트 세트를 어떻게 짜야 하는가?

#### 문제 6 (다음)

제117강 실험안 한 줄을 “가설 / 조작 변수 / 고정 변수” 형식으로 작성하시오.

---

### 정답 및 해설

#### 문제 1

예: 환경 스냅샷·단일 노드 smoke(헬스/채팅). 네트워크·NCCL 검증도 TP 앞이다.

#### 문제 2

첫 바이트≈완료에 가깝게 보이거나, 서버 버퍼링 때문에 첫 토큰 시점을 관측할 수 없다.

#### 문제 3

예: 긴 prefill/프롬프트 길이, 캐시 미스, 스케줄 대기, 콜드 스타트. (두 가지)

#### 문제 4

안 된다. mock은 파이프라인 검증용이며 하드웨어 성능이 아니다.

#### 문제 5

공유 긴 prefix + 짧은 가변 꼬리 세트와, prefix가 서로 다른 대조 세트를 동일 동시성·버전으로 비교하고 히트율(가능 시)을 기록한다.

#### 문제 6

예: “양자화로 TPOT가 개선된다 / quantization=on / 모델·프롬프트·concurrency·버전 고정”.

### 17. 다음 강의와 연결

지금은 “올렸다·쟀다·해석했다”. 다음은 **의도적으로 한 변수만 바꿔** GPU·엔진 최적화를 실험한다.

이전 강의: **제115강. 2× DGX Spark 환경 구성**  
다음 강의: **제117강. 프로젝트 — GPU 최적화 실험**

제117강은 제116강 `report.md`의 §6 표를 입력으로 받는다고 가정한다. 측정 규율은 제118강에서 리포트 형식으로 고정된다.
