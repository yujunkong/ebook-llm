# 진행 상태

| 항목 | 상태 |
|---|---|
| 커리큘럼 확정 | 완료 |
| 프로젝트 지침 | 완료 |
| 용어집 초안 | 완료 |
| **1권 1~26강** | **완료** |
| **2권 27~54강** | **완료** |
| **3권 55~78강** | **완료** |
| **4권 79~98강** | **완료** |
| **5권 99~120강** | **완료** |
| EPUB 빌드 | **완료** (`epub/01`~`05_*.epub`, MathML) |
| Markdown 스타일 가이드 | 완료 (Apple Books 기준) |
| Apple Books 가이드 | **완료** (`docs/APPLE_BOOKS_GUIDE.md`) |
| 120강 구조 일괄 반영 | **완료** (`# N강` · 배우는 내용 · 왜 중요한가 · LLM 연결 · 핵심 요약 · 용어 사전 · 연습문제) |
| 10·11강 내용 보강 | **완료** |
| 12~15강 수식·예제 보강 | **완료** (∇L/GD, MSE/CE, Chain Rule, NN Affine+Activation) |
| 나머지 강의 수식·내용 보강 | **진행 중** (`cursor/enrich-remaining-lectures-90e1`, 다수 강에 수식 보강 섹션·핵심 공식 추가) |

## 작성 순서

1. Apple Books 가이드 구조는 120강에 일괄 반영됨.
2. 10·11강은 내용(예제·도식·실습·LLM 연결)을 심화 보강.
3. 재빌드: `python3 scripts/build_epub.py --book all`

### 커리큘럼 변경 (5권, 번호 유지)

| 강 | 이전 | 변경 |
|---:|---|---|
| 108 | GLM 계열 모델 구조 | 모델 알고리즘 계열별 모델 구조 |
| 112 | GB10과 DGX Spark 구조 | Inference Engine 비교 — vLLM · TensorRT-LLM · SGLang |

GB10/DGX Spark 하드웨어 설명은 **115강**에서 다룬다.
