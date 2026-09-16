# 《밑바닥부터 LLM》 프로젝트 지침

이 문서는 120강 전자책 프로젝트의 고정 규칙을 요약한다.
상세 원칙은 원본 지침(사용자 제공)을 따른다.

## 고정 구조

| 권 | 강의 | 주제 |
|---|---|---|
| 1권 | 1~26강 | Python · Tensor · 수학 · PyTorch 기초 |
| 2권 | 27~54강 | Transformer와 Tokenizer |
| 3권 | 55~78강 | GPT Pretraining과 SFT |
| 4권 | 79~98강 | RLHF · PPO · GRPO |
| 5권 | 99~120강 | vLLM · GLM · DGX Spark 최적화 |

- 5권 / 120강 구조를 임의로 변경하지 않는다.
- 강의 번호를 삭제·재배치하지 않는다.
- 새 내용은 기존 강의와 연결하여 설명한다.

## 학습 경로

Python → 수학 → Tensor → Neural Network → Backpropagation → PyTorch → Tokenizer → Embedding → Attention → Transformer → GPT → Pretraining → SFT → Preference Learning → RLHF → PPO → DPO → GRPO → RLVR → Reasoning → LLM Inference → vLLM → GLM → DGX Spark → LLM 최적화

## 핵심 원칙

1. 쉽게 설명하되 얕게 설명하지 않는다.
2. 직관 → 용어 → 수학 → 숫자 예제 → 코드 → LLM 연결 → 실습.
3. 최종 결과물은 Sigil에서 편집 가능한 EPUB3이다.
4. ChatGPT 대화체를 쓰지 않는다. 기술 서적 문체를 유지한다.
5. 성능 수치는 임의로 만들지 않는다.

## 파일 규칙

- 강의 원고: `books/0N_.../NN강_제목.md`
- 용어집: `glossary/`
- EPUB 산출물: `epub/`
- 공통 문서: `docs/`
