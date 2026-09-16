# 《밑바닥부터 LLM》

Python과 수학 기초부터 Transformer, GPT, RLHF, vLLM·DGX Spark 최적화까지  
**120강 · 5권** 실전형 전자책 프로젝트입니다.

## 학습 경로

```
Python → 수학 → Tensor → Neural Network → Backpropagation → PyTorch
→ Tokenizer → Embedding → Attention → Transformer
→ GPT → Pretraining → SFT
→ RLHF → PPO → DPO → GRPO → RLVR → Reasoning
→ LLM Inference → vLLM → GLM → DGX Spark → 최적화
```

## 구성

| 권 | 강의 | 주제 |
|---|---|---|
| [1권](books/01_python_tensor_math_pytorch/) | 1~26 | Python · Tensor · 수학 · PyTorch |
| [2권](books/02_tokenizer_transformer/) | 27~54 | Tokenizer와 Transformer |
| [3권](books/03_gpt_pretraining_sft/) | 55~78 | GPT Pretraining과 SFT |
| [4권](books/04_rlhf_ppo_grpo/) | 79~98 | RLHF · PPO · GRPO |
| [5권](books/05_vllm_glm_dgx/) | 99~120 | vLLM · GLM · DGX Spark |

## 문서

- [전체 커리큘럼](docs/CURRICULUM.md)
- [프로젝트 지침](docs/PROJECT_GUIDELINES.md)
- [작성 체크리스트](docs/WRITING_CHECKLIST.md)
- [진행 상태](docs/STATUS.md)
- [용어집](glossary/TERMS.md)

## 디렉터리

```
books/          # 강의 원고 (Markdown)
docs/           # 커리큘럼·지침
glossary/       # 용어집
assets/images/  # 그림
scripts/        # EPUB 변환·검증 스크립트
epub/           # 최종 EPUB 산출물
```

## 문체

친절한 기술 서적 문체를 유지합니다.  
쉽게 설명하되 얕게 설명하지 않으며, 직관 → 수학 → 숫자 예제 → 코드 → LLM 연결 순서를 따릅니다.

## 수식 렌더링

원고의 `$$…$$` 는 의도된 LaTeX다. 뷰어가 수식 엔진을 안 쓰면 원문이 그대로 보인다.

```bash
# 브라우저에서 스크린샷처럼 보기 (KaTeX)
python3 scripts/build_math_preview.py books/01_python_tensor_math_pytorch/09강_Scalar_Vector_Matrix_Tensor.md

# EPUB3 (MathML)
python3 scripts/build_epub.py --book 1
```

자세한 설명: [docs/MATH_RENDERING.md](docs/MATH_RENDERING.md)

## 최종 산출물

- `epub/01_밑바닥부터_LLM_1권.epub` ~ `05_밑바닥부터_LLM_5권.epub`
- Sigil에서 편집 가능한 EPUB3
- 미리보기 HTML: `epub/preview/`