# 《밑바닥부터 LLM》 120강 커리큘럼

## 📗 1권 — Python · Tensor · 수학 · PyTorch (1~26강)

| 강 | 제목 |
|---:|---|
| 1 | LLM을 밑바닥부터 배운다는 것 |
| 2 | Python 환경 준비와 첫 프로그램 |
| 3 | 변수, 자료형, 연산 |
| 4 | 조건문과 반복문 |
| 5 | 함수와 모듈 |
| 6 | 리스트, 튜플, 딕셔너리 |
| 7 | 파일 입출력과 데이터 다루기 |
| 8 | NumPy로 배열 다루기 |
| 9 | Scalar, Vector, Matrix, Tensor |
| 10 | 선형대수 기초 — 내적과 행렬곱 |
| 11 | 미분과 편미분 |
| 12 | Gradient와 Gradient Descent |
| 13 | Loss Function |
| 14 | Chain Rule |
| 15 | Neural Network의 구조 |
| 16 | Forward Propagation |
| 17 | Backpropagation 직접 계산하기 |
| 18 | Backpropagation NumPy 구현 |
| 19 | PyTorch Tensor |
| 20 | Autograd — 자동 미분 |
| 21 | nn.Module로 모델 만들기 |
| 22 | Dataset과 DataLoader |
| 23 | 학습 루프와 최적화기 |
| 24 | 과적합과 정규화 |
| 25 | 프로젝트 — 작은 Neural Network 직접 구현 |
| 26 | 1권 총정리 — LLM으로 가는 다리 |

## 📘 2권 — Tokenizer와 Transformer (27~54강)

| 강 | 제목 |
|---:|---|
| 27 | 텍스트가 숫자가 되는 과정 |
| 28 | Character / Word / Subword Tokenization |
| 29 | BPE Tokenizer 직접 구현 |
| 30 | Vocabulary와 Special Tokens |
| 31 | Embedding — 토큰을 벡터로 |
| 32 | Language Model과 Next Token Prediction |
| 33 | Softmax와 Logit |
| 34 | Cross Entropy Loss |
| 35 | Attention이 필요한 이유 |
| 36 | Query, Key, Value |
| 37 | Dot-Product Attention 계산 |
| 38 | Softmax Attention과 작은 숫자 예제 |
| 39 | Self-Attention 구현 |
| 40 | Causal Mask |
| 41 | Multi-Head Attention |
| 42 | Positional Encoding |
| 43 | RoPE |
| 44 | LayerNorm과 Residual Connection |
| 45 | Feed-Forward Network (MLP) |
| 46 | Transformer Block 조립 |
| 47 | Encoder와 Decoder |
| 48 | Causal Language Model 구조 |
| 49 | 프로젝트 — Mini Transformer 구현 (1) |
| 50 | 프로젝트 — Mini Transformer 구현 (2) |
| 51 | Attention 시각화 |
| 52 | 계산 복잡도와 메모리 |
| 53 | 최신 Transformer 변형 개요 |
| 54 | 2권 총정리 — GPT로 가는 길 |

## 📙 3권 — GPT Pretraining과 SFT (55~78강)

| 강 | 제목 |
|---:|---|
| 55 | GPT란 무엇인가 |
| 56 | GPT 아키텍처 구현 |
| 57 | Causal LM Training 목표 |
| 58 | Text Generation — Greedy와 Sampling |
| 59 | Temperature, Top-K, Top-P |
| 60 | Pretraining Dataset 구성 |
| 61 | Tokenization Pipeline과 Dataset Packing |
| 62 | Training Loop 설계 |
| 63 | Optimizer, Learning Rate, Scheduler |
| 64 | Mixed Precision과 Gradient Accumulation |
| 65 | Checkpoint 관리 |
| 66 | Validation과 Evaluation |
| 67 | Perplexity와 생성 품질 |
| 68 | 프로젝트 — Mini GPT Pretraining |
| 69 | Instruction Tuning의 개념 |
| 70 | Instruction Dataset 형식 |
| 71 | SFT 구현 |
| 72 | Chat Template과 Special Tokens |
| 73 | LoRA |
| 74 | QLoRA |
| 75 | SFT 평가와 실패 사례 |
| 76 | 프로젝트 — Mini GPT + SFT |
| 77 | Pretraining과 SFT의 역할 정리 |
| 78 | 3권 총정리 — Post-Training으로 |

## 📕 4권 — RLHF · PPO · GRPO (79~98강)

| 강 | 제목 |
|---:|---|
| 79 | Post-Training 지도 |
| 80 | 강화학습 기초 — State, Action, Reward |
| 81 | Policy와 Value Function |
| 82 | Policy Gradient |
| 83 | Advantage |
| 84 | Preference Dataset |
| 85 | Reward Model 구현 |
| 86 | RLHF 전체 구조 |
| 87 | PPO 직관과 수식 |
| 88 | PPO 구현 |
| 89 | KL Divergence의 역할 |
| 90 | DPO — Preference를 직접 학습하기 |
| 91 | DPO 구현 |
| 92 | GRPO |
| 93 | RLVR과 Verifiable Reward |
| 94 | Reasoning Training |
| 95 | 프로젝트 — Preference / RL 실습 |
| 96 | Alignment의 한계와 부작용 |
| 97 | 논문·실무 흐름 정리 |
| 98 | 4권 총정리 — Inference와 Serving으로 |

## 📔 5권 — vLLM · GLM · DGX Spark (99~120강)

| 강 | 제목 |
|---:|---|
| 99 | Training과 Inference의 차이 |
| 100 | Prefill과 Decode |
| 101 | KV Cache |
| 102 | Continuous Batching |
| 103 | Quantization — INT8, INT4, FP8 |
| 104 | vLLM 개요와 구조 |
| 105 | PagedAttention |
| 106 | vLLM Scheduler |
| 107 | LLM Serving 성능 지표 — TTFT, TPOT, Throughput |
| 108 | GLM 계열 모델 구조 |
| 109 | MoE |
| 110 | MTP와 Speculative Decoding |
| 111 | GPU 아키텍처 — CUDA, SM, Memory Bandwidth |
| 112 | GB10과 DGX Spark 구조 |
| 113 | Tensor Parallel |
| 114 | NCCL과 RoCE |
| 115 | 2× DGX Spark 환경 구성 |
| 116 | 프로젝트 — 실제 LLM Serving |
| 117 | 프로젝트 — GPU 최적화 실험 |
| 118 | 성능 측정 리포트 작성법 |
| 119 | 실무 체크리스트와 장애 대응 |
| 120 | 5권·전권 총정리 — 밑바닥에서 Serving까지 |

## 10강 단위 주요 프로젝트

| 권 | 프로젝트 |
|---|---|
| 1권 | Neural Network 직접 구현 (25강) |
| 2권 | Mini Transformer (49~50강) |
| 3권 | Mini GPT + SFT (68, 76강) |
| 4권 | Preference / RL / Reasoning 실습 (95강) |
| 5권 | 실제 LLM Serving + GPU 최적화 (116~117강) |
