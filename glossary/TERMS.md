# 《밑바닥부터 LLM》 용어집 (초안)

용어는 처음 등장하는 강의에서 자세히 설명하고, 이후 강의에서는 필요한 수준만 반복한다.
최종 EPUB에는 전권 용어집을 포함한다.

| 용어 (EN) | 한국어 | 첫 등장(예정) | 한 줄 정의 |
|---|---|---:|---|
| AI | 인공지능 | 1 | 사람이 하던 판단·생성 작업을 시스템이 수행하는 기술 |
| Machine Learning | 머신러닝 | 1 | 데이터로부터 규칙을 학습하는 방법 |
| Deep Learning | 딥러닝 | 1 | 여러 층의 Neural Network로 학습하는 방법 |
| LLM | 대규모 언어 모델 | 1 | 대규모 텍스트로 학습한 언어 모델 |
| Parameter | 파라미터 | 1 | 모델이 학습으로 조정하는 숫자 |
| Token | 토큰 | 1 | 텍스트를 나눈 최소 처리 단위 |
| Virtual Environment | 가상환경 | 2 | 프로젝트별 패키지를 격리하는 Python 실행 환경 |
| Variable | 변수 | 3 | 값을 가리키는 이름 |
| Control Flow | 제어 흐름 | 4 | 조건·반복으로 실행 순서를 바꾸는 구조 |
| Function | 함수 | 5 | 입력을 받아 작업을 수행하고 출력을 반환하는 코드 묶음 |
| Module | 모듈 | 5 | 재사용 가능한 Python 파일·패키지 단위 |
| List / Dict | 리스트 / 딕셔너리 | 6 | 순서 있는 모음 / 키-값 매핑 자료구조 |
| JSON / CSV | 제이슨 / 씨에스브이 | 7 | 구조화 데이터를 저장·교환하는 대표 형식 |
| NumPy | 넘파이 | 8 | 다차원 배열 연산 라이브러리 |
| Broadcasting | 브로드캐스팅 | 8 | 모양이 다른 배열을 규칙에 맞게 확장해 연산하는 방식 |
| Scalar | 스칼라 | 9 | 단일 숫자 (0차원) |
| Vector | 벡터 | 9 | 1차원 숫자 배열 |
| Matrix | 행렬 | 9 | 2차원 숫자 배열 |
| Tensor | 텐서 | 9 | 다차원 배열 형태의 데이터 |
| Dot Product | 내적 | 10 | 두 벡터의 대응 원소 곱의 합 |
| Matrix Multiplication | 행렬곱 | 10 | 행·열 내적으로 새 행렬을 만드는 연산 |
| Derivative | 미분(도함수) | 11 | 입력이 바뀔 때 출력이 얼마나 변하는지를 나타내는 값 |
| Partial Derivative | 편미분 | 11 | 여러 변수 중 하나만 바꿔 보는 미분 |
| Gradient | 그래디언트 | 12 | Loss가 각 Parameter에 대해 변하는 방향과 크기 |
| Gradient Descent | 경사하강법 | 12 | Gradient 반대 방향으로 Parameter를 갱신하는 최적화 |
| Learning Rate | 학습률 | 12 | 한 번에 Parameter를 얼마나 옮길지 정하는 계수 |
| Loss | 손실 | 13 | 예측과 정답의 차이를 수치화한 값 |
| MSE | 평균제곱오차 | 13 | 오차 제곱의 평균으로 정의한 손실 |
| Cross Entropy | 교차엔트로피 | 13 | 확률 분포 예측에 자주 쓰는 분류·언어모델 손실 |
| Chain Rule | 연쇄법칙 | 14 | 합성함수의 미분을 곱으로 나누어 계산하는 규칙 |
| Computational Graph | 계산 그래프 | 14 | 연산을 노드로 연결해 미분 경로를 추적하는 구조 |
| Neural Network | 신경망 | 15 | 층과 가중치로 입력을 변환하는 학습 가능한 모델 |
| Activation Function | 활성화 함수 | 15 | 비선형성을 넣어 표현력을 높이는 함수 |
| Forward Propagation | 순전파 | 16 | 입력에서 출력·Loss까지 앞으로 계산하는 과정 |
| Backpropagation | 역전파 | 17 | Chain Rule로 Gradient를 뒤에서부터 계산하는 방법 |
| Autograd | 자동 미분 | 20 | 연산 그래프로 Gradient를 자동 계산하는 기능 |
| nn.Module | 엔엔 모듈 | 21 | PyTorch에서 모델을 정의하는 기본 클래스 |
| Dataset / DataLoader | 데이터셋 / 데이터로더 | 22 | 샘플 정의와 배치 공급을 담당하는 인터페이스 |
| Batch | 배치 | 22 | 한 번에 묶어 학습·추론하는 샘플 묶음 |
| Optimizer | 최적화기 | 23 | Parameter 갱신 규칙을 수행하는 객체 (SGD, Adam 등) |
| Overfitting | 과적합 | 24 | 훈련 데이터에만 맞고 새 데이터에 약한 상태 |
| Regularization | 정규화(규제) | 24 | 과적합을 줄이기 위한 Dropout·Weight Decay 등 기법 |
| Dropout | 드롭아웃 | 24 | 학습 중 일부 유닛을 무작위로 끄는 규제 |
| Embedding | 임베딩 | 31 | Token을 벡터로 매핑하는 표현 |
| Attention | 어텐션 | 35 | 토큰 간 참조 가중치를 계산하는 메커니즘 |
| Transformer | 트랜스포머 | 46 | Attention 기반 신경망 구조 |
| Pretraining | 사전학습 | 55 | 대규모 데이터로 일반 능력을 먼저 학습하는 단계 |
| SFT | 지도 미세조정 | 71 | 지시-응답 데이터로 모델을 맞추는 학습 |
| LoRA | 로라 | 73 | 저랭크 행렬로 일부를 효율적으로 미세조정하는 방법 |
| RLHF | 인간 피드백 강화학습 | 86 | 선호 신호로 정책을 개선하는 포스트트레이닝 |
| PPO | 근사 정책 최적화 | 87 | 정책 업데이트를 안정적으로 제한하는 RL 알고리즘 |
| DPO | 직접 선호 최적화 | 90 | Reward Model 없이 Preference를 직접 학습하는 방법 |
| GRPO | 그룹 상대 정책 최적화 | 92 | 그룹 내 상대 보상으로 정책을 학습하는 방법 |
| KV Cache | KV 캐시 | 101 | Decode 시 Key/Value를 재사용하는 캐시 |
| vLLM | 브이엘엘엠 | 104 | 고성능 LLM Inference 엔진 |
| PagedAttention | 페이지드 어텐션 | 105 | 가상 메모리처럼 KV Cache를 관리하는 기법 |
| MoE | 전문가 혼합 | 109 | 토큰별로 일부 전문가를 선택해 계산하는 구조 |
| TTFT | 첫 토큰 지연 | 107 | 요청 후 첫 토큰이 나오기까지의 시간 |
| TPOT | 토큰당 시간 | 107 | 이후 토큰 하나를 생성하는 평균 시간 |

| Softmax | 소프트맥스 | 33 | 점수를 합이 1인 확률로 바꾸는 함수 |
| Logit | 로짓 | 33 | Softmax 직전의 점수 |
| Self-Attention | 셀프 어텐션 | 39 | 같은 시퀀스 안 토큰끼리 참조 가중치를 계산 |
| Causal Mask | 인과 마스크 | 40 | 미래 토큰을 보지 못하게 가리는 마스크 |
| Multi-Head Attention | 멀티헤드 어텐션 | 41 | 여러 Attention head를 병렬로 쓰는 구조 |
| Positional Encoding | 위치 인코딩 | 42 | 토큰 위치 정보를 벡터에 더하는 방법 |
| RoPE | 로프 | 43 | 회전으로 상대 위치 정보를 Q/K에 주입하는 방법 |
| LayerNorm | 레이어 정규화 | 44 | 특징 차원에서 평균·분산을 맞춰 안정화 |
| Residual Connection | 잔차 연결 | 44 | 입력을 출력에 더해 깊은 망 학습을 돕는 경로 |
| BPE | 바이트쌍 인코딩 | 29 | 빈번한 쌍을 병합해 서브워드를 만드는 토크나이징 |

| Causal LM | 인과 언어모델 | 55 | 과거 토큰만 보고 다음 토큰을 예측하는 언어모델 |
| Temperature | 온도 | 59 | Softmax 전에 logits를 나눠 분포 날카로움/완화를 조절 |
| Top-K | 탑케이 | 59 | 확률 상위 K개 토큰만 남기고 샘플링하는 방법 |
| Top-P | 탑피(Nucleus) | 59 | 누적확률 P를 채울 때까지 토큰을 남기는 샘플링 |
| Dataset Packing | 데이터셋 패킹 | 61 | 여러 문서를 이어 붙여 시퀀스 길이를 채우는 기법 |
| AdamW | 아담더블유 | 63 | Weight Decay를 분리해 적용하는 Adam 계열 Optimizer |
| Mixed Precision | 혼합 정밀도 | 64 | FP16/BF16 등으로 계산해 메모리·속도를 돕는 학습 방식 |
| Checkpoint | 체크포인트 | 65 | 학습 재개를 위해 모델·상태를 저장한 스냅샷 |
| Perplexity | 퍼플렉서티 | 67 | 평균 NLL의 지수로 표현한 언어모델 평가 지표 |
| Instruction Tuning | 인스트럭션 튜닝 | 69 | 지시-응답 형태로 모델을 맞추는 학습 |
| Chat Template | 채팅 템플릿 | 72 | 대화 메시지를 모델 입력 문자열로 직렬화하는 형식 |
| QLoRA | 큐로라 | 74 | 양자화 베이스 모델 위에 LoRA를 올려 미세조정하는 방법 |

| Post-Training | 포스트트레이닝 | 79 | 사전학습 이후 정렬·선호·추론 능력을 다듬는 단계 |
| Policy | 정책 | 81 | 상태(문맥)에서 행동(토큰)을 고르는 확률 분포 |
| Value Function | 가치함수 | 81 | 상태에서 앞으로 얻을 보상의 기댓 |
| Policy Gradient | 정책 경사 | 82 | 보상이 커지도록 정책 파라미터를 미분으로 갱신하는 방법 |
| Advantage | 어드밴티지 | 83 | 어떤 행동이 평균보다 얼마나 나은지를 나타내는 신호 |
| Preference Dataset | 선호 데이터셋 | 84 | chosen/rejected처럼 상대적 선호를 담은 데이터 |
| KL Divergence | 클다이버전스 | 89 | 두 확률분포의 차이; 정책이 참조에서 너무 멀어지지 않게 함 |
| Reference Model | 참조 모델 | 89 | KL 제약의 기준이 되는 고정(또는 느리게 갱신되는) 정책 |
| Reasoning Training | 추론 학습 | 94 | 다단계 사고·검증 신호를 활용해 추론 능력을 키우는 학습 |
| Reward Hacking | 보상 해킹 | 96 | 보상 지표만 올리고 실제 목표는 왜곡하는 현상 |

| Prefill | 프리필 | 100 | 프롬프트 전체를 한 번에 처리해 KV를 채우는 Inference 단계 |
| Continuous Batching | 연속 배칭 | 102 | 요청이 끝나는 대로 새 요청을 끼워 넣는 동적 배치 |
| Quantization | 양자화 | 103 | 가중치·활성값을 낮은 비트로 표현해 메모리·대역을 줄이는 기법 |
| Scheduler | 스케줄러 | 106 | 어떤 요청을 언제 GPU에 올릴지 정하는 구성요소 |
| Speculative Decoding | 추론적 디코딩 | 110 | 초안 토큰을 빠르게 제안하고 검증하는 가속 기법 |
| Tensor Parallel | 텐서 병렬 | 113 | 레이어 내 텐서를 GPU들에 나눠 계산하는 병렬화 |
| NCCL | 엔씨씨엘 | 114 | GPU 간 집합통신을 위한 NVIDIA 라이브러리 |
| RoCE | 로시 | 114 | 이더넷 위에서 RDMA를 쓰는 네트워크 기술 |
| DGX Spark | 디지엑스 스파크 | 115 | 소형/데스크사이드 AI 시스템에 가까운 NVIDIA 플랫폼(문서 기준 설명) |

작성 중 용어가 추가되면 이 표를 갱신한다.
