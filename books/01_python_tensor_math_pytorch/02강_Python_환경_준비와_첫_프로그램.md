# 제2강. Python 환경 준비와 첫 프로그램

> **학습 목표**
> - Python 3.11 이상을 설치하고 버전을 확인한다
> - venv(virtual environment, 가상환경)로 프로젝트 전용 공간을 만든다
> - pip으로 패키지를 설치·확인한다
> - 에디터(또는 IDE)에서 `.py` 파일을 작성하고 실행한다
> - `print`로 Hello World를 출력한다
> - (선택) Jupyter Notebook을 열어 본다

---
## 1. 왜 이것을 배우는가

LLM 학습은 결국 **코드 실행 → 숫자 확인 → 수정 → 재실행**의 반복이다.

환경이 흔들리면 학습 자체가 흔들린다.

- 전역 Python에 패키지를 마구 설치하면 프로젝트마다 버전이 충돌한다
- “내 PC에서는 되는데 다른 곳에서는 안 된다”가 반복된다
- NumPy, PyTorch처럼 무거운 라이브러리는 버전·플랫폼 의존이 크다

그래서 첫 실습은 모델이 아니라 **재현 가능한 작업대**를 만드는 일이다.

## 2. 먼저 알아야 할 개념

### 2.1 Python이란

**Python(파이썬)**은 읽기 쉬운 문법을 가진 범용 프로그래밍 언어이다.

딥러닝·데이터 과학 생태계가 Python 중심으로 모여 있다. NumPy, PyTorch, Hugging Face Transformers, vLLM 등이 대표적이다.

이 책이 Python을 고른 이유:

1. LLM 연구·실무의 사실상 표준 언어이다
2. 문법이 상대적으로 짧아 “아이디어 → 코드”가 빠르다
3. 나중에 C++/CUDA로 내려가기 전에, 먼저 개념을 Python으로 고정하기 좋다

### 2.2 Interpreter와 Script

**Interpreter(인터프리터)**는 소스 코드를 한 줄씩(또는 한 단위씩) 해석해 실행하는 프로그램이다.  
**Script(스크립트)**는 보통 `.py` 파일에 적어 둔 실행 가능한 Python 코드이다.

```text
hello.py  작성
  → python hello.py 실행
    → Interpreter가 코드를 읽어 결과 출력
```

### 2.3 Package와 Dependency

**Package(패키지)**는 다른 사람이 만들어 배포한 코드 묶음이다.  
**Dependency(의존성)**는 내 프로젝트가 필요로 하는 외부 패키지와 그 버전이다.

나중에 `numpy`, `torch`를 설치하는 순간부터, “어떤 버전을 썼는가”가 실험 재현성의 일부가 된다.

## 3. 핵심 개념 설명

### 3.1 Python 버전: 3.11 이상을 권장하는 이유

이 책의 실습은 **Python 3.11+**를 기준으로 한다.

이유는 단순하다.

- 최신 PyTorch·관련 도구가 비교적 잘 맞는다
- 문법·성능·타입 힌트 경험이 안정적이다
- 너무 오래된 3.8/3.9만 쓰면 나중에 패키지 설치에서 막힐 수 있다

설치 후 반드시 확인한다.

```bash
python --version
# 또는
python3 --version
```

기대 예:

```text
Python 3.11.9
```

운영체제마다 `python`과 `python3` 중 하나만 있을 수 있다. 아래에서 `python`이라 쓰면, 실제로는 본인 환경의 명령을 쓰면 된다.

### 3.2 설치 개요 (OS별)

자세한 클릭 경로는 OS·시점마다 바뀐다. 여기서는 **확인해야 할 결과**를 기준으로 본다.

**Windows**

1. [python.org](https://www.python.org/downloads/)에서 설치 프로그램을 받는다
2. 설치 시 “Add python.exe to PATH”를 켠다
3. PowerShell 또는 터미널에서 `python --version`을 확인한다

**macOS**

- 공식 설치 프로그램, 또는 `brew install python@3.11` 등을 사용할 수 있다
- 터미널에서 `python3 --version`을 확인한다

**Linux**

배포판마다 다르다. 예:

```bash
# Ubuntu/Debian 계열 예시 (패키지명은 배포판에 맞게 조정)
sudo apt update
sudo apt install python3 python3-venv python3-pip
python3 --version
```

핵심은 “설치했다”가 아니라 **터미널에서 버전이 출력되는가**이다.

### 3.3 venv — 프로젝트마다 격리된 Python

**venv(virtual environment, 가상환경)**는 프로젝트 전용으로 패키지를 설치할 수 있게 만드는 격리 공간이다.

왜 필요한가?

- 프로젝트 A는 `torch==2.3`, 프로젝트 B는 다른 버전이 필요할 수 있다
- 시스템 Python을 더럽히지 않는다
- “이 폴더에서는 이 조합”이라는 규칙을 강제한다

생성:

```bash
# 프로젝트 폴더로 이동했다고 가정
python -m venv .venv
```

활성화:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

활성화되면 보통 프롬프트 앞에 `(.venv)`가 붙는다.

비활성화:

```bash
deactivate
```

### 3.4 pip — 패키지 설치 도구

**pip**은 Python 패키지를 설치·업그레이드·확인하는 표준 도구이다.

가상환경을 켠 뒤:

```bash
python -m pip install --upgrade pip
python -m pip install numpy
```

설치된 목록 확인:

```bash
python -m pip list
```

특정 패키지 정보:

```bash
python -m pip show numpy
```

`python -m pip` 형태를 권장한다. 어떤 `pip`이 어느 Python에 붙었는지 헷갈리는 사고를 줄인다.

### 3.5 에디터 / IDE

코드를 쓰려면 편집기가 필요하다.

| 도구 | 특징 |
|---|---|
| **VS Code** | 가볍고 확장성이 큼. Python 확장으로 충분 |
| **PyCharm** | Python 특화 IDE. 초보에게도 친절 |
| **Cursor** | AI 보조 코딩에 강한 편집기. 이 책의 실습에도 적합 |
| 기본 메모장 | 가능은 하나, 자동완성·실행 통합이 약함 |

처음에는 기능을 많이 알 필요 없다. 다음만 되면 충분하다.

1. `.py` 파일 저장
2. 터미널에서 실행
3. 에러 메시지를 읽을 수 있음

### 3.6 Jupyter — 선택 도구

**Jupyter Notebook(주피터 노트북)**은 코드 셀과 설명을 섞어 실행하는 대화형 환경이다.

장점:

- 중간 결과를 바로 본다
- 그래프·표를 끼워 넣기 쉽다

단점:

- 실행 순서가 꼬이면 재현이 어렵다
- 큰 프로젝트 구조(모듈 분리)에는 `.py`가 더 적합하다

이 책은 **기본은 `.py` 스크립트**, 탐색이 필요할 때만 Jupyter를 권한다.

설치 예:

```bash
python -m pip install jupyter
jupyter notebook
```

## 4. 직관적으로 이해하기

개발 환경을 집필실에 비유하면 이해하기 쉽다.

| 요소 | 비유 |
|---|---|
| Python | 언어와 기본 도구 |
| venv | 책상 위의 전용 작업 공간 |
| pip | 필요한 도구를 서랍에 채워 넣는 일 |
| 에디터 | 원고를 쓰는 타자기 |
| 스크립트 실행 | 원고를 소리 내어 읽어 보는 일 |

“도구를 전역으로 마구 쌓아 두는 사람”과 “프로젝트마다 작업대를 나누는 사람”은, 나중에 같은 코드를 돌려도 결과가 갈린다.

## 5. 숫자로 확인하기 — 환경 점검 체크리스트

설치가 끝났다면 아래를 순서대로 확인한다. 숫자가 맞는지보다 **통과/실패**가 중요하다.

| 단계 | 명령 | 기대 |
|---:|---|---|
| 1 | `python --version` | 3.11 이상 |
| 2 | `python -m venv .venv` | `.venv` 폴더 생성 |
| 3 | 활성화 | 프롬프트에 `(.venv)` |
| 4 | `python -m pip --version` | pip 버전 출력 |
| 5 | `python hello.py` | `Hello, LLM` 출력 |

한 단계라도 실패하면 다음으로 넘어가지 않는다. 환경 문제는 뒤로 미룰수록 비용이 커진다.

## 6. 코드로 첫 프로그램 작성하기

프로젝트 폴더를 하나 만든다. 예: `llm_scratch`.

```bash
mkdir llm_scratch
cd llm_scratch
python -m venv .venv
source .venv/bin/activate   # Windows는 활성화 명령이 다름
```

### 6.1 Hello World

`hello.py` 파일을 만든다.

```python
# hello.py
# 목적: Python이 정상 실행되는지 확인한다.

def main() -> None:
    # print: 화면에 문자열을 출력하는 내장 함수
    print("Hello, LLM")
    print("Python environment is ready.")

if __name__ == "__main__":
    # 이 파일을 직접 실행할 때만 main()을 호출한다
    main()
```

실행:

```bash
python hello.py
```

출력 예:

```text
Hello, LLM
Python environment is ready.
```

### 6.2 버전과 경로를 출력해 보기

“지금 돌아가는 Python이 가상환경의 것인지”를 확인하는 짧은 스크립트다.

```python
# check_env.py
# 목적: 사용 중인 Python 실행 파일 경로와 버전을 확인한다.

import sys

def main() -> None:
    print("executable:", sys.executable)
    print("version:", sys.version)
    print("prefix:", sys.prefix)

if __name__ == "__main__":
    main()
```

`sys.executable` 경로에 `.venv`가 보이면, 가상환경을 쓰고 있을 가능성이 높다.

### 6.3 (나중을 위한) NumPy / PyTorch 설치 점검

지금은 필수가 아니다. 다만 이후 강의에서 쓸 것이므로, **설치 가능 여부**만 미리 알아 둔다.

```bash
# 가상환경 활성화 후
python -m pip install numpy
# PyTorch는 OS/CUDA 조합에 따라 설치 명령이 다르다.
# 공식 안내(https://pytorch.org)의 명령을 따르는 것이 안전하다.
# 예: CPU 전용 환경에서는 안내된 CPU 휠을 설치한다.
```

설치 후 확인:

```python
# check_libs.py
# 목적: 이후 강의에서 쓸 라이브러리가 import 되는지 확인한다.

def main() -> None:
    try:
        import numpy as np

        print("numpy:", np.__version__)
    except ImportError:
        print("numpy: not installed")

    try:
        import torch

        print("torch:", torch.__version__)
    except ImportError:
        print("torch: not installed yet (ok for now)")

if __name__ == "__main__":
    main()
```

NumPy는 8강에서 본격적으로 다룬다. PyTorch는 1권 후반이다.  
지금 단계에서 torch 설치가 실패해도, Python·venv·pip만 되면 2~7강은 진행할 수 있다.

## 7. 실제 LLM에서는 어떻게 사용하는가

환경 준비는 “기초 교양”처럼 보이지만, 실제 LLM 실험에서도 같은 원리가 그대로 쓰인다.

| LLM 작업 | 환경이 중요한 이유 |
|---|---|
| Tokenizer / 데이터 전처리 | `tokenizers`, `datasets` 등 버전 차이로 결과가 달라질 수 있다 |
| GPT Pretraining | `torch`, CUDA, 드라이버 조합이 학습 속도·가능 여부를 좌우한다 |
| SFT / LoRA | `transformers`, `peft`, `bitsandbytes` 의존성이 복잡하다 |
| vLLM Serving | Python·CUDA·GPU 드라이버가 맞아야 서버가 뜬다 |

연구 노트에 보통 다음을 함께 적는다.

```text
Python 버전
패키지 목록 (pip freeze)
GPU / CUDA 버전
커밋 해시
랜덤 시드
```

2강에서 venv와 pip을 익히는 이유는, 나중에 “모델이 안 돌아가요”를 “환경이 안 맞아요”와 분리하기 위해서다.

## 8. 실습

### 실습 1 — 가상환경 만들기

**목표:** 프로젝트 전용 환경을 직접 만든다.

1. `llm_scratch` 폴더를 만든다
2. `.venv`를 생성하고 활성화한다
3. `python -m pip list`를 실행해 기본 패키지를 확인한다
4. `deactivate` 후 다시 활성화해 본다

**확인 기준:** 활성화 상태에서 `sys.executable`이 `.venv` 쪽을 가리킨다.

### 실습 2 — Hello World 변형

**목표:** 스크립트 실행 루프에 익숙해진다.

1. `hello.py`의 인사말을 본인 이름으로 바꾼다
2. `print`를 한 줄 더 추가해 오늘 날짜 문자열을 출력한다 (하드코딩으로 충분)
3. 파일을 저장하고 다시 실행한다

### 실습 3 — 환경 점검 스크립트

**목표:** 문제 진단용 최소 도구를 갖는다.

1. `check_env.py`를 작성·실행한다
2. 출력을 메모장에 붙여 둔다
3. (선택) `numpy`를 설치하고 `check_libs.py`로 버전을 확인한다

## 9. 자주 하는 실수

1. **가상환경을 켜지 않은 채 pip install을 한다**  
   시스템에 패키지가 깔리거나, 엉뚱한 Python에 설치된다. 항상 활성화 여부를 먼저 본다.

2. **`python`과 `python3`를 섞어 쓴다**  
   A 명령으로 설치하고 B 명령으로 실행하면 “설치했는데 import가 안 된다”가 생긴다.

3. **PATH 설정이 안 된 채로 GUI만 설치한다**  
   터미널에서 `python`이 안 잡히면 이후 실습이 전부 막힌다. 버전 출력이 첫 관문이다.

4. **프로젝트 폴더 밖에 파일을 흩뿌린다**  
   나중에 경로가 꼬인다. 이 책 실습은 한 폴더(또는 책 권 단위 폴더)를 기준으로 모은다.

5. **에러 메시지를 읽지 않고 재설치만 반복한다**  
   `ModuleNotFoundError`, `PermissionError`, `command not found`는 각각 원인이 다르다. 메시지를 먼저 읽는다.

## 10. 핵심 정리

- Python 3.11+를 설치하고, 터미널에서 버전을 확인하는 것이 출발점이다
- venv로 프로젝트마다 패키지를 격리한다
- pip으로 의존성을 관리하며, `python -m pip` 형태를 권장한다
- 에디터에서 `.py`를 작성하고 `python file.py`로 실행하는 루프가 기본 작업 단위다
- Jupyter는 선택 도구이며, 이 책은 스크립트 중심이다
- NumPy/PyTorch는 이후 강의에서 본격 사용하되, 설치·import 점검 방법은 지금 알아 둔다

## 11. 핵심 용어

| 용어 | 의미 |
|---|---|
| Python | 이 책의 기본 프로그래밍 언어 |
| Interpreter | 소스 코드를 해석해 실행하는 프로그램 |
| Script (`.py`) | 실행 가능한 Python 코드 파일 |
| venv | 프로젝트 전용 가상환경 |
| pip | Python 패키지 설치·관리 도구 |
| Package / Dependency | 외부 코드 묶음과 그 필요 관계 |
| IDE / Editor | 코드를 작성·실행하는 도구 |
| Jupyter Notebook | 셀 단위 대화형 실행 환경 |
| `print` | 표준 출력으로 값을 보여주는 내장 함수 |
| `sys.executable` | 현재 실행 중인 Python 바이너리 경로 |

## 12. 연습 문제
### 문제 1 (개념)

venv가 필요한 이유를 “버전 충돌”과 “재현성” 관점에서 두 문장으로 설명하시오.

### 문제 2 (개념)

`python -m pip install numpy`에서 `-m pip`을 쓰는 이유는 무엇인가?

### 문제 3 (실습)

가상환경을 활성화한 뒤 `check_env.py`를 실행했을 때, `executable` 경로에 `.venv`가 보인다면 무엇을 의미하는가?

### 문제 4 (코드)

다음 코드의 실행 결과를 쓰시오.

```python
print("A")
print("B")
```

### 문제 5 (연결)

LLM Pretraining 실험 노트에 Python 버전과 `pip freeze` 결과를 남기는 이유를 한 문장으로 쓰시오.

---

## 정답 및 해설

### 문제 1

프로젝트마다 필요한 패키지 버전이 다를 수 있어, 전역 설치는 충돌을 만든다. venv로 격리하면 같은 코드를 같은 의존성으로 다시 실행하기 쉬워진다.

### 문제 2

현재 사용 중인 그 Python 인터프리터에 연결된 pip을 호출하기 위해서다. 이름만 `pip`인 실행 파일이 다른 Python을 가리키는 실수를 줄인다.

### 문제 3

지금 스크립트를 실행하는 Python이 가상환경 안의 인터프리터라는 뜻이다. 패키지 설치·실행이 그 환경 기준으로 이뤄진다.

### 문제 4

```text
A
B
```

### 문제 5

학습 결과나 버그는 코드만이 아니라 환경(패키지·버전)에도 의존하므로, 재현과 비교를 위해 환경을 기록해야 한다.

## 13. 다음 강의와 연결

이전 **제1강. LLM을 밑바닥부터 배운다는 것**에서 전체 지도를 그렸고, 이번 강의에서 실행 환경을 고정했다.

다음 **제3강. 변수, 자료형, 연산**에서는 Python의 가장 작은 재료인 숫자·문자열·불리언과 연산을 다룬다.  
토큰 ID, 확률, Loss처럼 이후 LLM에서 끝없이 마주칠 값들을 코드로 담는 연습이 시작된다.

> 작업대가 준비되었다. 이제 재료를 담을 그릇을 만든다.

<!-- LECTURE_NAV -->

---

### 강의 이동

- **이전 강:** [1강. LLM을 밑바닥부터 배운다는 것](01강_LLM을_밑바닥부터_배운다는_것.md)
- **다음 강:** [3강. 변수, 자료형, 연산](03강_변수_자료형_연산.md)

<!-- /LECTURE_NAV -->
