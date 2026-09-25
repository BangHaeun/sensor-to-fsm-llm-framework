# Sensor-to-FSM LLM Framework

센서 데이터를 입력으로 받아 LLM이 스마트홈 제어용 FSM(Finite State Machine)을 JSON 형태로 생성하고, 생성 결과의 구조적 품질과 응답 지연을 평가하는 Python 프레임워크입니다. 동일한 파이프라인에서 GPT-5.2와 Claude Sonnet 4.6을 실행하여 모델별 결과도 비교합니다.

## Framework Flow

1. 공개 센서 데이터셋 로드
2. 센서 샘플을 LLM에 입력
3. 스마트홈 제어 FSM JSON 생성
4. FSM 정적 품질 평가
   - JSON 구조
   - 도메인 제어 상태 추론
   - Safety/Fallback 상태
   - 그래프 무결성
5. 모델별 품질 점수 및 latency 비교

## Files

- `test.py`: 데이터 로드, LLM 호출, FSM 생성, 정적 평가, latency 측정
- `.env.example`: OpenRouter API key 환경변수 예시
- `.gitignore`: 실제 `.env`와 Python cache 제외
- `requirements.txt`: 실행 의존성
- `LLM_FSM_벤치마크_분석보고서.docx`: 구현 과정, 실험 결과 및 분석 보고서

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env`에서 OpenRouter API key를 설정한 뒤:

```bash
python test.py
```

## Environment Variable

```text
OPENROUTER_API_KEY=your-key-here
```

실제 `.env` 파일은 Git에 포함되지 않습니다.

## Current Experiment

현재 구현은 공개된 `daily-min-temperatures.csv`의 상위 5개 샘플을 입력으로 사용합니다. 따라서 현 실험은 완전한 온도·습도·CO2 멀티센서 데이터셋에 대한 성능 검증이라기보다, **센서 데이터 → LLM 해석 → FSM 생성 → 정적 검증 → 모델 비교**로 이어지는 전체 프레임워크를 구축하고 동작을 확인한 초기 실험입니다.
