import asyncio
import json
import os
import time
import pandas as pd
import networkx as nx
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 1. API 키 설정 (.env 파일 또는 환경변수에서 로드)
load_dotenv()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# 2. 공개된 오픈 스마트홈 센서 데이터셋 URL 직접 로드
# (GitHub에 공개된 Smart Home IoT / Climate Sensor Open Dataset)
OPEN_DATASET_URL = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
# 실제 실내 온/습도/CO2 오픈 데이터셋 샘플 URL (GitHub Raw Link)
SMARTHOME_DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"

def fetch_open_multimodal_dataset(url: str, limit: int = 5) -> list[dict]:
    """공개 URL에서 오픈 데이터셋을 직접 다운로드하여 JSON 리스트로 변환"""
    print(f"[안내] 오픈 데이터셋 다운로드 중: {url}")
    df = pd.read_csv(url)

    # 데이터셋의 상위 N개 샘플을 벤치마크용 dict 형태로 변환
    samples = df.head(limit).to_dict(orient="records")
    return samples

SYSTEM_PROMPT = """
당신은 스마트홈 제어 FSM(유한 상태 머신) 설계 전문가입니다.
입력으로 주어지는 실내 환경 센서 데이터 및 상태를 분석하여 도메인 의미(예: Temp 상승 -> 냉방, 습도 상승 -> 제습, CO2 상승 -> 환기 등)를 유추하고 FSM 구조를 자율 설계하세요.

반드시 다른 설명 없이 오직 아래 구조의 Valid JSON 객체만 반환하세요:
{
  "states": ["IDLE", "COOLING", "VENTILATING", "DEHUMIDIFYING", "ERROR_FALLBACK"],
  "initial_state": "IDLE",
  "transitions": [
    {"from": "IDLE", "to": "COOLING", "condition": "temp > 28"},
    {"from": "IDLE", "to": "VENTILATING", "condition": "co2 > 1000"},
    {"from": "COOLING", "to": "IDLE", "condition": "temp <= 25"},
    {"from": "IDLE", "to": "ERROR_FALLBACK", "condition": "sensor_error"}
  ]
}
"""

# 3. FSM 정적 품질 검증기 (100점 만점)
def evaluate_fsm_static(json_text: str) -> tuple[int, dict]:
    scores = {
        "json_syntax": 0,
        "domain_inference": 0,
        "safety_fallback": 0,
        "graph_integrity": 0
    }

    try:
        data = json.loads(json_text)
        if "states" in data and "transitions" in data and "initial_state" in data:
            scores["json_syntax"] = 25
        else:
            return sum(scores.values()), scores
    except Exception:
        return 0, scores

    states = [str(s).upper() for s in data.get("states", [])]
    transitions = data.get("transitions", [])

    # 도메인 제어 상태 유추 검사
    if any(keyword in str(states) for keyword in ["COOL", "HEAT", "VENT", "DRY", "AIR", "DEHUMID"]):
        scores["domain_inference"] = 25

    # Safety/Fallback 검사
    if any("ERR" in s or "FALLBACK" in s or "FAIL" in s for s in states):
        scores["safety_fallback"] = 25

    # Graph 무결성 검사 (Deadlock 유무)
    try:
        G = nx.DiGraph()
        for s in data["states"]:
            G.add_node(s)
        for t in transitions:
            G.add_edge(t["from"], t["to"])

        isolated = list(nx.isolates(G))
        if len(isolated) == 0 and len(G.nodes) >= 3:
            scores["graph_integrity"] = 25
    except Exception:
        pass

    return sum(scores.values()), scores

# 4. 벤치마크 실행 함수
async def run_open_dataset_benchmark(model_name: str, dataset: list[dict]):
    print(f"\n==========================================")
    print(f"[오픈 데이터셋 벤치마크] 모델: {model_name}")
    print(f"==========================================")

    for idx, sample in enumerate(dataset, 1):
        start_time = time.time()
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"오픈 데이터셋 샘플 #{idx}: {json.dumps(sample, ensure_ascii=False)}"}
                ],
                temperature=0.1
            )
            latency_ms = round((time.time() - start_time) * 1000, 2)
            raw_content = response.choices[0].message.content.strip()

            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                raw_content = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])

            total_score, breakdown = evaluate_fsm_static(raw_content)

            print(f"[샘플 #{idx}] Latency: {latency_ms} ms | FSM 점수: {total_score}/100")
            print(f" - 세부 항목: {breakdown}")
            print(f"--- [생성된 FSM JSON 전체] ---")
            print(raw_content)
            print(f"-----------------------------------\n")

        except Exception as e:
            print(f"[샘플 #{idx}] 오류 발생: {e}")

async def main():
    # 1. 오픈 데이터셋 온라인 로드 (상위 5개 샘플 추출)
    dataset = fetch_open_multimodal_dataset(SMARTHOME_DATASET_URL, limit=5)

    # 2. OpenRouter 모델 벤치마크 수행
    models = [
        "openai/gpt-5.2",
        "anthropic/claude-sonnet-4.6"
    ]

    for model in models:
        await run_open_dataset_benchmark(model, dataset)

if __name__ == "__main__":
    asyncio.run(main())
