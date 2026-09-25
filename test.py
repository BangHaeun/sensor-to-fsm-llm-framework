import asyncio
import json
import os
import time

import networkx as nx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ucimlrepo import fetch_ucirepo


# 1. API key
load_dotenv()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# 2. Dataset
# UCI Room Occupancy Estimation (id=864)
# Raw environmental sensors: temperature, light, sound, CO2, PIR, etc.
# The occupancy target is intentionally NOT given to the LLM.
DATASET_ID = 864
NUM_WINDOWS = 5
WINDOW_SIZE = 6


def load_sensor_windows():
    dataset = fetch_ucirepo(id=DATASET_ID)

    # Only feature columns are used. The target (Room_Occupancy_Count) is excluded.
    features = dataset.data.features.copy()

    # Date/Time may be useful for ordering, but they are not sensor variables.
    non_sensor_columns = {"Date", "Time"}
    sensor_columns = [
        column for column in features.columns
        if column not in non_sensor_columns
    ]

    if len(features) < NUM_WINDOWS * WINDOW_SIZE:
        raise ValueError("Dataset is too small for the requested windows.")

    # Fixed, deterministic windows so both models receive exactly the same inputs.
    max_start = len(features) - WINDOW_SIZE
    starts = [
        round(i * max_start / (NUM_WINDOWS - 1))
        for i in range(NUM_WINDOWS)
    ]

    windows = []
    for start in starts:
        frame = features.iloc[start:start + WINDOW_SIZE]

        # Convert through JSON so numpy/pandas scalar types become standard Python types.
        records = json.loads(frame.to_json(orient="records"))

        windows.append({
            "start_index": int(start),
            "sensor_columns": sensor_columns,
            "observations": records,
        })

    return windows, sensor_columns


# 3. Neutral prompt
# No state names, actuator mappings, thresholds, or domain-specific answer examples are provided.
SYSTEM_PROMPT = """
You are given a short sequence of measurements from indoor environmental sensors.

Infer a finite state machine (FSM) that summarizes meaningful operational modes
and transitions supported by the provided measurements.

Rules:
1. Base the FSM only on the sensor fields and values present in the input.
2. Do not assume sensors, actuator types, labels, states, or external variables
   that are not present in the input.
3. Choose state names independently from the observed data. No state names are predefined.
4. Every transition condition must reference one or more sensor field names
   that appear in the input.
5. Transition conditions should use explicit numeric or boolean comparisons.
6. Return only valid JSON. Do not include markdown or explanatory text.

Required JSON structure:
{
  "states": ["..."],
  "initial_state": "...",
  "transitions": [
    {
      "from": "...",
      "to": "...",
      "condition": "..."
    }
  ]
}
""".strip()


# 4. Model-independent static evaluator
# This evaluator checks structure and grounding only.
# It does NOT reward particular state names such as COOLING, HEATING, etc.
def evaluate_fsm_static(json_text: str, sensor_columns: list[str]) -> tuple[int, dict]:
    scores = {
        "schema_validity": 0,
        "transition_integrity": 0,
        "sensor_grounding": 0,
        "graph_integrity": 0,
    }

    try:
        data = json.loads(json_text)
    except Exception:
        return 0, scores

    states = data.get("states")
    initial_state = data.get("initial_state")
    transitions = data.get("transitions")

    # 1) Basic JSON/schema validity
    if (
        isinstance(states, list)
        and len(states) >= 2
        and isinstance(initial_state, str)
        and isinstance(transitions, list)
        and len(transitions) >= 1
    ):
        scores["schema_validity"] = 25
    else:
        return sum(scores.values()), scores

    state_set = set(str(state) for state in states)

    # 2) Transition integrity
    transition_ok = True
    for transition in transitions:
        if not isinstance(transition, dict):
            transition_ok = False
            break

        source = str(transition.get("from", ""))
        target = str(transition.get("to", ""))
        condition = transition.get("condition")

        if (
            source not in state_set
            or target not in state_set
            or not isinstance(condition, str)
            or condition.strip() == ""
        ):
            transition_ok = False
            break

    if initial_state in state_set and transition_ok:
        scores["transition_integrity"] = 25

    # 3) Sensor grounding
    # Every transition should be supported by at least one real input sensor field.
    normalized_columns = [column.lower() for column in sensor_columns]
    grounded_count = 0

    for transition in transitions:
        condition = str(transition.get("condition", "")).lower()

        if any(column in condition for column in normalized_columns):
            grounded_count += 1

    if transitions:
        grounding_ratio = grounded_count / len(transitions)

        if grounding_ratio == 1.0:
            scores["sensor_grounding"] = 25
        elif grounding_ratio >= 0.75:
            scores["sensor_grounding"] = 20
        elif grounding_ratio >= 0.5:
            scores["sensor_grounding"] = 15
        elif grounding_ratio > 0:
            scores["sensor_grounding"] = 5

    # 4) Graph integrity
    try:
        graph = nx.DiGraph()
        graph.add_nodes_from(states)

        for transition in transitions:
            graph.add_edge(transition["from"], transition["to"])

        reachable = {initial_state} | nx.descendants(graph, initial_state)

        if (
            len(list(nx.isolates(graph))) == 0
            and reachable == state_set
        ):
            scores["graph_integrity"] = 25
    except Exception:
        pass

    return sum(scores.values()), scores


def clean_model_output(raw_content: str) -> str:
    raw_content = raw_content.strip()

    if raw_content.startswith("```"):
        lines = raw_content.splitlines()

        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            raw_content = "\n".join(lines[1:-1])
        else:
            raw_content = "\n".join(lines[1:])

    return raw_content.strip()


# 5. Benchmark
async def run_framework(model_name: str, windows: list[dict], sensor_columns: list[str]):
    print("\n==========================================")
    print(f"[Sensor-to-FSM] model: {model_name}")
    print("==========================================")

    results = []

    for index, window in enumerate(windows, 1):
        user_payload = {
            "sensor_fields": sensor_columns,
            "observations": window["observations"],
        }

        start_time = time.time()

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
                temperature=0.0,
            )

            latency_ms = round((time.time() - start_time) * 1000, 2)
            raw_content = response.choices[0].message.content
            cleaned_content = clean_model_output(raw_content)

            total_score, breakdown = evaluate_fsm_static(
                cleaned_content,
                sensor_columns,
            )

            result = {
                "sample": index,
                "latency_ms": latency_ms,
                "score": total_score,
                "breakdown": breakdown,
                "fsm": cleaned_content,
            }
            results.append(result)

            print(
                f"[window #{index}] "
                f"Latency: {latency_ms} ms | "
                f"Static score: {total_score}/100"
            )
            print(f" - breakdown: {breakdown}")
            print("--- generated FSM ---")
            print(cleaned_content)
            print("---------------------\n")

        except Exception as error:
            print(f"[window #{index}] error: {error}")

    return results


def print_summary(model_name: str, results: list[dict]):
    if not results:
        print(f"[summary] {model_name}: no successful results")
        return

    average_score = sum(result["score"] for result in results) / len(results)
    average_latency = sum(result["latency_ms"] for result in results) / len(results)

    print(
        f"[summary] {model_name} | "
        f"avg score: {average_score:.1f}/100 | "
        f"avg latency: {average_latency:.2f} ms"
    )


async def main():
    windows, sensor_columns = load_sensor_windows()

    print("[dataset] UCI Room Occupancy Estimation")
    print(f"[sensor fields] {sensor_columns}")
    print(f"[windows] {NUM_WINDOWS} x {WINDOW_SIZE} observations")

    models = [
        "openai/gpt-5.2",
        "anthropic/claude-sonnet-4.6",
    ]

    for model_name in models:
        results = await run_framework(
            model_name,
            windows,
            sensor_columns,
        )
        print_summary(model_name, results)


if __name__ == "__main__":
    asyncio.run(main())