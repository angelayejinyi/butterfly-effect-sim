from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI, APIConnectionError, APITimeoutError, BadRequestError, RateLimitError


# 1) Personas (selectable via --persona)
NEUTRAL_PERSONA = (
    "You are a participant in a repeated interaction. "
    "Use observed history to choose a reasonable action each round, balancing immediate outcomes and future interaction stability."
)

STRATEGIC_PERSONA = (
    "You are a strategic participant focused on maximizing your payoff. "
    "Study opponent behavior for exploitable patterns, and adjust your strategy to gain advantage while staying competitive in the long term."
)


# 2) Memory construction conditions (2 x 2)
MEMORY_CONDITIONS: dict[str, dict[str, Any]] = {
    "M1_TABLE_RAW": {"format": "table", "with_stats": False},
    "M2_NARRATIVE_RAW": {"format": "narrative", "with_stats": False},
    "M3_TABLE_RAW_STATS": {"format": "table", "with_stats": True},
    "M4_NARRATIVE_RAW_STATS": {"format": "narrative", "with_stats": True},
}


# 3) Fixed experiment setup
HORIZON = 10
SEEDS = list(range(30))
TEMPERATURE = 0.3
TOP_P = 0.95
ACTIONS = ["Cooperate", "Defect"]
PAYOFF = {
    ("Cooperate", "Cooperate"): (3, 3),
    ("Cooperate", "Defect"): (0, 5),
    ("Defect", "Cooperate"): (5, 0),
    ("Defect", "Defect"): (1, 1),
}


def load_env() -> tuple[str, str, str]:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
        print(f"Loaded env from: {env_path}")
    else:
        load_dotenv()
        print("Using current environment variables.")

    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5-chat")
    return endpoint, api_key, model


def make_client(endpoint: str, api_key: str) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-01",
    )


def build_two_agent_history_for_memory(role: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    role_hist = []
    for row in history:
        if role == "A":
            you_action, other_action = row["a_action"], row["b_action"]
            you_payoff, other_payoff = row["a_payoff"], row["b_payoff"]
        else:
            you_action, other_action = row["b_action"], row["a_action"]
            you_payoff, other_payoff = row["b_payoff"], row["a_payoff"]

        role_hist.append(
            {
                "round": row["round"],
                "you": you_action,
                "other": other_action,
                "you_payoff": you_payoff,
                "other_payoff": other_payoff,
            }
        )
    return role_hist


def build_memory_table(role_history: list[dict[str, Any]]) -> str:
    if not role_history:
        return "History: No previous rounds."

    lines = ["History:", "Round | You | Other agent | Your payoff | Other payoff"]
    for row in role_history:
        lines.append(
            f"{row['round']} | {row['you']} | {row['other']} | {row['you_payoff']} | {row['other_payoff']}"
        )
    return "\n".join(lines)


def build_memory_narrative(role_history: list[dict[str, Any]]) -> str:
    if not role_history:
        return "History: No previous rounds."

    lines = ["History:"]
    for row in role_history:
        lines.append(
            f"In round {row['round']}, you chose {row['you']} and the other agent chose {row['other']}. "
            f"You received {row['you_payoff']} points and the other agent received {row['other_payoff']}."
        )
    return "\n".join(lines)


def build_role_statistics_block(role_history: list[dict[str, Any]]) -> str:
    n = len(role_history)
    you_coop = sum(1 for r in role_history if r["you"] == "Cooperate")
    you_defect = sum(1 for r in role_history if r["you"] == "Defect")
    other_coop = sum(1 for r in role_history if r["other"] == "Cooperate")
    other_defect = sum(1 for r in role_history if r["other"] == "Defect")
    cc = sum(1 for r in role_history if r["you"] == "Cooperate" and r["other"] == "Cooperate")
    cd = sum(1 for r in role_history if r["you"] == "Cooperate" and r["other"] == "Defect")
    dc = sum(1 for r in role_history if r["you"] == "Defect" and r["other"] == "Cooperate")
    dd = sum(1 for r in role_history if r["you"] == "Defect" and r["other"] == "Defect")
    you_cum = sum(r["you_payoff"] for r in role_history)
    other_cum = sum(r["other_payoff"] for r in role_history)
    you_rate = (you_coop / n) if n > 0 else 0.0
    other_rate = (other_coop / n) if n > 0 else 0.0
    you_avg = (you_cum / n) if n > 0 else 0.0
    other_avg = (other_cum / n) if n > 0 else 0.0

    lines = [
        "Summary statistics:",
        f"- You cooperated {you_coop} times and defected {you_defect} times.",
        f"- Your cooperation rate: {you_rate:.3f}",
        f"- Opponent cooperated {other_coop} times and defected {other_defect} times.",
        f"- Opponent cooperation rate: {other_rate:.3f}",
        f"- Your cumulative payoff: {you_cum}",
        f"- Opponent cumulative payoff: {other_cum}",
        f"- Your average payoff per round: {you_avg:.3f}",
        f"- Opponent average payoff per round: {other_avg:.3f}",
        f"- Joint outcomes: CC={cc}, CD={cd}, DC={dc}, DD={dd}",
    ]
    return "\n".join(lines)


def build_two_agent_memory_block(
    role: str,
    history: list[dict[str, Any]],
    memory_format: str,
    with_stats: bool,
) -> str:
    role_history = build_two_agent_history_for_memory(role=role, history=history)
    if memory_format == "table":
        history_text = build_memory_table(role_history)
    elif memory_format == "narrative":
        history_text = build_memory_narrative(role_history)
    else:
        raise ValueError(f"Unknown memory format: {memory_format}")

    if with_stats:
        return f"{history_text}\n\n{build_role_statistics_block(role_history)}"
    return history_text


def build_two_agent_prompt(persona_text: str, role: str, memory_block: str, horizon: int) -> str:
    game_block = (
        "Game setup:\n"
        f"- Repeated Prisoner's Dilemma for {horizon} rounds.\n"
        "- Each round, both agents simultaneously choose one action: Cooperate or Defect.\n"
        "- You cannot observe the other agent's current-round action before choosing.\n"
        "- The other agent is another model-driven player under the same memory-construction condition.\n"
        "\n"
        "Payoffs (your payoff, other payoff):\n"
        "- If both choose Cooperate: (3, 3).\n"
        "- If you choose Cooperate and other chooses Defect: (0, 5).\n"
        "- If you choose Defect and other chooses Cooperate: (5, 0).\n"
        "- If both choose Defect: (1, 1).\n"
        "\n"
        "Task: choose a reasonable action this round based on memory and repeated-game context."
    )

    instruction_block = (
        "Output format:\n"
        "Return valid JSON with keys:\n"
        "- action: one of [\"Cooperate\", \"Defect\"]\n"
        "- reason: short string"
    )

    return (
        f"Persona:\n{persona_text}\n\n"
        f"Your role label this run: Agent {role}.\n\n"
        f"{game_block}\n\n"
        f"{memory_block}\n\n"
        f"{instruction_block}"
    )


def call_model(client: AzureOpenAI, model: str, prompt: str, temperature: float, top_p: float) -> str:
    max_attempts = 5
    base_delay_seconds = 1.0

    for attempt in range(1, max_attempts + 1):
        try:
            try:
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    top_p=top_p,
                )
                try:
                    text = response.output_text
                    if isinstance(text, str) and text:
                        return text
                except Exception:
                    pass

                texts: list[str] = []
                for item in getattr(response, "output", []) or []:
                    for part in getattr(item, "content", []) or []:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str) and part_text:
                            texts.append(part_text)

                if texts:
                    return "\n".join(texts)

                return str(response)
            except BadRequestError as e:
                if "Unsupported data type" not in str(e):
                    raise

            chat_resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=top_p,
            )
            return chat_resp.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            if attempt == max_attempts:
                raise

            delay_seconds = base_delay_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
            print(
                f"Transient API error on attempt {attempt}/{max_attempts}: {e}. "
                f"Retrying in {delay_seconds:.1f}s..."
            )
            time.sleep(delay_seconds)

    raise RuntimeError("Model call loop ended unexpectedly")


def parse_action(raw_text: str) -> tuple[str, bool, str]:
    try:
        obj = json.loads(raw_text)
        action = obj.get("action")
        if action in ACTIONS:
            return action, True, ""
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        try:
            obj = json.loads(match.group(0))
            action = obj.get("action")
            if action in ACTIONS:
                return action, True, ""
        except Exception:
            pass

    return "Defect", False, "parse_failed_fallback_to_defect"


def choose_action_with_retry(
    client: AzureOpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_retries: int = 1,
) -> tuple[str, bool, str, str]:
    raw_last = ""
    note_last = ""

    for retry_idx in range(max_retries + 1):
        suffix = "\nReminder: return valid JSON only." if retry_idx > 0 else ""
        raw = call_model(client, model, prompt + suffix, temperature, top_p)
        action, valid, note = parse_action(raw)
        raw_last = raw
        note_last = note
        if valid:
            return action, True, "", raw

    return "Defect", False, note_last, raw_last


def run_episode(
    task: tuple[str, str, bool, int, int, float, float, str, str, str, str]
) -> list[dict[str, Any]]:
    (
        memory_condition,
        memory_format,
        with_stats,
        seed,
        horizon,
        temperature,
        top_p,
        endpoint,
        api_key,
        model,
        persona_text,
    ) = task

    client = make_client(endpoint, api_key)
    rng = random.Random(seed)

    history: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for round_idx in range(1, horizon + 1):
        # Randomized prompt ordering helps reduce stable positional bias.
        if rng.random() < 0.5:
            memory_a = build_two_agent_memory_block("A", history, memory_format, with_stats)
            prompt_a = build_two_agent_prompt(persona_text, "A", memory_a, horizon)

            memory_b = build_two_agent_memory_block("B", history, memory_format, with_stats)
            prompt_b = build_two_agent_prompt(persona_text, "B", memory_b, horizon)
        else:
            memory_b = build_two_agent_memory_block("B", history, memory_format, with_stats)
            prompt_b = build_two_agent_prompt(persona_text, "B", memory_b, horizon)

            memory_a = build_two_agent_memory_block("A", history, memory_format, with_stats)
            prompt_a = build_two_agent_prompt(persona_text, "A", memory_a, horizon)

        a_action, a_valid, a_note, a_raw = choose_action_with_retry(
            client=client,
            model=model,
            prompt=prompt_a,
            temperature=temperature,
            top_p=top_p,
            max_retries=1,
        )
        b_action, b_valid, b_note, b_raw = choose_action_with_retry(
            client=client,
            model=model,
            prompt=prompt_b,
            temperature=temperature,
            top_p=top_p,
            max_retries=1,
        )

        a_payoff, b_payoff = PAYOFF[(a_action, b_action)]

        history.append(
            {
                "round": round_idx,
                "a_action": a_action,
                "b_action": b_action,
                "a_payoff": a_payoff,
                "b_payoff": b_payoff,
            }
        )

        rows.append(
            {
                "memory_condition": memory_condition,
                "memory_format": memory_format,
                "memory_abstraction": "raw_plus_stats" if with_stats else "raw_only",
                "seed": seed,
                "round": round_idx,
                "agent_a_action": a_action,
                "agent_b_action": b_action,
                "agent_a_payoff": a_payoff,
                "agent_b_payoff": b_payoff,
                "valid_parse_a": a_valid,
                "valid_parse_b": b_valid,
                "parse_note_a": a_note,
                "parse_note_b": b_note,
                "agent_a_raw_output": a_raw,
                "agent_b_raw_output": b_raw,
                "agent_a_prompt_input": prompt_a,
                "agent_b_prompt_input": prompt_b,
            }
        )

    return rows


def build_tasks(
    memory_conditions: dict[str, dict[str, Any]],
    seeds: list[int],
    horizon: int,
    temperature: float,
    top_p: float,
    endpoint: str,
    api_key: str,
    model: str,
    persona_text: str,
) -> list[tuple[str, str, bool, int, int, float, float, str, str, str, str]]:
    tasks = []
    for memory_condition, config in memory_conditions.items():
        for seed in seeds:
            tasks.append(
                (
                    memory_condition,
                    config["format"],
                    bool(config["with_stats"]),
                    seed,
                    horizon,
                    temperature,
                    top_p,
                    endpoint,
                    api_key,
                    model,
                    persona_text,
                )
            )
    return tasks


def dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["memory_condition", "seed", "round"],
        keep="last",
    )
    return df.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel memory two-agent Prisoner's Dilemma simulation")
    parser.add_argument("--workers", type=int, default=2, help="Thread workers")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--seed-count", type=int, default=len(SEEDS), help="Use seeds [0..seed_count-1]")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument(
        "--persona",
        type=str,
        choices=["neutral", "strategic"],
        default="strategic",
        help="Persona used for both agents in Block 3 two-agent run",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data/generated/PD/memory_construction_two_agent_strategic_persona.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed (memory_condition, seed) episodes already in out-file",
    )
    args = parser.parse_args()

    endpoint, api_key, model = load_env()
    print(f"Azure OpenAI deployment: {model}")

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    persona_text = STRATEGIC_PERSONA if args.persona == "strategic" else NEUTRAL_PERSONA

    seeds = list(range(args.seed_count))
    tasks = build_tasks(
        memory_conditions=MEMORY_CONDITIONS,
        seeds=seeds,
        horizon=args.horizon,
        temperature=args.temperature,
        top_p=args.top_p,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        persona_text=persona_text,
    )

    existing_rows: list[dict[str, Any]] = []
    if args.resume and out_file.exists():
        existing_df = pd.read_csv(out_file)
        existing_rows = existing_df.to_dict("records")
        done = {(r["memory_condition"], int(r["seed"])) for r in existing_rows}
        tasks = [t for t in tasks if (t[0], t[3]) not in done]
        print(f"Resume enabled: loaded {len(existing_rows)} existing rows")

    print(f"Total tasks (memory_condition, seed): {len(tasks)}")
    print(f"Running with workers={args.workers}")

    all_rows = existing_rows.copy()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_episode, task): task for task in tasks}

        for future in as_completed(future_to_task):
            memory_condition, _, _, seed, *_ = future_to_task[future]
            try:
                episode_rows = future.result()
                all_rows.extend(episode_rows)
                completed += 1

                if completed % 10 == 0 or completed == len(tasks):
                    all_rows = dedup_rows(all_rows)
                    pd.DataFrame(all_rows).to_csv(out_file, index=False)
                    print(
                        f"Progress: {completed}/{len(tasks)} episodes complete; "
                        f"latest={memory_condition}/seed={seed}"
                    )
            except Exception as e:
                print(f"Task failed for {memory_condition}/seed={seed}: {e}")
                print(traceback.format_exc())

    all_rows = dedup_rows(all_rows)
    pd.DataFrame(all_rows).to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    results_df = pd.DataFrame(all_rows)
    summary_df = (
        results_df.groupby(["memory_condition"], as_index=False)
        .agg(
            avg_agent_a_round_payoff=("agent_a_payoff", "mean"),
            avg_agent_b_round_payoff=("agent_b_payoff", "mean"),
            agent_a_coop_rate=("agent_a_action", lambda s: (s == "Cooperate").mean()),
            agent_b_coop_rate=("agent_b_action", lambda s: (s == "Cooperate").mean()),
            parse_success_rate_a=("valid_parse_a", "mean"),
            parse_success_rate_b=("valid_parse_b", "mean"),
        )
    )
    print(summary_df)


if __name__ == "__main__":
    main()
