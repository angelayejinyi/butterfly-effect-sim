from __future__ import annotations

import argparse
import json
import os
import random
import re
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError
from openai import RateLimitError, APIConnectionError, APITimeoutError


# 1) Persona definitions (three meaning sets: neutral, strategic, cooperative; three format variations each)

# Neutral personas (balance-focused)
NEUTRAL_BASE = (
    "You are a participant in a repeated interaction. "
    "Use observed history to choose a reasonable action each round, balancing immediate outcomes and future interaction stability."
)

# Strategic personas (exploit-focused, win-oriented)
STRATEGIC_BASE = (
    "You are a strategic participant focused on maximizing your long-term payoff. "
    "Study opponent behavior for exploitable patterns, and adjust your strategy to gain advantage while staying competitive."
)

# Cooperative personas (benign, trust-and-reciprocity focused)
COOPERATIVE_BASE = (
    "You are a cooperative participant focused on mutual benefit and long-term stability. "
    "Use observed history to maintain trust when possible, respond proportionally to harm, and favor mutually beneficial outcomes."
)

PERSONA_VARIATIONS = {
    # Neutral format variants
    "P1_PLAIN_NEUTRAL": NEUTRAL_BASE,
    "P2_DESCRIPTIVE_NEUTRAL": (
        "Persona profile:\n"
        "- Focus on observed history\n"
        "- Balance immediate and future outcomes\n"
        "- Prefer stable interaction when feasible\n"
        "- Avoid rigid one-rule behavior"
    ),
    "P3_TABULAR_NEUTRAL": (
        "Persona | Value\n"
        "Trait | Adaptive, history-aware\n"
        "Norm | Choose reasonable actions from evidence\n"
        "Decision focus | Balance immediate outcomes and future stability\n"
        "Context | Repeated interactions"
    ),
    # Strategic format variants
    "P4_PLAIN_STRATEGIC": STRATEGIC_BASE,
    "P5_DESCRIPTIVE_STRATEGIC": (
        "Persona profile:\n"
        "- Maximize your long-term payoff\n"
        "- Study opponent for exploitable patterns\n"
        "- Adjust strategy to gain advantage\n"
        "- Stay strategic and competitive"
    ),
    "P6_TABULAR_STRATEGIC": (
        "Persona | Value\n"
        "Goal | Maximize your long-term payoff\n"
        "Action | Study opponent for exploitable patterns\n"
        "Strategy | Adjust strategy to gain advantage\n"
        "Style | Stay strategic and competitive"
    ),
    # Cooperative/benign format variants
    "P7_PLAIN_COOPERATIVE": COOPERATIVE_BASE,
    "P8_DESCRIPTIVE_COOPERATIVE": (
        "Persona profile:\n"
        "- Prioritize mutual benefit and trust\n"
        "- Prefer cooperation when supported by history\n"
        "- Respond proportionally to repeated harm\n"
        "- Seek stable long-term outcomes"
    ),
    "P9_TABULAR_COOPERATIVE": (
        "Persona | Value\n"
        "Trait | Cooperative, trust-oriented\n"
        "Norm | Favor reciprocal cooperation when credible\n"
        "Decision focus | Maximize long-term joint stability with self-protection\n"
        "Context | Repeated interactions with memory"
    ),
}


PAPER_PERSONAS = {
    key: value for key, value in PERSONA_VARIATIONS.items()
    if key in ("P4_PLAIN_STRATEGIC", "P5_DESCRIPTIVE_STRATEGIC", "P6_TABULAR_STRATEGIC")
}


# 2) Fixed experiment setup
HORIZON = 10
SEEDS = list(range(30))
TEMPERATURE = 0.3
TOP_P = 0.95
ACTIONS = ["Cooperate", "Defect"]
OPPONENT_POLICIES = ["TitForTat", "Random", "AlwaysCooperate", "AlwaysDefect"]
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


# 3) External fixed opponent policies
def opponent_action(
    policy_name: str,
    round_idx: int,
    agent_past_actions: list[str],
    rng: random.Random,
) -> str:
    if policy_name == "AlwaysCooperate":
        return "Cooperate"
    if policy_name == "AlwaysDefect":
        return "Defect"
    if policy_name == "TitForTat":
        if round_idx == 1:
            return "Cooperate"
        return agent_past_actions[-1]
    if policy_name == "Random":
        return rng.choice(ACTIONS)
    raise ValueError(f"Unknown policy: {policy_name}")


# 4) History/prompt helpers
def build_full_structured_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "History: No previous rounds."

    lines = ["History (full structured):", "Round | You | Opponent | Your payoff | Opp payoff"]
    for row in history:
        lines.append(
            f"{row['round']} | {row['you']} | {row['opp']} | {row['you_payoff']} | {row['opp_payoff']}"
        )
    return "\n".join(lines)


def build_prompt(persona_text: str, history: list[dict[str, Any]], horizon: int) -> str:
    game_block = (
        "Game setup:\n"
        f"- Repeated Prisoner's Dilemma for {horizon} rounds.\n"
        "- Each round, both players simultaneously choose one action: Cooperate or Defect.\n"
        "- You cannot observe the opponent's current-round action before choosing.\n"
        "- Opponent follows a fixed external policy (unknown to you).\n"
        "\n"
        "Payoffs (your payoff, opponent payoff):\n"
        "- If both choose Cooperate: (3, 3).\n"
        "- If you choose Cooperate and opponent chooses Defect: (0, 5).\n"
        "- If you choose Defect and opponent chooses Cooperate: (5, 0).\n"
        "- If both choose Defect: (1, 1).\n"
        "\n"
        "Task: choose a reasonable action this round based on history and repeated-game context."
    )

    instruction_block = (
        "Output format:\n"
        "Return valid JSON with keys:\n"
        "- action: one of [\"Cooperate\", \"Defect\"]\n"
        "- reason: short string"
    )

    return (
        f"Persona:\n{persona_text}\n\n"
        f"{game_block}\n\n"
        f"{build_full_structured_history(history)}\n\n"
        f"{instruction_block}"
    )


def call_model(
    client: AzureOpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
) -> str:
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


def run_episode(task: tuple[str, str, str, int, int, float, float, str, str, str]) -> list[dict[str, Any]]:
    persona_name, persona_text, policy_name, seed, horizon, temperature, top_p, endpoint, api_key, model = task
    client = make_client(endpoint, api_key)

    history: list[dict[str, Any]] = []
    agent_actions: list[str] = []
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for round_idx in range(1, horizon + 1):
        prompt = build_prompt(persona_text=persona_text, history=history, horizon=horizon)
        agent_action, valid_parse, parse_note, raw_output = choose_action_with_retry(
            client=client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_retries=1,
        )
        opp_action = opponent_action(
            policy_name=policy_name,
            round_idx=round_idx,
            agent_past_actions=agent_actions,
            rng=rng,
        )
        agent_payoff, opp_payoff = PAYOFF[(agent_action, opp_action)]

        history.append(
            {
                "round": round_idx,
                "you": agent_action,
                "opp": opp_action,
                "you_payoff": agent_payoff,
                "opp_payoff": opp_payoff,
            }
        )
        agent_actions.append(agent_action)

        rows.append(
            {
                "persona_format": persona_name,
                "opponent_policy": policy_name,
                "seed": seed,
                "round": round_idx,
                "agent_action": agent_action,
                "opponent_action": opp_action,
                "agent_payoff": agent_payoff,
                "opponent_payoff": opp_payoff,
                "valid_parse": valid_parse,
                "parse_note": parse_note,
                "agent_raw_output": raw_output,
                "agent_prompt_input": prompt,
            }
        )

    return rows


def build_tasks(
    selected_personas: dict[str, str],
    opponent_policies: list[str],
    seeds: list[int],
    horizon: int,
    temperature: float,
    top_p: float,
    endpoint: str,
    api_key: str,
    model: str,
) -> list[tuple[str, str, str, int, int, float, float, str, str, str]]:
    tasks = []
    for persona_name, persona_text in selected_personas.items():
        for policy_name in opponent_policies:
            for seed in seeds:
                tasks.append(
                    (
                        persona_name,
                        persona_text,
                        policy_name,
                        seed,
                        horizon,
                        temperature,
                        top_p,
                        endpoint,
                        api_key,
                        model,
                    )
                )
    return tasks


def dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    df = pd.DataFrame(rows)
    # Keep the latest copy of each round record if duplicates exist.
    df = df.drop_duplicates(
        subset=["persona_format", "opponent_policy", "seed", "round"],
        keep="last",
    )
    return df.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel persona Prisoner's Dilemma simulation")
    parser.add_argument("--workers", type=int, default=2, help="Thread workers")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--seed-count", type=int, default=len(SEEDS), help="Use seeds [0..seed_count-1]")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument(
        "--out-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data/results/generated/PD/persona_format_single_agent.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed (persona_format, opponent_policy, seed) triplets already in out-file",
    )
    parser.add_argument(
        "--condition-set", choices=("paper", "all"), default="paper",
        help="paper: manuscript conditions; all: include exploratory conditions",
    )
    args = parser.parse_args()

    endpoint, api_key, model = load_env()
    print(f"Azure OpenAI deployment: {model}")

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_count))
    selected_personas = PAPER_PERSONAS if args.condition_set == "paper" else PERSONA_VARIATIONS

    tasks = build_tasks(
        selected_personas=selected_personas,
        opponent_policies=OPPONENT_POLICIES,
        seeds=seeds,
        horizon=args.horizon,
        temperature=args.temperature,
        top_p=args.top_p,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
    )

    existing_rows: list[dict[str, Any]] = []
    if args.resume and out_file.exists():
        existing_df = pd.read_csv(out_file)
        existing_rows = existing_df.to_dict("records")
        done = {
            (r["persona_format"], r["opponent_policy"], int(r["seed"]))
            for r in existing_rows
        }
        tasks = [t for t in tasks if (t[0], t[2], t[3]) not in done]
        print(f"Resume enabled: loaded {len(existing_rows)} existing rows")

    print(f"Total tasks (persona, policy, seed): {len(tasks)}")
    print(f"Running with workers={args.workers}")

    all_rows = existing_rows.copy()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_episode, task): task for task in tasks}

        for future in as_completed(future_to_task):
            persona_name, _, policy_name, seed, *_ = future_to_task[future]
            try:
                episode_rows = future.result()
                all_rows.extend(episode_rows)
                completed += 1

                if completed % 10 == 0 or completed == len(tasks):
                    all_rows = dedup_rows(all_rows)
                    pd.DataFrame(all_rows).to_csv(out_file, index=False)
                    print(
                        f"Progress: {completed}/{len(tasks)} episodes complete; latest={persona_name}/{policy_name}/seed={seed}"
                    )
            except Exception as e:
                print(f"Task failed for {persona_name}/{policy_name}/seed={seed}: {e}")
                print(traceback.format_exc())

    all_rows = dedup_rows(all_rows)
    pd.DataFrame(all_rows).to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    results_df = pd.DataFrame(all_rows)
    summary_df = (
        results_df.groupby(["persona_format", "opponent_policy"], as_index=False)
        .agg(
            avg_agent_round_payoff=("agent_payoff", "mean"),
            agent_coop_rate=("agent_action", lambda s: (s == "Cooperate").mean()),
            parse_success_rate=("valid_parse", "mean"),
        )
    )
    print(summary_df)


if __name__ == "__main__":
    main()
