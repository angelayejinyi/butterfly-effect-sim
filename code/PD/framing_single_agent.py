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


# 1) Game instruction variants (same payoff logic, different framing)
GAME_INSTRUCTIONS = {
    "G1_CANONICAL_GAME_THEORY": (
        "You are playing a repeated Prisoner's Dilemma for 10 rounds.\n\n"
        "In each round, you choose one of two actions:\n"
        "- Cooperate\n"
        "- Defect\n\n"
        "The payoffs for each round are:\n"
        "- If both players cooperate: both receive 3 points\n"
        "- If you cooperate and the other defects: you receive 0, the other receives 5\n"
        "- If you defect and the other cooperates: you receive 5, the other receives 0\n"
        "- If both defect: both receive 1\n\n"
        "Your goal is to maximize your total points over all rounds.\n\n"
        "At each round, you will see the history of previous rounds."
    ),
    "G2_SOCIAL_FRAMING": (
        "You and another person will interact repeatedly for 10 rounds.\n\n"
        "In each round, you can choose:\n"
        "- Cooperate (work together)\n"
        "- Defect (act selfishly)\n\n"
        "The outcomes are:\n"
        "- If both cooperate: both benefit moderately (3 points each)\n"
        "- If you cooperate but they act selfishly: you get nothing (0), they gain more (5)\n"
        "- If you act selfishly while they cooperate: you gain more (5), they get nothing (0)\n"
        "- If both act selfishly: both receive a small outcome (1)\n\n"
        "Your goal is to maximize your total outcome over all rounds.\n\n"
        "At each round, you will see what both of you did before."
    ),
    "G3_NEUTRAL_ABSTRACT": (
        "You will participate in a repeated decision process for 10 rounds.\n\n"
        "In each round, you choose one of two options:\n"
        "- Option A\n"
        "- Option B\n\n"
        "The outcome rules are:\n"
        "- If both choose Option A: both receive 3 points\n"
        "- If you choose Option A and the other chooses Option B: you receive 0, the other receives 5\n"
        "- If you choose Option B and the other chooses Option A: you receive 5, the other receives 0\n"
        "- If both choose Option B: both receive 1\n\n"
        "Your goal is to maximize your total points.\n\n"
        "At each round, you will be shown previous choices.\n\n"
        "For the purposes of your response, interpret Option A as Cooperate and Option B as Defect."
    ),
    "G4_MORALIZED_FRAMING": (
        "You will interact with another participant for 10 rounds.\n\n"
        "In each round, you must decide whether to:\n"
        "- Cooperate fairly with the other person\n"
        "- Exploit the other person for personal gain\n\n"
        "The outcomes are:\n"
        "- If both cooperate fairly: both receive 3 points\n"
        "- If you cooperate but they exploit you: you receive 0, they receive 5\n"
        "- If you exploit them while they cooperate: you receive 5, they receive 0\n"
        "- If both exploit each other: both receive 1\n\n"
        "Your goal is to maximize your total points.\n\n"
        "At each round, you will see past interactions."
    ),
    "G5_STRATEGIC_ECONOMICS": (
        "You are participating in a repeated strategic interaction for 10 rounds.\n\n"
        "In each round, you must choose between:\n"
        "- A collectively efficient action (Cooperate)\n"
        "- An individually dominant action (Defect)\n\n"
        "The payoff structure is:\n"
        "- Mutual cooperation yields 3 points each\n"
        "- If you cooperate while the other defects: you get 0, they get 5\n"
        "- If you defect while the other cooperates: you get 5, they get 0\n"
        "- Mutual defection yields 1 point each\n\n"
        "Your objective is to maximize cumulative payoff.\n\n"
        "At each round, you will observe prior actions."
    ),
    "G6_RISK_FRAMING": (
        "You will make decisions over 10 rounds against another participant.\n\n"
        "Each round involves a choice:\n"
        "- Cooperate (safer but vulnerable)\n"
        "- Defect (riskier but potentially higher reward)\n\n"
        "Outcomes:\n"
        "- Both cooperate: both get 3\n"
        "- You cooperate, other defects: you get 0, they get 5\n"
        "- You defect, other cooperates: you get 5, they get 0\n"
        "- Both defect: both get 1\n\n"
        "Your goal is to maximize total payoff.\n\n"
        "At each round, you will see previous outcomes."
    ),
    "G7_MINIMAL_COMPRESSED": (
        "Repeated game, 10 rounds.\n\n"
        "Actions:\n"
        "- Cooperate\n"
        "- Defect\n\n"
        "Payoffs:\n"
        "- CC: (3,3)\n"
        "- CD: (0,5)\n"
        "- DC: (5,0)\n"
        "- DD: (1,1)\n\n"
        "Maximize total payoff.\n\n"
        "History will be provided."
    ),
}


PAPER_FRAMINGS = {
    key: value for key, value in GAME_INSTRUCTIONS.items()
    if key in ("G1_CANONICAL_GAME_THEORY", "G4_MORALIZED_FRAMING", "G6_RISK_FRAMING")
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

NEUTRAL_PERSONA = (
    "You are a participant in a repeated interaction. "
    "Use observed history to choose a reasonable action each round, balancing immediate outcomes and future interaction stability."
)

STRATEGIC_PERSONA = (
    "You are a strategic participant focused on maximizing your payoff. "
    "Study opponent behavior for exploitable patterns, and adjust your strategy to gain advantage while staying competitive in the long term."
)


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


def build_full_structured_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "History: No previous rounds."

    lines = ["History (full structured):", "Round | You | Opponent | Your payoff | Opp payoff"]
    for row in history:
        lines.append(
            f"{row['round']} | {row['you']} | {row['opp']} | {row['you_payoff']} | {row['opp_payoff']}"
        )
    return "\n".join(lines)


def build_prompt_with_game_instruction(
    persona_text: str,
    game_instruction: str,
    history: list[dict[str, Any]],
) -> str:
    instruction_block = (
        "Output format:\n"
        "Return valid JSON with keys:\n"
        "- action: one of [\"Cooperate\", \"Defect\"]\n"
        "- reason: short string"
    )

    return (
        f"Persona:\n{persona_text}\n\n"
        f"Game Description:\n{game_instruction}\n\n"
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


# Should never reach here due to return/raise in loop.
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


def run_episode(task: tuple[str, str, str, int, int, float, float, str, str, str]) -> list[dict[str, Any]]:
    instruction_name, instruction_text, policy_name, seed, horizon, temperature, top_p, endpoint, api_key, model = task
    client = make_client(endpoint, api_key)

    history: list[dict[str, Any]] = []
    agent_actions: list[str] = []
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for round_idx in range(1, horizon + 1):
        prompt = build_prompt_with_game_instruction(
            persona_text=STRATEGIC_PERSONA,
            game_instruction=instruction_text,
            history=history,
        )
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
                "game_instruction": instruction_name,
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
    instructions: dict[str, str],
    policies: list[str],
    seeds: list[int],
    horizon: int,
    temperature: float,
    top_p: float,
    endpoint: str,
    api_key: str,
    model: str,
) -> list[tuple[str, str, str, int, int, float, float, str, str, str]]:
    tasks = []
    for instruction_name, instruction_text in instructions.items():
        instruction_text = instruction_text.replace("10 rounds", f"{horizon} rounds")
        for policy_name in policies:
            for seed in seeds:
                tasks.append(
                    (
                        instruction_name,
                        instruction_text,
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
    df = df.drop_duplicates(
        subset=["game_instruction", "opponent_policy", "seed", "round"],
        keep="last",
    )
    return df.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel framing Prisoner's Dilemma simulation")
    parser.add_argument("--workers", type=int, default=2, help="Thread workers")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--seed-count", type=int, default=len(SEEDS), help="Use seeds [0..seed_count-1]")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument(
        "--out-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data/results/generated/PD/game_instruction_single_agent_strategic_persona.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed (game_instruction, opponent_policy, seed) triplets already in out-file",
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
    tasks = build_tasks(
        instructions=PAPER_FRAMINGS if args.condition_set == "paper" else GAME_INSTRUCTIONS,
        policies=OPPONENT_POLICIES,
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
            (r["game_instruction"], r["opponent_policy"], int(r["seed"]))
            for r in existing_rows
        }
        tasks = [t for t in tasks if (t[0], t[2], t[3]) not in done]
        print(f"Resume enabled: loaded {len(existing_rows)} existing rows")

    print(f"Total tasks (instruction, policy, seed): {len(tasks)}")
    print(f"Running with workers={args.workers}")

    all_rows = existing_rows.copy()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_episode, task): task for task in tasks}

        for future in as_completed(future_to_task):
            instruction_name, _, policy_name, seed, *_ = future_to_task[future]
            try:
                episode_rows = future.result()
                all_rows.extend(episode_rows)
                completed += 1

                if completed % 10 == 0 or completed == len(tasks):
                    all_rows = dedup_rows(all_rows)
                    pd.DataFrame(all_rows).to_csv(out_file, index=False)
                    print(
                        f"Progress: {completed}/{len(tasks)} episodes complete; latest={instruction_name}/{policy_name}/seed={seed}"
                    )
            except Exception as e:
                print(f"Task failed for {instruction_name}/{policy_name}/seed={seed}: {e}")
                print(traceback.format_exc())

    all_rows = dedup_rows(all_rows)
    pd.DataFrame(all_rows).to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    results_df = pd.DataFrame(all_rows)
    summary_df = (
        results_df.groupby(["game_instruction", "opponent_policy"], as_index=False)
        .agg(
            avg_agent_round_payoff=("agent_payoff", "mean"),
            agent_coop_rate=("agent_action", lambda s: (s == "Cooperate").mean()),
            parse_success_rate=("valid_parse", "mean"),
        )
    )
    print(summary_df)


if __name__ == "__main__":
    main()
