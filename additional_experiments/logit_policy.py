"""Per-decision cooperation probabilities from token log-probabilities (persona-format PD experiment).

Instead of estimating cooperation from sampled actions, we read the model's probability of choosing
Cooperate vs Defect at the action position of its JSON answer. The prompt is byte-identical to the paper's
(persona_two_agent.build_two_agent_prompt / persona_single_agent.build_prompt): we do NOT ask a Yes/No
question, because relabelling the action is itself a framing perturbation (Option A/B shifted GPT-5.2
cooperation by +0.36 in block 2).

Reading the decision: the answer starts {"action": "<value>", ...}. The decision is made at the first token
whose text contains the first letter of <value>. "Cooperate" may span several tokens, but its first token
already differs from Defect's, so P(Cooperate) = sum of top-k alternatives at that position that are prefixes
of "Cooperate" (same for Defect). We store both class log-probs (logp_C, logp_D) so the analysis can also
apply the sampling temperature: p_T(C) = sigmoid((logp_C - logp_D) / T).

Modes
  fixed   : round 1 (no history) and fixed one-round histories CC / DD / CD / DC, --reps calls each
  replay  : every decision along the paper's own trajectories (data/results/PD/persona/<model>/...),
            i.e. p(Cooperate | the exact history the agent saw). Averaging p over trajectories estimates the
            paper's cooperation rate without the extra noise of one sampled action per decision.

Run (from the repository root; credentials from gpt5.env as in single_perturbation.py):
  python additional_experiments/logit_policy.py run --model gpt-5.2 --mode fixed --reps 2 --dry-run
  python additional_experiments/logit_policy.py run --model gpt-5.2 --mode fixed
  python additional_experiments/logit_policy.py run --model gpt-5.2 --mode replay --regime two
  python additional_experiments/logit_policy.py run --model deepseek-v3.2 --deployment <name> --mode fixed
  python additional_experiments/logit_policy.py analyze --models gpt-5.2 deepseek-v3.2

Claude's API does not return log-probabilities; the script stops early for any deployment that returns none.
Outputs: additional_experiments/results/PD_logit_policy/<model>/<mode>_<regime>.csv (one row per call,
prompt and raw output kept), and summary_*.csv from `analyze`.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "PD_logit_policy"
VALUE_START = re.compile(r'"action"\s*:\s*"\s*[A-Za-z]')
COLUMNS = ["key", "model", "persona", "history", "role", "rep", "opponent_policy", "seed", "round", "sampled_action",
           "status", "p_coop_raw", "logp_C", "logp_D", "covered_mass", "decision_token", "system_message",
           "raw_output", "prompt"]
FIXED_HISTORIES = {"none": [], "CC": [("Cooperate", "Cooperate")], "DD": [("Defect", "Defect")],
                   "CD": [("Cooperate", "Defect")], "DC": [("Defect", "Cooperate")]}  # (self, other)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sp = _load("single_perturbation", HERE / "single_perturbation.py")
NAMES = sp.NAMES


# =============================================================================
# Prompts (paper code, unchanged)
# =============================================================================
def two_agent_prompt(two, persona: str, role: str, pairs: list[tuple[str, str]], horizon: int) -> str:
    """pairs = [(self_action, other_action), ...] from this agent's point of view."""
    history = []
    for r, (me, other) in enumerate(pairs, start=1):
        a, b = (me, other) if role == "A" else (other, me)
        pa, pb = two.PAYOFF[(a, b)]
        history.append({"round": r, "a_action": a, "b_action": b, "a_payoff": pa, "b_payoff": pb})
    return two.build_two_agent_prompt(persona_text=persona, role=role, history=history, horizon=horizon)


def single_agent_prompt(one, persona: str, pairs: list[tuple[str, str]], horizon: int) -> str:
    history = [{"round": r, "you": me, "opp": opp, "you_payoff": one.PAYOFF[(me, opp)][0],
                "opp_payoff": one.PAYOFF[(me, opp)][1]} for r, (me, opp) in enumerate(pairs, start=1)]
    return one.build_prompt(persona_text=persona, history=history, horizon=horizon)


# =============================================================================
# Log-probability extraction
# =============================================================================
def _class_of(token: str) -> str | None:
    core = re.sub(r"^[^A-Za-z]+", "", token).lower()
    if not core:
        return None
    if "cooperate".startswith(core) or core.startswith("cooperate"):
        return "C"
    if "defect".startswith(core) or core.startswith("defect"):
        return "D"
    return None


def decision_logprobs(content: list[Any]) -> tuple[float, float, float, str]:
    """Return (logp_C, logp_D, covered_mass, chosen_token) at the action-value position."""
    text = ""
    for tok in content:
        if VALUE_START.search(text + tok.token):
            mass = {"C": 0.0, "D": 0.0}
            for alt in tok.top_logprobs:
                cls = _class_of(alt.token)
                if cls:
                    mass[cls] += math.exp(alt.logprob)
            lc = math.log(mass["C"]) if mass["C"] > 0 else -math.inf
            ld = math.log(mass["D"]) if mass["D"] > 0 else -math.inf
            return lc, ld, mass["C"] + mass["D"], tok.token
        text += tok.token
    raise ValueError("action value not found in the returned tokens")


def p_coop(lc: float, ld: float, temperature: float = 1.0) -> float:
    if lc == -math.inf and ld == -math.inf:
        return math.nan
    if ld == -math.inf:
        return 1.0
    if lc == -math.inf:
        return 0.0
    return 1.0 / (1.0 + math.exp(-(lc - ld) / temperature))


def query(client, model: str, prompt: str, system: str | None, temperature: float, top_p: float) -> dict[str, Any]:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    kw = dict(model=model, messages=messages, temperature=temperature, logprobs=True, top_logprobs=5)
    if model.lower() != "claude-haiku-4.5":
        kw["top_p"] = top_p
    resp = client.chat.completions.create(**kw)
    choice = resp.choices[0]
    raw = choice.message.content or ""
    if choice.logprobs is None or not choice.logprobs.content:
        return {"status": "no_logprobs", "raw_output": raw}
    try:
        lc, ld, mass, tok = decision_logprobs(choice.logprobs.content)
    except ValueError as e:
        return {"status": f"parse_error: {e}", "raw_output": raw}
    return {"status": "ok", "logp_C": lc, "logp_D": ld, "covered_mass": mass, "decision_token": tok,
            "p_coop_raw": p_coop(lc, ld), "raw_output": raw}


# =============================================================================
# Tasks
# =============================================================================
def build_tasks(args, two, one) -> list[dict[str, Any]]:
    personas = dict(two.PAPER_PERSONAS)
    tasks = []
    if args.mode == "fixed":
        for p, text in personas.items():
            for h, pairs in FIXED_HISTORIES.items():
                for rep in range(args.reps):
                    if args.regime == "two":
                        for role in ("A", "B"):
                            tasks.append(dict(key=f"{p}|{h}|{role}|{rep}", persona=p, history=h, role=role, rep=rep,
                                              prompt=two_agent_prompt(two, text, role, pairs, args.horizon)))
                    else:
                        tasks.append(dict(key=f"{p}|{h}|single|{rep}", persona=p, history=h, role="single", rep=rep,
                                          prompt=single_agent_prompt(one, text, pairs, args.horizon)))
        return tasks
    # replay the paper's trajectories
    base = sp.BUNDLED / args.model
    for p, text in personas.items():
        if args.regime == "two":
            for f in sorted((base / "two_agent" / p).glob("seed*/rounds.csv")):
                d = pd.read_csv(f).sort_values("round")
                acts = list(zip(d["agent_a_action"], d["agent_b_action"]))
                for t in range(len(acts)):
                    for role, (i, j) in (("A", (0, 1)), ("B", (1, 0))):
                        pairs = [(x[i], x[j]) for x in acts[:t]]
                        tasks.append(dict(key=f"{p}|{d['seed'].iloc[0]}|{t + 1}|{role}", persona=p,
                                          seed=int(d["seed"].iloc[0]), round=t + 1, role=role,
                                          sampled_action=acts[t][i],
                                          prompt=two_agent_prompt(two, text, role, pairs, args.horizon)))
        else:
            for f in sorted((base / "single_agent" / p).glob("*/seed*/rounds.csv")):
                d = pd.read_csv(f).sort_values("round")
                acts = list(zip(d["agent_action"], d["opponent_action"]))
                opp = d["opponent_policy"].iloc[0]
                for t in range(len(acts)):
                    tasks.append(dict(key=f"{p}|{opp}|{d['seed'].iloc[0]}|{t + 1}", persona=p, opponent_policy=opp,
                                      seed=int(d["seed"].iloc[0]), round=t + 1, role="single",
                                      sampled_action=acts[t][0],
                                      prompt=single_agent_prompt(one, text, acts[:t], args.horizon)))
    return tasks


def cmd_run(args) -> None:
    two, one = sp._load_pd_module("persona_two_agent"), sp._load_pd_module("persona_single_agent")
    out_path = OUT / args.model / f"{args.mode}_{args.regime}.csv"
    tasks = build_tasks(args, two, one)
    done = set(pd.read_csv(out_path)["key"]) if out_path.exists() else set()
    todo = [t for t in tasks if t["key"] not in done]
    print(f"model={args.model} mode={args.mode} regime={args.regime} calls={len(tasks)} remaining={len(todo)}")
    if args.dry_run or not todo:
        return

    endpoint, api_key, deployment = sp.load_credentials(args.env_file, args.deployment)
    azure = os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY")
    client = two.make_client(endpoint, api_key) if azure else sp.make_openai_client(endpoint, api_key)
    system = None if args.no_system else "You are a helpful assistant."  # same as single_perturbation.call_chat_model

    probe = query(client, deployment, todo[0]["prompt"], system, args.temperature, args.top_p)
    if probe["status"] == "no_logprobs":
        raise SystemExit(f"{deployment}: the API returned no log-probabilities; this model cannot be used.")
    print(f"Probe OK: status={probe['status']} p_coop_raw={probe.get('p_coop_raw')} token={probe.get('decision_token')!r}")

    def run_one(t):
        return {**{k: v for k, v in t.items()}, "model": args.model, "system_message": system or "",
                **query(client, deployment, t["prompt"], system, args.temperature, args.top_p)}

    rows, failed = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, t) for t in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                rows.append(fut.result())
            except Exception as e:
                failed += 1
                print(f"FAILED: {e}\n{traceback.format_exc()}")
            if len(rows) >= 200 or i == len(futures):  # flush in chunks so the run can resume
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
                rows = []
                print(f"Progress: {i}/{len(futures)} (failed={failed})")
    print(f"Done. Saved to {out_path}. Rerun the same command to resume.")


# =============================================================================
# Analysis
# =============================================================================
def boot_gap(x: pd.Series, y: pd.Series, rng, n_boot: int) -> tuple[float, float, float]:
    """Mean of x - y paired by seed, percentile bootstrap over seeds."""
    d = (x - y).dropna().to_numpy()
    if len(d) == 0:
        return (np.nan,) * 3
    bs = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return d.mean(), *np.percentile(bs, [2.5, 97.5])


def cmd_analyze(args) -> None:
    rng = np.random.default_rng(args.boot_seed)
    fixed_rows, replay_rows = [], []
    for model in args.models:
        for regime in ("two", "single"):
            f = OUT / model / f"fixed_{regime}.csv"
            if f.exists():
                d = pd.read_csv(f).query("status == 'ok'")
                d["p_T"] = [p_coop(c, dd, args.temperature) for c, dd in zip(d.logp_C, d.logp_D)]
                g = d.groupby(["persona", "history"]).agg(p_raw=("p_coop_raw", "mean"), p_T=("p_T", "mean"),
                                                          sd_raw=("p_coop_raw", "std"), mass=("covered_mass", "mean"),
                                                          n=("p_coop_raw", "size")).reset_index()
                g.insert(0, "regime", regime)
                g.insert(0, "model", model)
                g["persona"] = g["persona"].map(NAMES)
                fixed_rows.append(g)
            f = OUT / model / f"replay_{regime}.csv"
            if f.exists():
                d = pd.read_csv(f).query("status == 'ok'")
                d["p_T"] = [p_coop(c, dd, args.temperature) for c, dd in zip(d.logp_C, d.logp_D)]
                d["sampled"] = (d["sampled_action"] == "Cooperate").astype(float)
                per_seed = d.groupby(["persona", "seed"])[["p_coop_raw", "p_T", "sampled"]].mean().unstack(0)
                for c0, c1 in (("P4_PLAIN_STRATEGIC", "P5_DESCRIPTIVE_STRATEGIC"),
                               ("P4_PLAIN_STRATEGIC", "P6_TABULAR_STRATEGIC"),
                               ("P5_DESCRIPTIVE_STRATEGIC", "P6_TABULAR_STRATEGIC")):
                    for est in ("sampled", "p_coop_raw", "p_T"):
                        if (est, c0) in per_seed and (est, c1) in per_seed:
                            e, lo, hi = boot_gap(per_seed[(est, c1)], per_seed[(est, c0)], rng, args.n_boot)
                            replay_rows.append(dict(model=model, regime=regime, contrast=f"{NAMES[c1]}-{NAMES[c0]}",
                                                    estimator=est, estimate=e, ci_low=lo, ci_high=hi,
                                                    ci_width=hi - lo))
    pd.set_option("display.width", 250)
    if fixed_rows:
        fx = pd.concat(fixed_rows)
        fx.to_csv(OUT / "summary_fixed.csv", index=False)
        print("p(Cooperate) after fixed histories (none = round 1)\n", fx.round(3).to_string(index=False))
    if replay_rows:
        rp = pd.DataFrame(replay_rows)
        rp.to_csv(OUT / "summary_replay.csv", index=False)
        print("\nFormat gaps on the paper's trajectories: sampled actions vs log-prob estimates\n",
              rp.round(3).to_string(index=False))


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, help="label matching data/results/PD/persona/<model>")
    r.add_argument("--deployment")
    r.add_argument("--env-file")
    r.add_argument("--mode", choices=["fixed", "replay"], default="fixed")
    r.add_argument("--regime", choices=["two", "single"], default="two")
    r.add_argument("--reps", type=int, default=5, help="fixed mode: calls per (persona, history, role)")
    r.add_argument("--horizon", type=int, default=10)
    r.add_argument("--temperature", type=float, default=0.3)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--no-system", action="store_true", help="omit the 'You are a helpful assistant.' system message")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--dry-run", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--temperature", type=float, default=0.3, help="for p_T = sigmoid((logp_C - logp_D) / T)")
    a.add_argument("--n-boot", type=int, default=10_000)
    a.add_argument("--boot-seed", type=int, default=20260925)
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_analyze(args)


if __name__ == "__main__":
    main()
