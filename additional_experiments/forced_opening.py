"""Forced-opening test of the opening-move mechanism in the two-agent repeated Prisoner's Dilemma.

Hypothesis: persona format mainly shifts the round-1 move, and reciprocity then locks the dyad into
mutual cooperation or mutual defection. If so, fixing round 1 should remove most of the format gap
in rounds 2-10.

Design: both agents use the same persona format (as in the paper's homogeneous LLM-LLM condition).
Round 1 is NOT queried: a fixed outcome (CC or DD by default) is written into the shared history and
the models play rounds 2..horizon. Prompt template, persona texts, payoffs, horizon, sampling parameters,
parser and retries are imported unchanged from code/PD/persona_two_agent.py; credentials and the
OpenAI-compatible client are reused from single_perturbation.py (gpt5.env).

Run (from the repository root):
  python additional_experiments/forced_opening.py run --model gpt-5.2 --dry-run
  python additional_experiments/forced_opening.py run --model gpt-5.2 --seed-count 2 --horizon 3   # smoke test
  python additional_experiments/forced_opening.py run --model gpt-5.2
  python additional_experiments/forced_opening.py run --model claude-haiku-4-5-20251001 --deployment claude-haiku-4.5

Analyze (compares with the paper's free-play runs in data/results/PD/persona/<model>/two_agent):
  python additional_experiments/forced_opening.py analyze --models gpt-5.2 claude-haiku-4-5-20251001

Outputs: additional_experiments/results/PD_reason_analysis/forced_opening/
  <model>/<opening>/<persona>/seed<k>/rounds.csv     one file per episode (round 1 rows have forced=True)
  summary.csv                                        cooperation and format gaps per condition, with CIs
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "PD_reason_analysis" / "forced_opening"
OPENINGS = {"CC": ("Cooperate", "Cooperate"), "DD": ("Defect", "Defect"),
            "CD": ("Cooperate", "Defect"), "DC": ("Defect", "Cooperate")}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sp = _load("single_perturbation", HERE / "single_perturbation.py")  # credentials, client, paths, names
NAMES = sp.NAMES  # P4_PLAIN_STRATEGIC -> Plain, ...


# =============================================================================
# Running
# =============================================================================
def run_forced_episode(two, persona_text: str, task: dict[str, Any]) -> list[dict[str, Any]]:
    """persona_two_agent.run_episode with round 1 replaced by a fixed outcome."""
    client = two.make_client(task["endpoint"], task["api_key"])
    rng = random.Random(task["seed"])
    horizon = task["horizon"]
    a_open, b_open = OPENINGS[task["opening"]]
    history: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for round_idx in range(1, horizon + 1):
        rng.random()  # the original script draws a prompt-construction order each round; prompts do not depend on it
        if round_idx == 1:
            a_action, b_action = a_open, b_open
            a_valid = b_valid = True
            a_note = b_note = "forced_opening"
            a_raw = b_raw = prompt_a = prompt_b = ""
        else:
            prompt_a = two.build_two_agent_prompt(persona_text=persona_text, role="A", history=history, horizon=horizon)
            prompt_b = two.build_two_agent_prompt(persona_text=persona_text, role="B", history=history, horizon=horizon)
            kw = dict(client=client, model=task["model"], temperature=task["temperature"], top_p=task["top_p"], max_retries=1)
            a_action, a_valid, a_note, a_raw = two.choose_action_with_retry(prompt=prompt_a, **kw)
            b_action, b_valid, b_note, b_raw = two.choose_action_with_retry(prompt=prompt_b, **kw)
        a_payoff, b_payoff = two.PAYOFF[(a_action, b_action)]
        history.append({"round": round_idx, "a_action": a_action, "b_action": b_action,
                        "a_payoff": a_payoff, "b_payoff": b_payoff})
        rows.append({
            "persona_format": task["persona"], "opening": task["opening"], "seed": task["seed"],
            "round": round_idx, "forced": round_idx == 1,
            "agent_a_action": a_action, "agent_b_action": b_action,
            "agent_a_payoff": a_payoff, "agent_b_payoff": b_payoff,
            "valid_parse_a": a_valid, "valid_parse_b": b_valid,
            "parse_note_a": a_note, "parse_note_b": b_note,
            "agent_a_raw_output": a_raw, "agent_b_raw_output": b_raw,
            "agent_a_prompt_input": prompt_a, "agent_b_prompt_input": prompt_b,
        })
    return rows


def episode_path(model: str, t: dict[str, Any]) -> Path:
    return OUT / model / t["opening"] / t["persona"] / f"seed{t['seed']}" / "rounds.csv"


def cmd_run(args) -> None:
    two = sp._load_pd_module("persona_two_agent")
    personas = dict(two.PAPER_PERSONAS)
    tasks = [dict(persona=p, opening=o, seed=s, horizon=args.horizon, temperature=args.temperature, top_p=args.top_p)
             for o in args.openings for p in personas for s in range(args.seed_count)]
    todo = [t for t in tasks if not sp.is_complete(episode_path(args.model, t), args.horizon)]
    print(f"model={args.model} openings={args.openings} episodes={len(tasks)} remaining={len(todo)} "
          f"API calls~{len(todo) * (args.horizon - 1) * 2}")
    if args.dry_run or not todo:
        return

    endpoint, api_key, deployment = sp.load_credentials(args.env_file, args.deployment)
    if not (sp.os.getenv("AZURE_OPENAI_ENDPOINT") and sp.os.getenv("AZURE_OPENAI_API_KEY")):
        two.make_client, two.call_model = sp.make_openai_client, sp.call_chat_model
    print(f"Deployment: {deployment}")

    def run_one(t: dict[str, Any]) -> list[dict[str, Any]]:
        rows = run_forced_episode(two, personas[t["persona"]], {**t, "endpoint": endpoint, "api_key": api_key, "model": deployment})
        for r in rows:
            r["model"] = args.model
        return rows

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, t): t for t in todo}
        for fut in as_completed(futures):
            path = episode_path(args.model, futures[fut])
            try:
                rows = fut.result()
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                pd.DataFrame(rows).to_csv(tmp, index=False)
                tmp.replace(path)
                done += 1
            except Exception as e:
                failed += 1
                print(f"FAILED {path}: {e}\n{traceback.format_exc()}")
            if (done + failed) % 10 == 0 or done + failed == len(todo):
                print(f"Progress: {done + failed}/{len(todo)} (failed={failed})")
    print(f"Done: {done} written, {failed} failed. Rerun the same command to resume.")


# =============================================================================
# Analysis
# =============================================================================
def seed_coop(df: pd.DataFrame) -> pd.Series:
    """Dyad-mean cooperation in rounds >= 2, one value per seed."""
    d = df[df["round"] >= 2]
    coop = ((d["agent_a_action"] == "Cooperate").astype(float) + (d["agent_b_action"] == "Cooperate")) / 2
    return coop.groupby(d["seed"]).mean()


def load_conditions(model: str) -> dict[str, dict[str, pd.Series]]:
    """{condition: {persona: per-seed cooperation}}; 'free' = paper runs, 'free|CC' = paper runs that opened CC."""
    conds: dict[str, dict[str, pd.Series]] = {"free": {}, "free|CC": {}}
    for p in NAMES:
        files = sorted((sp.BUNDLED / model / "two_agent" / p).glob("seed*/rounds.csv"))
        df = pd.concat(pd.read_csv(f) for f in files)
        r1 = df[df["round"] == 1].set_index("seed")
        cc = r1.index[(r1["agent_a_action"] == "Cooperate") & (r1["agent_b_action"] == "Cooperate")]
        conds["free"][p] = seed_coop(df)
        conds["free|CC"][p] = seed_coop(df[df["seed"].isin(cc)])
    for odir in sorted(d for d in (OUT / model).glob("*") if d.is_dir()):
        conds[f"forced_{odir.name}"] = {
            p: seed_coop(pd.concat(pd.read_csv(f) for f in sorted((odir / p).glob("seed*/rounds.csv"))))
            for p in NAMES if (odir / p).exists()}
    return conds


def boot_ci(x: np.ndarray, y: np.ndarray, rng, n_boot: int) -> tuple[float, float, float]:
    """Mean difference x - y with an independent percentile bootstrap (different runs, so no pairing)."""
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan
    d = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n_boot)]
    return x.mean() - y.mean(), *np.percentile(d, [2.5, 97.5])


def cmd_analyze(args) -> None:
    rng = np.random.default_rng(args.boot_seed)
    rows = []
    for model in args.models:
        conds = load_conditions(model)
        for cond, by_p in conds.items():
            for p, v in by_p.items():
                rows.append(dict(model=model, condition=cond, quantity=f"coop[{NAMES[p]}]", estimate=v.mean(),
                                 ci_low=np.nan, ci_high=np.nan, n_seeds=len(v)))
            for c0, c1 in combinations(NAMES, 2):
                if c0 in by_p and c1 in by_p:
                    e, lo, hi = boot_ci(by_p[c1].to_numpy(), by_p[c0].to_numpy(), rng, args.n_boot)
                    rows.append(dict(model=model, condition=cond, quantity=f"gap[{NAMES[c1]}-{NAMES[c0]}]",
                                     estimate=e, ci_low=lo, ci_high=hi, n_seeds=min(len(by_p[c0]), len(by_p[c1]))))
    s = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "summary.csv", index=False)
    s["value"] = [f"{e:+.3f}" if np.isnan(lo) else f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  for e, lo, hi in zip(s.estimate, s.ci_low, s.ci_high)]
    pd.set_option("display.width", 250)
    print("Cooperation in rounds 2-10 (dyad mean) and format gaps; 'free' = paper runs, "
          "'free|CC' = paper runs whose round 1 was mutual cooperation\n")
    print(s.pivot_table(index=["model", "quantity"], columns="condition", values="value", aggfunc="first").to_string())
    print(f"\nSaved {OUT / 'summary.csv'}")


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run simulations")
    r.add_argument("--model", required=True, help="label matching data/results/PD/persona/<model>, e.g. gpt-5.2")
    r.add_argument("--deployment", help="deployment name (default: AZURE_OPENAI_DEPLOYMENT_NAME in gpt5.env)")
    r.add_argument("--env-file", help="default: gpt5.env")
    r.add_argument("--openings", nargs="+", choices=list(OPENINGS), default=["CC", "DD"])
    r.add_argument("--seed-count", type=int, default=30)
    r.add_argument("--horizon", type=int, default=10)
    r.add_argument("--temperature", type=float, default=0.3)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--dry-run", action="store_true")

    a = sub.add_parser("analyze", help="compute cooperation and format gaps")
    a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--n-boot", type=int, default=10_000)
    a.add_argument("--boot-seed", type=int, default=20260925)

    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_analyze(args)


if __name__ == "__main__":
    main()
