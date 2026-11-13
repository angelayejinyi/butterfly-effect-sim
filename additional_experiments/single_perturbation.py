"""Perturb only ONE agent in the two-agent repeated Prisoner's Dilemma.

The two agents use the same model but different persona formats (e.g. PLAIN vs TABULAR), so only one
side carries the perturbation. Prompt template, persona texts, payoffs, horizon, sampling parameters,
parser and retries are imported unchanged from code/PD/persona_two_agent.py and persona_single_agent.py.
Credentials are read from gpt5.env.

Run (from the repository root):
  python additional_experiments/single_perturbation.py run --model gpt-5.2 --dry-run
  python additional_experiments/single_perturbation.py run --model gpt-5.2 --seed-count 2 --horizon 2   # smoke test
  python additional_experiments/single_perturbation.py run --model gpt-5.2
  # DeepSeek V4 Flash also needs the paper's original conditions:
  python additional_experiments/single_perturbation.py run --model deepseek-v4-flash --deployment <name> --mode single
  python additional_experiments/single_perturbation.py run --model deepseek-v4-flash --deployment <name> --mode homogeneous
  python additional_experiments/single_perturbation.py run --model deepseek-v4-flash --deployment <name>

Analyze:
  python additional_experiments/single_perturbation.py analyze --models gpt-5.2 claude-haiku-4-5-20251001 \
      gemini-2.5-flash deepseek-v4-flash

Metrics per model and contrast c0 -> c1 (c0 = baseline persona, c1 = perturbed persona):
  direct     coop(c1 agent with a c0 partner) - coop(agent in a c0/c0 dyad)
  spillover  coop(c0 agent with a c1 partner) - coop(agent in a c0/c0 dyad)   <- the unperturbed partner
  I          Y(c1,c1) - Y(c1,c0) - Y(c0,c1) + Y(c0,c0), dyad-mean cooperation (non-additivity of two doses)
  S_fixed, S_LLM, A   as in the paper (reproduced and checked against amplification_old_vs_new.csv)
  A_single   |direct| - S_fixed   (one perturbed agent in both regimes)
CIs: paired percentile bootstrap over seeds, 10,000 resamples, seed 20260925 (as in the paper).

All runs and analysis outputs go to additional_experiments/results/.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"
BUNDLED = REPO / "data" / "results" / "PD" / "persona"
PAPER_AMP = REPO / "data" / "plots_and_tables" / "amplification_old_vs_new.csv"

SHORT = {"PLAIN": "P4_PLAIN_STRATEGIC", "DESCRIPTIVE": "P5_DESCRIPTIVE_STRATEGIC", "TABULAR": "P6_TABULAR_STRATEGIC"}
NAMES = {"P4_PLAIN_STRATEGIC": "Plain", "P5_DESCRIPTIVE_STRATEGIC": "Descriptive", "P6_TABULAR_STRATEGIC": "Tabular"}
OPPONENTS = ["TitForTat", "Random", "AlwaysCooperate", "AlwaysDefect"]
MODE_DIRS = {"mixed": "two_agent_mixed", "homogeneous": "two_agent", "single": "single_agent"}


def _load_pd_module(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "code" / "PD" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Running
# =============================================================================
def load_credentials(env_file: str | None, deployment: str | None) -> tuple[str, str, str]:
    from dotenv import load_dotenv

    candidates = [Path(env_file)] if env_file else [REPO / "gpt5.env", HERE / "gpt5.env", Path.cwd() / "gpt5.env"]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(f"gpt5.env not found; looked in {[str(p) for p in candidates]} (or pass --env-file)")
    load_dotenv(dotenv_path=str(found), override=True)
    print(f"Loaded credentials from: {found}")
    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
        endpoint, api_key = os.environ["AZURE_OPENAI_ENDPOINT"], os.environ["AZURE_OPENAI_API_KEY"]
    elif os.getenv("OPENAI_ENDPOINT") and os.getenv("OPENAI_API_KEY"):
        endpoint, api_key = os.environ["OPENAI_ENDPOINT"], os.environ["OPENAI_API_KEY"]
    else:
        raise EnvironmentError(f"Missing endpoint/key pair in {found}: set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY or OPENAI_ENDPOINT/OPENAI_API_KEY")
    model = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not model:
        raise EnvironmentError("Pass --deployment or set AZURE_OPENAI_DEPLOYMENT_NAME")
    return endpoint, api_key, model


def make_openai_client(endpoint: str, api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=endpoint, api_key=api_key)


def call_chat_model(client, model: str, prompt: str, temperature: float, top_p: float) -> str:
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    sampling = {"temperature": temperature}
    if model.lower() != "claude-haiku-4.5":
        sampling["top_p"] = top_p
    for attempt in range(1, 6):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                **sampling,
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError, APITimeoutError):
            if attempt == 5:
                raise
            delay = 2 ** (attempt - 1) + random.uniform(0.0, 0.5)
            print(f"Transient API error on attempt {attempt}/5; retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("Model call loop ended unexpectedly")


def run_mixed_episode(two, personas: dict[str, str], task: dict[str, Any]) -> list[dict[str, Any]]:
    """Same loop as persona_two_agent.run_episode, but agent A and agent B get different persona texts."""
    client = two.make_client(task["endpoint"], task["api_key"])
    rng = random.Random(task["seed"])
    text_a, text_b = personas[task["persona_a"]], personas[task["persona_b"]]
    horizon = task["horizon"]
    history: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for round_idx in range(1, horizon + 1):
        if rng.random() < 0.5:  # same construction-order draw as the original script
            prompt_a = two.build_two_agent_prompt(persona_text=text_a, role="A", history=history, horizon=horizon)
            prompt_b = two.build_two_agent_prompt(persona_text=text_b, role="B", history=history, horizon=horizon)
        else:
            prompt_b = two.build_two_agent_prompt(persona_text=text_b, role="B", history=history, horizon=horizon)
            prompt_a = two.build_two_agent_prompt(persona_text=text_a, role="A", history=history, horizon=horizon)
        kw = dict(client=client, model=task["model"], temperature=task["temperature"], top_p=task["top_p"], max_retries=1)
        a_action, a_valid, a_note, a_raw = two.choose_action_with_retry(prompt=prompt_a, **kw)
        b_action, b_valid, b_note, b_raw = two.choose_action_with_retry(prompt=prompt_b, **kw)
        a_payoff, b_payoff = two.PAYOFF[(a_action, b_action)]
        history.append({"round": round_idx, "a_action": a_action, "b_action": b_action,
                        "a_payoff": a_payoff, "b_payoff": b_payoff})
        rows.append({
            "persona_format": f"{task['persona_a']}__{task['persona_b']}",
            "persona_a": task["persona_a"], "persona_b": task["persona_b"],
            "seed": task["seed"], "round": round_idx,
            "agent_a_action": a_action, "agent_b_action": b_action,
            "agent_a_payoff": a_payoff, "agent_b_payoff": b_payoff,
            "valid_parse_a": a_valid, "valid_parse_b": b_valid,
            "parse_note_a": a_note, "parse_note_b": b_note,
            "agent_a_raw_output": a_raw, "agent_b_raw_output": b_raw,
            "agent_a_prompt_input": prompt_a, "agent_b_prompt_input": prompt_b,
        })
    return rows


def episode_path(model: str, task: dict[str, Any]) -> Path:
    base = RESULTS / model / MODE_DIRS[task["mode"]]
    if task["mode"] == "mixed":
        return base / f"A-{task['persona_a']}__B-{task['persona_b']}" / f"seed{task['seed']}" / "rounds.csv"
    if task["mode"] == "single":
        return base / task["persona_a"] / task["policy"] / f"seed{task['seed']}" / "rounds.csv"
    return base / task["persona_a"] / f"seed{task['seed']}" / "rounds.csv"


def build_tasks(args, personas: list[str], policies: list[str]) -> list[dict[str, Any]]:
    common = dict(mode=args.mode, horizon=args.horizon, temperature=args.temperature, top_p=args.top_p)
    tasks = []
    for seed in range(args.seed_count):
        if args.mode == "mixed":
            for p, q in combinations(personas, 2):
                # counterbalance slot: even seeds put the first persona in slot A, odd seeds in slot B
                orders = [(p, q), (q, p)] if args.both_orders else ([(p, q)] if seed % 2 == 0 else [(q, p)])
                tasks += [{**common, "persona_a": a, "persona_b": b, "seed": seed} for a, b in orders]
        elif args.mode == "homogeneous":
            tasks += [{**common, "persona_a": p, "persona_b": p, "seed": seed} for p in personas]
        else:
            tasks += [{**common, "persona_a": p, "policy": o, "seed": seed} for p in personas for o in policies]
    return tasks


def is_complete(path: Path, horizon: int) -> bool:
    try:
        return path.exists() and len(pd.read_csv(path)) >= horizon
    except Exception:
        return False


def cmd_run(args) -> None:
    two = _load_pd_module("persona_two_agent")
    single = _load_pd_module("persona_single_agent")
    personas_all = dict(two.PAPER_PERSONAS)
    if personas_all != dict(single.PAPER_PERSONAS):
        sys.exit("PAPER_PERSONAS differ between persona_two_agent.py and persona_single_agent.py")

    # Each agent's prompt must equal the homogeneous prompt except for the persona block.
    template = two.build_two_agent_prompt(persona_text="<PERSONA>", role="A", history=[], horizon=args.horizon)
    for text in personas_all.values():
        prompt = two.build_two_agent_prompt(persona_text=text, role="A", history=[], horizon=args.horizon)
        assert prompt.replace(text, "<PERSONA>") == template

    personas = [SHORT[p] for p in args.personas]
    tasks = build_tasks(args, personas, single.OPPONENT_POLICIES)
    todo = [t for t in tasks if not is_complete(episode_path(args.model, t), args.horizon)]
    calls = len(todo) * args.horizon * (1 if args.mode == "single" else 2)
    print(f"mode={args.mode} model={args.model} episodes={len(tasks)} remaining={len(todo)} API calls~{calls}")
    if args.dry_run or not todo:
        return

    endpoint, api_key, deployment = load_credentials(args.env_file, args.deployment)
    if not (os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY")):
        for module in (two, single):
            module.make_client = make_openai_client
            module.call_model = call_chat_model
    print(f"Deployment: {deployment}")

    def run_one(t: dict[str, Any]) -> list[dict[str, Any]]:
        t = {**t, "endpoint": endpoint, "api_key": api_key, "model": deployment}
        if t["mode"] == "mixed":
            rows = run_mixed_episode(two, personas_all, t)
        elif t["mode"] == "homogeneous":
            rows = two.run_episode((t["persona_a"], personas_all[t["persona_a"]], t["seed"], t["horizon"],
                                    t["temperature"], t["top_p"], endpoint, api_key, deployment))
        else:
            rows = single.run_episode((t["persona_a"], personas_all[t["persona_a"]], t["policy"], t["seed"],
                                       t["horizon"], t["temperature"], t["top_p"], endpoint, api_key, deployment))
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
def _read_all(model: str, mode_dir: str) -> pd.DataFrame:
    root = RESULTS / model / mode_dir  # new runs take precedence over bundled paper data
    if not root.exists():
        root = BUNDLED / model / mode_dir
    files = sorted(root.glob("**/seed*/rounds.csv"))
    if not files:
        raise FileNotFoundError(f"No rounds.csv under {root}")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def load_two_agent_long(model: str, mixed: bool) -> pd.DataFrame:
    """One row per agent x seed x round."""
    df = _read_all(model, "two_agent_mixed" if mixed else "two_agent")
    if not mixed:
        df["persona_a"] = df["persona_b"] = df["persona_format"]
    parts = [pd.DataFrame({"seed": df["seed"], "round": df["round"],
                           "self": df[f"persona_{me}"], "partner": df[f"persona_{other}"],
                           "coop": (df[f"agent_{me}_action"] == "Cooperate").astype(float),
                           "valid": df[f"valid_parse_{me}"].astype(bool)})
             for me, other in (("a", "b"), ("b", "a"))]
    return pd.concat(parts, ignore_index=True)


def contrast_arrays(two_h, two_m, single, c0: str, c1: str, by_round: bool) -> dict[str, np.ndarray]:
    seeds = np.array(sorted(two_h["seed"].unique()))

    def per_seed(d: pd.DataFrame) -> np.ndarray:
        if by_round:
            return d.groupby(["seed", "round"])["coop"].mean().unstack("round").reindex(seeds).to_numpy()
        return d.groupby("seed")["coop"].mean().reindex(seeds).to_numpy()[:, None]

    out = {
        "hom0": per_seed(two_h[two_h.self == c0]),
        "hom1": per_seed(two_h[two_h.self == c1]),
        "focal": per_seed(two_m[(two_m.self == c1) & (two_m.partner == c0)]),
        "partner": per_seed(two_m[(two_m.self == c0) & (two_m.partner == c1)]),
    }
    for o in OPPONENTS:
        out[f"fix0_{o}"] = per_seed(single[(single.persona_format == c0) & (single.opponent_policy == o)])
        out[f"fix1_{o}"] = per_seed(single[(single.persona_format == c1) & (single.opponent_policy == o)])
    return out


def metrics(a: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict[str, np.ndarray]:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = {k: np.nanmean(v if idx is None else v[idx], axis=0) for k, v in a.items()}
    deltas = {o: m[f"fix1_{o}"] - m[f"fix0_{o}"] for o in OPPONENTS}
    s_fixed = np.mean([np.abs(d) for d in deltas.values()], axis=0)
    direct = m["focal"] - m["hom0"]
    res = {
        "direct": direct,
        "spillover": m["partner"] - m["hom0"],
        "I": m["hom1"] - (m["focal"] + m["partner"]) + m["hom0"],  # = Y11 - 2*Ymix + Y00 on dyad means
        "S_fixed": s_fixed,
        "S_fixed_TfT_Random": np.mean([np.abs(deltas[o]) for o in ("TitForTat", "Random")], axis=0),
        "S_LLM": np.abs(m["hom1"] - m["hom0"]),
        "A": np.abs(m["hom1"] - m["hom0"]) - s_fixed,
        "A_single": np.abs(direct) - s_fixed,
    }
    res.update({f"delta_fixed_{o}": d for o, d in deltas.items()})
    return res


def bootstrap(a: dict[str, np.ndarray], n_boot: int, seed: int) -> dict[str, tuple]:
    point = metrics(a)
    n = next(iter(a.values())).shape[0]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        for k, v in metrics(a, rng.integers(0, n, n)).items():
            draws[k].append(v)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN metrics when mixed runs are absent
        return {k: (point[k], *np.nanpercentile(np.array(draws[k]), [2.5, 97.5], axis=0)) for k in point}


def cmd_analyze(args) -> None:
    paper = pd.read_csv(PAPER_AMP).query("dimension == 'persona'")
    summary, roundwise = [], []
    for model in args.models:
        two_h = load_two_agent_long(model, mixed=False)
        single = _read_all(model, "single_agent")
        single["coop"] = (single["agent_action"] == "Cooperate").astype(float)
        try:
            two_m = load_two_agent_long(model, mixed=True)
            print(f"[{model}] mixed-run parse success rate: {two_m['valid'].mean():.3f}")
        except FileNotFoundError:
            print(f"[{model}] no mixed runs yet; only the paper reproduction check is computed")
            two_m = two_h.iloc[0:0]

        for c0, c1 in permutations(NAMES, 2):
            label = f"{NAMES[c0]} -> {NAMES[c1]}"
            res = bootstrap(contrast_arrays(two_h, two_m, single, c0, c1, False), args.n_boot, args.boot_seed)
            summary += [{"model": model, "contrast": label, "metric": k, "estimate": e[0], "ci_low": lo[0], "ci_high": hi[0]}
                        for k, (e, lo, hi) in res.items()]

            ref = paper[(paper.model == model) & (paper.contrast == label)]
            for ours, theirs in (("S_fixed", "S_fixed"), ("S_LLM", "S_LLM"), ("A", "A_new")):
                if len(ref) and abs(res[ours][0][0] - ref[theirs].iloc[0]) > 1e-6:
                    print(f"  WARNING {label} {ours}: {res[ours][0][0]:.4f} != paper {ref[theirs].iloc[0]:.4f}")
            if len(ref):
                print(f"  {label}: S_fixed/S_LLM/A match the paper")

            if len(two_m):
                res_r = bootstrap(contrast_arrays(two_h, two_m, single, c0, c1, True), min(args.n_boot, 2000), args.boot_seed)
                for k in ("direct", "spillover", "I"):
                    e, lo, hi = res_r[k]
                    roundwise += [{"model": model, "contrast": label, "metric": k, "round": t + 1,
                                   "estimate": e[t], "ci_low": lo[t], "ci_high": hi[t]} for t in range(len(e))]

    RESULTS.mkdir(parents=True, exist_ok=True)
    s = pd.DataFrame(summary)
    s.to_csv(RESULTS / "summary.csv", index=False)
    if roundwise:
        pd.DataFrame(roundwise).to_csv(RESULTS / "roundwise.csv", index=False)
    key = ["direct", "spillover", "I", "S_fixed", "S_LLM", "A", "A_single"]
    view = s[s.metric.isin(key)].copy()
    view["value"] = [f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]" for e, lo, hi in zip(view.estimate, view.ci_low, view.ci_high)]
    print(view.pivot_table(index=["model", "contrast"], columns="metric", values="value", aggfunc="first")
          .reindex(columns=key).to_string())
    print(f"\nSaved {RESULTS / 'summary.csv'}" + (f" and {RESULTS / 'roundwise.csv'}" if roundwise else ""))


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run simulations")
    r.add_argument("--model", required=True, help="label used in results/, e.g. gpt-5.2, deepseek-v4-flash")
    r.add_argument("--deployment", help="deployment name (default: AZURE_OPENAI_DEPLOYMENT_NAME in gpt5.env)")
    r.add_argument("--env-file", help="default: gpt5.env")
    r.add_argument("--mode", choices=list(MODE_DIRS), default="mixed")
    r.add_argument("--personas", nargs="+", choices=list(SHORT), default=list(SHORT))
    r.add_argument("--both-orders", action="store_true", help="mixed: run both slot orders for every seed")
    r.add_argument("--seed-count", type=int, default=30)
    r.add_argument("--horizon", type=int, default=10)
    r.add_argument("--temperature", type=float, default=0.3)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--dry-run", action="store_true")

    a = sub.add_parser("analyze", help="compute metrics")
    a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--n-boot", type=int, default=10_000)
    a.add_argument("--boot-seed", type=int, default=20260925)

    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_analyze(args)


if __name__ == "__main__":
    main()
