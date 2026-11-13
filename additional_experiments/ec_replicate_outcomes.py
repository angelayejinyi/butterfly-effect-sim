"""Calibrate echo-chamber persona effects against run-to-run noise from same-condition replicates.

Replicates: BULLET persona, seed 0, identical configuration, re-run k times
(additional_experiments/results/EC_replicates/<replicate_dir>/run*).
Paper runs: data/results/EC/persona/<paper_dir>/{bullet,prose}/seed*.

Usage (from the repository root):
  python additional_experiments/ec_replicate_outcomes.py gpt-5.2 gpt-5.2
  python additional_experiments/ec_replicate_outcomes.py deepseek-v4-flash DeepSeek-V4-Flash

Metric definitions follow code/plots/plot_main_representation_figure.py and Appendix C.2
(stance groups for the same-group edge ratio are oppose / neutral / support, i.e. the sign of the stance).
"""
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET_ACTIONS = {"REPOST", "REPLY"}
STANCE = {r["agent_id"]: int(r["ai_regulation_stance"])
          for r in json.load(open(ROOT / "code/EC/inputs/ai_regulation_minimal_personas_100.json"))}


def cumulative_edges(a: pd.DataFrame, through_round: int = 15) -> Counter:
    t = a[a["round"].le(through_round) & a["action"].isin(TARGET_ACTIONS) & a["target_author_id"].notna()]
    return Counter(zip(t["agent_id"], t["target_author_id"]))


def weighted_stance_assortativity(edges: Counter) -> float:  # same as plot_main_representation_figure.py
    src = np.array([STANCE[s] for s, _ in edges], float)
    tgt = np.array([STANCE[t] for _, t in edges], float)
    w = np.array(list(edges.values()), float)
    sm, tm = np.average(src, weights=w), np.average(tgt, weights=w)
    cov = np.average((src - sm) * (tgt - tm), weights=w)
    return float(cov / np.sqrt(np.average((src - sm) ** 2, weights=w) * np.average((tgt - tm) ** 2, weights=w)))


def same_group_ratio(edges: Counter) -> float:
    return sum(v for (s, t), v in edges.items() if np.sign(STANCE[s]) == np.sign(STANCE[t])) / sum(edges.values())


def outcomes(run_dir: Path) -> dict:
    a = pd.read_csv(run_dir / "agent_rounds.csv")
    s = pd.read_csv(run_dir / "round_summary.csv")
    e = cumulative_edges(a)
    return {"cross_group_pp": s.loc[s["round"].between(2, 15), "cross_group_interaction_ratio"].mean() * 100,
            "assortativity": weighted_stance_assortativity(e), "same_group_ratio": same_group_ratio(e),
            "targets": a[["round", "agent_id", "target_author_id"]]}


def main(replicate_dir: str, paper_dir: str) -> None:
    reps = [outcomes(d) for d in sorted((ROOT / "additional_experiments/results/EC_replicates" / replicate_dir).glob("run*"))]
    paper = ROOT / "data/results/EC/persona" / paper_dir
    pairs = [(outcomes(paper / "bullet" / f"seed{k}"), outcomes(paper / "prose" / f"seed{k}")) for k in range(30)]
    b0, p0 = pairs[0]
    rows = []
    for m in ["cross_group_pp", "assortativity", "same_group_ratio"]:
        r = np.array([x[m] for x in reps])
        eff = np.mean([p[m] - b[m] for b, p in pairs])
        gaps = np.array([abs(x[m] - y[m]) for x, y in combinations(reps, 2)])
        rows.append({"metric": m, "n_replicates": len(r), "replicate_mean": r.mean(), "replicate_sd": r.std(ddof=1),
                     "replicate_min": r.min(), "replicate_max": r.max(), "max_replicate_pair_gap": gaps.max(),
                     "paper_effect_30seeds": eff, "effect_in_sd": eff / r.std(ddof=1),
                     "bullet_seed0": b0[m], "prose_seed0": p0[m],
                     "n_replicates_beyond_prose_seed0": int((r >= p0[m]).sum() if eff > 0 else (r <= p0[m]).sum())})
    out = pd.DataFrame(rows)
    path = ROOT / "additional_experiments/results/EC_replicates" / f"outcome_noise_baseline_{replicate_dir}.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
