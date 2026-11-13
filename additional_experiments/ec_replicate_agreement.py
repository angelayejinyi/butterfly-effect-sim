"""Run-to-run (same condition, same seed) agreement baseline for the echo-chamber persona experiment."""
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "additional_experiments/results/EC_replicates"
ORIG = ROOT / "data/results/EC/persona"
MODELS = {"gpt-5.2": "gpt-5.2", "claude-haiku-4.5": "claude-haiku-4.5", "gemini-2.5-flash": "gemini-2.5-flash", "deepseek-v4-flash": "DeepSeek-V4-Flash"}
K = ["round", "agent_id"]
def load(p): return pd.read_csv(p / "agent_rounds.csv")[K + ["action", "target_author_id"]]
def agree(a, b):
    m = a.merge(b, on=K, suffixes=("_0", "_1"), validate="one_to_one")
    assert len(m) == len(a) == len(b)
    both = m.target_author_id_0.notna() & m.target_author_id_1.notna()
    return (m.action_0.eq(m.action_1).mean(),
            m.target_author_id_0.eq(m.target_author_id_1).mean(),
            m.loc[both, "target_author_id_0"].eq(m.loc[both, "target_author_id_1"]).mean())
def ci(x):
    x = np.asarray(x); return x.mean(), *(np.percentile(x, [2.5, 97.5]) if len(x) > 1 else (np.nan, np.nan))
rows = []
for rep_name, orig_name in MODELS.items():
    runs = [load(d) for d in sorted((REP / rep_name).glob("run*"), key=lambda d: int(d.name[3:]))]
    rr = [agree(a, b) for a, b in combinations(runs, 2)]
    b0 = load(ORIG / orig_name / "bullet/seed0"); p0 = load(ORIG / orig_name / "prose/seed0")
    ro = [agree(b0, r) for r in runs]
    pert0 = agree(b0, p0)
    pert_all = [agree(load(ORIG / orig_name / "bullet" / f"seed{s}"), load(ORIG / orig_name / "prose" / f"seed{s}")) for s in range(30)]
    for lab, vals in [x for x in [("replicate_vs_replicate", rr), ("replicate_vs_original_bullet_seed0", ro), ("bullet_vs_prose_seed0", [pert0]), ("bullet_vs_prose_30seeds", pert_all)] if x[1]]:
        for i, met in enumerate(["action", "target", "conditional_target"]):
            m, lo, hi = ci([v[i] for v in vals])
            rows.append(dict(replicate_model=rep_name, paper_model=orig_name, comparison=lab, metric=met, n_pairs=len(vals), mean=m, min=min(v[i] for v in vals), max=max(v[i] for v in vals), p2_5=lo, p97_5=hi))
df = pd.DataFrame(rows)
df.to_csv(REP / "agreement_noise_baseline.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False))
