"""Extra metrics on the single-perturbation runs (robustness to opponent set, dyad-level one-dose effect)."""
import importlib.util, warnings
from itertools import permutations
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sp", HERE / "single_perturbation.py"); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
MIX = HERE / "results" / "PD_single_perturbation"; AL = {"claude-haiku-4-5-20251001": "claude-haiku-4.5"}
def ra(m, d):
    root = (MIX / AL.get(m, m) / d) if d == "two_agent_mixed" else sp.BUNDLED / m / d
    return pd.concat(pd.read_csv(f) for f in sorted(root.glob("**/seed*/rounds.csv")))
sp._read_all = ra
def met(a, idx=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = {k: np.nanmean(v if idx is None else v[idx], axis=0)[0] for k, v in a.items()}
    d = {o: m[f"fix1_{o}"] - m[f"fix0_{o}"] for o in sp.OPPONENTS}
    direct, spill = m["focal"] - m["hom0"], m["partner"] - m["hom0"]
    dyad1 = (direct + spill) / 2            # dyad-mean effect, one agent perturbed
    dyad2 = m["hom1"] - m["hom0"]           # dyad-mean effect, both perturbed (signed S_LLM)
    S_all = np.mean([abs(v) for v in d.values()]); S_TR = np.mean([abs(d["TitForTat"]), abs(d["Random"])])
    return {"direct": direct, "spillover": spill, "dyad_one": dyad1, "dyad_two": dyad2,
            "spill_share": spill / direct if abs(direct) > 0.05 else np.nan,
            "A_single": abs(direct) - S_all, "A_single_TfT_Random": abs(direct) - S_TR,
            "A_single_TfT": abs(direct) - abs(d["TitForTat"]),
            "S_fixed": S_all, "delta_TfT": d["TitForTat"]}
rows = []
rng0 = 20260925
for model in ["gpt-5.2", "claude-haiku-4-5-20251001", "gemini-2.5-flash", "deepseek-v4-flash"]:
    h = sp.load_two_agent_long(model, False); x = sp.load_two_agent_long(model, True)
    s = ra(model, "single_agent"); s["coop"] = (s.agent_action == "Cooperate").astype(float)
    for c0, c1 in permutations(sp.NAMES, 2):
        a = sp.contrast_arrays(h, x, s, c0, c1, False); n = a["hom0"].shape[0]
        pt = met(a); rng = np.random.default_rng(rng0)
        bs = [met(a, rng.integers(0, n, n)) for _ in range(10000)]
        for k in pt:
            v = np.array([b[k] for b in bs], float)
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            rows.append(dict(model=model, contrast=f"{sp.NAMES[c0]} -> {sp.NAMES[c1]}", metric=k, estimate=pt[k], ci_low=lo, ci_high=hi))
df = pd.DataFrame(rows); out = MIX / "analysis" / "extra_metrics.csv"; df.to_csv(out, index=False)
df["v"] = [f"{e:+.3f} [{l:+.3f},{h:+.3f}]" for e, l, h in zip(df.estimate, df.ci_low, df.ci_high)]
pd.set_option("display.width", 400)
for ks in (["direct", "spillover", "dyad_one", "dyad_two"], ["A_single", "A_single_TfT_Random", "A_single_TfT", "delta_TfT"]):
    print(df[df.metric.isin(ks)].pivot_table(index=["model", "contrast"], columns="metric", values="v", aggfunc="first")[ks].to_string()); print()
sig = lambda k, sgn: ((df.metric == k) & ((df.ci_low > 0) if sgn > 0 else (df.ci_high < 0))).sum()
for k in ["spillover", "A_single", "A_single_TfT_Random", "A_single_TfT"]:
    print(k, "sig+", sig(k, 1), "sig-", sig(k, -1), "of", (df.metric == k).sum())
