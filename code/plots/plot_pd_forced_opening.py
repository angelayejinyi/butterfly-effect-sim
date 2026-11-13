"""Appendix D.4 figure: forced openings and stated reasons.

Top row: two-agent cooperation in rounds 2-10 under free play and after a forced mutual-cooperation (CC)
or mutual-defection (DD) opening, with 95% percentile-bootstrap CIs over 30 seeds.
Bottom row: share of stated reasons that cite the risk of being exploited (free play: round 1;
forced openings: rounds 2-10), from additional_experiments/reason_analysis.py.

Usage: python additional_experiments/reason_analysis.py && python code/plots/plot_pd_forced_opening.py
"""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
FO = ROOT / "additional_experiments" / "results" / "PD_reason_analysis"
OUT = ROOT / "data" / "plots_and_tables" / "figs"
MODELS = [("gpt-5.2", "GPT-5.2"), ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
          ("gemini-2.5-flash", "Gemini 2.5 Flash"), ("deepseek-v4-flash", "DeepSeek V4 Flash")]
FMT = {"P4_PLAIN_STRATEGIC": "Plain", "P5_DESCRIPTIVE_STRATEGIC": "Descriptive", "P6_TABULAR_STRATEGIC": "Tabular"}
COLOR = {"Plain": "#4878D0", "Descriptive": "#EE854A", "Tabular": "#6ACC64"}
STAGES = [("free", "Free"), ("CC", "CC"), ("DD", "DD")]


def coop_runs(files) -> pd.DataFrame:
    out = []
    for f in files:
        d = pd.read_csv(f, usecols=["persona_format", "seed", "round", "agent_a_action", "agent_b_action"])
        d = d[d["round"] >= 2]
        c = ((d.agent_a_action == "Cooperate").mean() + (d.agent_b_action == "Cooperate").mean()) / 2
        out.append(dict(persona=FMT[d.persona_format.iloc[0]], seed=d.seed.iloc[0], coop=c))
    return pd.DataFrame(out)


def boot(x, rng, b=10000):
    x = np.asarray(x); m = rng.choice(x, (b, len(x))).mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def main():
    plt.rcParams.update({"font.family": ["Verdana", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    rng = np.random.default_rng(0)
    reasons = pd.read_csv(FO / "reason_summary.csv")
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharey="row", sharex=True)
    rows = []
    for j, (mkey, mname) in enumerate(MODELS):
        for si, (stage, _) in enumerate(STAGES):
            pat = (ROOT / "data/results/PD/persona" / mkey / "two_agent" / "P*_STRATEGIC" / "seed*" / "rounds.csv"
                   if stage == "free" else FO / "forced_opening" / mkey / stage / "P*_STRATEGIC" / "seed*" / "rounds.csv")
            runs = coop_runs(glob.glob(str(pat)))
            for k, fmt in enumerate(["Plain", "Descriptive", "Tabular"]):
                m, lo, hi = boot(runs[runs.persona == fmt].coop, rng)
                x = si + (k - 1) * 0.16
                axes[0, j].errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=4, color=COLOR[fmt], elinewidth=0.9)
                rstage = "free_r1" if stage == "free" else stage
                r = reasons[(reasons.model == mkey) & (reasons.stage == rstage) & (reasons.persona == fmt)]
                share = r.exploitation_risk.iloc[0] if len(r) else np.nan
                if len(r):
                    axes[1, j].bar(x, share, width=0.15, color=COLOR[fmt])
                rows.append(dict(model=mname, stage=stage, persona=fmt, coop=m, coop_lo=lo, coop_hi=hi, risk_share=share))
        if mkey == "gemini-2.5-flash":
            axes[1, j].text(0, 0.05, "n/a", ha="center", va="bottom", fontsize=7, color="#777777")
        axes[0, j].set_title(mname)
        for i in range(2):
            axes[i, j].set_xticks(range(3), [s for _, s in STAGES]); axes[i, j].set_xlim(-0.5, 2.5)
            axes[i, j].grid(axis="y", color="#e5e5e5", lw=0.6); axes[i, j].set_axisbelow(True)
    axes[0, 0].set_ylim(-0.05, 1.05); axes[1, 0].set_ylim(0, 1.0)
    axes[0, 0].set_ylabel("Cooperation\n(rounds 2–10)")
    axes[1, 0].set_ylabel("Reasons citing\nexploitation risk")
    handles = [Line2D([], [], color=COLOR[f], marker="o", ls="", ms=5, label=f) for f in COLOR]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "pd_forced_opening.pdf", bbox_inches="tight", metadata={"Creator": None, "Producer": None})
    fig.savefig(OUT / "pd_forced_opening.png", bbox_inches="tight", dpi=200)
    T = pd.DataFrame(rows); T.to_csv(ROOT / "data/plots_and_tables/pd_forced_opening_reasons.csv", index=False)
    v = T.dropna()
    rho = v.coop.rank().corr(v.risk_share.rank())
    print(T.round(2).to_string(index=False)); print("cells", len(v), "spearman coop vs risk", round(rho, 2))


if __name__ == "__main__":
    main()
