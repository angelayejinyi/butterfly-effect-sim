"""Section 4.3 figure: the round-1 cooperation gap predicts the interaction effect gap A.

For each model and persona-representation contrast (Plain/Descriptive/Tabular), the
x-axis is the absolute difference in round-1 two-agent cooperation between the two
representations, and the y-axis is A = S_LLM - S_fixed (cooperation, pp), with
S_fixed averaged over the four fixed opponents. Error bars are paired-bootstrap 95% CIs
(30 seeds, 10,000 replicates). Colors (models) follow Figure 3; marker shapes show the
three persona-representation contrasts.

Usage (from the repository root):
    python code/plots/plot_pd_round1_mechanism.py
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "results" / "PD" / "persona"
OUT = ROOT / "data" / "plots_and_tables" / "figs"
FORMATS = {"Plain": "P4_PLAIN_STRATEGIC", "Descriptive": "P5_DESCRIPTIVE_STRATEGIC", "Tabular": "P6_TABULAR_STRATEGIC"}
OPPONENTS = ["TitForTat", "Random", "AlwaysCooperate", "AlwaysDefect"]
MODELS = {  # label: (results dir, color)
    "GPT-5.2": ("gpt-5.2", "#E83E68"),
    "Claude Haiku 4.5": ("claude-haiku-4-5-20251001", "#3F6FDE"),
    "Gemini 2.5 Flash": ("gemini-2.5-flash", "#19A17F"),
    "DeepSeek V3.2": ("deepseek-v3.2", "#E08B2C"),
}
CONTRAST_MARKERS = {"Plain vs Descriptive": "o", "Plain vs Tabular": "s", "Descriptive vs Tabular": "^"}
N_BOOT, BOOT_SEED = 10_000, 20260925


def _two_agent(model_dir, fmt):
    rows = {}
    for path in (DATA / model_dir / "two_agent" / fmt).glob("seed*/rounds.csv"):
        d = pd.read_csv(path)
        coop = ((d.agent_a_action == "Cooperate").astype(float) + (d.agent_b_action == "Cooperate")) / 2
        rows[int(d.seed.iloc[0])] = (coop.mean(), coop[d["round"] == 1].mean())
    return pd.DataFrame(rows, index=["all", "round1"]).T.sort_index()


def _single_agent(model_dir, fmt, opponent):
    rows = {}
    for path in (DATA / model_dir / "single_agent" / fmt / opponent).glob("seed*/rounds.csv"):
        d = pd.read_csv(path)
        rows[int(d.seed.iloc[0])] = (d.agent_action == "Cooperate").mean()
    return pd.Series(rows).sort_index()


def compute():
    rng = np.random.default_rng(BOOT_SEED)
    records = []
    for label, (model_dir, *_rest) in MODELS.items():
        two = {f: _two_agent(model_dir, d) for f, d in FORMATS.items()}
        single = {f: {o: _single_agent(model_dir, d, o) for o in OPPONENTS} for f, d in FORMATS.items()}
        for first, second in combinations(FORMATS, 2):
            seeds = sorted(set(two[first].index) & set(two[second].index)
                           .intersection(*[set(single[f][o].index) for f in (first, second) for o in OPPONENTS]))
            d_llm = (two[second].loc[seeds, "all"] - two[first].loc[seeds, "all"]).to_numpy()
            d_fix = {o: (single[second][o].loc[seeds] - single[first][o].loc[seeds]).to_numpy() for o in OPPONENTS}

            def gap(idx):
                return abs(d_llm[idx].mean()) - np.mean([abs(d_fix[o][idx].mean()) for o in OPPONENTS])

            n = len(seeds)
            boot = np.array([gap(rng.integers(0, n, n)) for _ in range(N_BOOT)])
            round1_gap = abs(two[second].loc[seeds, "round1"].mean() - two[first].loc[seeds, "round1"].mean())
            records.append(dict(model=label, contrast=f"{first} vs {second}", round1_gap=round1_gap,
                                A=gap(np.arange(n)), A_low=np.quantile(boot, 0.025), A_high=np.quantile(boot, 0.975)))
    return pd.DataFrame(records)


def _spearman(x, y):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def plot(results):
    OUT.mkdir(parents=True, exist_ok=True)
    with sns.axes_style("ticks"), plt.rc_context({
        "font.family": "Verdana", "axes.labelsize": 11, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "legend.fontsize": 8.5,
    }):
        fig, ax = plt.subplots(figsize=(4.0, 3.6), dpi=300)
        ax.axhspan(0, 70, color="#E83E68", alpha=0.04, zorder=0)
        ax.axhline(0, color="#777777", linewidth=1.2, linestyle="--", zorder=1)
        for label, (_, color) in MODELS.items():
            for contrast, marker in CONTRAST_MARKERS.items():
                row = results[(results.model == label) & (results.contrast == contrast)].iloc[0]
                ax.errorbar(
                    row.round1_gap * 100, row.A * 100,
                    yerr=[[(row.A - row.A_low) * 100], [(row.A_high - row.A) * 100]],
                    fmt=marker, color=color, markersize=6.5, elinewidth=1.3, capsize=2.5,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3,
                )
        rho = _spearman(results.round1_gap, results.A)
        ax.text(0.03, 0.96, rf"Spearman $\rho$ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=9.5)
        ax.text(0.98, 0.03, "attenuation", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#777777")
        ax.text(0.98, 0.97, "amplification", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#777777")
        from matplotlib.lines import Line2D
        model_handles = [Line2D([], [], color=c, marker="o", linestyle="none", markersize=6.5, label=m)
                         for m, (_, c) in MODELS.items()]
        contrast_handles = [Line2D([], [], color="#555555", marker=mk, linestyle="none", markersize=6.5,
                                   label=ct.replace("Descriptive", "Desc.").replace("Tabular", "Tab."))
                            for ct, mk in CONTRAST_MARKERS.items()]
        legend = ax.legend(handles=model_handles, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
                           ncol=1, handletextpad=0.1, labelspacing=0.2, borderaxespad=0.1)
        ax.add_artist(legend)
        ax.legend(handles=contrast_handles, frameon=False, loc="lower right", bbox_to_anchor=(1.02, 1.0),
                  ncol=1, handletextpad=0.1, labelspacing=0.2, borderaxespad=0.1)
        ax.set_xlim(-5, 80)
        ax.set_ylim(-65, 65)
        ax.set_xlabel("Round-1 cooperation gap (pp)")
        ax.set_ylabel(r"Interaction effect gap $A$ (pp)")
        ax.grid(True, color="#C7C7C7", linewidth=0.8, linestyle="--", alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", top=True, right=True)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"pd_round1_mechanism.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    return rho


if __name__ == "__main__":
    results = compute()
    results.to_csv(OUT.parent / "pd_round1_mechanism.csv", index=False)
    print(results.round(3).to_string(index=False))
    print("Spearman rho:", round(plot(results), 3))
