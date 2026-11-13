"""Appendix D.2 figures: cooperation under alternative personas and game descriptions.

Reads the logged runs in additional_experiments/results/PD_other_formats (block1) and
PD_other_framings (block2). Each point is the mean cooperation rate over 30 seeds with a
95% percentile-bootstrap CI over seeds. Fixed-opponent values average the four policies
within each seed; two-agent values average both agents within each dyad.

Usage: python code/plots/plot_pd_alt_implementations.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "additional_experiments" / "results"
OUT = ROOT / "data" / "plots_and_tables" / "figs"
MODELS = [("gpt-5.2", "GPT-5.2"), ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
          ("gemini-2.5-flash", "Gemini 2.5 Flash"), ("deepseek-v3.2", "DeepSeek V3.2")]
FORMATS = ["Plain", "Descriptive", "Tabular"]
FMT_COLOR = dict(zip(FORMATS, ["#4878D0", "#EE854A", "#6ACC64"]))
PERSONAS = ["Neutral", "Cooperative", "Strategic"]
PERSONA_KEYS = {("Plain", "Neutral"): "P1_PLAIN_NEUTRAL", ("Descriptive", "Neutral"): "P2_DESCRIPTIVE_NEUTRAL",
                ("Tabular", "Neutral"): "P3_TABULAR_NEUTRAL", ("Plain", "Strategic"): "P4_PLAIN_STRATEGIC",
                ("Descriptive", "Strategic"): "P5_DESCRIPTIVE_STRATEGIC", ("Tabular", "Strategic"): "P6_TABULAR_STRATEGIC",
                ("Plain", "Cooperative"): "P7_PLAIN_COOPERATIVE", ("Descriptive", "Cooperative"): "P8_DESCRIPTIVE_COOPERATIVE",
                ("Tabular", "Cooperative"): "P9_TABULAR_COOPERATIVE"}
GAMES = [("G1_CANONICAL_GAME_THEORY", "Canonical"), ("G4_MORALIZED_FRAMING", "Moralized"), ("G6_RISK_FRAMING", "Risk"),
         ("G2_SOCIAL_FRAMING", "Social"), ("G3_NEUTRAL_ABSTRACT", "Abstract"),
         ("G5_STRATEGIC_ECONOMICS", "Economic"), ("G7_MINIMAL_COMPRESSED", "Minimal")]
REGIMES = [("two_agent", "Two-agent", "o"), ("single_agent", "Fixed opponent", "s")]


def seed_means(path: Path, key: str) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=lambda c: not c.endswith(("raw_output", "prompt_input")))
    if "agent_a_action" in d:
        d["coop"] = ((d.agent_a_action == "Cooperate").astype(float) + (d.agent_b_action == "Cooperate")) / 2
    else:
        d["coop"] = (d.agent_action == "Cooperate").astype(float)
    return d.groupby([key, "seed"]).coop.mean().reset_index().rename(columns={key: "cond"})


def ci(x: np.ndarray, rng: np.random.Generator, b: int = 10000) -> tuple[float, float, float]:
    boots = rng.choice(x, size=(b, len(x)), replace=True).mean(axis=1)
    return x.mean(), *np.percentile(boots, [2.5, 97.5])


def summarize(folder: str, block: str, key: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for mdir, mname in MODELS:
        for reg, _, _ in REGIMES:
            sm = seed_means(RES / folder / f"{block}_{reg}_{mdir}.csv", key)
            for cond, g in sm.groupby("cond"):
                m, lo, hi = ci(g.coop.to_numpy(), rng)
                rows.append(dict(model=mname, regime=reg, cond=cond, mean=m, lo=lo, hi=hi, n=len(g)))
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update({"font.family": ["Verdana", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def panel_axes(n_rows: int, height: float):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, height), sharey=True, sharex=True)
    for ax, (_, mname) in zip(axes, MODELS):
        ax.set_title(mname)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Cooperation rate")
    return fig, axes


def errpoint(ax, x, lo, hi, y, color, marker, filled):
    ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt=marker, ms=4.2, color=color, mfc=color if filled else "white",
                mec=color, mew=1.0, elinewidth=0.9, capsize=0)


def plot_personas(df: pd.DataFrame):
    fig, axes = panel_axes(len(PERSONAS), 2.0)
    for ax, (_, mname) in zip(axes, MODELS):
        for yi, persona in enumerate(PERSONAS):
            for fi, fmt in enumerate(FORMATS):
                for ri, (reg, _, mk) in enumerate(REGIMES):
                    r = df[(df.model == mname) & (df.regime == reg) & (df.cond == PERSONA_KEYS[(fmt, persona)])].iloc[0]
                    y = yi + (fi - 1) * 0.2 + (ri - 0.5) * 0.08
                    errpoint(ax, r["mean"], r.lo, r.hi, y, FMT_COLOR[fmt], mk, reg == "two_agent")
    axes[0].set_yticks(range(len(PERSONAS)), PERSONAS)
    handles = [Line2D([], [], color=FMT_COLOR[f], marker="o", ls="", ms=5, label=f) for f in FORMATS]
    handles += [Line2D([], [], color="#555555", marker=mk, mfc="#555555" if reg == "two_agent" else "white",
                       ls="", ms=5, label=lab) for reg, lab, mk in REGIMES]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_games(df: pd.DataFrame):
    fig, axes = panel_axes(len(GAMES), 2.3)
    reg_color = {"two_agent": "#3B5B92", "single_agent": "#9A9A9A"}
    for ax, (_, mname) in zip(axes, MODELS):
        ax.axhspan(2.5, len(GAMES) - 0.5, color="#f4f4f4", zorder=0)
        for yi, (key, _) in enumerate(GAMES):
            for ri, (reg, _, mk) in enumerate(REGIMES):
                r = df[(df.model == mname) & (df.regime == reg) & (df.cond == key)].iloc[0]
                errpoint(ax, r["mean"], r.lo, r.hi, yi + (ri - 0.5) * 0.28, reg_color[reg], mk, True)
    axes[0].set_yticks(range(len(GAMES)), [lab for _, lab in GAMES])
    handles = [Line2D([], [], color=reg_color[reg], marker=mk, ls="", ms=5, label=lab) for reg, lab, mk in REGIMES]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": None, "Producer": None}
    per = summarize("PD_other_formats", "block1", "persona_format")
    gam = summarize("PD_other_framings", "block2", "game_instruction")
    pd.concat([per.assign(experiment="persona"), gam.assign(experiment="game_description")]).to_csv(
        ROOT / "data" / "plots_and_tables" / "pd_alt_implementations_ci.csv", index=False)
    for name, fig in [("pd_other_personas", plot_personas(per)), ("pd_game_descriptions", plot_games(gam))]:
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata=meta)
        fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
