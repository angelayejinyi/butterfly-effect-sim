"""Appendix D.1 and D.3 figures.

(1) pd_amplification_forest: interaction effect gap A = S_LLM - S_fixed with 95% paired-bootstrap CIs
    for every prespecified representation contrast (data/plots_and_tables/amplification_old_vs_new.csv).
(2) pd_single_perturbation_scatter: direct vs. spillover effects in single-perturbation dyads
    (additional_experiments/results/PD_single_perturbation/analysis/summary.csv).

Usage: python code/plots/plot_pd_appendix_summaries.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "plots_and_tables" / "figs"
MODELS = [("gpt-5.2", "GPT-5.2"), ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
          ("gemini-2.5-flash", "Gemini 2.5 Flash"), ("deepseek-v3.2", "DeepSeek V3.2")]
GROUPS = [("persona", "Persona format", ["Plain -> Descriptive", "Plain -> Tabular", "Descriptive -> Tabular"]),
          ("framing", "Game framing", ["Canonical -> Moralized", "Canonical -> Risk", "Moralized -> Risk"]),
          ("memory", "Memory", ["Table-Raw -> Narrative-Raw", "Table-Raw -> Table+Stats", "Table-Raw -> Narrative+Stats",
                                "Narrative-Raw -> Table+Stats", "Narrative-Raw -> Narrative+Stats",
                                "Table+Stats -> Narrative+Stats"])]
AMP, ATT, NS = "#3B6FB6", "#D1603D", "#A8A8A8"
MODEL_COLOR = {"GPT-5.2": "#4878D0", "Claude Haiku 4.5": "#EE854A", "Gemini 2.5 Flash": "#6ACC64",
               "DeepSeek V4 Flash": "#956CB4"}
META = {"Creator": None, "Producer": None}


def style():
    plt.rcParams.update({"font.family": ["Verdana", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def short(c: str) -> str:
    return c.replace(" -> ", " vs ")


def forest():
    df = pd.read_csv(ROOT / "data" / "plots_and_tables" / "amplification_old_vs_new.csv")
    rows, labels, seps, ypos = [], [], [], 0
    for dim, name, contrasts in GROUPS:
        seps.append((ypos, name))
        for c in contrasts:
            rows.append((dim, c, ypos)); labels.append(short(c)); ypos += 1
        ypos += 0.6
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 3.6), sharey=True, sharex=True)
    for ax, (mkey, mname) in zip(axes, MODELS):
        ax.axvline(0, color="#555555", lw=0.7)
        ax.grid(axis="x", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
        for dim, c, y in rows:
            r = df[(df.model == mkey) & (df.dimension == dim) & (df.contrast == c)].iloc[0]
            lo, hi, a = r.A_new_ci_low, r.A_new_ci_high, r.A_new
            col = AMP if lo > 0 else ATT if hi < 0 else NS
            ax.plot([lo, hi], [y, y], color=col, lw=1.3, solid_capstyle="round")
            ax.plot(a, y, "o", ms=3.8, color=col)
        ax.set_title(mname)
        ax.set_xlim(-1.0, 1.0); ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xlabel(r"$A = S^{\mathrm{LLM}} - S^{\mathrm{fixed}}$")
    ys = [y for *_, y in rows]
    axes[0].set_yticks(ys, labels); axes[0].set_ylim(max(ys) + 0.6, -1.3)
    for y0, name in seps:
        axes[0].annotate(name, xy=(0, y0 - 0.85), xycoords=("axes fraction", "data"), xytext=(-4, 0),
                         textcoords="offset points", ha="right", va="center", fontweight="bold", fontsize=7.5,
                         annotation_clip=False)
        for ax in axes:
            if y0 > 0:
                ax.axhline(y0 - 0.8, color="#dddddd", lw=0.6)
    handles = [Line2D([], [], color=c, marker="o", ms=4, lw=1.3, label=l)
               for c, l in [(AMP, "Amplification"), (ATT, "Attenuation"), (NS, "CI includes 0")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def scatter():
    s = pd.read_csv(ROOT / "additional_experiments" / "results" / "PD_single_perturbation" / "analysis" / "summary.csv")
    names = dict(MODELS[:3]); names["deepseek-v4-flash"] = "DeepSeek V4 Flash"
    d = s[s.metric == "direct"].merge(s[s.metric == "spillover"], on=["model", "contrast"], suffixes=("_d", "_s"))
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    ax.plot([-1.05, 1.05], [-1.05, 1.05], ls="--", color="#999999", lw=0.8, zorder=0)
    ax.axhline(0, color="#cccccc", lw=0.6, zorder=0); ax.axvline(0, color="#cccccc", lw=0.6, zorder=0)
    for mkey, g in d.groupby("model"):
        col = MODEL_COLOR[names[mkey]]
        for _, r in g.iterrows():
            sig = r.ci_low_s > 0 or r.ci_high_s < 0
            ax.plot(r.estimate_d, r.estimate_s, "o", ms=5, color=col, mfc=col if sig else "white", mew=1.1)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Direct effect on the perturbed agent")
    ax.set_ylabel("Spillover effect on the unperturbed partner")
    ax.text(0.97, 0.03, "spillover = direct", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#777777")
    handles = [Line2D([], [], color=MODEL_COLOR[n], marker="o", ls="", ms=5, label=n)
               for n in ["GPT-5.2", "Claude Haiku 4.5", "Gemini 2.5 Flash", "DeepSeek V4 Flash"]]
    handles.append(Line2D([], [], color="#555555", marker="o", mfc="white", ls="", ms=5, label="Spillover CI includes 0"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.8, handletextpad=0.2)
    fig.tight_layout()
    return fig


def main():
    style(); OUT.mkdir(parents=True, exist_ok=True)
    for name, fig in [("pd_amplification_forest", forest()), ("pd_single_perturbation_scatter", scatter())]:
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata=META)
        fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
