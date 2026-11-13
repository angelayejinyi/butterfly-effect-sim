"""Appendix figure (homophily and hub assignment): run-level echo-chamber metrics.

Recomputes the two metric panels of figs/homophily_hub_panel with the paper's metric
definitions (Appendix C.2), adapted from EC_visualization.ipynb:
  * stance assortativity: weighted Pearson correlation of numeric stances (-2..2) over
    cumulative directed REPOST/REPLY edges through round 15 (same as
    code/plots/plot_main_representation_figure.py);
  * weighted same-group edge ratio: share of edge weight between agents whose stances
    have the same sign (oppose / neutral / support).
The notebook version used categorical side assortativity on an undirected graph, which
gives different values. Pairwise tests are Welch t-tests with Holm correction within each
panel; only significant pairs are annotated.

Data: GPT-5.2 homophily runs (low / medium / high) and GPT-5.2 hub-assignment runs
(anti / pro / mixed / random hubs). The network drawings in the composite figure are unchanged.

Usage (from the repository root):
    python code/plots/plot_homophily_hub_metrics.py
"""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EC = ROOT / "data" / "results" / "EC"
OUT = ROOT / "data" / "plots_and_tables" / "figs"
STANCE = {r["agent_id"]: int(r["ai_regulation_stance"])
          for r in json.load(open(ROOT / "code/EC/inputs/ai_regulation_minimal_personas_100.json"))}
TARGET_ACTIONS = {"REPOST", "REPLY"}

PANELS = {
    "homophily": {
        "dir": EC / "homophily" / "gpt-5.2",
        "groups": {"Low homophily": "low", "Medium homophily": "medium", "High homophily": "high"},
        "pairs": [("Low homophily", "Medium homophily"), ("Medium homophily", "High homophily"),
                  ("Low homophily", "High homophily")],
    },
    "hub": {
        "dir": EC / "hub_assignment" / "gpt-5.2",
        "groups": {"Anti hubs": "anti_hubs", "Pro hubs": "pro_hubs", "Mixed hubs": "mixed_hubs", "Random hubs": "random_hubs"},
        "pairs": None,  # all pairs
    },
}
METRICS = [("assortativity", "Stance assortativity"), ("same_group_ratio", "Weighted same-group edge ratio")]


def cumulative_edges(a: pd.DataFrame, through_round: int = 15) -> Counter:
    t = a[a["round"].le(through_round) & a["action"].isin(TARGET_ACTIONS) & a["target_author_id"].notna()]
    return Counter(zip(t["agent_id"], t["target_author_id"]))


def weighted_stance_assortativity(edges: Counter) -> float:
    src = np.array([STANCE[s] for s, _ in edges], float)
    tgt = np.array([STANCE[t] for _, t in edges], float)
    w = np.array(list(edges.values()), float)
    sm, tm = np.average(src, weights=w), np.average(tgt, weights=w)
    cov = np.average((src - sm) * (tgt - tm), weights=w)
    return float(cov / np.sqrt(np.average((src - sm) ** 2, weights=w) * np.average((tgt - tm) ** 2, weights=w)))


def same_group_ratio(edges: Counter) -> float:
    same = sum(v for (s, t), v in edges.items() if np.sign(STANCE[s]) == np.sign(STANCE[t]))
    return same / sum(edges.values())


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compute():
    rows = []
    for panel, spec in PANELS.items():
        for label, sub in spec["groups"].items():
            for run in sorted((spec["dir"] / sub).glob("run*")):
                e = cumulative_edges(pd.read_csv(run / "agent_rounds.csv"))
                rows.append({"panel": panel, "group": label, "run": run.name,
                             "assortativity": weighted_stance_assortativity(e), "same_group_ratio": same_group_ratio(e)})
    return pd.DataFrame(rows)


def tests(df, panel):
    spec = PANELS[panel]
    order = list(spec["groups"])
    pairs = spec["pairs"] or list(combinations(order, 2))
    out = []
    for metric, _ in METRICS:
        ps = []
        for a, b in pairs:
            va = df.loc[(df.panel == panel) & (df.group == a), metric]
            vb = df.loc[(df.panel == panel) & (df.group == b), metric]
            ps.append(stats.ttest_ind(va, vb, equal_var=False).pvalue)
        for (a, b), p, ph in zip(pairs, ps, holm(ps)):
            out.append({"panel": panel, "metric": metric, "a": a, "b": b, "p": p, "p_holm": ph})
    return pd.DataFrame(out)


def _bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.1, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.1, text, ha="center", va="bottom", fontsize=9, color="#222222")


def plot(df, tst, panel, fname):
    order = list(PANELS[panel]["groups"])
    palette = dict(zip(order, sns.color_palette("muted", n_colors=len(order))))
    sub = df[df.panel == panel]
    with sns.axes_style("ticks"), plt.rc_context({"font.family": "Verdana", "axes.labelsize": 12,
                                                  "xtick.labelsize": 11, "ytick.labelsize": 11}):
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 7), dpi=300)
        fig.subplots_adjust(hspace=0.5)
        for ax, (metric, ylabel) in zip(axes, METRICS):
            sns.boxplot(data=sub, x="group", y=metric, hue="group", order=order, hue_order=order, palette=palette,
                        width=0.4, linewidth=1.5, showfliers=False, showcaps=False, legend=False, ax=ax)
            for i, g in enumerate(order):
                v = sub.loc[sub.group == g, metric]
                ax.hlines(np.median(v), i - 0.2, i + 0.2, colors="black", linewidth=2.0, zorder=3)
            sns.stripplot(data=sub, x="group", y=metric, hue="group", order=order, hue_order=order, palette=palette,
                          size=3.5, alpha=0.5, jitter=True, legend=False, ax=ax)
            lo, hi = float(sub[metric].min()), float(sub[metric].max())
            rng = max(hi - lo, 1e-6)
            sig = tst[(tst.panel == panel) & (tst.metric == metric) & (tst.p_holm < 0.05)]
            y = hi + 0.05 * rng
            for k, r in enumerate(sig.itertuples()):
                text = "$p$ < 0.001" if r.p_holm < 0.001 else f"$p$ = {r.p_holm:.3f}"
                _bracket(ax, order.index(r.a), order.index(r.b), y + k * 0.09 * rng, 0.022 * rng, text)
            ax.set_ylim(top=y + max(len(sig), 1) * 0.09 * rng + 0.1 * rng)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=20, length=0)
            ax.tick_params(axis="y", length=0)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
            ax.set_axisbelow(True)
            sns.despine(ax=ax)
        OUT.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{fname}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    df = compute()
    df.to_csv(OUT.parent / "ec_homophily_hub_metrics.csv", index=False)
    tst = pd.concat([tests(df, "homophily"), tests(df, "hub")], ignore_index=True)
    tst.to_csv(OUT.parent / "ec_homophily_hub_tests.csv", index=False)
    print(df.groupby(["panel", "group"])[["assortativity", "same_group_ratio"]].agg(["mean", "count"]).round(3))
    print(tst.round(4).to_string(index=False))
    plot(df, tst, "homophily", "ec_homophily_metrics")
    plot(df, tst, "hub", "ec_hub_metrics")
