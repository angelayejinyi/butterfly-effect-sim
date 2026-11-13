"""Figure 2: effect of persona format on Prisoner's Dilemma outcomes (GPT-5.2).

Adapted from the four-panel summary in PD_visualization.ipynb. Columns show the
single-agent (fixed-opponent) and two-agent (LLM--LLM) settings; rows show average
payoff and cooperation rate. Heatmaps report pairwise differences between persona
formats; p-values are Holm-corrected within each heatmap, and non-significant cells
(adjusted p >= 0.05) are gray. Bars show run-level means
with 95% confidence intervals; dots show individual runs.

Changes from the notebook version:
  * reads the per-seed files in data/results/PD/persona/ instead of the merged CSVs;
  * error bars show 95% bootstrap confidence intervals instead of standard deviations;
  * p-values are Holm-corrected within each heatmap (one family per setting and metric);
  * single-agent tests use run-level means (one value per seed), as in the two-agent setting;
  * p-values below 0.001 are printed as "p<.001" instead of "p=0.000";
  * saves PDF and PNG to data/plots_and_tables/figs/.

Usage (from the repository root):
    python code/plots/plot_pd_persona_format_panel.py
"""

from pathlib import Path
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "results" / "PD" / "persona"
FIG_DIR = REPO_ROOT / "data" / "plots_and_tables" / "figs"
OUTPUT_STEM = "pd_persona_format_panel"

STRATEGIC_KEYS = ["P4_PLAIN_STRATEGIC", "P5_DESCRIPTIVE_STRATEGIC", "P6_TABULAR_STRATEGIC"]
LABEL_MAP = {
    "P4_PLAIN_STRATEGIC": "Plain",
    "P5_DESCRIPTIVE_STRATEGIC": "Descriptive",
    "P6_TABULAR_STRATEGIC": "Tabular",
}
FORMAT_ORDER = ["Plain", "Descriptive", "Tabular"]
POLICY_ORDER = ["TitForTat", "Random", "AlwaysCooperate", "AlwaysDefect"]
COMPARISON_ORDER = ["PLAIN $-$ DESCRIPTIVE", "PLAIN $-$ TABULAR", "DESCRIPTIVE $-$ TABULAR"]
METRICS = [("Payoff", "payoff"), ("Cooperation rate", "coop_rate")]
ALPHA = 0.05
NONSIG_GRAY = "lightgray"
ERRORBAR = ("ci", 95)
BOOTSTRAP_SEED = 0


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values; NaNs are left unchanged."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    m, running = len(order), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _apply_holm(effects: pd.DataFrame) -> pd.DataFrame:
    effects = effects.copy()
    effects["p_raw"] = effects["p"]
    for _, idx in effects.groupby("metric_key").groups.items():
        effects.loc[idx, "p"] = holm_adjust(effects.loc[idx, "p_raw"].to_numpy())
    return effects


def _comparison_label(key_a: str, key_b: str) -> str:
    return f"{key_a.split('_')[1]} $-$ {key_b.split('_')[1]}"


def _format_p(p: float) -> str:
    return "p<.001" if p < 0.001 else f"p={p:.3f}"


def safe_mwu(v1, v2):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            v1_clean = pd.Series(v1).dropna()
            v2_clean = pd.Series(v2).dropna()
            if len(v1_clean) == 0 or len(v2_clean) == 0:
                return np.nan, np.nan
            u_stat, p_val = mannwhitneyu(v1_clean, v2_clean, alternative="two-sided")
            if any("Precision loss" in str(w.message) for w in caught):
                return np.nan, np.nan
            return u_stat, p_val
        except Exception:
            return np.nan, np.nan


def load_rounds(model_dir: Path, setting: str) -> pd.DataFrame:
    files = sorted((model_dir / setting).glob("*/**/rounds.csv"))
    if not files:
        raise FileNotFoundError(f"No rounds.csv files under {model_dir / setting}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return df[df["persona_format"].isin(STRATEGIC_KEYS)].copy()


def load_single_agent(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["coop_rate"] = (df["agent_action"] == "Cooperate").astype(float)
    df["Format"] = df["persona_format"].map(LABEL_MAP)
    df = (
        df.groupby(["seed", "Format", "opponent_policy"])
        .agg(payoff=("agent_payoff", "mean"), coop_rate=("coop_rate", "mean"))
        .reset_index()
    )
    df["Format"] = pd.Categorical(df["Format"], categories=FORMAT_ORDER, ordered=True)
    df["opponent_policy"] = pd.Categorical(df["opponent_policy"], categories=POLICY_ORDER, ordered=True)
    return df


def load_two_agent(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Format"] = df["persona_format"].map(LABEL_MAP)
    df["agent_a_coop"] = (df["agent_a_action"] == "Cooperate").astype(float)
    df["agent_b_coop"] = (df["agent_b_action"] == "Cooperate").astype(float)
    df = (
        df.groupby(["seed", "Format"])
        .agg(
            agent_a_payoff=("agent_a_payoff", "mean"),
            agent_a_coop=("agent_a_coop", "mean"),
            agent_b_payoff=("agent_b_payoff", "mean"),
            agent_b_coop=("agent_b_coop", "mean"),
        )
        .reset_index()
    )
    rows_a = df.assign(Agent="A", payoff=df["agent_a_payoff"], coop_rate=df["agent_a_coop"])
    rows_b = df.assign(Agent="B", payoff=df["agent_b_payoff"], coop_rate=df["agent_b_coop"])
    out = pd.concat(
        [rows_a[["seed", "Format", "Agent", "payoff", "coop_rate"]],
         rows_b[["seed", "Format", "Agent", "payoff", "coop_rate"]]],
        ignore_index=True,
    )
    out["Format"] = pd.Categorical(out["Format"], categories=FORMAT_ORDER, ordered=True)
    out["Agent"] = pd.Categorical(out["Agent"], categories=["A", "B"], ordered=True)
    return out


def single_agent_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise contrasts against each fixed opponent, on run-level means.

    Each seed contributes one value (its mean over the 10 rounds), so rounds of the same
    run are not treated as independent observations.
    """
    rows = []
    for metric_label, metric_key in METRICS:
        for policy in POLICY_ORDER:
            for i in range(len(STRATEGIC_KEYS)):
                for j in range(i + 1, len(STRATEGIC_KEYS)):
                    v1 = df.loc[(df["Format"] == LABEL_MAP[STRATEGIC_KEYS[i]]) & (df["opponent_policy"] == policy), metric_key]
                    v2 = df.loc[(df["Format"] == LABEL_MAP[STRATEGIC_KEYS[j]]) & (df["opponent_policy"] == policy), metric_key]
                    u_stat, p_val = safe_mwu(v1, v2)
                    rows.append({
                        "metric": metric_label,
                        "metric_key": metric_key,
                        "policy": policy,
                        "comparison": _comparison_label(STRATEGIC_KEYS[i], STRATEGIC_KEYS[j]),
                        "effect": v1.mean() - v2.mean(),
                        "u": u_stat,
                        "p": p_val,
                    })
    return pd.DataFrame(rows)


def two_agent_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise contrasts for each agent slot, on run-level means."""
    rows = []
    for metric_label, metric_key in METRICS:
        for agent in ["A", "B"]:
            for i in range(len(STRATEGIC_KEYS)):
                for j in range(i + 1, len(STRATEGIC_KEYS)):
                    v1 = df.loc[(df["Format"] == LABEL_MAP[STRATEGIC_KEYS[i]]) & (df["Agent"] == agent), metric_key]
                    v2 = df.loc[(df["Format"] == LABEL_MAP[STRATEGIC_KEYS[j]]) & (df["Agent"] == agent), metric_key]
                    u_stat, p_val = safe_mwu(v1, v2)
                    rows.append({
                        "metric": metric_label,
                        "metric_key": metric_key,
                        "Agent": agent,
                        "comparison": _comparison_label(STRATEGIC_KEYS[i], STRATEGIC_KEYS[j]),
                        "effect": v1.mean() - v2.mean(),
                        "u": u_stat,
                        "p": p_val,
                    })
    return pd.DataFrame(rows)


def _annotation(effect_df: pd.DataFrame, p_df: pd.DataFrame) -> pd.DataFrame:
    ann = effect_df.copy().astype(object)
    for r in effect_df.index:
        for c in effect_df.columns:
            e, p = effect_df.loc[r, c], p_df.loc[r, c]
            ann.loc[r, c] = f"{e:+.2f}\n{_format_p(p)}" if pd.notna(p) and p < ALPHA else ""
    return ann


def _heatmap(ax, effects, column, columns, cmap, vmax):
    sub = effects.copy()
    effect_mat = sub.pivot(index="comparison", columns=column, values="effect").reindex(index=COMPARISON_ORDER, columns=columns)
    p_mat = sub.pivot(index="comparison", columns=column, values="p").reindex(index=COMPARISON_ORDER, columns=columns)
    plot_mat = effect_mat.astype(float).copy()
    plot_mat[p_mat.isna() | (p_mat >= ALPHA)] = np.nan
    sns.heatmap(
        plot_mat, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
        annot=_annotation(effect_mat, p_mat), fmt="", annot_kws={"size": 9, "weight": "normal"},
        linewidths=1.1, linecolor="white", cbar=False, ax=ax,
    )


def _bars(ax, data, x, order, y, palette):
    sns.barplot(
        data=data, x=x, y=y, hue="Format", order=order, hue_order=FORMAT_ORDER, palette=palette,
        errorbar=ERRORBAR, seed=BOOTSTRAP_SEED, capsize=0.08,
        err_kws={"linewidth": 1.4, "color": "black"}, alpha=0.8, ax=ax,
    )
    sns.stripplot(
        data=data, x=x, y=y, hue="Format", order=order, hue_order=FORMAT_ORDER, palette=palette,
        dodge=True, size=3.25, alpha=0.45, jitter=True, legend=False, ax=ax,
    )
    if ax.get_legend():
        ax.get_legend().remove()


def plot_pd_persona_format_panel(model: str = "gpt-5.2", data_dir: Path = DATA_DIR, fig_dir: Path = FIG_DIR):
    model_dir = data_dir / model
    single_raw = load_rounds(model_dir, "single_agent")
    two_raw = load_rounds(model_dir, "two_agent")
    single_df = load_single_agent(single_raw)
    two_df = load_two_agent(two_raw)
    single_fx = _apply_holm(single_agent_effects(single_df))
    two_fx = _apply_holm(two_agent_effects(two_df))

    all_fx = pd.concat([single_fx, two_fx], ignore_index=True)
    vmax = {
        key: (lambda v: v if np.isfinite(v) and v > 0 else 1.0)(np.nanmax(np.abs(all_fx.loc[all_fx["metric_key"] == key, "effect"])))
        for _, key in METRICS
    }
    cmaps = {
        "payoff": sns.diverging_palette(15, 240, s=85, l=45, as_cmap=True).copy(),
        "coop_rate": sns.diverging_palette(10, 260, s=80, l=45, as_cmap=True).copy(),
    }
    for cmap in cmaps.values():
        cmap.set_bad(NONSIG_GRAY)

    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": ["Verdana", "DejaVu Sans"],
        "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.dpi": 150,
    })
    palette = dict(zip(FORMAT_ORDER, sns.color_palette("muted", len(FORMAT_ORDER))))

    fig, axes = plt.subplots(2, 4, figsize=(15, 5), constrained_layout=True)
    for row, (_, key) in enumerate(METRICS):
        last_row = row == len(METRICS) - 1

        ax = axes[row, 0]
        _heatmap(ax, single_fx[single_fx["metric_key"] == key], "policy", POLICY_ORDER, cmaps[key], vmax[key])
        ax.set_xlabel("Opponent policy" if last_row else "", labelpad=5)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20, length=0, pad=2)
        ax.tick_params(axis="y", rotation=0, length=0, pad=2)

        ax = axes[row, 1]
        _bars(ax, single_df, "opponent_policy", POLICY_ORDER, key, palette)
        ax.set_xlabel("Opponent policy" if last_row else "", labelpad=5)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20, length=0)
        ax.tick_params(axis="y", length=0)
        if key == "coop_rate":
            ax.set_ylim(-0.05, 1.05)

        ax = axes[row, 2]
        _heatmap(ax, two_fx[two_fx["metric_key"] == key], "Agent", ["A", "B"], cmaps[key], vmax[key])
        ax.set_xlabel("Agent" if last_row else "", labelpad=5)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0, length=0, pad=2)
        ax.tick_params(axis="y", rotation=0, length=0, pad=2)

        ax = axes[row, 3]
        _bars(ax, two_df, "Agent", ["A", "B"], key, palette)
        ax.set_xlabel("Agent" if last_row else "", labelpad=5)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0, length=0)
        ax.tick_params(axis="y", length=0)
        if key == "coop_rate":
            ax.set_ylim(-0.05, 1.05)

    fig.canvas.draw()
    p00, p01, p02, p03 = (axes[0, k].get_position() for k in range(4))
    top_y = p00.y1 + 0.04
    fig.text((p00.x0 + p01.x1) / 2, top_y, "Single-agent setting", ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text((p02.x0 + p03.x1) / 2, top_y, "Two-agent setting", ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(-0.012, 0.8, "Average payoff", rotation=90, va="center", ha="left", fontsize=13, fontweight="bold")
    fig.text(-0.012, 0.3, "Cooperation rate", rotation=90, va="center", ha="left", fontsize=13, fontweight="bold")

    legend = fig.legend(
        handles=[Patch(facecolor=palette[label], label=label) for label in FORMAT_ORDER],
        loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(FORMAT_ORDER), frameon=False, title="Persona format",
    )
    legend.get_title().set_fontsize(11)
    legend.get_title().set_fontweight("normal")

    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("pdf", "png"):
        path = fig_dir / f"{OUTPUT_STEM}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=500)
        outputs.append(path)
    plt.close(fig)
    return outputs, single_fx, two_fx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt-5.2", help="Model directory under data/results/PD/persona/")
    args = parser.parse_args()
    paths, _, _ = plot_pd_persona_format_panel(model=args.model)
    for p in paths:
        print(f"Saved {p}")
