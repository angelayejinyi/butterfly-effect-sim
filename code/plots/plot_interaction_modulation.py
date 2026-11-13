"""Publication figure for representation sensitivity across interaction regimes."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_NAMES = {
    "gpt-5.2": "GPT-5.2",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "deepseek-v3.2": "DeepSeek V3.2",
}
MODEL_ORDER = ["GPT-5.2", "Claude Haiku 4.5", "Gemini 2.5 Flash", "DeepSeek V3.2"]
MODEL_COLORS = {
    "GPT-5.2": "#E83E68",
    "Claude Haiku 4.5": "#3F6FDE",
    "Gemini 2.5 Flash": "#19A17F",
    "DeepSeek V3.2": "#E08B2C",
}
REGIME_STYLES = {
    "Fixed opponents": {"color": "#777777", "marker": "o"},
    "LLM-LLM": {"color": "#E83E68", "marker": "D"},
}


def plot_interaction_modulation(
    data_dir: Path,
    *,
    dimension: str,
    contrast: str,
    effect_axis_label: str,
    gap_title: str,
    trajectory_title: str,
    output_stem: str,
    magnitude_xlim: tuple[float, float],
    magnitude_ticks: list[float],
    gap_xlim: tuple[float, float],
    gap_ticks: list[float],
    trajectory_ylim: tuple[float, float],
    trajectory_ticks: list[float],
    regime_legend_location: str,
) -> tuple[Path, Path]:
    overall = pd.read_csv(data_dir / "amplification_old_vs_new.csv")
    roundwise = pd.read_csv(data_dir / "amplification_roundwise_updated.csv")
    headline = overall.query("dimension == @dimension and contrast == @contrast").copy()
    trajectory = roundwise.query("dimension == @dimension and contrast == @contrast").copy()
    for frame in (headline, trajectory):
        frame["Model"] = frame["model"].map(MODEL_NAMES)
    for column in ["S_fixed", "S_LLM", "A_new", "A_new_ci_low", "A_new_ci_high"]:
        headline[f"{column}_pp"] = 100 * headline[column]
    for column in ["A_new", "A_new_ci_low", "A_new_ci_high"]:
        trajectory[f"{column}_pp"] = 100 * trajectory[column]
    headline["Model"] = pd.Categorical(headline["Model"], MODEL_ORDER, ordered=True)
    trajectory["Model"] = pd.Categorical(trajectory["Model"], MODEL_ORDER, ordered=True)
    headline = headline.sort_values("Model")
    trajectory = trajectory.sort_values(["Model", "round"])

    if len(headline) != 4 or len(trajectory) != 40:
        raise ValueError(f"Expected 4 overall and 40 round rows for {dimension}: {contrast}")
    if not trajectory.groupby("Model", observed=True)["round"].nunique().eq(10).all():
        raise ValueError(f"Incomplete round coverage for {dimension}: {contrast}")
    if headline[["A_new_ci_low", "A_new_ci_high"]].isna().any().any():
        raise ValueError(f"Missing overall confidence intervals for {dimension}: {contrast}")
    if trajectory[["A_new_ci_low", "A_new_ci_high"]].isna().any().any():
        raise ValueError(f"Missing round confidence intervals for {dimension}: {contrast}")

    with sns.axes_style("ticks"), plt.rc_context(
        {
            "font.family": "Verdana",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    ):
        fig = plt.figure(figsize=(10, 8.4), dpi=300)
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.08], hspace=0.48, wspace=0.38)
        ax_regimes = fig.add_subplot(grid[0, 0])
        ax_gap = fig.add_subplot(grid[0, 1])
        ax_rounds = fig.add_subplot(grid[1, :])
        y_positions = {model: 3 - index for index, model in enumerate(MODEL_ORDER)}

        for row in headline.itertuples():
            y_position = y_positions[str(row.Model)]
            ax_regimes.hlines(
                y_position, row.S_fixed_pp, row.S_LLM_pp,
                color="#B8B8B8", linewidth=2.0, zorder=1,
            )
        regime_data = pd.concat(
            [
                headline[["Model", "S_fixed_pp"]]
                .rename(columns={"S_fixed_pp": "Magnitude"})
                .assign(Regime="Fixed opponents"),
                headline[["Model", "S_LLM_pp"]]
                .rename(columns={"S_LLM_pp": "Magnitude"})
                .assign(Regime="LLM-LLM"),
            ],
            ignore_index=True,
        )
        regime_data["y"] = regime_data["Model"].map(y_positions).astype(float)
        for regime, style in REGIME_STYLES.items():
            subset = regime_data.loc[regime_data["Regime"].eq(regime)]
            sns.scatterplot(
                data=subset, x="Magnitude", y="y", color=style["color"],
                marker=style["marker"], s=88, edgecolor="white", linewidth=0.8,
                label=regime, ax=ax_regimes, zorder=3,
            )
            for point in subset.itertuples():
                ax_regimes.annotate(
                    f"{point.Magnitude:.1f}", (point.Magnitude, point.y),
                    xytext=(0, 8 if regime == "Fixed opponents" else -9),
                    textcoords="offset points", ha="center",
                    va="bottom" if regime == "Fixed opponents" else "top",
                    color=style["color"], fontsize=8.5,
                )
        ax_regimes.set_xlim(*magnitude_xlim)
        ax_regimes.set_xticks(magnitude_ticks)
        ax_regimes.set_ylim(-0.45, 3.45)
        ax_regimes.set_yticks(
            [y_positions[model] for model in MODEL_ORDER], labels=MODEL_ORDER
        )
        ax_regimes.set_xlabel(effect_axis_label)
        ax_regimes.set_ylabel("")
        ax_regimes.set_title("a   Effect across interaction regimes", loc="left", fontweight="bold")
        ax_regimes.legend(frameon=False, loc=regime_legend_location, handletextpad=0.4)

        for row in headline.itertuples():
            y_position = y_positions[str(row.Model)]
            color = MODEL_COLORS[str(row.Model)]
            ax_gap.hlines(
                y_position, row.A_new_ci_low_pp, row.A_new_ci_high_pp,
                color=color, linewidth=10, alpha=0.28, zorder=2,
            )
            ax_gap.scatter(
                row.A_new_pp, y_position, s=88, color=color,
                edgecolor="white", linewidth=0.8, zorder=3,
            )
            ax_gap.annotate(
                f"{row.A_new_pp:+.1f}", (row.A_new_pp, y_position),
                xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", color=color, fontsize=8.5,
            )
        ax_gap.axvline(0, color="#777777", linewidth=1.4, linestyle="--", zorder=1)
        ax_gap.set_xlim(*gap_xlim)
        ax_gap.set_xticks(gap_ticks)
        ax_gap.set_ylim(-0.45, 3.45)
        ax_gap.set_yticks([y_positions[model] for model in MODEL_ORDER])
        ax_gap.set_yticklabels([])
        ax_gap.set_xlabel(r"Interaction effect gap $A$ (pp)")
        ax_gap.set_ylabel("")
        ax_gap.set_title(f"b   {gap_title}", loc="left", fontweight="bold")

        for model in MODEL_ORDER:
            subset = trajectory.loc[trajectory["Model"].eq(model)].sort_values("round")
            rounds = subset["round"].to_numpy(dtype=float)
            estimates = subset["A_new_pp"].to_numpy(dtype=float)
            ci_low = subset["A_new_ci_low_pp"].to_numpy(dtype=float)
            ci_high = subset["A_new_ci_high_pp"].to_numpy(dtype=float)
            ax_rounds.fill_between(
                rounds, ci_low, ci_high,
                color=MODEL_COLORS[model], alpha=0.12, linewidth=0, zorder=1,
            )
            sns.lineplot(
                x=rounds, y=estimates, color=MODEL_COLORS[model],
                marker="o", markersize=5.5, linewidth=2.0,
                label=model, ax=ax_rounds, zorder=3,
            )
        ax_rounds.axhline(0, color="#777777", linewidth=1.4, linestyle="--", zorder=2)
        ax_rounds.set_xlim(1, 10)
        ax_rounds.set_xticks(range(1, 11))
        ax_rounds.set_ylim(*trajectory_ylim)
        ax_rounds.set_yticks(trajectory_ticks)
        ax_rounds.set_xlabel("Round")
        ax_rounds.set_ylabel(r"$A_t$ (pp)")
        ax_rounds.set_title(f"c   {trajectory_title}", loc="left", fontweight="bold")
        ax_rounds.legend(
            frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20),
            ncol=4, columnspacing=1.3, handletextpad=0.5,
        )

        for axis in (ax_regimes, ax_gap, ax_rounds):
            axis.grid(
                True, axis="both", color="#C7C7C7",
                linewidth=0.8, linestyle="--", alpha=0.75,
            )
            axis.set_axisbelow(True)
            axis.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.96)
        png_path = data_dir / f"{output_stem}.png"
        pdf_path = data_dir / f"{output_stem}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        plt.show()
    return png_path, pdf_path


def plot_interaction_modulation_panels(
    data_dir: Path,
    output_dir: Path,
    *,
    dimension: str,
    contrast: str,
    effect_axis_label: str,
    output_stem: str,
    magnitude_xlim: tuple[float, float],
    magnitude_ticks: list[float],
    gap_xlim: tuple[float, float],
    gap_ticks: list[float],
    trajectory_ylim: tuple[float, float],
    trajectory_ticks: list[float],
) -> list[tuple[Path, Path]]:
    overall = pd.read_csv(data_dir / "amplification_old_vs_new.csv")
    roundwise = pd.read_csv(data_dir / "amplification_roundwise_updated.csv")
    headline = overall.query("dimension == @dimension and contrast == @contrast").copy()
    trajectory = roundwise.query("dimension == @dimension and contrast == @contrast").copy()
    for frame in (headline, trajectory):
        frame["Model"] = frame["model"].map(MODEL_NAMES)
    for column in ["S_fixed", "S_LLM", "A_new", "A_new_ci_low", "A_new_ci_high"]:
        headline[f"{column}_pp"] = 100 * headline[column]
    for column in ["A_new", "A_new_ci_low", "A_new_ci_high"]:
        trajectory[f"{column}_pp"] = 100 * trajectory[column]
    headline["Model"] = pd.Categorical(headline["Model"], MODEL_ORDER, ordered=True)
    trajectory["Model"] = pd.Categorical(trajectory["Model"], MODEL_ORDER, ordered=True)
    headline = headline.sort_values("Model")
    trajectory = trajectory.sort_values(["Model", "round"])

    if len(headline) != 4 or len(trajectory) != 40:
        raise ValueError(f"Expected 4 overall and 40 round rows for {dimension}: {contrast}")
    if not trajectory.groupby("Model", observed=True)["round"].nunique().eq(10).all():
        raise ValueError(f"Incomplete round coverage for {dimension}: {contrast}")

    output_dir.mkdir(parents=True, exist_ok=True)
    y_positions = {model: 3 - index for index, model in enumerate(MODEL_ORDER)}
    saved_paths = []

    with sns.axes_style("ticks"), plt.rc_context(
        {
            "font.family": "Verdana",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    ):
        fig, axis = plt.subplots(figsize=(4.5, 4.5), dpi=300)
        for row in headline.itertuples():
            y_position = y_positions[str(row.Model)]
            axis.hlines(y_position, row.S_fixed_pp, row.S_LLM_pp, color="#B8B8B8", linewidth=2.0, zorder=1)
        regime_data = pd.concat(
            [
                headline[["Model", "S_fixed_pp"]].rename(columns={"S_fixed_pp": "Sensitivity"}).assign(Regime="Fixed opponents"),
                headline[["Model", "S_LLM_pp"]].rename(columns={"S_LLM_pp": "Sensitivity"}).assign(Regime="LLM-LLM"),
            ],
            ignore_index=True,
        )
        regime_data["y"] = regime_data["Model"].map(y_positions).astype(float)
        for regime, style in REGIME_STYLES.items():
            subset = regime_data.loc[regime_data["Regime"].eq(regime)]
            legend_label = (
                r"Fixed-opponent $S^{\mathrm{fixed}}$"
                if regime == "Fixed opponents"
                else r"LLM-LLM $S^{\mathrm{LLM}}$"
            )
            sns.scatterplot(
                data=subset, x="Sensitivity", y="y", color=style["color"], marker=style["marker"],
                s=88, edgecolor="white", linewidth=0.8, label=legend_label, ax=axis, zorder=3,
            )
            for point in subset.itertuples():
                axis.annotate(
                    f"{point.Sensitivity:.1f}", (point.Sensitivity, point.y),
                    xytext=(0, 8 if regime == "Fixed opponents" else -9), textcoords="offset points",
                    ha="center", va="bottom" if regime == "Fixed opponents" else "top",
                    color=style["color"], fontsize=10,
                )
        wrapped_effect_label = effect_axis_label.replace("Representation sensitivity", "Representation\nsensitivity")
        axis.set(xlim=magnitude_xlim, xticks=magnitude_ticks, ylim=(-0.45, 3.45), xlabel=wrapped_effect_label, ylabel="")
        axis.set_yticks([y_positions[model] for model in MODEL_ORDER], labels=MODEL_ORDER)
        axis.legend(
            frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25),
            ncol=2, columnspacing=1.0, handletextpad=0.4,
        )
        saved_paths.append(_finish_panel(fig, axis, output_dir, f"{output_stem}_regimes", bottom=0.31))

        fig, axis = plt.subplots(figsize=(4.5, 4.5), dpi=300)
        for row in headline.itertuples():
            y_position = y_positions[str(row.Model)]
            color = MODEL_COLORS[str(row.Model)]
            axis.hlines(y_position, row.A_new_ci_low_pp, row.A_new_ci_high_pp, color=color, linewidth=10, alpha=0.28, zorder=2)
            axis.scatter(row.A_new_pp, y_position, s=88, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            axis.annotate(
                f"{row.A_new_pp:+.1f}", (row.A_new_pp, y_position), xytext=(0, 8),
                textcoords="offset points", ha="center", va="bottom", color=color, fontsize=10,
            )
        axis.axvline(0, color="#777777", linewidth=1.4, linestyle="--", zorder=1)
        axis.set(xlim=gap_xlim, xticks=gap_ticks, ylim=(-0.45, 3.45), xlabel="Interaction effect\ngap $A$ (pp)", ylabel="")
        axis.set_yticks([y_positions[model] for model in MODEL_ORDER], labels=MODEL_ORDER)
        axis.text(
            -1.5, 3.34, "attenuation\n$A<0$",
            ha="right", va="top", color="#777777", fontsize=9, linespacing=1.0,
        )
        axis.text(
            1.5, 3.34, "amplification\n$A>0$",
            ha="left", va="top", color="#777777", fontsize=9, linespacing=1.0,
        )
        saved_paths.append(_finish_panel(fig, axis, output_dir, f"{output_stem}_gap"))

        fig, axis = plt.subplots(figsize=(4.5, 4.5), dpi=300)
        for model in MODEL_ORDER:
            subset = trajectory.loc[trajectory["Model"].eq(model)].sort_values("round")
            rounds = subset["round"].to_numpy(dtype=float)
            estimates = subset["A_new_pp"].to_numpy(dtype=float)
            ci_low = subset["A_new_ci_low_pp"].to_numpy(dtype=float)
            ci_high = subset["A_new_ci_high_pp"].to_numpy(dtype=float)
            axis.fill_between(rounds, ci_low, ci_high, color=MODEL_COLORS[model], alpha=0.12, linewidth=0, zorder=1)
            sns.lineplot(
                x=rounds, y=estimates, color=MODEL_COLORS[model], marker="o", markersize=4.5,
                linewidth=1.8, label=model, ax=axis, zorder=3,
            )
        axis.axhline(0, color="#777777", linewidth=1.4, linestyle="--", zorder=2)
        axis.set(
            xlim=(1, 10), xticks=range(1, 11), ylim=trajectory_ylim, yticks=trajectory_ticks,
            xlabel="Round", ylabel=r"$A_t$ (pp)",
        )
        axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handletextpad=0.4)
        saved_paths.append(_finish_panel(fig, axis, output_dir, f"{output_stem}_trajectory", bottom=0.24))

    return saved_paths


def _finish_panel(
    fig: plt.Figure,
    axis: plt.Axes,
    output_dir: Path,
    output_stem: str,
    *,
    bottom: float = 0.14,
) -> tuple[Path, Path]:
    axis.grid(True, axis="both", color="#C7C7C7", linewidth=0.8, linestyle="--", alpha=0.75)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    fig.subplots_adjust(left=0.25, right=0.97, bottom=bottom, top=0.97)
    fig.tight_layout(pad=0.8)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.show()
    return png_path, pdf_path
    return png_path, pdf_path