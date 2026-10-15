"""Main-paper figure for EC persona-representation sensitivity."""

from collections import Counter
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


FORMATS = ["BULLET_ORIGINAL", "NARRATIVE_PROSE"]
SEEDS = list(range(30))
ROUNDS = list(range(1, 16))
TARGET_ACTIONS = {"REPOST", "REPLY"}
PRIMARY_COLOR = "#E83E68"
MODEL_STYLES = {
    "GPT-5.2": {"results_dir": "gpt-5.2", "color": PRIMARY_COLOR, "marker": "o"},
    "Claude Haiku 4.5": {
        "results_dir": "claude-haiku-4.5",
        "color": "#168AAD",
        "marker": "s",
    },
    "Gemini 2.5 Flash": {
        "results_dir": "gemini-2.5-flash",
        "color": "#D97706",
        "marker": "^",
    },
    "DeepSeek V4 Flash": {
        "results_dir": "DeepSeek-V4-Flash",
        "color": "#2A9D8F",
        "marker": "D",
    },
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260925


def _bootstrap_mean(values: np.ndarray, generator: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    draws = generator.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    bootstrap_means = values[draws].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _bootstrap_rounds(values: np.ndarray, generator: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    draws = generator.integers(0, values.shape[0], size=(BOOTSTRAP_REPLICATES, values.shape[0]))
    bootstrap_means = values[draws, :].mean(axis=1)
    return (
        values.mean(axis=0),
        np.quantile(bootstrap_means, 0.025, axis=0),
        np.quantile(bootstrap_means, 0.975, axis=0),
    )


def _t_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    low, high = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
    return mean, float(low), float(high)


def _weighted_stance_assortativity(edge_counts: Counter, stance: dict[str, int]) -> float:
    if not edge_counts:
        return np.nan
    sources = np.array([stance[source] for source, _ in edge_counts], dtype=float)
    targets = np.array([stance[target] for _, target in edge_counts], dtype=float)
    weights = np.array(list(edge_counts.values()), dtype=float)
    source_mean = np.average(sources, weights=weights)
    target_mean = np.average(targets, weights=weights)
    covariance = np.average((sources - source_mean) * (targets - target_mean), weights=weights)
    source_variance = np.average((sources - source_mean) ** 2, weights=weights)
    target_variance = np.average((targets - target_mean) ** 2, weights=weights)
    if source_variance == 0 or target_variance == 0:
        return np.nan
    return float(covariance / np.sqrt(source_variance * target_variance))


def _cumulative_edges(data: pd.DataFrame, through_round: int) -> Counter:
    targeted = data.loc[
        data["round"].le(through_round)
        & data["action"].isin(TARGET_ACTIONS)
        & data["target_author_id"].notna()
    ]
    return Counter(zip(targeted["agent_id"], targeted["target_author_id"]))


def _load_raw_paired_results(
    data_dir: Path,
    results_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    agent_frames = []
    summary_frames = []
    for persona_format in FORMATS:
        for seed in SEEDS:
            condition_dir = "bullet" if persona_format == "BULLET_ORIGINAL" else "prose"
            run_dir = data_dir / "persona" / results_dir / condition_dir / f"seed{seed}"
            agent_frames.append(pd.read_csv(run_dir / "agent_rounds.csv"))
            summary_frames.append(pd.read_csv(run_dir / "round_summary.csv"))
    agent_rounds = pd.concat(agent_frames, ignore_index=True)
    round_summary = pd.concat(summary_frames, ignore_index=True)

    key_columns = ["seed", "round", "agent_id"]
    bullet_keys = agent_rounds.loc[agent_rounds["persona_format"].eq(FORMATS[0]), key_columns]
    narrative_keys = agent_rounds.loc[agent_rounds["persona_format"].eq(FORMATS[1]), key_columns]
    assert not bullet_keys.duplicated().any() and not narrative_keys.duplicated().any()
    assert set(map(tuple, bullet_keys.to_numpy())) == set(map(tuple, narrative_keys.to_numpy()))
    assert sorted(agent_rounds["seed"].unique()) == SEEDS
    assert sorted(agent_rounds["round"].unique()) == ROUNDS

    repo_root = data_dir.parents[2]
    with (repo_root / "code" / "EC" / "inputs" / "ai_regulation_minimal_personas_100.json").open() as file:
        personas = json.load(file)
    stance = {row["agent_id"]: int(row["ai_regulation_stance"]) for row in personas}
    return agent_rounds, round_summary, stance


def _compute_paired_metrics(
    agent_rounds: pd.DataFrame,
    round_summary: pd.DataFrame,
    stance: dict[str, int],
    *,
    condition_column: str = "persona_format",
    conditions: tuple[str, str] = (FORMATS[0], FORMATS[1]),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key_columns = ["seed", "round", "agent_id"]
    bullet = agent_rounds.loc[agent_rounds[condition_column].eq(conditions[0])].copy()
    narrative = agent_rounds.loc[agent_rounds[condition_column].eq(conditions[1])].copy()
    paired = bullet[key_columns + ["action", "target_author_id"]].merge(
        narrative[key_columns + ["action", "target_author_id"]],
        on=key_columns,
        suffixes=("_bullet", "_narrative"),
        validate="one_to_one",
    )
    paired["action_agreement"] = paired["action_bullet"].eq(paired["action_narrative"])
    paired["target_agreement"] = paired["target_author_id_bullet"].eq(paired["target_author_id_narrative"])
    agreement_by_seed = paired.groupby("seed", as_index=False).agg(
        action_agreement=("action_agreement", "mean"),
        target_agreement=("target_agreement", "mean"),
    )

    targeting_by_seed_round = (
        round_summary.pivot(
            index=["seed", "round"],
            columns=condition_column,
            values="cross_group_interaction_ratio",
        )
        .reset_index()
        .sort_values(["seed", "round"])
    )
    targeting_by_seed_round["difference"] = (
        targeting_by_seed_round[conditions[1]] - targeting_by_seed_round[conditions[0]]
    )

    network_rows = []
    for seed in SEEDS:
        seed_data = {
            persona_format: agent_rounds.loc[
                agent_rounds["seed"].eq(seed) & agent_rounds[condition_column].eq(persona_format)
            ]
            for persona_format in conditions
        }
        for round_index in ROUNDS:
            bullet_assortativity = _weighted_stance_assortativity(
                _cumulative_edges(seed_data[conditions[0]], round_index), stance
            )
            narrative_assortativity = _weighted_stance_assortativity(
                _cumulative_edges(seed_data[conditions[1]], round_index), stance
            )
            network_rows.append(
                {
                    "seed": seed,
                    "round": round_index,
                    "difference": narrative_assortativity - bullet_assortativity,
                }
            )
    network_by_seed_round = pd.DataFrame(network_rows)
    return agreement_by_seed, targeting_by_seed_round, network_by_seed_round


def plot_ec_representation_sensitivity(
    data_dir: Path,
    output_dir: Path,
) -> list[tuple[Path, Path]]:
    model_results = {}
    for model_index, (model, style) in enumerate(MODEL_STYLES.items()):
        agent_rounds, round_summary, stance = _load_raw_paired_results(
            data_dir, str(style["results_dir"])
        )
        agreements, targeting, network = _compute_paired_metrics(
            agent_rounds, round_summary, stance
        )
        agreement_generator = np.random.default_rng(BOOTSTRAP_SEED + 10 * model_index)
        agreement_results = {
            metric: _bootstrap_mean(agreements[metric].to_numpy(), agreement_generator)
            for metric in ["action_agreement", "target_agreement"]
        }
        targeting_matrix = (
            targeting.pivot(index="seed", columns="round", values="difference")
            .loc[SEEDS, ROUNDS]
            .to_numpy()
        )
        network_matrix = (
            network.pivot(index="seed", columns="round", values="difference")
            .loc[SEEDS, ROUNDS]
            .to_numpy()
        )
        targeting_summary = _bootstrap_rounds(
            targeting_matrix, np.random.default_rng(BOOTSTRAP_SEED + 10 * model_index + 1)
        )
        network_summary = _bootstrap_rounds(
            network_matrix, np.random.default_rng(BOOTSTRAP_SEED + 10 * model_index + 2)
        )
        overall_targeting = (
            targeting.loc[targeting["round"].gt(1)]
            .groupby("seed")["difference"]
            .mean()
            .loc[SEEDS]
        )
        model_results[model] = {
            "agreement": agreement_results,
            "targeting": targeting_summary,
            "network": network_summary,
            "overall": _t_interval(overall_targeting.to_numpy()),
            "final": _t_interval(network_matrix[:, -1]),
        }

    gpt_results = model_results["GPT-5.2"]
    assert np.allclose(gpt_results["overall"], [0.051474, 0.042242, 0.060705], atol=5e-7)
    assert np.allclose(gpt_results["final"], [-0.193880, -0.220352, -0.167407], atol=5e-7)

    return _plot_paired_sensitivity_panels(model_results, output_dir, "ec_persona_representation")


def _plot_paired_sensitivity_panels(model_results, output_dir, output_stem):
    output_dir.mkdir(parents=True, exist_ok=True)
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
        saved_paths = []

        figure, agreement_axis = plt.subplots(figsize=(4.5, 4.5), dpi=300)

        agreement_specs = [
            ("action_agreement", 1, "Action-type\nagreement"),
            ("target_agreement", 0, "Interaction-target\nagreement"),
        ]
        model_offsets = {
            "GPT-5.2": 0.18,
            "Claude Haiku 4.5": 0.06,
            "Gemini 2.5 Flash": -0.06,
            "DeepSeek V4 Flash": -0.18,
        }
        for metric, y_position, _ in agreement_specs:
            for model, style in MODEL_STYLES.items():
                estimate, ci_low, ci_high = model_results[model]["agreement"][metric]
                point_y = y_position + model_offsets[model]
                agreement_axis.errorbar(
                    100 * estimate,
                    point_y,
                    xerr=[[100 * (estimate - ci_low)], [100 * (ci_high - estimate)]],
                    fmt=str(style["marker"]),
                    color=str(style["color"]),
                    markersize=7.5,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    elinewidth=2.0,
                    capsize=4,
                    capthick=1.5,
                    label=model if metric == "action_agreement" else None,
                    zorder=3,
                )
        agreement_axis.set_xlim(0, 105)
        agreement_axis.set_xticks([0, 20, 40, 60, 80, 100])
        agreement_axis.set_ylim(-0.38, 1.38)
        agreement_axis.set_yticks(
            [spec[1] for spec in agreement_specs],
            labels=[spec[2] for spec in agreement_specs],
        )
        agreement_axis.set_xlabel("Agreement across paired seeds (%)")
        agreement_axis.set_ylabel("")
        agreement_axis.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(0.01, 0.5),
            handletextpad=0.3,
            labelspacing=0.3,
        )
        saved_paths.append(
            _finish_panel(
                figure,
                agreement_axis,
                output_dir,
                f"{output_stem}_agreement",
                left=0.35,
            )
        )

        rounds = np.asarray(ROUNDS)
        figure, targeting_axis = plt.subplots(figsize=(4.5, 4.5), dpi=300)
        for model, style in MODEL_STYLES.items():
            targeting_mean, targeting_low, targeting_high = model_results[model]["targeting"]
            targeting_axis.fill_between(
                rounds, 100 * targeting_low, 100 * targeting_high,
                color=str(style["color"]), alpha=0.10, linewidth=0, zorder=1,
            )
            targeting_axis.plot(
                rounds, 100 * targeting_mean, color=str(style["color"]),
                marker=str(style["marker"]), markersize=5.0, linewidth=2.0,
                label=model, zorder=3,
            )
        targeting_axis.axhline(0, color="#777777", linewidth=1.4, linestyle="--", zorder=2)
        targeting_axis.set_xlim(1, 15)
        targeting_axis.set_xticks([1, 3, 6, 9, 12, 15])
        targeting_axis.set_xlabel("Round")
        targeting_axis.set_ylabel(r"Cross-group targeting difference $\Delta_{\mathrm{target},t}$ (pp)")
        saved_paths.append(
            _finish_panel(
                figure,
                targeting_axis,
                output_dir,
                f"{output_stem}_targeting",
            )
        )

        figure, network_axis = plt.subplots(figsize=(4.5, 4.5), dpi=300)
        for model, style in MODEL_STYLES.items():
            network_mean, network_low, network_high = model_results[model]["network"]
            network_axis.fill_between(
                rounds, network_low, network_high,
                color=str(style["color"]), alpha=0.10, linewidth=0, zorder=1,
            )
            network_axis.plot(
                rounds, network_mean, color=str(style["color"]),
                marker=str(style["marker"]), markersize=5.0, linewidth=2.0,
                label=model, zorder=3,
            )
        network_axis.axhline(0, color="#777777", linewidth=1.4, linestyle="--", zorder=2)
        network_axis.set_xlim(1, 15)
        network_axis.set_xticks(range(1, 16))
        network_axis.set_xlabel("Round")
        network_axis.set_ylabel(r"Weighted stance-assortativity difference $\Delta_{r,t}$")
        saved_paths.append(
            _finish_panel(
                figure,
                network_axis,
                output_dir,
                f"{output_stem}_collective_divergence",
            )
        )

    return saved_paths


def plot_ec_design_perturbation_sensitivity(data_dir: Path, output_dir: Path):
    with (data_dir.parents[2] / "code" / "EC" / "inputs" / "ai_regulation_minimal_personas_100.json").open() as file:
        personas = json.load(file)
    stance = {persona["agent_id"]: int(persona["ai_regulation_stance"]) for persona in personas}
    designs = [
        ("Activation probability", "activation_0.5/BULLET_ORIGINAL", "activation_probability", "activation", "0.3", "0.5"),
        ("Memory window", "memory4", "memory_window", "memory", "2", "4"),
        ("Recommendation feed size", "feed10", "recommendation_feed_size", "feed_size", "5", "10"),
    ]
    saved_paths = []
    trajectory_rows = []
    agreement_rows = []
    for perturbation, higher_condition, output_stem, dimension, lower_dir, higher_dir in designs:
        model_results = {}
        for model_index, (model, style) in enumerate(MODEL_STYLES.items()):
            results_dir = str(style["results_dir"])
            agent_frames = []
            summary_frames = []
            for condition in ["BULLET_ORIGINAL", higher_condition]:
                for seed in SEEDS:
                    condition_dir = lower_dir if condition == "BULLET_ORIGINAL" else higher_dir
                    run_dir = data_dir / dimension / results_dir / condition_dir / f"seed{seed}"
                    metadata = pd.read_csv(run_dir / "run_metadata.csv").iloc[0]
                    assert int(metadata["seed"]) == seed and int(metadata["rounds"]) == 15
                    assert float(metadata["activation_prob"]) == (0.5 if condition.startswith("activation") else 0.3)
                    assert int(metadata["memory_rounds"]) == (4 if condition == "memory4" else 2)
                    assert int(metadata["feed_size"]) == (10 if condition == "feed10" else 5)
                    agents = pd.read_csv(run_dir / "agent_rounds.csv")
                    summary = pd.read_csv(run_dir / "round_summary.csv")
                    assert sorted(agents["round"].unique()) == ROUNDS
                    assert sorted(summary["round"].tolist()) == ROUNDS
                    agent_frames.append(agents.assign(seed=seed, condition=condition))
                    summary_frames.append(summary.assign(seed=seed, condition=condition))
            agreements, targeting, network = _compute_paired_metrics(
                pd.concat(agent_frames, ignore_index=True),
                pd.concat(summary_frames, ignore_index=True),
                stance, condition_column="condition",
                conditions=("BULLET_ORIGINAL", higher_condition),
            )
            agreements = agreements.set_index("seed").loc[SEEDS]
            agreement_generator = np.random.default_rng(BOOTSTRAP_SEED + 10 * model_index)
            agreement_results = {}
            for metric in ["action_agreement", "target_agreement"]:
                estimate = _bootstrap_mean(agreements[metric].to_numpy(), agreement_generator)
                agreement_results[metric] = estimate
                agreement_rows.append({
                    "perturbation": perturbation, "model": model, "metric": metric,
                    "mean": estimate[0], "ci_low": estimate[1], "ci_high": estimate[2], "n_pairs": 30,
                })
            model_results[model] = {"agreement": agreement_results}
            for offset, (metric, values) in enumerate([("targeting", targeting), ("network", network)], 1):
                matrix = values.pivot(index="seed", columns="round", values="difference").loc[SEEDS, ROUNDS].to_numpy()
                assert matrix.shape == (30, 15) and np.isfinite(matrix[:, 1:]).all()
                mean, ci_low, ci_high = _bootstrap_rounds(
                    matrix, np.random.default_rng(BOOTSTRAP_SEED + 10 * model_index + offset)
                )
                model_results[model][metric] = (mean, ci_low, ci_high)
                for round_index, round_number in enumerate(ROUNDS):
                    trajectory_rows.append({
                        "perturbation": perturbation, "model": model, "metric": metric,
                        "round": round_number, "signed_effect": mean[round_index],
                        "ci_low": ci_low[round_index], "ci_high": ci_high[round_index],
                        "n_pairs": int(np.isfinite(matrix[:, round_index]).sum()),
                        "unit": "proportion" if metric == "targeting" else "assortativity",
                    })
        saved_paths.extend(_plot_paired_sensitivity_panels(model_results, output_dir, f"ec_design_{output_stem}"))
    pd.DataFrame(agreement_rows).to_csv(output_dir.parent / "ec_design_perturbation_agreements.csv", index=False)
    pd.DataFrame(trajectory_rows).to_csv(output_dir.parent / "ec_design_perturbation_trajectories.csv", index=False)
    assert len(saved_paths) == 9
    return saved_paths


def _finish_panel(
    figure: plt.Figure,
    axis: plt.Axes,
    output_dir: Path,
    output_stem: str,
    *,
    left: float = 0.21,
) -> tuple[Path, Path]:
    axis.grid(True, color="#C7C7C7", linewidth=0.8, linestyle="--", alpha=0.75)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    figure.subplots_adjust(left=left, right=0.96, bottom=0.16, top=0.96)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.show()
    return png_path, pdf_path
