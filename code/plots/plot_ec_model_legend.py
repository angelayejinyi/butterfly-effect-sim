"""Shared model legend for the echo-chamber design-perturbation figures (Appendix D.5).

Colors and markers mirror MODEL_STYLES in plot_main_representation_figure.py.
Usage: python code/plots/plot_ec_model_legend.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[2] / "data" / "plots_and_tables" / "figs"
STYLES = [("GPT-5.2", "#E83E68", "o"), ("Claude Haiku 4.5", "#168AAD", "s"),
          ("Gemini 2.5 Flash", "#D97706", "^"), ("DeepSeek V4 Flash", "#2A9D8F", "D")]


def main():
    plt.rcParams.update({"font.family": ["Verdana", "DejaVu Sans"], "font.size": 9, "pdf.fonttype": 42})
    handles = [Line2D([], [], color=c, marker=m, ms=5, lw=2.0, label=n) for n, c, m in STYLES]
    handles.append(Line2D([], [], color="#777777", lw=1.4, ls="--", label="No difference"))
    fig = plt.figure(figsize=(7.0, 0.3))
    fig.legend(handles=handles, loc="center", ncol=5, frameon=False, handlelength=2.2, columnspacing=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "ec_model_legend.pdf", bbox_inches="tight", metadata={"Creator": None, "Producer": None})
    fig.savefig(OUT / "ec_model_legend.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    main()
