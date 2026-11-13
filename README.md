# The Butterfly Effect in LLM Social Simulations

LLM-based social simulations offer a way to explore how individual decisions give rise to collective behavior. But their outcomes may also depend on seemingly minor choices in how agents are described and how interactions are organized. Understanding this sensitivity is essential for deciding which conclusions a simulation can support.

This project studies how such choices affect agent behavior and collective outcomes in repeated Prisoner's Dilemma (PD) games and social-media echo chambers (EC). The repository includes simulation code, inputs, recorded results, and visualization tools to support reproducible investigation.

## Repository Structure

| Path | Contents |
| --- | --- |
| [code/PD/](code/PD/README.md) | Persona, framing, and memory experiments with fixed opponents or two LLM agents |
| [code/EC/](code/EC/README.md) | Echo-chamber simulator, paired experiment runner, and offline tests |
| [code/EC/inputs/](code/EC/inputs/README.md) | Synthetic personas, network inputs, and network provenance |
| [code/plots/](code/plots/) | Plotting functions used by the paper notebook and standalone figure scripts |
| [code/analysis/](code/analysis/) | Standalone analyses for the Tit-for-Tat reference and the single-perturbation test |
| [additional_experiments/](additional_experiments/) | Scripts and results for the additional experiments listed below |
| [data/results/](data/results/) | Recorded PD and EC runs |
| [data/plots_and_tables/](data/plots_and_tables/) | Paper notebook, summary CSVs, appendix tables, and figures |
| [audit-card.html](audit-card.html) | Social Simulation Audit Card; open directly in a browser |

## Install

Use Python 3.10 or later. Run commands below from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Inspect and Reproduce Results

These commands do not call model APIs or require credentials:

```bash
python -m unittest discover -s code/EC/tests -v
python code/EC/run_experiments.py persona --model-family gpt --seeds 0 --dry-run
jupyter lab data/plots_and_tables/paper_visualizations.ipynb
```

Run the notebook cells in order, using a kernel started at the repository root or a subdirectory. It displays bundled sensitivity and appendix tables, regenerates directional pairwise tables, and computes EC figure metrics and bootstrap intervals from recorded runs. PD figures use bundled amplification summaries. Outputs are 15 PNG/PDF figure pairs in `data/plots_and_tables/figs/`, with generated table and EC design CSVs in `data/plots_and_tables/`.

## Included Experiments and Data

| Simulation | Recorded comparisons | Coverage |
| --- | --- | --- |
| PD | Three persona formats, three game framings, four memory representations; four fixed opponent policies and a two-agent setting | Four models, seeds 0-29, ten rounds; 6,000 seed files |
| EC | Bullet/prose personas, activation 0.3/0.5, memory 2/4 rounds, feed size 5/10 | Four models, 30 paired seeds, 15 rounds; 960 runs |

Models are GPT-5.2, Claude Haiku 4.5, Gemini 2.5 Flash, and DeepSeek (V3.2 for PD; V4 Flash for EC). The PD payoff matrix is CC = (3, 3), CD = (0, 5), DC = (5, 0), DD = (1, 1). EC uses 100 synthetic agents with fixed AI-regulation stances.

Results are organized by dimension, model, condition, and seed; PD also separates interaction settings and fixed opponents. Distributed PD files retain actions, payoffs, and parsing status. EC files retain decisions, interaction targets, messages, reasons, exposure metrics, and run metadata. Full prompts and raw model responses are omitted from distributed decision files.

Historical schemas differ: 1,500 PD files identify their model only through the directory, and 30 EC metadata files omit `persona_format`. Network homophily and hub-assignment runs are bundled in `data/results/EC/homophily/` and `data/results/EC/hub_assignment/` (5-30 runs per condition); `python code/plots/plot_homophily_hub_metrics.py` recomputes their stance assortativity and same-group edge ratio and writes the Appendix D.6 panels and test tables to `data/plots_and_tables/`. The EC runner's `paper` and `recorded` profiles select different HIGH networks; see the [network provenance](code/EC/inputs/networks/homophily/paper_high_assortativity/PROVENANCE.md).

## Additional Experiments

These experiments extend the main results (paper Appendices D and F). Scripts read the bundled data and, where noted, call model APIs with the same prompts, payoffs, and sampling settings as the main runs.

| Experiment | Paper | Script | Results |
| --- | --- | --- | --- |
| Tit-for-Tat as the fixed-opponent reference ($A_{\mathrm{TfT}}$) | Sec. 4.2, App. D.1 | `code/analysis/pd_tft_reference_gap.py` | printed to stdout |
| Neutral and cooperative personas in the three formats | Sec. 4.3, App. D.2 | `code/PD/persona_*_agent.py --condition-set all` | `additional_experiments/results/PD_other_formats/` |
| Four additional game descriptions | App. D.2 | `code/PD/framing_*_agent.py --condition-set all` | `additional_experiments/results/PD_other_framings/` |
| Figure for both D.2 experiments | App. D.2 | `code/plots/plot_pd_alt_implementations.py` | `data/plots_and_tables/figs/pd_other_personas.pdf`, `pd_game_descriptions.pdf` |
| Amplification forest plot and single-perturbation scatter | App. D.1, D.3 | `code/plots/plot_pd_appendix_summaries.py` | `data/plots_and_tables/figs/pd_amplification_forest.pdf`, `pd_single_perturbation_scatter.pdf` (estimates: `data/plots_and_tables/amplification_old_vs_new.csv`, `additional_experiments/results/PD_single_perturbation/analysis/summary.csv`) |
| Stated-reason coding and forced-opening figure | App. D.4 | `additional_experiments/reason_analysis.py`, then `code/plots/plot_pd_forced_opening.py` | `additional_experiments/results/PD_reason_analysis/reason_summary.csv`, `data/plots_and_tables/figs/pd_forced_opening.pdf` |
| Single-perturbation dyads (one agent perturbed) | Sec. 4.2, App. D.3 | `additional_experiments/single_perturbation.py`, `analyze_single_perturbation_wrapper.py`, `single_perturbation_extra.py`; test in `code/analysis/pd_single_perturbation_spillover_test.py` | `additional_experiments/results/PD_single_perturbation/` |
| Round-1 gap vs. interaction effect gap | Sec. 4.3 (Figure 5), App. D.4 | `code/plots/plot_pd_round1_mechanism.py` | `data/plots_and_tables/pd_round1_mechanism.csv` |
| Forced openings (round 1 fixed to CC or DD) | App. D.4 | `additional_experiments/forced_opening.py` | `additional_experiments/results/PD_reason_analysis/forced_opening/` |
| Action-token probabilities | App. D.4 | `additional_experiments/logit_policy.py` | `additional_experiments/results/PD_logit_policy/` |
| Echo-chamber same-condition replicates (noise baseline) | Sec. 4.2 (Figure 4a), App. D.5 | `additional_experiments/ec_replicate_agreement.py`, `ec_replicate_outcomes.py`; figure in `code/plots/plot_main_representation_figure.py` | `additional_experiments/results/EC_replicates/` |

Replicates repeat the BULLET persona condition with seed 0 and the paper configuration (activation probability 0.3): 30 runs for GPT-5.2 and 5 runs for each other model. Forced-opening and single-perturbation runs keep full prompts and raw model responses.

```bash
python code/analysis/pd_tft_reference_gap.py
python code/plots/plot_pd_round1_mechanism.py
python additional_experiments/ec_replicate_agreement.py
python additional_experiments/ec_replicate_outcomes.py gpt-5.2 gpt-5.2
python additional_experiments/forced_opening.py analyze --models gpt-5.2 claude-haiku-4-5-20251001 gemini-2.5-flash deepseek-v4-flash
```

## Run New Simulations

Model execution requires API access and incurs provider charges. Configure the relevant template:

| Simulation | Configuration | Variables |
| --- | --- | --- |
| PD | Copy `.env.example` to `.env` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME` |
| EC | Copy `code/EC/.env.example` to `code/EC/.env` | `OPENAI_ENDPOINT`, `OPENAI_API_KEY` |

Environment variables can also be supplied directly. Example small runs:

```bash
python code/PD/persona_single_agent.py --seed-count 1 --horizon 2
python code/EC/run_experiments.py persona --model-family gpt --seeds 0
```

New simulation outputs default to `data/results/generated/`, separate from bundled results and excluded from Git. See the [PD guide](code/PD/README.md) and [EC guide](code/EC/README.md) for conditions, request settings, and output options.
