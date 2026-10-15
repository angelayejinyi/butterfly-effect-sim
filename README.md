# The Butterfly Effect in LLM Social Simulations

LLM-based social simulations offer a way to explore how individual decisions give rise to collective behavior. But their outcomes may also depend on seemingly minor choices in how agents are described and how interactions are organized. Understanding this sensitivity is essential for deciding which conclusions a simulation can support.

This project studies how such choices affect agent behavior and collective outcomes in repeated Prisoner's Dilemma (PD) games and social-media echo chambers (EC). The repository includes simulation code, inputs, recorded results, and visualization tools to support reproducible investigation.

## Repository Structure

| Path | Contents |
| --- | --- |
| [code/PD/](code/PD/README.md) | Persona, framing, and memory experiments with fixed opponents or two LLM agents |
| [code/EC/](code/EC/README.md) | Echo-chamber simulator, paired experiment runner, and offline tests |
| [code/EC/inputs/](code/EC/inputs/README.md) | Synthetic personas, network inputs, and network provenance |
| [code/plots/](code/plots/) | Plotting functions used by the paper notebook |
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

Historical schemas differ: 1,500 PD files identify their model only through the directory, and 30 EC metadata files omit `persona_format`. Network homophily and hub-assignment experiments are supported by the EC runner but have no bundled simulation results. Its `paper` and `recorded` profiles select different HIGH networks; see the [network provenance](code/EC/inputs/networks/homophily/paper_high_assortativity/PROVENANCE.md).

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
