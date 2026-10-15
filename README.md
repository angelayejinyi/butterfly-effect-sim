# The Butterfly Effect in LLM Social Simulations

Code and data for studying how agent representations and simulation settings affect outcomes in two environments: repeated Prisoner's Dilemma (PD) and a social-media echo chamber (EC).

## Repository structure

| Path | Contents |
| --- | --- |
| [code/PD/](code/PD/README.md) | Persona, framing, and memory experiments with fixed opponents and two LLM agents |
| [code/EC/](code/EC/README.md) | Echo-chamber simulation, experiment runner, and offline tests |
| [code/EC/inputs/](code/EC/inputs/README.md) | Synthetic agent personas and network inputs |
| [data/results/](data/results/) | Per-seed simulation results |
| [data/plots_and_tables/](data/plots_and_tables/) | Statistical summaries and appendix tables |
| [audit-card.html](audit-card.html) | Interactive Social Simulation Audit Card; open in a browser |

## Experiments

**Prisoner's Dilemma.** Agents choose Cooperate or Defect over 10 rounds. The payoff matrix is CC = (3, 3), CD = (0, 5), DC = (5, 0), and DD = (1, 1). In the fixed-opponent setting, an LLM plays against Tit-for-Tat, Random, Always-Cooperate, or Always-Defect. In the two-agent setting, two LLM players receive role-specific interaction histories. The experiments vary strategic persona format (plain, descriptive, tabular), game framing (canonical, moralized, risk), and memory representation (table or narrative, with or without summary statistics). Outcomes include cooperation rate and average payoff.

**Echo chamber.** One hundred agents interact for 15 rounds on a fixed social network. Each agent has a synthetic persona and a fixed stance toward AI regulation. Active agents view recent neighbor posts and choose POST, REPOST, REPLY, or SILENT. The paired comparisons vary persona format (bullet or prose), activation probability (0.3 or 0.5), memory window (2 or 4 rounds), and feed size (5 or 10 posts). Outcomes describe action and target agreement, exposure, and the structure of realized interactions.

Both experiments use seeds 0–29 across four model families: GPT-5.2, Claude Haiku 4.5, Gemini 2.5 Flash, and DeepSeek. PD uses DeepSeek V3.2; EC uses DeepSeek V4 Flash.

## Setup

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To execute simulations, configure the relevant environment file:

| Simulation | Configuration file | Variables |
| --- | --- | --- |
| PD | Copy `.env.example` to `.env` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME` |
| EC | Copy `code/EC/.env.example` to `code/EC/.env` | `OPENAI_ENDPOINT`, `OPENAI_API_KEY` |

These variables can also be supplied through the environment.

## Inspect and run

All commands below run from the repository root.

Inspect available options, an EC experiment plan, and offline checks:

```bash
python code/PD/persona_single_agent.py --help
python code/EC/run_experiments.py persona --model-family gpt --dry-run
python -m unittest discover -s code/EC/tests -v
```

Run a short PD example or one seed of the EC persona comparison:

```bash
python code/PD/persona_single_agent.py --seed-count 1 --horizon 2
python code/EC/run_experiments.py persona --model-family gpt --seeds 0
```

These simulation commands make model API calls. Full configurations use 30 seeds, with 10 rounds for PD and 15 rounds for EC. See the [PD guide](code/PD/README.md) and [EC guide](code/EC/README.md) for conditions, model selection, and output options. New outputs are written under `data/results/generated/`.

## Included results

| Simulation | Included conditions | Coverage |
| --- | --- | --- |
| PD | Three strategic persona formats, three game framings, four memory representations; fixed-opponent and two-agent settings | Four models, seeds 0–29, 10 rounds per seed |
| EC | Bullet/prose, activation 0.3/0.5, memory 2/4 rounds, feed size 5/10 | Four models, seeds 0–29, 15 rounds per seed |

Results are organized by experiment dimension, model, condition, and seed. PD CSVs contain actions, payoffs, and parsing status. EC CSVs contain agent decisions, interaction targets, messages, reasons, exposure measures, and run configuration. Statistical summaries and appendix tables are available in `data/plots_and_tables/`.
