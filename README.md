# Stop Drawing Scientific Claims from LLM Social Simulations Without Robustness Audits

Jinyi Ye<sup>1*</sup>, Lei Cao<sup>2*</sup>, Ding Chen<sup>3</sup>, Emilio Ferrara<sup>1,2</sup>

<sup>1</sup> Thomas Lord Department of Computer Science, University of Southern California<br>
<sup>2</sup> Annenberg School for Communication and Journalism, University of Southern California<br>
<sup>3</sup> Marshall School of Business, University of Southern California

<sup>*</sup> Equal contribution.<br>
Corresponding author: [jinyiy@usc.edu](mailto:jinyiy@usc.edu)

[![NeurIPS SocialAgent Workshop 2026](https://img.shields.io/badge/NeurIPS-SocialAgent_Workshop_2026-blue.svg)](https://neurips.cc/Conferences/2026)
[![arXiv](https://img.shields.io/badge/arXiv-2605.18890-b31b1b.svg)](https://arxiv.org/abs/2605.18890)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TL;DR:** Small changes in LLM social simulation design can change collective outcomes. We study this sensitivity in cooperation and polarization experiments and introduce TRAILS, a framework for auditing robustness before drawing scientific conclusions.

## Overview

This repository accompanies [*Stop Drawing Scientific Claims from LLM Social Simulations Without Robustness Audits*](https://arxiv.org/abs/2605.18890). It provides simulation code, inputs, and recorded results for two case studies:

- **Prisoner's Dilemma (PD):** repeated games with fixed opponent policies or two LLM agents, varying persona format, game framing, and memory representation.
- **Echo chambers (EC):** 100 synthetic social-media agents with fixed AI-regulation stances, varying network homophily, hub assignment, activation probability, feed size, and memory window.

The experiments compare GPT-5.2, Claude Haiku 4.5, Gemini 2.5 Flash, and DeepSeek V3.2 to examine the robustness of conclusions drawn from social simulations.

## Key Findings

- Small changes in persona format and game framing can substantially shift cooperation in repeated games.
- Network homophily and hub assignment affect polarization in social-media simulations.
- Sensitivity varies across models and design choices; robustness must be evaluated for each claim and model.

The paper introduces **TRAILS (Taxonomy for Robustness Audits In LLM Simulations)**, organizing audits across agent, interaction, and system design.

## Repository Structure

| Path | Contents |
| --- | --- |
| [code/PD/](code/PD/) | Fixed-opponent and two-agent game simulators |
| [code/EC/](code/EC/) | Social-media simulator and experiment runner |
| [code/EC/inputs/](code/EC/inputs/) | Synthetic personas and network inputs |
| [code/EC/tests/](code/EC/tests/) | Offline tests |
| [data/PD/](data/PD/) | Persona, framing, and memory results |
| [data/EC/](data/EC/) | Activation, feed-size, memory, homophily, and hub results |

See the [PD guide](code/PD/README.md), [EC guide](code/EC/README.md), and [input documentation](code/EC/inputs/README.md) for experiment options and input formats.

## Installation

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running a Simulation

### 1. Configure API Access

Copy the template for the simulation you want to run and fill in its variables. Live runs incur provider charges.

| Setting | Template | Variables |
| --- | --- | --- |
| PD | [.env.example](.env.example) | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME` |
| EC | [code/EC/.env.example](code/EC/.env.example) | `OPENAI_ENDPOINT`, `OPENAI_API_KEY` |

```bash
cp .env.example .env
cp code/EC/.env.example code/EC/.env
```

### 2. Preview an Experiment

Check an EC run configuration without calling the API:

```bash
python code/EC/run_experiments.py homophily --model-family gpt --seeds 0 --dry-run
```

### 3. Run a Simulation

Start with a short PD game or a single-seed EC comparison:

```bash
python code/PD/persona_single_agent.py --seed-count 1 --horizon 2
python code/EC/run_experiments.py homophily --model-family gpt --seeds 0
python code/EC/run_experiments.py hub_assignment --model-family gpt --seeds 0
```

Outputs are saved under `data/generated/`. Use `--model` to specify an EC provider model ID and `--seeds` to choose the runs.

## Data and Results

Recorded results are organized by experiment, model, condition, and run. PD files contain round-level actions and payoffs; EC files contain agent decisions, interactions, round summaries, and run metadata.

| Data | Contents | Coverage |
| --- | --- | --- |
| [PD results](data/PD) | Persona, framing, and memory comparisons; actions, payoffs, and parsing status | Four models, ten rounds; 6,000 files |
| [EC design results](data/EC) | Activation, feed-size, and memory comparisons; decisions, interactions, and round summaries | Four models, five runs per condition, 15 rounds; 120 runs |
| [EC homophily](data/EC/homophily) and [hub results](data/EC/hub_assignment) | Network-structure comparisons with run metadata | Four models; 195 runs |

## Tests

Run the offline EC tests without API credentials:

```bash
python -m unittest discover -s code/EC/tests -v
```

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{ye2026stop,
  title     = {Stop Drawing Scientific Claims from LLM Social Simulations Without Robustness Audits},
  author    = {Ye, Jinyi and Cao, Lei and Chen, Ding and Ferrara, Emilio},
  booktitle = {NeurIPS 2026 Workshop on Large Language Models for Social Reasoning and Simulation (SocialAgent)},
	year      = {2026},
	url       = {https://arxiv.org/abs/2605.18890}
}
```

## Contact

For questions about the code or experiments, contact Jinyi Ye at [jinyiy@usc.edu](mailto:jinyiy@usc.edu).

## License

The code is released under the [MIT License](LICENSE).
