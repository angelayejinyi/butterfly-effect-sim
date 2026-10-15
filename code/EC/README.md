# Network Experiments

The bundled simulator and `network_generators_powerlaw.py` support the three
homophily conditions (`low`, `medium`, `high`) and four hub assignments
(`random`, `mixed_hubs`, `pro_hubs`, `anti_hubs`). Inspect their configurations
without credentials or model calls:

```bash
python code/EC/run_experiments.py homophily --profile recorded --seeds 0 --dry-run
python code/EC/run_experiments.py hub_assignment --profile recorded --seeds 0 --dry-run
```

Run these commands from the repository root. Remove `--dry-run` to run new
simulations with configured API credentials. `--profile recorded` selects the
legacy HIGH input; the default `paper` profile selects a reconstructed HIGH
network whose historical edge identity has not been verified. Neither profile
guarantees identical model responses to historical runs.

# Echo-chamber simulation

This directory implements a fixed-network social-media simulation with 100 agents and fixed AI-regulation stances. Each active agent receives a neighbor feed and chooses POST, REPOST, REPLY, or SILENT.

## Files

| File or directory | Purpose |
| --- | --- |
| `EC_experiment.py` | Simulation loop, prompt construction, feed selection, response parsing, and per-round logging |
| `run_experiments.py` | Condition selection and batch execution with paired seeds |
| `requirements.txt`, `.env.example` | Python dependencies and API configuration template |
| `inputs/` | Agent personas and network inputs; see the [input guide](inputs/README.md) |
| `network_generators_powerlaw.py` | Network generation with degree and assortativity controls |
| `run_all_assortativity.sh` | Shell entry point for the homophily experiment |
| `tests/test_experiments.py` | Offline checks for configurations, graph properties, prompts, and model request parameters |

## Experiments

| Dimension | Conditions |
| --- | --- |
| `persona` | Bullet / prose |
| `homophily` | Low / medium / high input-network homophily |
| `hub_assignment` | Random / mixed / pro / anti hubs |
| `activation` | 0.3 / 0.5 for all four models |
| `memory` | 2 / 4 rounds |
| `feed_size` | Up to 5 / 10 posts |

Defaults are 15 rounds and paired seeds 0–29. The baseline uses bullet personas, the medium-homophily network, activation probability 0.3, memory of 2 rounds, and a feed size of 5. The runner changes the selected dimension while keeping the other settings fixed.

The `paper` and `recorded` profiles select different HIGH network inputs. The default `paper` profile uses the regenerated graph in the manuscript's homophily band; its construction is documented in the [network provenance](inputs/networks/homophily/paper_high_assortativity/PROVENANCE.md).

## Verification and execution

Run the following commands inside this `EC/` directory. Inputs and API configuration are local to this directory; experiment outputs are centralized under `../../data/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_ENDPOINT` and `OPENAI_API_KEY` in this directory's `.env`, or supply them as environment variables. Model execution requires access to the configured API.

1. Inspect the experiment plan and an example prompt without API calls:

   ```bash
   python run_experiments.py all --model-family gpt --dry-run
   python EC_experiment.py --persona-format NARRATIVE_PROSE --dry-run
   ```

2. Run the offline checks:

   ```bash
   python -m unittest discover -s tests -v
   ```

3. Run a paired comparison for one seed:

   ```bash
   python run_experiments.py persona --model-family gpt --seeds 0
   ```

   Omit `--seeds 0` to use all 30 seeds. Replace `persona` with another dimension or `all`. Model families are `gpt`, `claude`, `gemini`, and `deepseek`; `--model` accepts a provider-specific model ID. Use `--help` for additional options.

## Outputs

Recorded activation, memory, and feed-size outputs are in [data/EC](../../data/EC), organized by dimension, model, condition, and historical run ID. Each condition contains run1-run5 only; the 30-run extension results are not bundled. All five runs record seed 0. Baseline activation is 0.3 for GPT and 0.35 for Claude, Gemini, and DeepSeek V3.2; the runner defaults above describe new simulations, not these historical runs. Source paths and parameters are listed in [the design manifest](../../data/EC/design_results_manifest.csv). Distributed decision CSVs omit full prompt, raw-response, and bio columns. Recorded homophily and hub-assignment outputs are also bundled separately.

New runs write to `data/generated/EC/` by default, separate from the supplied results. Each run saves its configuration, agent decisions and prompts, round summaries, metadata, and final agent state. Use `--out-root` (batch) or `--out-dir` (single run) to choose another destination.
