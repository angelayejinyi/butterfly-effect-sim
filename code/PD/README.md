# Repeated Prisoner's Dilemma

This directory implements three prompt perturbations in two interaction settings: one LLM playing against a fixed policy, and two LLM-controlled players interacting with each other.

## Files and conditions

| Dimension | Fixed-opponent script | Two-agent script | Default conditions |
| --- | --- | --- | --- |
| Persona | `persona_single_agent.py` | `persona_two_agent.py` | Plain / descriptive / tabular strategic persona |
| Framing | `framing_single_agent.py` | `framing_two_agent.py` | Canonical / moralized / risk |
| Memory | `memory_single_agent.py` | `memory_two_agent.py` | Table / narrative history, each with or without summary statistics |

Each script contains its prompt definitions, simulation loop, response parser, and CSV output logic. The fixed-opponent scripts evaluate Tit-for-Tat, Random, Always-Cooperate, and Always-Defect. The two-agent scripts give each player a role-specific view of the preceding rounds.

Defaults are 10 rounds, 30 seeds, temperature 0.3, and top-p 0.95. The payoff matrix is CC = (3, 3), CD = (0, 5), DC = (5, 0), and DD = (1, 1). Prompt conditions and opponent policies are defined in code; no external input dataset is required.

## Verification and execution

Run the following commands from the repository root. Install dependencies with `pip install -r requirements.txt`. Configure `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_DEPLOYMENT_NAME` in `.env` or the environment.

1. Inspect the prompt definitions and available options:

   ```bash
   python code/PD/persona_single_agent.py --help
   python code/PD/persona_two_agent.py --help
   ```

2. Run a short configuration in each interaction setting:

   ```bash
   python code/PD/persona_single_agent.py --seed-count 1 --horizon 2 --out-file data/results/generated/PD/persona_single_check.csv
   python code/PD/persona_two_agent.py --seed-count 1 --horizon 2 --out-file data/results/generated/PD/persona_two_check.csv
   ```

   These commands make model API calls. The same options apply to the framing and memory scripts.

3. Run the default 30-seed, 10-round configuration:

   ```bash
   python code/PD/persona_single_agent.py --out-file data/results/generated/PD/persona_single.csv
   python code/PD/persona_two_agent.py --out-file data/results/generated/PD/persona_two.csv
   ```

   Replace the script names to evaluate framing or memory. Persona and framing default to `--condition-set paper`; `--condition-set all` includes additional conditions. Memory defaults to `--persona strategic` and also supports `--persona neutral`.

## Outputs

Supplied results are in [data/results/PD](../../data/results/PD), organized by dimension, model, interaction setting, condition, opponent policy where applicable, and seed. The distributed PD CSVs retain actions, payoffs, and parsing status, with full prompt and raw-response columns omitted. Each supplied condition includes seeds 0–29.

New runs default to `data/results/generated/PD/`. Each script writes a round-level CSV to `--out-file`, recording the condition, seed, actions, payoffs, prompts, model responses, and parsing status. Fixed-opponent outputs identify the opponent policy; two-agent outputs contain both players' decisions. Use these records to inspect individual trajectories and calculate condition-level outcomes.
