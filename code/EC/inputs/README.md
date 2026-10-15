# EC simulation inputs

This directory contains the personas and networks used by the EC simulation. The runner selects input files according to the experiment dimension and network profile.

## Files

| File or directory | Contents and use |
| --- | --- |
| `ai_regulation_minimal_personas_100.json` | 100 synthetic agents with ID, bio, follower count, and fixed AI-regulation stance; used in every run |
| `networks/homophily/` | Low, medium, and high homophily networks |
| `networks/hub_assignment/` | Random, mixed, pro, and anti hub-assignment networks; also includes the optional follower-hub condition |
| `degree_sequence.json`, `*degree_assignment.json` | Network degree construction records |
| `graph_summary.csv` | Network measurements |

## Input selection

The runner uses each condition's `_01.edgelist`. Files numbered `_02`–`_05` provide alternative graph realizations. Persona, activation, memory, and feed-size experiments use the medium-homophily network.

For the HIGH condition, `--profile paper` selects `homophily/paper_high_assortativity/`, a regenerated network with numeric stance assortativity 0.26044. `--profile recorded` selects `homophily/high_assortativity/`, whose first graph has assortativity 0.38612. See the [HIGH network construction details](networks/homophily/paper_high_assortativity/PROVENANCE.md).

To inspect selected paths without running a simulation, execute from the `EC/` directory:

```bash
python run_experiments.py all --dry-run
```

Network summaries also include measurements for variants not bundled in this directory.
