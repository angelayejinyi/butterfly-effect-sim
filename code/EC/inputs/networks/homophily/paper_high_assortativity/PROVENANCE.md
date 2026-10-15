# HIGH network construction

`paper_high_assortativity_01.edgelist` is a regenerated input for the HIGH homophily condition, selected by `--profile paper`. It was constructed using the bundled power-law network generator.

| Parameter | Value |
| --- | --- |
| Target numeric stance assortativity | 0.22–0.30 |
| Hub assignment | Mixed hubs |
| Target mean degree | 8 |
| Power-law exponent | 2.4 |
| Base seed / graph seed | 42 / 21042 |
| Nodes / edges | 100 / 411 |
| Resulting numeric stance assortativity | 0.2604407192952882 |

The network preserves the per-agent degrees of the bundled LOW and MEDIUM inputs. Its assortativity matches the historical summary entry `slightly_higher_assortativity_01`; historical edge identity has not been verified.

To generate networks from the `EC/` directory:

```bash
python network_generators_powerlaw.py
```

The HIGH output is written under `../../data/generated/networks/homophily/paper_high_assortativity/`.
