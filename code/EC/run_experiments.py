"""Run the six EC perturbation families using explicit paired seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from EC_experiment import run_simulation

EC_ROOT = Path(__file__).resolve().parent
DIMENSIONS = ("persona", "homophily", "hub_assignment", "activation", "memory", "feed_size")
MODEL_IDS = {
    "gpt": "gpt-5.2",
    "claude": "claude-haiku-4.5",
    "gemini": "gemini-2.5-flash",
    "deepseek": "DeepSeek-V4-Flash",
}


def conditions(dimension: str, model_family: str, profile: str) -> list[tuple[str, dict]]:
    """Only the named perturbation changes within a comparison family."""
    if dimension == "persona":
        return [(fmt, {"persona_format": fmt}) for fmt in ("BULLET_ORIGINAL", "NARRATIVE_PROSE")]
    if dimension == "homophily":
        levels = ("low_assortativity", "medium_assortativity",
                  "paper_high_assortativity" if profile == "paper" else "high_assortativity")
        return [(label, {"network": f"homophily/{folder}/{folder}_01.edgelist"})
                for label, folder in zip(("low", "medium", "high"), levels)]
    if dimension == "hub_assignment":
        return [(hub, {"network": f"hub_assignment/{hub}/{hub}_01.edgelist"})
                for hub in ("random", "mixed_hubs", "pro_hubs", "anti_hubs")]
    if dimension == "activation":
        low = 0.3  # All four models use 0.3 vs 0.5 (seeds 0–29).
        return [(f"activation_{v}", {"activation_prob": v}) for v in (low, 0.5)]
    if dimension == "memory":
        return [(f"memory{v}", {"memory_rounds": v}) for v in (2, 4)]
    if dimension == "feed_size":
        return [(f"feed{v}", {"feed_size": v}) for v in (5, 10)]
    raise ValueError(dimension)


def build_plan(args: argparse.Namespace) -> list[dict]:
    dimensions = DIMENSIONS if args.dimension == "all" else (args.dimension,)
    plan = []
    for dimension in dimensions:
        variants = conditions(dimension, args.model_family, args.profile)
        for seed in args.seeds:
            # Alternate condition execution order while retaining the same paired seed.
            for label, overrides in (variants if seed % 2 == 0 else list(reversed(variants))):
                config = dict(rounds=15, activation_prob=0.3, feed_size=5, memory_rounds=2,
                              workers=args.workers, seed=seed, temperature=0.3, top_p=0.95,
                              persona_format="BULLET_ORIGINAL", model=args.model or MODEL_IDS[args.model_family])
                config.update(overrides)
                network = config.pop("network", "homophily/medium_assortativity/medium_assortativity_01.edgelist")
                config["personas_path"] = str(args.personas.resolve())
                config["edgelist_path"] = str((args.network_root / network).resolve())
                config["out_dir"] = str((args.out_root / args.profile / args.model_family / dimension / label / f"seed{seed}").resolve())
                plan.append(config)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dimension", choices=(*DIMENSIONS, "all"))
    parser.add_argument("--model-family", choices=MODEL_IDS, default="gpt")
    parser.add_argument("--model", help="Override provider model ID, keeping the selected family configuration")
    parser.add_argument("--profile", choices=("paper", "recorded"), default="paper",
                        help="Network selection only: paper uses reconstructed HIGH; recorded uses legacy HIGH")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(30)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--personas", type=Path, default=EC_ROOT / "inputs/ai_regulation_minimal_personas_100.json")
    parser.add_argument("--network-root", type=Path, default=EC_ROOT / "inputs/networks")
    parser.add_argument("--out-root", type=Path, default=EC_ROOT.parents[1] / "data/generated/EC")
    parser.add_argument("--dry-run", action="store_true", help="Print the entire run plan; no credentials or API calls")
    args = parser.parse_args()
    if args.workers < 1 or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("workers must be positive; seeds must be distinct non-negative integers")
    plan = build_plan(args)
    for config in plan:
        for key in ("personas_path", "edgelist_path"):
            if not Path(config[key]).is_file():
                parser.error(f"Missing {key}: {config[key]}")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    for config in plan:
        out = Path(config["out_dir"])
        manifest = out / "experiment_config.json"
        if out.exists() and any(out.iterdir()):
            if not manifest.exists() or json.loads(manifest.read_text()) != config:
                raise ValueError(f"Existing output has a different or unknown configuration: {out}. Use a new --out-root.")
            if (out / "COMPLETE").exists() and all((out / name).is_file() for name in
                    ("agent_rounds.csv", "round_summary.csv", "run_metadata.csv", "final_agent_state.csv")):
                print(f"Skipping completed run: {out}")
                continue
        out.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(config, indent=2) + "\n")
        run_simulation(**config)
        (out / "COMPLETE").write_text("Simulation outputs saved.\n")


if __name__ == "__main__":
    main()
