"""Package historical network runs without prompts or raw model responses."""
import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "gpt-5.2": "EC_results_gpt",
    "claude-haiku-4.5": "EC_results_claude/echo_chamber_claude/claude-haiku-4-5-20251001",
    "gemini-2.5-flash": "EC_results_gemini/echo_chamber_gemini/gemini-2.5-flash",
    "deepseek-v3.2": "EC_results_deepseek/echo_chamber_deepseek/deepseek-v3.2",
}
CONDITIONS = {
    "homophily": {"low": "low_assortativity", "medium": "medium_assortativity",
                  "high": "high_assortativity"},
    "hub_assignment": {name: name for name in
                       ("random", "mixed_hubs", "pro_hubs", "anti_hubs")},
}
OMIT = {"bio", "agent_prompt_input", "agent_raw_output"}
FILES = ("agent_rounds.csv", "round_summary.csv", "run_metadata.csv")


def export_results(source_root: Path, destination: Path) -> list[dict]:
    manifest = []
    for model, relative_root in SOURCES.items():
        for dimension, conditions in CONDITIONS.items():
            for condition, source_condition in conditions.items():
                source = source_root / relative_root / source_condition
                runs = sorted(source.glob("*/agent_rounds.csv"))
                if not runs:
                    raise ValueError(f"No recorded runs in {source}")
                for run_index, agent_file in enumerate(runs):
                    run = agent_file.parent
                    target = destination / dimension / model / condition / run.name
                    target.mkdir(parents=True, exist_ok=True)
                    metadata = None
                    for filename in FILES:
                        with (run / filename).open(newline="") as handle:
                            reader = csv.DictReader(handle)
                            fields = [field for field in reader.fieldnames if field not in OMIT]
                            rows = [{key: row[key] for key in fields} for row in reader]
                        if filename == "run_metadata.csv":
                            if len(rows) != 1:
                                raise ValueError(f"Expected one metadata row: {run}")
                            metadata = rows[0].copy()
                            for field in ("edgelist", "personas"):
                                if field in rows[0]:
                                    basename = Path(rows[0][field].replace("\\", "/")).name
                                    input_root = ROOT / "code/EC/inputs"
                                    matches = list(input_root.rglob(basename))
                                    if len(matches) != 1:
                                        raise ValueError(f"Expected one bundled input for {basename}: {matches}")
                                    rows[0][field] = matches[0].relative_to(ROOT).as_posix()
                        with (target / filename).open("w", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=fields)
                            writer.writeheader()
                            writer.writerows(rows)
                    manifest.append(dict(
                        dimension=dimension, model=model, condition=condition,
                        run_id=run.name, source_run=str(run.relative_to(source_root)),
                        seed=metadata.get("seed", ""),
                        recorded_model=metadata.get("model", ""),
                        rounds=metadata.get("rounds", ""),
                        activation_prob=metadata.get("activation_prob", ""),
                        original_edgelist=metadata.get("edgelist", ""),
                        network_matches_condition=Path(metadata["edgelist"].replace("\\", "/")).parent.name
                            == source_condition,
                        historical_first10=run_index < 10,
                    ))
    with (destination / "network_results_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Original EC_data directory; needed only for repackaging")
    parser.add_argument("--out-root", type=Path, default=ROOT / "data/EC")
    args = parser.parse_args()
    manifest = export_results(args.source_root, args.out_root)
    print(f"Exported {len(manifest)} network runs to {args.out_root}")


if __name__ == "__main__":
    main()