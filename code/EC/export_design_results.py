"""Package the historical five-run design study, not the 30-seed extension."""
import argparse
import csv
import shutil
import tempfile
from pathlib import Path

from export_network_results import FILES, OMIT, ROOT, SOURCES


CONDITIONS = {
    "activation": ("medium_assortativity", "medium_assortativity_prob0.5"),
    "feed_size": ("medium_assortativity", "medium_assortativity_feedsize10"),
    "memory": ("medium_assortativity", "medium_assortativity_memrounds4"),
}
PARAMETERS = {"activation": "activation_prob", "feed_size": "feed_size",
              "memory": "memory_rounds"}


def export_results(source_root: Path, destination: Path, replace: bool = False) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    if not replace and any((destination / name).exists() for name in CONDITIONS):
        raise FileExistsError("Design results already exist; use --replace to replace only these three dimensions")
    manifest = []
    with tempfile.TemporaryDirectory(prefix="ec-five-run-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for model, relative_root in SOURCES.items():
            for dimension, conditions in CONDITIONS.items():
                for source_condition in conditions:
                    for run_number in range(1, 6):
                        run = source_root / relative_root / source_condition / f"medium_assortativity_01_run{run_number}"
                        with (run / "run_metadata.csv").open(newline="") as handle:
                            metadata_rows = list(csv.DictReader(handle))
                        if len(metadata_rows) != 1:
                            raise ValueError(f"Expected one metadata row: {run}")
                        metadata = metadata_rows[0]
                        baseline_activation = "0.3" if model == "gpt-5.2" else "0.35"
                        expected = dict(rounds="15", seed="0", activation_prob=baseline_activation,
                                        feed_size="5", memory_rounds="2")
                        if source_condition.endswith("prob0.5"):
                            expected["activation_prob"] = "0.5"
                        elif source_condition.endswith("feedsize10"):
                            expected["feed_size"] = "10"
                        elif source_condition.endswith("memrounds4"):
                            expected["memory_rounds"] = "4"
                        for field, value in expected.items():
                            if metadata.get(field) != value:
                                raise ValueError(f"Unexpected {field} in {run}: {metadata.get(field)!r}, expected {value!r}")
                        condition = metadata[PARAMETERS[dimension]]
                        target = staging / dimension / model / condition / run.name
                        target.mkdir(parents=True)
                        for filename in FILES:
                            with (run / filename).open(newline="") as handle:
                                reader = csv.DictReader(handle)
                                fields = [field for field in reader.fieldnames if field not in OMIT]
                                rows = [{field: row[field] for field in fields} for row in reader]
                            if not rows:
                                raise ValueError(f"Empty result: {run / filename}")
                            if filename == "run_metadata.csv":
                                for field in ("edgelist", "personas"):
                                    basename = Path(metadata[field].replace("\\", "/")).name
                                    matches = list((ROOT / "code/EC/inputs").rglob(basename))
                                    if len(matches) != 1:
                                        raise ValueError(f"Expected one bundled input for {basename}: {matches}")
                                    rows[0][field] = matches[0].relative_to(ROOT).as_posix()
                            with (target / filename).open("w", newline="") as handle:
                                writer = csv.DictWriter(handle, fieldnames=fields)
                                writer.writeheader()
                                writer.writerows(rows)
                        manifest.append(dict(
                            dimension=dimension, model=model, condition=condition,
                            run_id=run.name, source_run=run.relative_to(source_root).as_posix(),
                            seed=metadata["seed"], rounds=metadata["rounds"],
                            activation_prob=metadata["activation_prob"],
                            feed_size=metadata["feed_size"], memory_rounds=metadata["memory_rounds"],
                            original_edgelist=metadata["edgelist"],
                        ))
        for dimension in CONDITIONS:
            target = destination / dimension
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(staging / dimension), str(target))
        with (destination / "design_results_manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=ROOT / "data/EC")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    rows = export_results(args.source_root, args.out_root, args.replace)
    print(f"Exported {len(rows)} historical design runs to {args.out_root}")


if __name__ == "__main__":
    main()