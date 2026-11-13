"""Run single_perturbation.py's analysis on results/PD_single_perturbation (mixed runs) + bundled paper data."""
import sys, importlib.util
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sp", HERE / "single_perturbation.py")
sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
MIXED_ROOT = HERE / "results" / "PD_single_perturbation"
ALIAS = {"claude-haiku-4-5-20251001": "claude-haiku-4.5"}
def _read_all(model, mode_dir):
    if mode_dir == "two_agent_mixed":
        root = MIXED_ROOT / ALIAS.get(model, model) / mode_dir
    else:
        root = sp.BUNDLED / model / mode_dir
    files = sorted(root.glob("**/seed*/rounds.csv"))
    if not files: raise FileNotFoundError(root)
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
sp._read_all = _read_all
sp.RESULTS = MIXED_ROOT / "analysis"
sys.argv = ["x", "analyze", "--models", "gpt-5.2", "claude-haiku-4-5-20251001", "gemini-2.5-flash", "deepseek-v4-flash"] + sys.argv[1:]
sp.main()
