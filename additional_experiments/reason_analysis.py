"""Keyword coding of the agents' stated reasons in the Prisoner's Dilemma (Appendix D.4).

Each two-agent decision returns JSON with an action and a short reason. We code every reason with
fixed regular-expression dictionaries (CODES below) and report, by model, persona representation,
and stage, the share of reasons in each category:
  free_r1  round-1 reasons in free play (GPT-5.2 and Claude Haiku 4.5 from the logged main runs in
           results/PD_other_formats; DeepSeek V4 Flash from data/results/PD/persona). Gemini 2.5 Flash
           free-play reasons were logged truncated and are excluded.
  CC / DD  reasons in rounds 2-10 after a forced opening (results/PD_reason_analysis/forced_opening).

Outputs: results/PD_reason_analysis/reason_codes.csv (one row per decision) and reason_summary.csv.
Usage: python additional_experiments/reason_analysis.py
"""
import glob
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"
OUT = RES / "PD_reason_analysis"
FMT = {"P4_PLAIN_STRATEGIC": "Plain", "P5_DESCRIPTIVE_STRATEGIC": "Descriptive", "P6_TABULAR_STRATEGIC": "Tabular"}
CODES = {
    "exploitation_risk": r"\b(be|being|getting|get)\s+exploited|exploitation|avoid(ing)?\s+(being\s+)?exploit|"
                         r"risk(s|ing)?\s+(being\s+)?(exploit|sucker|getting)|sucker|taken advantage of|vulnerab|protect",
    "trust_reciprocity": r"trust|reciproc|mutual(ly)?\s+(benefi|cooperat|gain|high)|goodwill|"
                         r"signal\w*\s+(willingness|cooperat|good|intent)|build\w*\s+cooperat",
    "exploit_partner": r"exploit(ing)?\s+(the|their|a|an|its|opponent|other|partner|later|cooperat|this|that|them|it)|"
                       r"before exploiting|take advantage of (the|their|opponent|other|a|this)|capitaliz|temptation|"
                       r"opportunit(y|ies)\s+to\s+(gain|earn|get|maximi[sz]e|exploit|score)|endgame|end-?game",
}
RX = {k: re.compile(v, re.I) for k, v in CODES.items()}


def reason_text(raw) -> str:
    m = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)', str(raw))
    return m.group(1) if m else str(raw)


def code_rows(d: pd.DataFrame, model: str, stage: str) -> list[dict]:
    rows = []
    for side in "ab":
        for _, r in d.iterrows():
            txt = reason_text(r[f"agent_{side}_raw_output"])
            rows.append(dict(model=model, stage=stage, persona=FMT[r.persona_format], seed=r.seed, round=r["round"],
                             action=r[f"agent_{side}_action"], reason=txt,
                             **{k: bool(x.search(txt)) for k, x in RX.items()}))
    return rows


def main():
    rows = []
    free = {"gpt-5.2": RES / "PD_other_formats" / "block1_two_agent_gpt-5.2.csv",
            "claude-haiku-4-5-20251001": RES / "PD_other_formats" / "block1_two_agent_claude-haiku-4-5-20251001.csv"}
    for m, f in free.items():
        d = pd.read_csv(f)
        rows += code_rows(d[d.persona_format.isin(FMT) & (d["round"] == 1)], m, "free_r1")
    v4 = pd.concat(pd.read_csv(f) for f in glob.glob(str(ROOT / "data/results/PD/persona/deepseek-v4-flash/two_agent/*/seed*/rounds.csv")))
    rows += code_rows(v4[v4.persona_format.isin(FMT) & (v4["round"] == 1)], "deepseek-v4-flash", "free_r1")
    for f in glob.glob(str(OUT / "forced_opening" / "*" / "*" / "*" / "seed*" / "rounds.csv")):
        model, opening = Path(f).parts[-5], Path(f).parts[-4]
        d = pd.read_csv(f)
        rows += code_rows(d[d["round"] >= 2], model, opening)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "reason_codes.csv", index=False)
    S = R.groupby(["model", "stage", "persona"])[list(CODES)].mean().reset_index()
    S["n"] = R.groupby(["model", "stage", "persona"]).size().values
    S.to_csv(OUT / "reason_summary.csv", index=False)
    print(S.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
