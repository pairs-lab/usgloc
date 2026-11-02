#!/usr/bin/env python3
"""Regenerate the prior-refinement (multi-network) tables in both papers.

Idempotent: the generated blocks sit between
    % BEGIN AUTO prior-table
    % END AUTO prior-table
markers; on the first run the markers are created in place of the old
tab:prior table (report) and inserted before the calibration subsection
(letter). Cells whose result directory does not exist yet print as \\res, so
the script can be re-run as runs land.

    python3 scripts/analysis/update_prior_tables.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_prior_table import GSCPR, RUNS, SC, avg, cell, med  # noqa: E402

ROOT = Path("/home/thanhnc/gauss_ws/src/usgloc")
RAL = ROOT / "paper/usgloc_ral_draft.tex"
FULL = ROOT / "paper/usgloc_supplementary.tex"
B, E = "% BEGIN AUTO prior-table\n", "% END AUTO prior-table\n"


def rows(prior, with_full):
    p_dir, c_dir, f_dir = RUNS[prior]
    P = [med(f"{ROOT}/{p_dir}/7scenes_{s}.json") for s in SC]
    C = [med(f"{ROOT}/{c_dir}/7scenes_{s}.json") for s in SC]
    F = [med(f"{ROOT}/{f_dir}/7scenes_{s}.json") for s in SC]
    G = GSCPR[prior]
    name = "ACE~\\cite{ace}" if prior == "ace" else "Marepo~\\cite{marepo}"
    out = [(f"{name} prior$^\\ddagger$", P + [avg(P)]),
           ("\\quad $+$ GS-CPR$^\\dagger$", G + [avg(G)]),
           ("\\quad $+$ \\method{} correction", C + [avg(C)])]
    if with_full:
        out.append(("\\quad $+$ \\method{} full pipeline", F + [avg(F)]))
    lines = []
    for label, vals in out:
        lines.append(label + " & " + " & ".join(cell(v) for v in vals) + "\\\\")
    return "\n".join(lines)


def table(with_full, star):
    env = "table*" if star else "table"
    cap = ("Prior refinement on 7Scenes (median cm/$^\\circ$, SfM pseudo-GT). "
           "$\\ddagger$: prior re-run by us on the identical 100 frames per scene; "
           "$\\dagger$: as reported by GS-CPR~\\cite{gscpr} from the same priors "
           "(its own prior rows agree with our re-runs to within 0.1\\,cm). "
           "``correction'': our evidential matching correction alone, seeded at the prior"
           + ("; ``full pipeline'': correction followed by the weighted refinement and polish." if with_full else "."))
    return (f"\\begin{{{env}}}[t]\n\\centering\n\\caption{{{cap}}}\n\\label{{tab:prior}}\n"
            "\\setlength{\\tabcolsep}{2.2pt}\n\\footnotesize\n"
            "\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
            "Method & Chess & Fire & Heads & Office & Pumpkin & Kitchen & Stairs & Avg.\\\\\n\\midrule\n"
            + rows("ace", with_full) + "\n\\midrule\n" + rows("marepo", with_full)
            + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


def splice(path, block, first_time_anchor):
    s = path.read_text()
    if B in s and E in s:
        i, j = s.index(B), s.index(E) + len(E)
        s = s[:i] + B + block + E + s[j:]
        what = "updated"
    else:
        i, j = first_time_anchor(s)
        s = s[:i] + B + block + E + s[j:]
        what = "created"
    path.write_text(s)
    print(f"{path.name}: prior table {what}")


def full_anchor(s):
    # replace the old tab:prior float
    i = s.index("\\label{tab:prior}")
    a = s.rfind("\\begin{table", 0, i)
    b = s.index("\\end{table}", i) + len("\\end{table}\n")
    return a, b


def ral_anchor(s):
    i = s.index("\\subsection{Uncertainty Calibration}")
    return i, i


if __name__ == "__main__":
    splice(FULL, table(with_full=True, star=True), full_anchor)
    splice(RAL, table(with_full=False, star=False), ral_anchor)
