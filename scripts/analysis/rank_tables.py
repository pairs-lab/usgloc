#!/usr/bin/env python3
r"""Mark the best and second-best entry of every results column in the paper.

Convention: \textbf{} = best, \underline{} = second best, applied to the
TRANSLATION and ROTATION numbers of a "t/R" cell INDEPENDENTLY, per column.
Ties share a rank, so two rows can both be bold or both underlined.

Re-runnable: it strips any \textbf/\underline it previously added to numeric
cells before recomputing, so it can be applied again after the numbers change.
Row LABELS (everything before the first '&') are never touched, so
\textbf{\method{} (Ours...)} survives.

    python3 scripts/analysis/rank_tables.py paper/usgloc_ral_draft.tex [--check]

--check reports what it would do and writes nothing.
"""
import argparse
import re
import sys

CELL = re.compile(r'^\s*(?:\\textbf\{|\\underline\{)*\s*'
                  r'(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)'
                  r'\s*\}*\s*$')
ONE = re.compile(r'^\s*(?:\\textbf\{|\\underline\{)*\s*'
                 r'(-?\d+(?:\.\d+)?)\s*\}*\s*$')
# column headers whose metric is HIGHER-is-better (success / recall rates)
HIGHER = ('SR', 'Recall', 'recall', 'Acc')


def strip_fmt(s):
    r"""Remove \textbf{...} / \underline{...} wrappers, keeping the content."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'\\(?:textbf|underline)\{([^{}]*)\}', r'\1', s)
    return s


def split_row(line):
    r"""(label, [cells]) for a data row, else None. Skips rules and group headers."""
    body = line.rstrip()
    if not body or body.lstrip().startswith('%'):
        return None
    if '\\multicolumn' in body or '\\midrule' in body or '\\toprule' in body \
       or '\\bottomrule' in body or '&' not in body:
        return None
    tail = body
    trail = ''
    m = re.search(r'(\\\\\s*)$', tail)
    if m:
        trail = m.group(1)
        tail = tail[: m.start()]
    parts = tail.split('&')
    return parts[0], parts[1:], trail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tex')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--tabulars', default='',
                    help='comma-separated 1-based line numbers of the \\begin{tabular} '
                         'lines to rank. Default: none -- ranking is OPT-IN, because '
                         'ablation tables (whose rows remove components) and calibration '
                         'tables (where the target is the NOMINAL value, not the extremum) '
                         'must not be ranked.')
    args = ap.parse_args()

    lines = open(args.tex).read().split('\n')
    # group consecutive data rows into tables by tracking tabular blocks
    want = {int(x) for x in args.tabulars.split(',') if x.strip()}
    if not want:
        print('no --tabulars given; nothing to do (ranking is opt-in)')
        return
    blocks, cur, hdr = [], [], None
    in_tab = False
    tab_line = None
    for i, ln in enumerate(lines):
        if '\\begin{tabular}' in ln:
            tab_line = i + 1
            in_tab, cur, hdr = (tab_line in want), [], None
            continue
        if '\\end{tabular}' in ln:
            if cur:
                blocks.append((hdr, cur))
            in_tab, cur, hdr = False, [], None
            continue
        if in_tab:
            r = split_row(ln)
            if r:
                # the first row with no numeric cell is the header
                if hdr is None and not any(CELL.match(strip_fmt(c)) or ONE.match(strip_fmt(c))
                                           for c in r[1]):
                    hdr = r[1]
                    continue
                cur.append((i, r))

    changed = 0
    for hdr, blk in blocks:
        ncol = max(len(r[1][1]) for r in blk)
        # collect per column: list of (line_idx, cell_idx, t, r) for "t/R" cells
        cols = {c: [] for c in range(ncol)}
        ones = {c: [] for c in range(ncol)}
        for li, (label, cells, trail) in blk:
            for c, cell in enumerate(cells):
                raw = strip_fmt(cell)
                m = CELL.match(raw)
                if m:
                    cols[c].append((li, c, float(m.group(1)), float(m.group(2))))
                    continue
                m1 = ONE.match(raw)
                if m1:
                    ones[c].append((li, c, float(m1.group(1))))
        # rank each column, each metric independently
        marks = {}          # (line_idx, cell_idx) -> {'t'|'r'|'one': rank}
        # higher-is-better single-number columns (success / recall rates)
        for c, entries in ones.items():
            if len(entries) < 2:
                continue
            h = hdr[c] if hdr and c < len(hdr) else ''
            if not any(k in h for k in HIGHER):
                continue
            vals = sorted({e[2] for e in entries}, reverse=True)
            best = vals[0]
            second = vals[1] if len(vals) > 1 else None
            for e in entries:
                rank = 1 if e[2] == best else (2 if second is not None and e[2] == second else None)
                marks.setdefault((e[0], e[1]), {})['one'] = rank
        for c, entries in cols.items():
            if len(entries) < 2:
                continue
            for pos, key in ((2, 't'), (3, 'r')):
                vals = sorted({e[pos] for e in entries})
                best = vals[0]
                second = vals[1] if len(vals) > 1 else None
                for e in entries:
                    rank = 1 if e[pos] == best else (2 if second is not None and e[pos] == second else None)
                    d = marks.setdefault((e[0], e[1]), {})
                    d[key] = rank
        # rewrite lines
        for li, (label, cells, trail) in blk:
            out = [label]
            touched = False
            for c, cell in enumerate(cells):
                raw = strip_fmt(cell)
                m = CELL.match(raw)
                m1 = None if m else ONE.match(raw)
                if (li, c) in marks and m1 and 'one' in marks[(li, c)]:
                    rk1 = marks[(li, c)]['one']
                    lead = re.match(r'^(\s*)', raw).group(1) or ' '
                    tailsp = re.search(r'(\s*)$', raw).group(1)
                    v = m1.group(1)
                    if rk1 == 1:
                        v = '\\textbf{%s}' % v
                    elif rk1 == 2:
                        v = '\\underline{%s}' % v
                    out.append('%s%s%s' % (lead, v, tailsp))
                    touched = True
                    continue
                if not m or (li, c) not in marks:
                    out.append(cell if not (m or m1) else raw)
                    if m or m1:
                        touched = True
                    continue
                rk = marks[(li, c)]
                lead = re.match(r'^(\s*)', raw).group(1) or ' '
                tailsp = re.search(r'(\s*)$', raw).group(1)
                t_s, r_s = m.group(1), m.group(2)

                def wrap(s, r):
                    if r == 1:
                        return '\\textbf{%s}' % s
                    if r == 2:
                        return '\\underline{%s}' % s
                    return s
                new = '%s%s/%s%s' % (lead, wrap(t_s, rk.get('t')),
                                     wrap(r_s, rk.get('r')), tailsp)
                out.append(new)
                touched = True
            if touched:
                newline = '&'.join(out) + trail
                if newline != lines[li]:
                    lines[li] = newline
                    changed += 1

    if args.check:
        print(f'would rewrite {changed} row(s)')
        return
    open(args.tex, 'w').write('\n'.join(lines))
    print(f'rewrote {changed} row(s) in {args.tex}')


if __name__ == '__main__':
    main()
