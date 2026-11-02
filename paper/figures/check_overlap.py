#!/usr/bin/env python3
# Usage (from paper/figures): pdflatex fig1_overview.tex && pdftotext -bbox fig1_overview.pdf words.html && python3 check_overlap.py fig1_overview.log words.html
"""Exact text-vs-image overlap check for a TikZ standalone figure.
Image boxes come from TikZ (\typeout BBOX), word boxes from pdftotext -bbox."""
import re, sys, html
log, words_html = sys.argv[1], sys.argv[2]
PT2BP = 72/72.27; BORDER = 2.0
L = open(log, encoding='utf-8', errors='ignore').read()
pic = [float(v) for v in re.search(r'PICBOX ([-\d.]+)pt ([-\d.]+)pt ([-\d.]+)pt ([-\d.]+)pt', L).groups()]
def to_pdf(x, y):  # tikz pt -> pdf bp, origin top-left
    return ((x - pic[0] + BORDER)*PT2BP, (pic[3] - y + BORDER)*PT2BP)
imgs = {}
for n, a, b, c, d in re.findall(r'BBOX (\w+) ([-\d.]+)pt ([-\d.]+)pt ([-\d.]+)pt ([-\d.]+)pt', L):
    x0, y1 = to_pdf(float(a), float(b)); x1, y0 = to_pdf(float(c), float(d))
    imgs[n] = (x0, y0, x1, y1)
W = []
for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>',
                     open(words_html, encoding='utf-8').read()):
    W.append((tuple(float(v) for v in m.groups()[:4]), html.unescape(m.group(5))))
# text deliberately drawn ON an image (white-on-black tags, the vacuity key)
TAGS = {'query', 'render', 'at', 'Iq', 'I', 'q', 'T', '0', 'k', 'Ur', 'U', 'r', 'w', 'ũ', 'u', '˜', 'T0k', 'T0'}
def inter(a, b, tol=0.25):
    return min(a[2], b[2]) - max(a[0], b[0]) > tol and min(a[3], b[3]) - max(a[1], b[1]) > tol
bad = []
for wb, t in W:
    for n, ib in imgs.items():
        if inter(wb, ib) and t.strip() not in TAGS:
            bad.append((n, t, wb))
print(f"{len(W)} words, {len(imgs)} images")
if not bad: print("NO text overlaps any image")
for n, t, wb in bad:
    print(f"  OVERLAP  '{t}'  with image {n}   word y {wb[1]:.1f}-{wb[3]:.1f}  image y {imgs[n][1]:.1f}-{imgs[n][3]:.1f}")
# text-vs-text: words from different lines that intersect
tt = []
for i in range(len(W)):
    for j in range(i+1, len(W)):
        a, b = W[i][0], W[j][0]
        same_line = abs(a[1]-b[1]) < 1.0 and abs(a[3]-b[3]) < 1.0
        if not same_line and inter(a, b, tol=0.6):
            tt.append((W[i][1], W[j][1]))
print("text-vs-text overlaps:", tt if tt else "none")

# ---- block borders (drawn rectangles, from the figure's geometry, cm) and near-misses
CM = 28.45274
BW, RH, cA, cB, cC, rC = 3.827, 3.30, 4.92, 9.547, 14.174, 4.10
blocks = {'offline': (0, 0, 4.52, 7.40),
          'B1': (cA, rC, cA+BW, rC+RH), 'B2': (cB, rC, cB+BW, rC+RH),
          'C3': (cA, 0, cA+BW, RH), 'C2': (cB, 0, cB+BW, RH), 'C1': (cC, 0, cC+BW, RH)}
crossing = []
for name, (x0, y0, x1, y1) in blocks.items():
    X0, Y1 = to_pdf(x0*CM, y0*CM); X1, Y0 = to_pdf(x1*CM, y1*CM)
    for wb, t in W:
        # a word crosses a border if it straddles one of the four edges
        for edge in ('L', 'R', 'T', 'B'):
            if edge in 'LR':
                ex = X0 if edge == 'L' else X1
                if wb[0] < ex - 0.4 < wb[2] - 0.8 and wb[1] < Y1 and wb[3] > Y0:
                    crossing.append((name, edge, t))
            else:
                ey = Y0 if edge == 'T' else Y1
                if wb[1] < ey - 0.4 < wb[3] - 0.8 and wb[0] < X1 and wb[2] > X0:
                    crossing.append((name, edge, t))
print("text crossing a block border:", crossing if crossing else "none")
near = []
for wb, t in W:
    if t.strip() in TAGS: continue
    for n, ib in imgs.items():
        dx = max(ib[0]-wb[2], wb[0]-ib[2], 0); dy = max(ib[1]-wb[3], wb[1]-ib[3], 0)
        if max(dx, dy) < 1.5 and not inter(wb, ib):
            near.append((t, n, round(max(dx, dy), 2)))
print("text within 1.5 bp of an image (reads as touching):", near if near else "none")
