"""Fig 1, Introduction: the benchmark map, four tasks on one text-to-numbers axis. No inputs (values restate the source studies; sizes asserted in fig2-fig5).
Out: figs/fig1_gradient.{pdf,png}. Run from merge_jev/paper/: python3 scripts/fig1_gradient.py"""
from matplotlib.patches import FancyBboxPatch
from style import save, plt, JEV, POSTHOC, REF, WIDTH

# rows of each card; values restate the source studies (sizes asserted in the per-case scripts)
ROWS = ["model reads", "compared with", "measure"]
CASES = [
    ("Study screening", "Fig. 2", JEV, "no detectable\ndifference",
     ["title, abstract,\nreview criteria", "ASReview\n(active learning)", "work saved at\n95% recall"]),
    ("Drug-trial claims", "Fig. 3", JEV, "no detectable\ndifference",
     ["two claims from\ntrial abstracts", "published models\non the benchmark", "detection F$_1$"]),
    ("Cell types", "Fig. 4", POSTHOC, "depends on\nthe baseline",
     ["marker gene\nnames", "CellMarker\n(curated database)", "cells named\ncorrectly"]),
    ("Mass spectrometry", "Fig. 5", REF, "fewer matches\naccepted",
     ["engine numbers\nwritten as text", "mokapot\n(trained on the run)", "matches accepted\nat 1% error"]),
]
f, (b, a) = plt.subplots(2, 1, figsize=(WIDTH, 5.0), gridspec_kw=dict(height_ratios=[2.3, 2.7], hspace=.08))
a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
# panel a: what a decision model takes in and gives back (one worked example)
b.axis("off"); b.set_xlim(0, 1); b.set_ylim(0, 1)
def box(x, y, w, h, title, body, col):
    b.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.02", fc="none", ec=col, lw=.9))
    b.text(x + w / 2, y + h - .075, title, ha="center", va="center", fontsize=6.6, weight="bold", color=col, linespacing=1.15)
    b.text(x + w / 2, y + .5 * (h - .12), body, ha="center", va="center", fontsize=6.2, linespacing=1.3)
b.text(.0, 1.0, "a", fontsize=9, weight="bold", va="top")
b.text(.03, 1.0, "The same question, two kinds of model: is this study eligible?", fontsize=7.2, weight="bold", va="top")
ROWS_A = [
    (.53, "Chatbot: free text in, free text out", REF,
     "title and abstract\n+ \"Is this study eligible?\"",
     "\"Probably yes: it is a\nrandomised trial of ...\"",
     "words: nothing\nto rank or cut at"),
    (.04, "Decision model: closed question in, probability per answer out", JEV,
     "title and abstract\n+ answers: eligible / not",
     "eligible        0.91\nnot eligible  0.09",
     "rank, set a cut-off\n(confidence to be checked)"),
]
for y, name, col, inp, out, use in ROWS_A:
    b.text(.02, y + .375, name, ha="left", va="center", fontsize=6.8, weight="bold", color=col)
    xs = [.02, .37, .70]; ws = [.28, .26, .28]
    box(xs[0], y, ws[0], .32, "Input", inp, REF)
    box(xs[1], y, ws[1], .32, "Output", out, col)
    box(xs[2], y, ws[2], .32, "What you get", use, col)
    for k in range(2):
        b.annotate("", (xs[k + 1], y + .14), (xs[k] + ws[k], y + .14), arrowprops=dict(arrowstyle="-|>", lw=1, color=REF))
a.text(.0, .985, "b", fontsize=9, weight="bold", va="top", transform=a.transAxes)
a.annotate("", (.995, .955), (.17, .955), arrowprops=dict(arrowstyle="-|>", lw=1, color=REF))
a.text(.17, .975, "input is language", ha="left", va="bottom", fontsize=6.8, style="italic")
a.text(.995, .975, "input is numbers from one experiment", ha="right", va="bottom", fontsize=6.8, style="italic")
y0, dy = .70, .155
for j, r in enumerate(ROWS):
    a.text(.0, y0 - j * dy, r, ha="left", va="center", fontsize=6.6, color="#444444", weight="bold")
for i, (t, fig, col, verdict, vals) in enumerate(CASES):
    xc = .265 + i * .21
    a.add_patch(FancyBboxPatch((xc - .1, .02), .2, .905, boxstyle="round,pad=0,rounding_size=.012",
                               fc="none", ec=col, lw=.9, transform=a.transAxes))
    a.text(xc, .875, f"{t}", ha="center", va="center", fontsize=6.5, weight="bold")
    a.text(xc, .815, fig, ha="center", va="center", fontsize=6.3, color="#555555")
    for j, v in enumerate(vals):
        a.text(xc, y0 - j * dy, v, ha="center", va="center", fontsize=6.4)
    a.text(xc, .14, verdict, ha="center", va="center", fontsize=6.6, color=col, weight="bold")
a.text(.0, .14, "result", ha="left", va="center", fontsize=6.6, color="#444444", weight="bold")
a.plot([.168, .995], [.255, .255], color="#cccccc", lw=.5)
save(f, "fig1_gradient")
print("ok")
