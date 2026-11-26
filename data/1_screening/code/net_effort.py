"""Net WSS@95 when development labels count as read. $0, cached.
Pool = dev + test records of a review (holdout excluded). Reading dev labels = n_dev records read.
Test-part read share r = 0.95 - WSS@95. net_read = (n_dev + r*n_test)/(n_dev+n_test); net_WSS = 0.95 - net_read.
Jev: dev labels are needed for the cut-off. Ultra: run on the test part only, so the dev labels are
charged to both arms for a like-for-like pool. Output: results_net_effort.json."""
import json, statistics as st
from layout import P
rows = json.load(open(P("results/jev/results_test.union39.json")))
split = json.load(open(P("data/processed/split.union.json")))["reviews"]
out = {}
for key in ("wss95", "asr_u4"):
    net, read = [], []
    for r in rows:
        nd = len(split[r["review"]]["ids"]["dev"]); nt = r["n_test"]
        rs = 0.95 - r[key]; nr = (nd + rs * nt) / (nd + nt)
        read.append(nr); net.append(0.95 - nr)
    out[key] = {"median_read_share": st.median(read), "median_net_wss95": st.median(net),
                "min": min(net), "max": max(net), "n_positive": sum(x > 0 for x in net), "n": len(net)}
json.dump(out, open(P("results/results_net_effort.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
