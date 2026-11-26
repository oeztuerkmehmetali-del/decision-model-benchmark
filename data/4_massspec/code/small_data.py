"""Shrink-the-run test: does a zero-shot scorer overtake mokapot when the run is too small for mokapot to learn from?
Run from anywhere:  python code/small_data.py > compare/small_data_result.txt   (needs mokapot next to sys.executable: conda env x2pilot)
Random PSM subsets (targets + decoys; not stratified, truth labels never used for drawing) of the Tier B run. Per subset
mokapot is re-trained with the paper's command; the zero-shot answers are the stored ones (no model calls).
Yardstick as in master_eval.py: targets accepted at entrapment FDP <= 1% (here as a share of the subset's targets), plus AUROC.
Fixed before any result: SIZES, DRAWS, SEED, the yardstick, and the reading rule - a scorer "overtakes" mokapot at a size
if it accepts more in >= 15 of the 20 draws (two-sided sign test p = 0.04; six sizes, uncorrected: exploratory).
The 1% rule cannot accept one known-wrong match below 225 accepted targets, so the accepted share is coarse at the
smallest sizes; AUROC is not. Writes compare/small_data.tsv (one row per draw).
"""
import os, subprocess, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from master_eval import IN, CMP, accepted, auroc  # noqa: E402

SIZES, DRAWS, SEED = (500, 1000, 2000, 5000, 10000, 20000), 20, 20261120
FALLBACK = "did not improve over the best feature"  # mokapot 0.10.0 brew.py: it then scores by its single best feature
ZERO = {"d3": "d3_p", "jev": "jev_p", "engine": "hs"}  # stored zero-shot answers and the raw engine score
FULL = (37061, 3579, 27047, 12157, 21939)  # master_result.txt: targets, known wrong, accepted by mokapot, d3, engine


def mokapot(lines, tmp):
    """re-train on these PIN lines -> (status, target scores by SpecId, higher = better; None if the run failed, mokapot's last error line)"""
    pin = os.path.join(tmp, "s.pin")
    with open(pin, "w") as f:
        f.write("\n".join(lines) + "\n")
    p = subprocess.run([os.path.join(os.path.dirname(sys.executable), "mokapot"), pin, "--dest_dir", tmp, "--file_root", "s",
                        "--seed", "1", "--max_workers", "4"], capture_output=True, text=True)
    if p.returncode:
        return "failed", None, (p.stderr.strip().splitlines() or ["no stderr"])[-1][:120]
    m = pd.read_csv(os.path.join(tmp, "s.mokapot.psms.txt"), sep="\t").astype({"SpecId": str}).set_index("SpecId")
    s = m["mokapot score"]
    if s.corr(m["mokapot q-value"], method="spearman") > 0:  # best-feature fallback on a lower-is-better feature
        s = -s
    return ("best feature" if FALLBACK in p.stdout + p.stderr else "trained"), s, ""


def main():
    d = pd.read_csv(f"{CMP}/d3_all.tsv", sep="\t").astype({"SpecId": str}).set_index("SpecId")
    pin = pd.read_csv(f"{IN}/sage_ev.pin", sep="\t", usecols=["SpecId", "Proteins", "ln(hyperscore)"]).astype({"SpecId": str}).drop_duplicates("SpecId").set_index("SpecId").reindex(d.index)
    d["hs"] = pin["ln(hyperscore)"]; d["ent"] = (pin.Proteins.str.contains("_ARATH") & ~pin.Proteins.str.contains("_HUMAN")).astype(int)
    with open(f"{IN}/sage_nopep.pin") as f:
        head, *body = f.read().splitlines()
    body = pd.Series(body, index=[b.split("\t", 1)[0] for b in body]).reindex(d.index)
    assert d.index.is_unique and body.notna().all(), "PIN rows and stored answers do not match one to one"
    assert (body.str.split("\t").str[1].eq("-1") == d.is_decoy.eq(1)).all(), "target/decoy labels disagree"

    def evaluate(ids):
        with tempfile.TemporaryDirectory() as tmp:
            status, s, why = mokapot([head, *body[ids]], tmp)
        t = d.loc[ids]; t = t[t.is_decoy == 0]; ent = t.ent.to_numpy(); y = 1 - ent
        # a failed mokapot run leaves the user with the engine score
        S = {"mokapot": (t.hs if s is None else s.reindex(t.index)).to_numpy(float), **{k: t[c].to_numpy(float) for k, c in ZERO.items()}}
        assert not any(np.isnan(v).any() for v in S.values()), "a scorer is missing target PSMs"
        row = {"status": status, "why": why, "decoys": len(ids) - len(t), "targets": len(t), "known_wrong": int(ent.sum())}
        for k, v in S.items():
            row["acc_" + k] = accepted(v, ent); row["auc_" + k] = auroc(v, y)
        return row

    full = evaluate(d.index); tt = d[d.is_decoy == 0]  # the whole run must reproduce the master table
    got = (full["targets"], full["known_wrong"], accepted(tt.score.to_numpy(float), tt.ent.to_numpy()), full["acc_d3"], full["acc_engine"])
    assert got == FULL, f"master table not reproduced: {got}"
    assert abs(full["acc_mokapot"] / FULL[2] - 1) < 0.01, f"re-run mokapot accepts {full['acc_mokapot']}, stored run {FULL[2]}"

    rng = np.random.default_rng(SEED); rows = []
    for n in SIZES:
        for k in range(DRAWS):
            rows.append({"size": n, "draw": k, **evaluate(d.index[rng.choice(len(d), n, replace=False)])})
        print(f"size {n} done", file=sys.stderr, flush=True)
    r = pd.DataFrame(rows); r.to_csv(f"{CMP}/small_data.tsv", sep="\t", index=False, float_format="%.4f")
    r = pd.concat([r, pd.DataFrame([{"size": len(d), "draw": 0, **full}])], ignore_index=True)  # whole run: one run, no spread

    K = ["mokapot", *ZERO]; q = lambda x, p: np.percentile(x, p)
    print(f"shrink-the-run test: {DRAWS} random draws per size, seed {SEED}; last row = the whole run (mokapot re-run once)")
    print("share = targets accepted at entrapment FDP <= 1%, as % of the subset's targets; median [IQR] over all draws\n")
    print(f"{'PSMs':>6s} {'targets':>7s} {'wrong':>5s} {'decoy %':>7s}  {'mokapot trained/best feature/failed':34s}  " + "  ".join(f"{k + ' share':20s}" for k in K) + "  " + "  ".join(f"{k + ' AUROC':13s}" for k in K))
    for n, g in r.groupby("size"):
        st = "/".join(str((g.status == s).sum()) for s in ("trained", "best feature", "failed"))
        sh = "  ".join(f"{q(x, 50):5.1f} [{q(x, 25):4.1f},{q(x, 75):5.1f}]   " for x in (100 * g["acc_" + k] / g.targets for k in K))
        au = "  ".join(f"{g['auc_' + k].median():.3f}        " for k in K)
        print(f"{n:6d} {g.targets.median():7.0f} {g.known_wrong.median():5.0f} {100 * (g.decoys / n).median():7.1f}  {st:34s}  {sh}  {au}")
    print(f"\ndraws (of {DRAWS}) in which the scorer accepts more / the same / fewer targets than the re-trained mokapot; 'overtakes' needs >= 15 more")
    for n, g in r[r["size"] < len(d)].groupby("size"):
        w = lambda a, b: f"{(g[a] > g[b]).sum():2d}/{(g[a] == g[b]).sum():2d}/{(g[a] < g[b]).sum():2d}"
        print(f"{n:6d}  " + "   ".join(f"{k} vs mokapot {w('acc_' + k, 'acc_mokapot')} (AUROC higher in {(g['auc_' + k] > g.auc_mokapot).sum():2d})" for k in ZERO)
              + f"   d3 vs engine {w('acc_d3', 'acc_engine')} (AUROC higher in {(g.auc_d3 > g.auc_engine).sum():2d})")


if __name__ == "__main__":
    main()
