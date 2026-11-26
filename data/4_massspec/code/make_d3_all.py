"""d3-ultra answers to Jev's exact requests (d3u/tB_all_choice.tsv) -> the files master_eval.py and small_data.py read.
Run from anywhere:  python code/make_d3_all.py
Writes compare/d3_all.tsv (Jev's jev/tB_all_choice.tsv rows + d3_p, d3_yes) and d3u/sage_d3.pin
(the 30-feature PIN + d3_p, d3_yes; same construction as jev/sage_jev.pin).
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from master_eval import IN, JEV, D3U, CMP  # noqa: E402


def main():
    d = pd.read_csv(f"{JEV}/tB_all_choice.tsv", sep="\t").astype({"SpecId": str})
    n = pd.read_csv(f"{D3U}/tB_all_choice.tsv", sep="\t").astype({"SpecId": str})
    assert d.SpecId.equals(n.SpecId), "d3-ultra run is not in Jev's row order"
    d["d3_p"], d["d3_yes"] = n.jev_p, n.jev_yes  # run_boundary.py names its answer columns jev_*
    d.to_csv(f"{CMP}/d3_all.tsv", sep="\t", index=False)
    add = d.set_index("SpecId")[["d3_p", "d3_yes"]]
    with open(f"{IN}/sage_nopep.pin") as f, open(f"{D3U}/sage_d3.pin", "w") as g:
        hdr = f.readline().rstrip("\n").split("\t"); k = hdr.index("Peptide")
        g.write("\t".join(hdr[:k] + ["d3_p", "d3_yes"] + hdr[k:]) + "\n")
        for line in f:
            x = line.rstrip("\n").split("\t")
            g.write("\t".join(x[:k] + [f"{add.d3_p[x[0]]:.6f}", str(int(add.d3_yes[x[0]]))] + x[k:]) + "\n")
    print(f"d3_all.tsv {len(d)} rows; d3 yes {int(d.d3_yes.sum())}")


if __name__ == "__main__":
    main()
