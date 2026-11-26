import argparse
import anndata as ad
import json
import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--pool-suffix", default="", help="input/pool/<t>_pool<suffix>.json; also renames the npz")
ap.add_argument("--tissue", required=True)
a = ap.parse_args()
t = a.tissue
base = f"../input/split_{t}/{t}"
ann = ad.read_h5ad(f"../input/external/{t}.h5ad")
spl = json.load(open(f"{base}_split.json"))
test_idx = set(spl["test_idx"])
pool = json.load(open(f"../input/pool/{t}_pool{a.pool_suffix}.json"))
pool_genes = sorted({g for gl in pool["pool"].values() for g in gl})
gcol = {g: i for i, g in enumerate(ann.var_names)}
present = [g for g in pool_genes if g in gcol]
missing = [g for g in pool_genes if g not in gcol]
if ann.raw is not None:  # raw.X must equal X or the lit vs cand npz families silently diverge
    rX, Xd = ann.raw.X, ann.X
    assert rX.shape == Xd.shape, f"raw.X {rX.shape} != X {Xd.shape}"
    if sp.issparse(rX):
        assert (rX != Xd).nnz == 0, "raw.X and X diverge (nnz)"
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(rX.size, min(10000, rX.size), replace=False)
        Xd_d = Xd.toarray() if sp.issparse(Xd) else np.asarray(Xd)
        assert np.array_equal(rX.flat[idx], Xd_d.flat[idx]), "raw.X and X diverge (sample)"
X = ann.raw.X if ann.raw is not None else ann.X
sub = X[:, [gcol[g] for g in present]]
if sp.issparse(sub):
    sub = sub.toarray()
is_test = np.array([i in test_idx for i in range(ann.n_obs)])
np.savez_compressed(f"../input/processed/{t}_fullcells_lit{a.pool_suffix}.npz",
    genes=np.array(present), X=sub.astype(np.float32),
    cell_type=ann.obs["cell_type"].values.astype(str),
    is_test=is_test)
print(f"full {sub.shape[0]} x {sub.shape[1]}; test {int(is_test.sum())}; missing {len(missing)}")
