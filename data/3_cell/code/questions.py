"""X7 state + question construction (PROPOSAL.md §5.3-§5.4, execution step G).

Turns a cluster's Wilcoxon marker table into the Jev request pair:

    state     = banded marker-gene profile + cluster identity + tissue context
    questions = one `choice` over the tissue's CL branch (<=255 options, §5.4)
                + one `noul` gate, batched into ONE request per cluster (§5.1)

Jaggedness rules honoured here: marker log2FC / pct-expressed / cluster size are
pre-bucketed into *named bands* (no raw floats in the state), and the author
label / CL id / dataset text is NEVER emitted into the state.

Pure functions only: no I/O, no network, no scanpy. Import-safe anywhere.
"""
import bisect

MAX_OPTIONS = 255  # Jev `choice` primitive cap; §5.4 narrowing policy is the lever

# ---- named bands (pre-bucketed; jag #2: no raw floats in the state) ----------

LOG2FC_EDGES = (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0)
LOG2FC_LABELS = ("<=-8", "-8..-4", "-4..-2", "-2..-1", "-1..1", "1..2", "2..4", "4..8", ">=8")

PCT_EDGES = (0.25, 0.50, 0.75, 0.95)
PCT_LABELS = ("0-25%", "25-50%", "50-75%", "75-95%", "95-100%")

SIZE_EDGES = (10, 50, 200, 1000)
SIZE_LABELS = ("<=10", "11-50", "51-200", "201-1000", ">1000")

CELLTYPE_INSTRUCTIONS = (
    "Which cell type does this cluster's marker-gene profile match? "
    "Choose one of the candidate Cell Ontology labels, or 'none of these' "
    "if none of them fit."
)
GATE_INSTRUCTIONS = (
    "Do these marker genes support exactly one specific cell type from the "
    "candidate list (rather than only a generic lineage or an ambiguous state)?"
)


def band(value, edges, labels):
    """Named band for a numeric value: edges ascending, len(labels) == len(edges) + 1.

    A value exactly equal to an edge falls into the higher band (bisect_right).
    """
    return labels[bisect.bisect_right(edges, value)]


def band_log2fc(x):
    """Named band for a marker log2 fold-change (see LOG2FC_LABELS)."""
    return band(float(x), LOG2FC_EDGES, LOG2FC_LABELS)


def band_pct(x):
    """Named band for the expressed-cell fraction; accepts 0-1 or 0-100 input."""
    x = float(x)
    if x > 1.0:
        x /= 100.0
    return band(x, PCT_EDGES, PCT_LABELS)


def band_size(n):
    """Named band for a cluster's cell count (§5.5; <=10 is the published accuracy floor)."""
    return band(float(n), SIZE_EDGES, SIZE_LABELS)


def _marker_row(row, string_cache=None):
    """One banded marker entry from a marker-table row (tolerant to key aliases).

    string_cache: optional dir with STRING JSON cache (string_net.py) — adds
    degree_band + top_partner(+score_band); missing/empty cache -> skipped.
    """
    gene = row.get("gene", row.get("name"))
    lfc = row.get("log2fc", row.get("logfc", row.get("log2FC")))
    pct = row.get("pct_expressed", row.get("pct", row.get("pct_nz_group")))
    lfc = None if lfc in (None, "") else lfc  # CSV round-trip: "" -> None
    pct = None if pct in (None, "") else pct
    out = {"gene": gene}
    if lfc is not None:
        out["log2fc_band"] = band_log2fc(lfc)
    if pct is not None:
        out["pct_expressed_band"] = band_pct(pct)
    if string_cache and gene:
        try:
            import os, string_net
            extra = string_net.row_extra(gene, string_cache)
            out.update(extra)
        except Exception:
            pass
    return out


def build_state(cluster_markers, tissue):
    """§5.3 state: banded marker profile + cluster label + tissue context.

    cluster_markers: {
        "cluster": <leiden label>,                 # required
        "size": <int>,                             # optional -> cluster_size_band
        "top":      [{"gene","log2fc","pct"}, ...],# required, top-k positive markers
        "negative": [{"gene","log2fc","pct"}, ...],# optional, top-k negative markers
    }
    The author label / CL id / dataset text must not appear in cluster_markers;
    this function only copies gene names and bands, so it cannot leak them.

    Returns {"tissue", "cluster", "cluster_size_band", "markers", "negative_markers"}.
    """
    top = cluster_markers.get("top", cluster_markers.get("markers", []))
    neg = cluster_markers.get("negative", cluster_markers.get("negative_markers", []))
    state = {
        "tissue": tissue,
        "cluster": cluster_markers.get("cluster"),
        "markers": [_marker_row(r) for r in top],
    }
    if cluster_markers.get("size") is not None:
        state["cluster_size_band"] = band_size(cluster_markers["size"])
    if neg:
        state["negative_markers"] = [_marker_row(r) for r in neg]
    return state


def markers_by_cluster(rows):
    """Group markers.py CSV rows (dicts with 'group') into the build_state input dicts.

    Optional 'size' on any row for the cluster is carried through.
    """
    out = {}
    for r in rows:
        c = out.setdefault(r["group"], {"cluster": r["group"], "top": []})
        c["top"].append({"gene": r["gene"], "log2fc": r.get("log2fc"), "pct": r.get("pct_expressed")})
        if r.get("size") is not None:
            c["size"] = r["size"]
    return out


def build_questions(cl_options):
    """§5.3-§5.4 questions: `choice` over the tissue CL branch + `noul` gate, one batch.

    cl_options: the tissue's narrowed CL branch — either an iterable of labels or
    a {label: cl_id} mapping (only the labels enter the question). Order is kept,
    duplicates dropped, and 'none of these' is always appended exactly once.

    Returns {"celltype": {"type": "choice", "instructions": ..., "criteria": {...}},
             "gate": {"type": "noul", "instructions": ...}}.
    Raises ValueError if the branch exceeds the 255-option cap (§5.4: narrow it).
    """
    labels = list(cl_options.keys()) if isinstance(cl_options, dict) else list(cl_options)
    seen, criteria = set(), {}
    for lab in labels:
        if lab not in seen:
            seen.add(lab)
            criteria[lab] = None
    if len(criteria) > MAX_OPTIONS:
        raise ValueError(f"{len(criteria)} CL options > {MAX_OPTIONS} cap; narrow the tissue branch (§5.4)")
    criteria["none of these"] = None
    return {
        "celltype": {"type": "choice", "instructions": CELLTYPE_INSTRUCTIONS, "criteria": criteria},
        "gate": {"type": "noul", "instructions": GATE_INSTRUCTIONS},
    }


def _selfcheck():
    assert band_log2fc(3.0) == "2..4" and band_log2fc(-3.0) == "-4..-2"
    assert band_log2fc(9.0) == ">=8" and band_log2fc(0.5) == "-1..1"
    assert band_pct(0.9) == "75-95%" and band_pct(90) == "75-95%"
    assert band_size(7) == "<=10" and band_size(12000) == ">1000"

    st = build_state({"cluster": "3", "size": 812,
                      "top": [{"gene": "INS", "log2fc": 6.2, "pct": 0.97}],
                      "negative": [{"gene": "GCG", "log2fc": -4.5, "pct": 0.4}]}, "pancreas")
    assert st["tissue"] == "pancreas" and st["cluster"] == "3"
    assert st["cluster_size_band"] == "201-1000"
    assert st["markers"] == [{"gene": "INS", "log2fc_band": "4..8", "pct_expressed_band": "95-100%"}]
    assert st["negative_markers"][0]["log2fc_band"] == "-8..-4"

    q = build_questions(["alpha cell", "beta cell", "ductal cell"])
    assert q["celltype"]["type"] == "choice" and q["gate"]["type"] == "noul"
    assert list(q["celltype"]["criteria"]) == ["alpha cell", "beta cell", "ductal cell", "none of these"]
    assert q["celltype"]["criteria"]["none of these"] is None
    assert list(build_questions({"a": "CL:1", "b": "CL:2"})["celltype"]["criteria"]) == ["a", "b", "none of these"]
    try:
        build_questions([f"t{i}" for i in range(MAX_OPTIONS + 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("over-cap branch must raise")
    print("questions selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
