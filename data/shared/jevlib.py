"""Shared Jev pilot harness: pinned client + sha256 JSONL cache + evaluation metrics.

Contract (all pilots):
  resp = ask(state, questions, cache="pX/cache.jsonl")        # one request, cached
  resps = ask_many([(state, questions), ...], cache=...)       # order-preserving fan-out
  USAGE -> {"requests", "cache_hits", "input_tokens"};  cost_usd()
Model pinned to jev-1.13.0. Numbers must be pre-bucketed by the caller (Jev jaggedness #2).
Stdlib only. Self-check: `python3 jevlib.py` (offline metrics) / `python3 jevlib.py --live`.
"""
import hashlib, json, math, os, random, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = os.environ.get("X3_MODEL", "jev-1.13.0")
URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1/systemone")
if not URL.rstrip("/").endswith("/systemone"):
    URL = URL.rstrip("/") + "/v1/systemone"
# Key env var name: TYPESAFE_API_KEY (api.typesafe.ai) or OPENCODE_API_KEY for the
# opencode-zen System One wire (https://opencode.ai/zen/v1/systemone, model
# jev-1.13-free). Explicit UA: zen's Cloudflare blocks Python-urllib (error 1010).
KEY_ENV = os.environ.get("JEV_KEY_ENV", "TYPESAFE_API_KEY")
UA = "curl/8.5.0"
PRICE_PER_MTOK = float(os.environ.get("X3_PRICE_PER_MTOK", "0.042"))
TOKEN_CAP = int(os.environ.get("JEV_TOKEN_CAP", "2000000"))  # per process; pilots stay <= ~$0.08
USAGE = {"requests": 0, "cache_hits": 0, "input_tokens": 0, "retries": 0}
_lock = threading.Lock()
_mem = {}  # cache path -> {key: response}


def _key(state, questions, model):
    blob = json.dumps({"model": model, "state": state, "questions": questions}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load(cache):
    if cache not in _mem:
        d = {}
        if cache and os.path.exists(cache):
            bad = 0
            with open(cache) as f:
                for line in f:
                    if line.strip():
                        try:
                            r = json.loads(line)
                            d[r["key"]] = r["response"]
                        except (json.JSONDecodeError, KeyError):
                            bad += 1
            if bad:
                print(f"[jevlib] cache {cache}: skipped {bad} corrupt line(s)", file=sys.stderr)
        _mem[cache] = d
    return _mem[cache]


def _post(payload, timeout=120, attempts=6):
    key = os.environ[KEY_ENV]
    body = json.dumps(payload).encode()
    for i in range(attempts):
        req = urllib.request.Request(URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < attempts - 1:
                wait = float(e.headers.get("retry-after") or 2 ** i)
                with _lock:
                    USAGE["retries"] += 1
                time.sleep(min(wait, 60))
                continue
            raise RuntimeError(f"Jev HTTP {e.code}: {e.read()[:500]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if i < attempts - 1:
                time.sleep(2 ** i)
                continue
            raise


def ask(state, questions, cache=None, model=MODEL):
    """One Jev request. Cached by sha256(model+state+questions); cache hits cost nothing."""
    k = _key(state, questions, model)
    store = _load(cache)
    if k in store:
        with _lock:
            USAGE["cache_hits"] += 1
        return store[k]
    if USAGE["input_tokens"] >= TOKEN_CAP:
        raise RuntimeError(f"JEV_TOKEN_CAP {TOKEN_CAP} reached")
    resp = _post({"model": model, "state": state, "questions": questions})
    if resp.get("model") != model:
        raise RuntimeError(f"answered by {resp.get('model')!r}, expected pinned {model}")
    with _lock:
        USAGE["requests"] += 1
        USAGE["input_tokens"] += resp.get("usage", {}).get("input_tokens", 0)
        store[k] = resp
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            with open(cache, "a") as f:
                f.write(json.dumps({"key": k, "response": resp}) + "\n")
    return resp


def ask_many(items, cache=None, workers=4, model=MODEL):
    """items: [(state, questions)] -> responses in the same order. workers<=4 keeps 8 pilots under 1,200 rpm."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda sq: ask(sq[0], sq[1], cache=cache, model=model), items))


def cost_usd(tokens=None):
    return (USAGE["input_tokens"] if tokens is None else tokens) / 1e6 * PRICE_PER_MTOK


# ---------------- metrics (pure functions) ----------------

def accuracy(pred, gold):
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def ece_top(conf, correct, bins=15):  # PROPOSAL H3: 15-bin reliability
    """Top-label ECE: conf = probability of the predicted option, correct = bool."""
    n = len(conf)
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            tot += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return tot


def ece_binary(p, y, bins=15):
    """Reliability ECE for Noul probabilities p against 0/1 outcomes y."""
    return ece_top(list(p), list(y), bins)


def brier_binary(p, y):
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(y)


def auroc(scores, y):
    """Mann-Whitney AUROC with tie correction."""
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    if not pos or not neg:
        return float("nan")
    w = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def coverage_risk(conf, correct, thresholds=(0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)):
    """Rows: threshold, coverage (share kept), accuracy on kept."""
    rows = []
    for t in thresholds:
        kept = [c for cf, c in zip(conf, correct) if cf >= t]
        rows.append({"tau": t, "coverage": len(kept) / len(conf),
                     "accuracy": (sum(kept) / len(kept)) if kept else None, "n": len(kept)})
    return rows


def aurc(conf, correct):
    """Area under the risk-coverage curve (lower = better abstention).
    Sort by confidence desc; risk(k) = error rate of top-k kept; AURC = mean risk over k."""
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    tot_err, tot = 0.0, 0
    risks = []
    for k, i in enumerate(order, 1):
        tot += 1
        tot_err += 0.0 if correct[i] else 1.0
        risks.append(tot_err / tot)
    return sum(risks) / len(risks) if risks else float("nan")


def auto_frac_at_loss(conf, correct, y, max_loss=0.05):
    """Max auto-decided (kept) fraction whose include-loss (missed includes / all includes) <= max_loss.
    correct = whether the auto-decision (argmax label) was right; y = gold 0/1."""
    n_inc = sum(y)
    if not n_inc:
        return float("nan")
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    best = 0.0
    for k in range(1, len(conf) + 1):
        kept = order[:k]
        missed_kept = sum(1 for j in kept if y[j] and not correct[j])
        if missed_kept / n_inc <= max_loss:
            best = k / len(conf)
    return best


def wss_at_recall(scores, y, recall_level=0.95):
    """Work saved over sampling at target recall: (TN+FN)/N - (1-recall_level).
    Rank by score desc, screen top-k until recall >= level, rest auto-excluded."""
    n = len(y)
    n_inc = sum(y)
    if not n_inc:
        return float("nan")
    order = sorted(range(n), key=lambda i: -scores[i])
    found = 0
    for k, i in enumerate(order, 1):
        found += 1 if y[i] else 0
        if found / n_inc >= recall_level:
            return (n - k) / n - (1 - recall_level)
    return 0.0 - (1 - recall_level)  # never reached: screened everything


def mcnemar_exact(b, c):
    """Exact McNemar: b = A right/B wrong, c = A wrong/B right. H0: b==c.
    Two-sided binomial test on min(b,c) against Bin(b+c, 0.5)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)

def paired_bootstrap_p(fn, colA, colB, n=10000, seed=0):
    """Two-sided bootstrap p-value for fn(A) - fn(B) on paired samples.
    p = 2 * min(frac(boot deltas > 0), frac(boot deltas < 0)), capped at 1."""
    rng = random.Random(seed)
    m = len(colA)
    obs = fn(colA) - fn(colB)
    pos = neg = 0
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        d = fn([colA[i] for i in idx]) - fn([colB[i] for i in idx])
        if d > 0:
            pos += 1
        elif d < 0:
            neg += 1
    return min(1.0, 2 * min(pos, neg) / n) if obs != 0 else 1.0


def bootstrap_ci(fn, *cols, n=1000, seed=0, alpha=0.05):
    """Percentile CI of fn(*resampled cols); seeded for reproducibility."""
    rng = random.Random(seed)
    m = len(cols[0])
    vals = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        vals.append(fn(*[[c[i] for i in idx] for c in cols]))
    vals.sort()
    return (vals[int(alpha / 2 * n)], vals[int((1 - alpha / 2) * n) - 1])


def hier_boot_ci(fn, by_review, n=1000, seed=0, alpha=0.05):
    """Hierarchical bootstrap CI: resample reviews, then records within reviews (0g).
    by_review: {review: [(pred, label), ...]}; fn takes flat (preds, labels)."""
    rng = random.Random(seed)
    revs = list(by_review.keys())
    vals = []
    for _ in range(n):
        preds, labels = [], []
        for _ in range(len(revs)):
            r = revs[rng.randrange(len(revs))]
            recs = by_review[r]
            for _ in range(len(recs)):
                p, l = recs[rng.randrange(len(recs))]
                preds.append(p); labels.append(l)
        vals.append(fn(preds, labels))
    vals.sort()
    return (vals[int(alpha / 2 * n)], vals[int((1 - alpha / 2) * n) - 1])


def dev_gap(pairs):
    """Crispness gap: mean(include p | y=1) - mean(include p | y=0) on dev pairs (0g)."""
    pos = [p for p, l in pairs if l == 1]
    neg = [p for p, l in pairs if l == 0]
    if not pos or not neg:
        return float("nan")
    return sum(pos) / len(pos) - sum(neg) / len(neg)

def _selfcheck():
    assert accuracy(["a", "b"], ["a", "c"]) == 0.5
    assert abs(auroc([0.9, 0.8, 0.1], [1, 1, 0]) - 1.0) < 1e-12
    assert abs(auroc([0.5, 0.5], [1, 0]) - 0.5) < 1e-12
    assert ece_top([1.0, 1.0], [True, True]) == 0.0
    assert abs(ece_top([0.9] * 10, [True] * 5 + [False] * 5) - 0.4) < 1e-9
    lo, hi = wilson_ci(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    cr = coverage_risk([0.9, 0.2], [True, False], thresholds=(0.0, 0.5))
    assert cr[0]["accuracy"] == 0.5 and cr[1]["coverage"] == 0.5 and cr[1]["accuracy"] == 1.0
    a, b = bootstrap_ci(lambda x: sum(x) / len(x), [0, 1] * 50)
    assert a < 0.5 < b
    assert abs(aurc([0.9, 0.9, 0.1, 0.1], [True, True, False, False]) - 5 / 24) < 1e-9
    assert abs(wss_at_recall([3, 2, 1, 0], [1, 1, 0, 0]) - 0.45) < 1e-9
    assert auto_frac_at_loss([0.9, 0.8, 0.2], [True, True, True], [1, 1, 0]) == 1.0
    assert abs(auto_frac_at_loss([0.9, 0.8, 0.2], [True, False, True], [1, 1, 0]) - 1 / 3) < 1e-9
    assert paired_bootstrap_p(lambda x: sum(x) / len(x), [1] * 20, [0] * 20, n=500) < 0.05
    # T1-T4 additions: McNemar exact known value; corrupt-tail skip
    assert abs(mcnemar_exact(10, 2) - 0.03857421875) < 1e-12
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(5, 5) - 1.0) < 1e-12
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        tf.write(json.dumps({"key": "k1", "response": {"a": 1}}) + "\n")
        tf.write("{corrupt\n")
        tf.write(json.dumps({"key": "k2", "response": {"b": 2}}) + "\n")
        bad_cache = tf.name
    d = _load(bad_cache)
    assert d == {"k1": {"a": 1}, "k2": {"b": 2}}, d
    os.unlink(bad_cache)
    print("jevlib selfcheck OK")


if __name__ == "__main__":
    import sys
    _selfcheck()
    if "--live" in sys.argv:
        c = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_selfcheck_cache.jsonl")
        q = {"loc": {"type": "choice", "instructions": "Where in the cell does this protein act, per `annotation`?",
                     "criteria": {"nucleus": None, "mitochondrion": None, "secreted": None, "none of these": None}},
             "tf": {"type": "noul", "instructions": "Does `annotation` describe a DNA-binding transcription factor?"}}
        r = ask({"annotation": "Binds promoter DNA via a zinc-finger domain and activates transcription of target genes."}, q, cache=c)
        print(json.dumps(r, indent=1))
        r2 = ask({"annotation": "Binds promoter DNA via a zinc-finger domain and activates transcription of target genes."}, q, cache=c)
        assert r2 == r and USAGE["cache_hits"] == 1, USAGE
        print("USAGE", USAGE, f"cost ${cost_usd():.6f}")
