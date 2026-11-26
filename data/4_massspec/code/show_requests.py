"""Write the first request each model received (state + 16 questions) and its stored answer to input/example_request_<model>.json.
python3 code/show_requests.py jev | d3u      (offline; no model call: the real scripts are run with the client patched to capture request #1)
Checks that sha256(model, state, questions) of the rebuilt request is a key of the stored answer cache, i.e. it is byte-for-byte what was sent."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
which = sys.argv[1]; assert which in ("jev", "d3u")
if which == "d3u":
    os.environ["X3_MODEL"] = "vllm-sr/d3-ultra"
sys.path[:0] = [HERE, os.path.join(os.path.dirname(ROOT), "shared")]
import jevlib


class Got(Exception):
    pass


def grab(state, questions, cache):
    key = jevlib._key(state, questions, jevlib.MODEL)
    hit = jevlib._load(cache).get(key)
    assert hit is not None, "rebuilt request is not in the stored cache"
    json.dump({"model": jevlib.MODEL, "cache_file": os.path.relpath(cache, ROOT), "cache_key_sha256": key, "state": state,
               "questions": questions, "stored_answer": hit}, open(os.path.join(ROOT, "input", f"example_request_{which}.json"), "w"), indent=1, ensure_ascii=False)
    raise Got


os.chdir(ROOT)
import run_boundary  # both models got the same requests; only model id and cache differ
jevlib.ask_many = lambda items, cache=None, workers=4, model=None: grab(items[0][0], items[0][1], cache)
sys.argv = ["run_boundary.py", "--all", "--qtype", "choice", "--variant", "sage_rel_ev", "--pin", "input/sage_nopep.pin", "--evidence", "input/evidence.tsv",
            "--mok", "baseline/tB.mokapot.psms.txt", "--mok-decoy", "baseline/tB.mokapot.decoy.psms.txt",
            "--cache", f"{which}/{'boundary_jev_cache_tB_all_choice' if which == 'jev' else 'cache_tB_all_choice'}.jsonl", "--out", "/dev/null"]
main = run_boundary.main
try:
    main()
except Got:
    print("wrote input/example_request_%s.json" % which)
