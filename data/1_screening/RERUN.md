# Offline re-run check (copy of this folder in a scratch dir, no model calls, no key set)

Ran steps 3-5, 7-10 of README.md from `code/` in the order fanout (jev, d3u; all cache hits: 0 requests) -> sweep (jev, d3u) -> score_final -> ablation -> net_effort -> gap_tests -> calib_sim -> v3_fixes -> d3u_arm. All exit 0; v3_fixes 53/53 asserts PASS. Compared with the stored files:

| file | result |
|---|---|
| `d3u/d3u_dev.json`, `d3u/d3u_test.json`, `d3u/dev_tau.d3u.json`, `d3u/results_d3u.json`, `d3u/d3u_numbers.md` | byte-identical |
| `jev/dev_tau.union.json`, `compare/results_test.union39.json`, `compare/results_net_effort.json`, `jev/results_gaps39.json` | byte-identical |
| `jev/jev_dev.json`, `jev/jev_test.json` | re-fan-out writes only the 62,093 / 18,183 split records; the stored files are supersets (64,947 / 19,902); every re-fanned record equals the stored one. Stored files shipped. |
| `jev/results_ablation39.json` | NOT reproduced: the stored file comes from an earlier ablation.py that did not restrict to split ids; re-run differs in a few reviews (overall-rule median WSS 0.743 identical; or_max 0.558 vs 0.539 stored). The paper figure uses the stored file only through the `old_*` comparison fields of `results_v3_fixes.json`. |
| `jev/results_calibsim39.json` | numerically identical except metadata (runtime_sec, source-path text) |
| `compare/results_v3_fixes.json`, `compare/v3_numbers.md` | see below |

`v3_fixes.py` run against the *stored* ablation/jev files: result in last line of this file.
