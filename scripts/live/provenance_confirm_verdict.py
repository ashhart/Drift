"""Verdict for configs/preregistration.provenance-confirm.json from a provenance_eval.py report (written before the data existed)."""
import json, sys
r, t = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))["thresholds_confirm"]
p, s = r["pooled"], r["tests"]["pooled_trained_vs_untrained"]
k1 = p["protocol_trained"] - p["protocol_untrained"] >= t["K1_gain"] and s["sign_test_p"] < 0.05
k2 = r["by_kind"]["conj"]["protocol_trained"] >= t["K2_conj"]
k3 = p["protocol_trained"] >= p["own_kv"] - t["K3_gap"]
k4 = p["wrong_memory"] <= t["K4_wrong"]
print(json.dumps({"K1_correction_fixes_provenance": k1, "K2_conjunction": k2, "K3_reaches_own_cache": k3, "K4_still_specific": k4,
                  "verdict": "PASSED" if k1 and k2 and k3 and k4 else "PARTIAL" if k1 and k4 else "FAILED", "pooled": p, "by_kind": r["by_kind"], "test": s}, indent=1))
