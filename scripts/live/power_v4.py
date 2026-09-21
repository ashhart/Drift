"""Power of Test A's criterion C2 under EXPLICIT assumptions about paired passage outcomes (simulation, no data).

Model (per run: `passages` clusters x 2 questions):
  - v3 correctness: passage p has its own rate r_p ~ Beta with mean p3 and intra-passage correlation icc (icc = 0: all equal),
    each of its questions is correct for v3 with probability r_p;
  - v4 given v3, per question: a v3 miss is FIXED with probability f, a v3 hit is BROKEN with probability b;
    with probability `shared` the two questions of a passage use the same random draw for fix/break (a memory is improved
    or damaged as a whole), otherwise independent draws.
  So p4 = p3 (1 - b) + (1 - p3) f. The two accuracy rates alone do NOT fix the power: b (churn) and the clustering do.
C2 = [EM(v4) - EM(v3) >= 0.04] AND [exact clustered sign-flip p < 0.05]."""
import argparse, json
import numpy as np
from drift.eval.paired_stats import sign_flip_p

parser = argparse.ArgumentParser()
parser.add_argument("--passages", type=int, default=120)
parser.add_argument("--sims", type=int, default=3000)
args = parser.parse_args()
rng = np.random.default_rng(0)


def simulate(p3, b, f, icc, shared):
    hits_test = hits_c2 = 0
    for _ in range(args.sims):
        if icc > 0:
            k = (1 - icc) / icc
            rates = rng.beta(p3 * k, (1 - p3) * k, size=args.passages)
        else:
            rates = np.full(args.passages, p3)
        v3 = rng.random((args.passages, 2)) < rates[:, None]
        u = rng.random((args.passages, 2))
        same = rng.random(args.passages) < shared
        u[same, 1] = u[same, 0]
        v4 = np.where(v3, u >= b, u < f)
        d = (v4.sum(1) - v3.sum(1)).tolist()
        p = sign_flip_p(d)
        hits_test += p < 0.05
        hits_c2 += (p < 0.05) and (v4.mean() - v3.mean() >= 0.04)
    return round(hits_test / args.sims, 3), round(hits_c2 / args.sims, 3)


scenarios = [("no true change, low churn", 0.88, 0.02, 0.147), ("no true change, high churn", 0.88, 0.06, 0.44),
             ("0.88 -> 0.93, no breakage", 0.88, 0.00, 0.417), ("0.88 -> 0.93, 2% breakage", 0.88, 0.02, 0.563), ("0.88 -> 0.93, 5% breakage", 0.88, 0.05, 0.783),
             ("0.88 -> 0.95 (target), 1% breakage", 0.88, 0.01, 0.657), ("0.88 -> 0.95 (target), 3% breakage", 0.88, 0.03, 0.803), ("0.88 -> 0.97, 1% breakage", 0.88, 0.01, 0.823)]
rows = []
for name, p3, b, f in scenarios:
    for icc, shared in ((0.0, 0.0), (0.3, 0.5)):
        test, c2 = simulate(p3, b, f, icc, shared)
        rows.append({"scenario": name, "p4": round(p3 * (1 - b) + (1 - p3) * f, 3), "break": b, "fix": f, "icc": icc, "shared_draw": shared, "P(test p<0.05)": test, "P(C2)": c2})
        print(rows[-1], flush=True)
print(json.dumps({"passages": args.passages, "questions_per_passage": 2, "sims": args.sims, "rows": rows}))
