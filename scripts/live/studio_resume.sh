#!/bin/sh
# Run from the orchestrating machine once the Studio answers ssh again. Step 1 decides whether v4 training can run there.
set -e
cd "$(dirname "$0")/../.."
STUDIO="${DRIFT_STUDIO:-studio}"
R=/Applications/oMLX.app/Contents/Resources
PY="PYTHONPATH=\"$R/Python/framework-mlx-base/lib/python3.11/site-packages:$R:.\" $R/Python/cpython-3.11/bin/python3"
rsync -a --exclude local --exclude .venv --exclude .venv-next --exclude .git --exclude node_modules ./ "$STUDIO:drift/"
ssh "$STUDIO" "mkdir -p drift/local/studio/translators drift/local/studio/train"
rsync -a local/live/stacked3.npz local/live/fanout3.npz "$STUDIO:drift/local/studio/translators/"
rsync -a local/live/train_taps/taps_glm local/live/train_taps/triples.json "$STUDIO:drift/local/studio/train/"
echo "== 1. gradient probe through Qwen3.8-Flash in oMLX's runtime"
ssh "$STUDIO" "cd ~/drift && $PY scripts/live/studio_grad_probe.py 2>&1 | grep -v Warning | tail -25"
echo "== 2. if train_grad.ok is true: untrained validation baseline for v4 (no training yet)"
echo "ssh $STUDIO \"cd ~/drift && $PY scripts/live/studio_train_answer_level.py --triples local/studio/train/triples.json --taps local/studio/train/taps_glm --out local/studio/train/baseline --eval-only\""
echo "== 3. provenance retest (both arms) from this machine:"
echo "PYTHONPATH=. .venv/bin/python scripts/live/d1_eval.py --passages local/live/gen_d1b.jsonl --translator local/live/stacked3.npz --fanout local/live/fanout3.npz --prereg configs/preregistration.d1b-provenance-retest.json --arm original --out local/live/d1b_original"
echo "PYTHONPATH=. .venv/bin/python scripts/live/d1_eval.py --passages local/live/gen_d1b.jsonl --translator local/live/stacked3.npz --fanout local/live/fanout3.npz --prereg configs/preregistration.d1b-provenance-retest.json --arm concise --max-new 96 --out local/live/d1b_concise"
