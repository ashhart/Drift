#!/bin/sh
# Runs on the Spark host: start generation (3 of the server's 4 slots) and the corpus taps side by side.
cd "$(dirname "$0")"
nohup ./spark_run.sh gen_corpus.py --n 3000 --seed 12 --out gen_train.jsonl --workers 3 > gen_train2.log 2>&1 < /dev/null &
nohup ./spark_run.sh spark_tap_glm.py --ids corpus2a.ids.json --out taps2_glm > tap2_glm.log 2>&1 < /dev/null &
