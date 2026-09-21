#!/bin/sh
# Runs on the Spark host. Reads the server's API key from the container environment into this
# process only; it is never printed or written to disk.
set -eu
cd "$(dirname "$0")"
DRIFT_GLM_KEY="$(docker inspect glm53-exl3-head --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^VLLM_API_KEY=//p')"
export DRIFT_GLM_KEY
exec python3 spark_tap_glm.py "$@"
