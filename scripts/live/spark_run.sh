#!/bin/sh
# Runs on the Spark host: spark_run.sh <script.py> [args]. The server's API key is read from the
# container environment into this process only; it is never printed or written to disk.
set -eu
cd "$(dirname "$0")"
DRIFT_GLM_KEY="$(docker inspect glm53-exl3-head --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^VLLM_API_KEY=//p')"
export DRIFT_GLM_KEY
script="$1"; shift
exec python3 "$script" "$@"
