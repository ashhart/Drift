#!/bin/sh
# Reproduce every automated check in this repository. Exit code is nonzero on any failure.
set -e
cd "$(dirname "$0")/.."
echo "== reference environment (transformers 4.56.2)"
.venv/bin/python -m pytest -q
echo "== next environment (transformers 5.17.0, mlx): everything but the 4.56.2-only HF guard"
PYTHONPATH=. .venv-next/bin/python -m pytest -q \
  tests/test_adapters_next.py tests/test_translate.py tests/test_core.py tests/test_transport_sync.py \
  tests/test_runtime_train_mail.py tests/test_properties.py tests/test_wire2_pool.py tests/test_hive.py \
  tests/test_mailbox.py tests/test_m1.py tests/test_service.py tests/test_workspace.py tests/test_e3_compete_registry.py \
  $( [ -f tests/test_adapters_mlx.py ] && echo tests/test_adapters_mlx.py )
echo "== omp-drift plugin"
( cd plugin/omp-drift && bun test && bunx tsc -p tsconfig.json )
echo "== all checks passed"
