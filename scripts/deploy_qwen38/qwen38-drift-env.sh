# Start Qwen3.8 with the Drift connector: source this, then run the recipe's start-drift.sh from its directory.
# The recipe's .env still supplies everything else, including the bf16 main cache the connector writes.
export DRIFT_CONNECTOR_DIR=/root/drift-live/qwen38-connector
export EXTRA_VLLM_ARGS="--kv-transfer-config '{\"kv_connector\":\"DriftQwen38Connector\",\"kv_role\":\"kv_producer\",\"kv_connector_module_path\":\"vllm_qwen38_drift\",\"engine_id\":\"qwen38-drift\",\"kv_connector_extra_config\":{\"handoff_path\":\"/dev/shm/qwen38-drift\"}}'"
