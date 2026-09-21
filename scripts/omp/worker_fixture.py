"""Exercise only the private worker protocol with deterministic fixture events."""
import json
import os
import sys
from pathlib import Path

facts = dict(open=0, own_prompt=0, tool_result=0, stream=0, close=0)

def save():
    Path(os.environ["DRIFT_WORKER_REPORT"]).write_text(json.dumps(facts))

for line in sys.stdin:
    request = json.loads(line)
    op, payload = request["op"], request["payload"]
    facts[op] = facts.get(op, 0) + 1
    def reply(kind, data):
        print(json.dumps({**request, "op": kind, "payload": data}), flush=True)
    if op == "open":
        reply("opened", {key: payload[key] for key in ("model_id", "model_sha256", "translator_sha256", "limits")} | {
            "backend": "fixture", "capabilities": {"native_state": "fixture", "tool_calls": True, "cancellation": True}})
    elif op == "own_prompt":
        reply("own_prompt_ack", {})
    elif op == "tool_result":
        if payload != {"call_id": "fixture-call", "text": "fixture-ok", "is_error": False}:
            raise RuntimeError("fixture tool mismatch")
        reply("tool_result_ack", {"call_id": payload["call_id"]})
    elif op == "stream":
        first = facts["stream"] == 1
        reply("text", {"text": "fixture-step" if first else "fixture-complete"})
        if first:
            reply("tool_call", {"call_id": "fixture-call", "name": "drift_contract_echo", "arguments": {}})
        reply("terminal", {"reason": "tool_use" if first else "stop", "usage": {"input_tokens": 3, "output_tokens": 2}})
    elif op == "cancel":
        reply("cancelled", {"target_seq": payload["target_seq"]})
    elif op == "close":
        save()
        reply("closed", {})
        break
    else:
        raise RuntimeError("fixture operation mismatch")
    save()
