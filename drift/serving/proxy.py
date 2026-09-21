"""OpenAI-compatible proxy in front of a connector-enabled server (docs/SERVING_INTEGRATION.md).

Any client (OMP, curl, an SDK) sends an ordinary chat request. If the body carries

    "drift": {"session": "<uuid>", "inject": "<name>" | null, "tap": "<name>" | null}

the proxy renders the chat template to token ids through the server's own `/tokenize`, prepends
one placeholder token per foreign entry, and calls `/v1/completions` with the token prompt,
`kv_transfer_params` for the connector and a per-session `cache_salt` so placeholder prefixes are
never shared across sessions by prefix caching. Without the field it is a transparent pass-through.
Standard library only. Non-streaming for drift requests (streaming passes through untouched
for stock requests).
"""
from __future__ import annotations
import argparse
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PASS_THROUGH = ("temperature", "top_p", "top_k", "max_tokens", "stop", "seed", "presence_penalty", "frequency_penalty", "n", "logprobs")


def _post(url: str, body: dict, headers: dict, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def build_handler(upstream: str, exchange_root: Path, placeholder_id: int, timeout: float = 600.0):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):   # quiet by default; servers log upstream
            return

        def _send(self, status: int, payload: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _forward_headers(self) -> dict:
            auth = self.headers.get("Authorization")
            return {"Authorization": auth} if auth else {}

        def do_GET(self):
            request = urllib.request.Request(upstream + self.path, headers=self._forward_headers())
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    self._send(response.status, response.read(), response.headers.get("Content-Type", "application/json"))
            except urllib.error.HTTPError as error:
                self._send(error.code, error.read())

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._send(400, b'{"error":"invalid JSON"}')
            drift = body.pop("drift", None) if isinstance(body, dict) else None
            if not drift or self.path.rstrip("/") != "/v1/chat/completions":
                status, payload = _post(upstream + self.path, body, self._forward_headers(), timeout)
                return self._send(status, payload)
            try:
                payload = self._drift_chat(body, drift)
            except (ValueError, KeyError, FileNotFoundError) as error:
                return self._send(400, json.dumps({"error": f"drift: {error}"}).encode())
            self._send(200, json.dumps(payload).encode())

        def _drift_chat(self, body: dict, drift: dict) -> dict:
            session = str(drift["session"])
            inject, tap = drift.get("inject"), drift.get("tap")
            if body.get("stream"):
                raise ValueError("streaming is not supported for drift requests yet")
            entries = 0
            if inject:
                manifest = exchange_root / session / "inject" / f"{inject}.json"
                entries = int(json.loads(manifest.read_text())["tokens"])
            status, tokenized = _post(upstream + "/tokenize", {"model": body.get("model"), "messages": body["messages"],
                                                               "add_generation_prompt": True}, self._forward_headers(), timeout)
            if status != 200:
                raise ValueError(f"upstream /tokenize failed with {status}")
            tokens = json.loads(tokenized)["tokens"]
            completion = {"model": body.get("model"), "prompt": [placeholder_id] * entries + tokens,
                          "kv_transfer_params": {"drift_session": session, "drift_inject": inject, "drift_tap": tap},
                          "cache_salt": f"drift:{session}:{inject or '-'}"}
            completion.update({k: body[k] for k in PASS_THROUGH if k in body})
            status, raw = _post(upstream + "/v1/completions", completion, self._forward_headers(), timeout)
            if status != 200:
                raise ValueError(f"upstream /v1/completions failed with {status}")
            result = json.loads(raw)
            return {"id": result.get("id", ""), "object": "chat.completion", "created": result.get("created", int(time.time())),
                    "model": result.get("model", body.get("model")),
                    "choices": [{"index": c.get("index", i), "finish_reason": c.get("finish_reason"),
                                 "message": {"role": "assistant", "content": c.get("text", "")}}
                                for i, c in enumerate(result.get("choices", []))],
                    "usage": result.get("usage"), "drift": {"session": session, "placeholders": entries, "tap": tap}}

    return Handler


def serve(listen: str, port: int, upstream: str, exchange_root: Path, placeholder_id: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((listen, port), build_handler(upstream.rstrip("/"), exchange_root, placeholder_id))
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", required=True, help="e.g. http://127.0.0.1:8000")
    parser.add_argument("--exchange-root", type=Path, required=True)
    parser.add_argument("--placeholder-token-id", type=int, required=True)
    args = parser.parse_args()
    serve(args.listen, args.port, args.upstream, args.exchange_root, args.placeholder_token_id).serve_forever()
