"""Public HTTP development fixture, not a hidden evaluation."""
API = '''import json
from http.server import BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        code, data = (500, {"status": "broken"}) if self.path == "/health" else (404, {"error": "not found"})
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass
'''
VERIFY = '''import http.client
import json
import threading
from http.server import HTTPServer
from api import Handler

server = HTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    for path, code, body in [("/health", 200, {"status": "ok"}), ("/missing", 404, {"error": "not found"})]:
        connection.request("GET", path)
        response = connection.getresponse()
        assert response.status == code, (path, response.status)
        assert response.getheader("Content-Type") == "application/json"
        assert json.loads(response.read()) == body
    connection.close()
    print("public HTTP checks passed")
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
'''
