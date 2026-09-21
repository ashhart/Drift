import { readFileSync, realpathSync } from "node:fs";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { scopedPath } from "./task_scope.mjs";
export function developmentVerify(config) {
  const run = config.runs.find(item => item.name === "verify"), hash = path => createHash("sha256").update(readFileSync(path)).digest("hex");
  if (process.platform !== "darwin" || !run || hash(run.executable) !== run.sha256 || !Number.isInteger(config.port) || config.port < 1024 || config.port > 65535 || config.max_run_ms > 5000 || config.max_output_bytes > 4096) throw new Error("DEVELOPMENT_SCOPE");
  for (const artifact of run.artifacts) if (hash(artifact.path) !== artifact.sha256) throw new Error("DEVELOPMENT_PIN");
  const roots = run.runtime_read_roots.map(path => { const actual = realpathSync(path); if (!["/System", "/usr", "/Library/Frameworks/Python.framework/Versions/3.11"].includes(actual)) throw new Error("DEVELOPMENT_RUNTIME"); return actual; });
  const q = JSON.stringify, address = `localhost:${config.port}`;
  const policy = ['(version 1)', '(allow default)', '(deny network*)', '(deny process-fork)', '(deny file-read-data (subpath "/Users") (subpath "/Volumes") (subpath "/private") (subpath "/Applications") (subpath "/Library") (subpath "/opt") (subpath "/cores"))', '(deny file-write*)', `(allow file-read-data file-write* (subpath ${q(config.root)}))`, `(allow file-read-data (literal ${q(realpathSync(run.executable))}))`, ...roots.map(path => `(allow file-read-data (subpath ${q(path)}))`), `(deny file-write* (literal ${q(scopedPath(config, "verify.py"))}))`, '(allow file-read-data file-write* (literal "/dev/null"))', `(allow network-bind network-inbound (local ip ${q(address)}))`, `(allow network-outbound (remote ip ${q(address)}))`].join("\n");
  const result = spawnSync("/usr/bin/sandbox-exec", ["-p", policy, run.executable, ...run.args], { cwd: config.root, env: { PATH: "/usr/bin:/bin", TMPDIR: config.root, PYTHONDONTWRITEBYTECODE: "1" }, encoding: "utf8", timeout: config.max_run_ms, maxBuffer: config.max_output_bytes, killSignal: "SIGKILL" });
  if (result.error || result.signal) throw new Error("DEVELOPMENT_VERIFY_LIMIT");
  return JSON.stringify({ exit_code: result.status, stdout: result.stdout, stderr: result.stderr });
}
