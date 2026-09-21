import { createHash } from "node:crypto";
import { readFileSync, realpathSync, statSync, writeFileSync, lstatSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { spawnSync } from "node:child_process";
const fail = () => { throw new Error("DRIFT_TASK_SCOPE"); };
const digest = data => createHash("sha256").update(data).digest("hex");
export function loadTaskScope(path, sha256) {
  const raw = readFileSync(path); if (digest(raw) !== sha256) fail();
  const config = JSON.parse(raw.toString("utf8"));
  const root = realpathSync(config.root);
  if (root !== config.root || !statSync(root).isDirectory()) fail();
  for (const key of ["max_file_bytes", "max_output_bytes", "max_run_ms"]) if (!Number.isSafeInteger(config[key]) || config[key] < 1) fail();
  if (config.max_file_bytes > 1048576 || config.max_output_bytes > 1048576 || config.max_run_ms > 60000 || !Array.isArray(config.runs)) fail();
  return { ...config, root };
}
export function scopedPath(config, input, create = false) {
  if (typeof input !== "string" || isAbsolute(input)) fail();
  const path = resolve(config.root, input), rel = relative(config.root, path);
  if (!rel || rel === ".." || rel.startsWith(".." + sep)) fail();
  const parent = realpathSync(dirname(path));
  if (parent !== config.root && !parent.startsWith(config.root + sep)) fail();
  if (!create || (() => { try { lstatSync(path); return true; } catch { return false; } })()) {
    if (lstatSync(path).isSymbolicLink() || !statSync(path).isFile() || realpathSync(path) !== path) fail();
  }
  return path;
}
export function taskRead(config, path) {
  const target = scopedPath(config, path); if (statSync(target).size > config.max_file_bytes) fail();
  return readFileSync(target, "utf8");
}
export function taskEdit(config, path, text) {
  if (typeof text !== "string" || Buffer.byteLength(text) > config.max_file_bytes) fail();
  if (config.edit_paths && !config.edit_paths.includes(path)) fail();
  writeFileSync(scopedPath(config, path, true), text, { flag: "w" }); return "written";
}
export function taskRun(config, name) {
  const run = config.runs.find(entry => entry.name === name);
  if (!run || !isAbsolute(run.executable) || digest(readFileSync(run.executable)) !== run.sha256 || !Array.isArray(run.args) || run.args.some(arg => typeof arg !== "string")) fail();
  if (process.platform !== "darwin" || !Array.isArray(run.runtime_read_roots)) fail();
  const roots = run.runtime_read_roots.map(path => {
    const resolved = realpathSync(path);
    if (!["/System", "/usr", "/bin", "/sbin", "/Library/Frameworks", "/Library/Developer"].some(base => resolved === base || resolved.startsWith(base + sep))) fail();
    return resolved;
  });
  for (const artifact of run.artifacts ?? []) if (digest(readFileSync(artifact.path)) !== artifact.sha256) fail();
  const q = JSON.stringify;
  const policy = ["(version 1)", "(allow default)", "(deny network*)", "(deny process-fork)", '(deny file-read-data (subpath "/Users") (subpath "/Volumes") (subpath "/private") (subpath "/Applications") (subpath "/Library") (subpath "/opt") (subpath "/cores"))', "(deny file-write*)", `(allow file-read-data file-write* (subpath ${q(config.root)}))`, `(allow file-read-data (literal ${q(realpathSync(run.executable))}))`, ...roots.map(path => `(allow file-read-data (subpath ${q(path)}))`), '(allow file-read-data file-write* (literal "/dev/null"))'].join("\n");
  const scopedPolicy = [policy, ...(config.read_only_paths ?? []).map(path => `(deny file-write* (literal ${q(scopedPath(config, path))}))`), ...(run.network_loopback === true ? ['(allow network-bind network-inbound (local ip "localhost:*"))', '(allow network-outbound (remote ip "localhost:*"))'] : [])].join("\n");
  const result = spawnSync("/usr/bin/sandbox-exec", ["-p", scopedPolicy, run.executable, ...run.args], { cwd: config.root, env: { PATH: "/usr/bin:/bin", TMPDIR: config.root, PYTHONDONTWRITEBYTECODE: "1" }, encoding: "utf8", timeout: config.max_run_ms, maxBuffer: config.max_output_bytes, killSignal: "SIGKILL" });
  if (result.error || result.signal) fail();
  return JSON.stringify({ exit_code: result.status, stdout: result.stdout, stderr: result.stderr });
}
