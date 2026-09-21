import { test, expect } from "bun:test";
import { mkdtempSync, writeFileSync, readFileSync, rmSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
import { taskRun } from "../../../scripts/omp/task_scope.mjs";
import { developmentVerify } from "../../../scripts/omp/development_verify.mjs";
const hash = (path: string) => createHash("sha256").update(readFileSync(path)).digest("hex");
test("development verifier permits its assigned listener and rejects other loopback ports and private files", () => {
  const outer = realpathSync(mkdtempSync(tmpdir()+"/development-scope-"));
  const root = realpathSync(mkdtempSync(outer+"/task-")), python = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3";
  writeFileSync(outer+"/private", "sentinel");
  const script = `import socket\ns=socket.socket()\ns.bind(('127.0.0.1',49271))\ns.listen()\ns.close()\nt=socket.socket()\ntry:\n t.bind(('127.0.0.1',49272))\n raise RuntimeError('other port permitted')\nexcept PermissionError: pass\ntry:\n open(${JSON.stringify(outer+"/private")}).read()\n raise RuntimeError('private read permitted')\nexcept PermissionError: pass\nprint('bounded')\n`;
  writeFileSync(root+"/verify.py", script);
  const config = { root, port: 49271, max_run_ms: 5000, max_output_bytes: 4096, runs: [{name:"verify", executable:python, sha256:hash(python), args:["-S","verify.py"], artifacts:[{path:root+"/verify.py",sha256:hash(root+"/verify.py")}], runtime_read_roots:["/System","/usr","/Library/Frameworks/Python.framework/Versions/3.11"]}] };
  try { const old = JSON.parse(taskRun({ ...config, runs: [{ ...config.runs[0], network_loopback: true }] }, "verify")); expect(old.exit_code).not.toBe(0); expect(old.stderr).toContain("other port permitted"); const result = JSON.parse(developmentVerify(config)); expect(result.exit_code).toBe(0); expect(result.stdout.trim()).toBe("bounded"); }
  finally { rmSync(outer,{recursive:true,force:true}); }
});
