import { test, expect } from "bun:test";
import { mkdtempSync, realpathSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { taskRun, taskEdit } from "../../../scripts/omp/task_scope.mjs";
test("public HTTP smoke is red before API fix and green under task sandbox after", () => {
  const root = realpathSync(mkdtempSync(tmpdir() + "/drift-http-test-"));
  const source = resolve(import.meta.dir, "../../../scripts/omp");
  const code = "import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from task_smoke import prepare;print(json.dumps(prepare(Path(sys.argv[2])/'task',Path('/Library/Frameworks/Python.framework/Versions/3.11/bin/python3'))))";
  try {
    const prepared = spawnSync("/usr/bin/python3", ["-S", "-c", code, source, root], { encoding: "utf8" });
    expect(prepared.status).toBe(0); const config = JSON.parse(prepared.stdout);
    expect(JSON.parse(taskRun(config, "verify")).exit_code).toBe(1);
    expect(() => taskEdit(config, "verify.py", "print('fake')")).toThrow();
    const path = config.root + "/api.py";
    writeFileSync(path, readFileSync(path, "utf8").replace('(500, {"status": "broken"})', '(200, {"status": "ok"})'));
    const green = JSON.parse(taskRun(config, "verify")); expect(green.exit_code).toBe(0); expect(green.stdout).toContain("public HTTP checks passed");
    const original = config.runs[0].args;
    config.runs[0].args = ["-S", "-c", "import socket\ntry:\n socket.create_connection(('192.0.2.1',80),.2)\nexcept PermissionError:\n print('denied')\nelse:\n raise AssertionError('external network permitted')\ntry:\n open('verify.py','w')\nexcept PermissionError:\n print('readonly')\nelse:\n raise AssertionError('public test writable')"];
    const scope = JSON.parse(taskRun(config, "verify")); expect(scope.exit_code).toBe(0); expect(scope.stdout).toBe("denied\nreadonly\n");
    config.runs[0].args = original;
    writeFileSync(config.root + "/verify.py", "print('tampered')"); expect(() => taskRun(config, "verify")).toThrow();
  } finally { rmSync(root, { recursive: true, force: true }); }
});
