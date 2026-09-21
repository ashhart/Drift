import { test, expect } from "bun:test";
import { mkdtempSync, writeFileSync, rmSync, symlinkSync, readFileSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
import { taskRead, taskEdit, taskRun } from "../../../scripts/omp/task_scope.mjs";
test("task-local file tools refuse escape and symlinks", () => {
  const root = realpathSync(mkdtempSync(tmpdir() + "/drift-task-"));
  const config = { root, max_file_bytes: 1024 };
  try {
    taskEdit(config, "sample.py", "print(1)"); expect(taskRead(config, "sample.py")).toBe("print(1)");
    expect(() => taskRead(config, "../outside")).toThrow();
    symlinkSync("/etc/hosts", root + "/outside"); expect(() => taskRead(config, "outside")).toThrow();
  } finally { rmSync(root, { recursive: true, force: true }); }
});
test("task command cannot read sibling evidence through the OS sandbox", () => {
  const outer = realpathSync(mkdtempSync(tmpdir() + "/drift-evidence-"));
  const root = realpathSync(mkdtempSync(outer + "/task-"));
  writeFileSync(outer + "/private", "sentinel");
  const executable = "/bin/cat";
  const config = { root, max_run_ms: 3000, max_output_bytes: 4096, runs: [{ name: "check", executable, sha256: createHash("sha256").update(readFileSync(executable)).digest("hex"), args: [outer + "/private"], runtime_read_roots: ["/System", "/usr"] }] };
  try {
    const result = JSON.parse(taskRun(config, "check")); expect(result.exit_code).not.toBe(0); expect(result.stdout).not.toContain("sentinel");
    writeFileSync(root + "/public", "allowed"); config.runs[0].args = [root + "/public"];
    const allowed = JSON.parse(taskRun(config, "check")); expect(allowed.exit_code).toBe(0); expect(allowed.stdout).toBe("allowed");
  } finally { rmSync(outer, { recursive: true, force: true }); }
});

test("task verification Python runs with bounded runtime roots", () => {
  const root = realpathSync(mkdtempSync(tmpdir() + "/drift-python-"));
  const executable = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3";
  const config = { root, max_run_ms: 3000, max_output_bytes: 4096, runs: [{ name: "check", executable, sha256: createHash("sha256").update(readFileSync(executable)).digest("hex"), args: ["-S", "-c", "print(2 + 2)"], runtime_read_roots: ["/System", "/usr", "/Library/Frameworks/Python.framework/Versions/3.11"] }] };
  try { const result = JSON.parse(taskRun(config, "check")); expect(result.exit_code).toBe(0); expect(result.stdout.trim()).toBe("4"); } finally { rmSync(root, { recursive: true, force: true }); }
});
