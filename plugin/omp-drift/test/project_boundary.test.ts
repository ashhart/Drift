import { test, expect } from "bun:test";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { createProjectBoundary } from "../../../scripts/omp/project_boundary_control.mjs";

const hash = (value: Uint8Array | string) => createHash("sha256").update(value).digest("hex");
function fixture(duration = 2000) {
  const root = realpathSync(mkdtempSync(join(tmpdir(), "project-boundary-"))), task = join(root, "task");
  mkdirSync(task, { mode: 0o700 });
  const path = join(root, "control.json"), started = Date.now();
  const actors = ["parent", "child"].map(role => ({ role, model: `drift-experimental/${role}`, worker: `${role}-worker`, session: `${role}-session` }));
  const config = { v: 1, root, task_root: task, started_at_ms: started, deadline_ms: started + duration, nonce: "a".repeat(32), actors };
  writeFileSync(path, JSON.stringify(config), { mode: 0o600 });
  return { root, path, config, gate: () => createProjectBoundary(path, hash(readFileSync(path))),
    context: (role: string) => ({ cwd: task, model: { provider: "drift-experimental", id: role } }),
    close: () => rmSync(root, { recursive: true, force: true }) };
}
async function ready(root: string, role: string) {
  for (let i = 0; i < 200 && !existsSync(join(root, `${role}-ready.json`)); i++) await Bun.sleep(5);
  const path = join(root, `${role}-ready.json`); expect(existsSync(path)).toBe(true); return hash(readFileSync(path));
}
function release(root: string, role: string, own: string, peer: string, nonce = "a".repeat(32)) {
  writeFileSync(join(root, `${role}-release.json`), JSON.stringify({ v: 1, action: "resume", nonce, ready_sha256: own, peer_ready_sha256: peer, exchange_sha256: "b".repeat(64) }), { mode: 0o600 });
}

test("parent launch is never parked and both later boundaries need bound release receipts", async () => {
  const f = fixture();
  try {
    const parent = f.gate(), child = f.gate();
    expect(await parent.wait("task", f.context("parent"))).toBe(false);
    expect(await parent.wait("hub", f.context("parent"))).toBe(false);
    const c = child.wait("drift_task_read", f.context("child")), ch = await ready(f.root, "child");
    expect(await parent.wait("task", f.context("parent"))).toBe(false);
    const p = parent.wait("hub", f.context("parent")), ph = await ready(f.root, "parent");
    release(f.root, "parent", ph, ch); release(f.root, "child", ch, ph);
    expect(await p).toBe(true); expect(await c).toBe(true);
    expect(await child.wait("hub", f.context("child"))).toBe(false);
  } finally { f.close(); }
});

test("wrong release binding poisons the boundary", async () => {
  const f = fixture();
  try {
    const child = f.gate(), waiting = child.wait("hub", f.context("child"));
    const checked = waiting.then(() => new Error("unexpected release"), error => error), started = performance.now();
    const own = await ready(f.root, "child"); release(f.root, "child", own, "c".repeat(64), "d".repeat(32));
    expect((await checked).message).toBe("PROJECT_BOUNDARY"); expect(performance.now() - started).toBeLessThan(500);
    await expect(child.wait("hub", f.context("child"))).rejects.toThrow("PROJECT_BOUNDARY");
  } finally { f.close(); }
});

test("concurrent calls share the parked boundary until release or abort", async () => {
  for (const abort of [false, true]) {
    const f = fixture();
    try {
      const child = f.gate(), parent = f.gate(); let settled = 0;
      const calls = ["hub", "drift_task_read"].map(tool => child.wait(tool, f.context("child")).then(value => { settled++; return value; }, error => { settled++; return error; }));
      const ch = await ready(f.root, "child"); await Bun.sleep(20);
      expect(settled).toBe(0);
      if (abort) child.abort();
      else {
        const p = parent.wait("hub", f.context("parent")), ph = await ready(f.root, "parent");
        release(f.root, "child", ch, ph); release(f.root, "parent", ph, ch); expect(await p).toBe(true);
      }
      const values = await Promise.all(calls);
      expect(values.every(value => abort ? value instanceof Error && value.message === "PROJECT_BOUNDARY" : value === true)).toBe(true);
    } finally { f.close(); }
  }
});

test("deadline, abort, changed configuration and wrong model fail closed", async () => {
  for (const fault of ["deadline", "abort", "config", "model"]) {
    const f = fixture(fault === "deadline" ? 40 : 2000);
    try {
      const gate = f.gate();
      if (fault === "model") { await expect(gate.wait("hub", f.context("wrong"))).rejects.toThrow("PROJECT_BOUNDARY"); continue; }
      const waiting = gate.wait("hub", f.context("child")), checked = waiting.then(() => new Error("unexpected release"), error => error), started = performance.now();
      await ready(f.root, "child");
      if (fault === "abort") gate.abort();
      if (fault === "config") writeFileSync(f.path, JSON.stringify({ ...f.config, nonce: "e".repeat(32) }));
      expect((await checked).message).toBe("PROJECT_BOUNDARY"); expect(performance.now() - started).toBeLessThan(500);
    } finally { f.close(); }
  }
});
