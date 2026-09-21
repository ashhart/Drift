import { expect, test } from "bun:test";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { createProviderProjectBoundary } from "../../../scripts/omp/provider_project_boundary.mjs";
import { boundaryFixture, sha } from "./provider_boundary_fixture";

test("provider boundary binds routes and releases both actors once", async () => {
  const f = boundaryFixture();
  try {
    const parent = createProviderProjectBoundary(f.path, f.expected, f.binding("parent"));
    const child = createProviderProjectBoundary(f.path, f.expected, f.binding("child"));
    const call = (role: string, toolNames: string[]) => ({ identity: f.binding(role).identity, toolNames, signal: new AbortController().signal });
    await parent(call("parent", ["task", "hub"])); expect(existsSync(f.root + "/parent-ready.json")).toBe(false);
    const c = child(call("child", ["hub"])), ch = await f.ready("child");
    const p = parent(call("parent", ["hub"])), ph = await f.ready("parent");
    f.release("parent", ph, ch); f.release("child", ch, ph); await Promise.all([p, c]);
    await child(call("child", ["hub"])); expect(existsSync(f.root + "/child-failed.json")).toBe(false);
    await child(call("child", [])); expect(existsSync(f.root + "/child-failed.json")).toBe(false);
  } finally { f.close(); }
});

test("abort invalidates readiness, preserves its bytes, and poisons the peer", async () => {
  const f = boundaryFixture();
  try {
    const parent = createProviderProjectBoundary(f.path, f.expected, f.binding("parent")), child = createProviderProjectBoundary(f.path, f.expected, f.binding("child"));
    const signal = new AbortController();
    const c = child({ identity: f.binding("child").identity, toolNames: ["hub"], signal: signal.signal }).catch((error: Error) => error);
    const ch = await f.ready("child");
    const p = parent({ identity: f.binding("parent").identity, toolNames: ["hub"], signal: new AbortController().signal }).catch((error: Error) => error);
    await f.ready("parent"); signal.abort();
    for (const result of await Promise.all([p, c])) expect(result.message).toBe("DRIFT_WORKER_CANCELLED");
    expect(existsSync(f.root + "/child-ready.json")).toBe(false); expect(existsSync(f.root + "/parent-ready.json")).toBe(false);
    expect(sha(readFileSync(f.root + "/child-invalidated.json"))).toBe(ch);
    expect(JSON.parse(readFileSync(f.root + "/child-failed.json", "utf8")).status).toBe("FAILED");
    await expect(child({ identity: f.binding("child").identity, toolNames: ["hub"], signal: new AbortController().signal })).rejects.toThrow("DRIFT_WORKER_CANCELLED");
  } finally { f.close(); }
});

test("wrong native identity and expired owner deadline never leave ready state", async () => {
  for (const wrong of [true, false]) {
    const f = boundaryFixture(wrong ? 2000 : 100);
    try {
      const hook = createProviderProjectBoundary(f.path, f.expected, f.binding("child"));
      const identity = { ...f.binding("child").identity, ...(wrong ? { model_sha256: "e".repeat(64) } : {}) };
      await expect(hook({ identity, toolNames: ["hub"], signal: new AbortController().signal })).rejects.toThrow("DRIFT_WORKER_CANCELLED");
      expect(existsSync(f.root + "/child-ready.json")).toBe(false);
      expect(existsSync(f.root + "/child-failed.json")).toBe(true);
    } finally { f.close(); }
  }
});

test("exchange runs only after both actors park and its digest binds release", async () => {
  const f = boundaryFixture();
  const calls: string[] = [];
  const signal = new AbortController();
  const pending: Promise<unknown>[] = [];
  try {
    const hook = (role: string) => createProviderProjectBoundary(f.path, f.expected, f.binding(role), {
      exchange: async () => {
        expect(existsSync(f.root + '/parent-ready.json')).toBe(true);
        expect(existsSync(f.root + '/child-ready.json')).toBe(true);
        calls.push(role);
        return { rows: 2, role };
      },
    });
    const parent = hook('parent'), child = hook('child');
    const call = (role: string, toolNames: string[]) => ({ identity: f.binding(role).identity, toolNames, signal: signal.signal });
    await parent(call('parent', ['task']));
    expect(calls).toEqual([]);
    pending.push(child(call('child', ['hub'])).catch((error: Error) => error));
    const ch = await f.ready('child');
    expect(calls).toEqual([]);
    pending.push(parent(call('parent', ['hub'])).catch((error: Error) => error));
    const ph = await f.ready('parent');
    for (let i = 0; i < 50 && calls.length < 2; i++) await Bun.sleep(5);
    expect(calls.sort()).toEqual(['child', 'parent']);
    for (const [role, own, peer] of [['parent', ph, ch], ['child', ch, ph]]) {
      const receipt = readFileSync(f.root + '/' + role + '-exchange.json');
      writeFileSync(f.root + '/' + role + '-release.json', JSON.stringify({
        v: 1, action: 'resume', nonce: f.config.nonce, ready_sha256: own,
        peer_ready_sha256: peer, exchange_sha256: sha(receipt),
      }), { mode: 0o600 });
    }
    expect(await Promise.all(pending)).toEqual([undefined, undefined]);
  } finally { signal.abort(); await Promise.all(pending); f.close(); }
});
