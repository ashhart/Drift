import { expect, test } from "bun:test";
import { chmodSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { createWorkerBoundary, workerBoundarySettings } from "../../../scripts/omp/worker_boundary_registration.mjs";
import { requireKvExchange } from "../src/worker_kv_registration";
import { kvBoundary } from "../src/worker_kv_policy";
import { boundaryFixture, sha } from "./provider_boundary_fixture";
import { createProviderProjectBoundary } from "../../../scripts/omp/provider_project_boundary.mjs";

function configure(f: ReturnType<typeof boundaryFixture>) {
  const workers = f.workers.map(entry => ({ ...entry, communication_mode: "kv_only" }));
  writeFileSync(process.env.DRIFT_WORKER_CONFIG!, JSON.stringify({ version: 1, experimental: true, workers }), { mode: 0o600 });
  process.env.DRIFT_WORKER_CONFIG_SHA256 = sha(readFileSync(process.env.DRIFT_WORKER_CONFIG!));
  writeFileSync(f.path, JSON.stringify({ ...f.config, v: 2, max_epochs: 2 }), { mode: 0o600 });
  return workers;
}

test("KV-only registration requires two matching workers and both pinned gates", () => {
  const f = boundaryFixture();
  try {
    const workers = configure(f), config = { version: 1, experimental: true, workers } as any;
    expect(requireKvExchange(config, { boundaryPath: "boundary", exchangePath: "exchange" })).toBe(true);
    for (const settings of [{}, { boundaryPath: "boundary" }, { exchangePath: "exchange" }]) {
      expect(() => requireKvExchange(config, settings)).toThrow("CAPABILITY");
    }
    expect(() => requireKvExchange({ ...config, workers: [workers[0]] }, {})).toThrow("CAPABILITY");
    const settings = workerBoundarySettings({ DRIFT_PROVIDER_BOUNDARY_CONFIG: f.path, DRIFT_PROVIDER_BOUNDARY_SHA256: sha(readFileSync(f.path)) });
    expect(() => createWorkerBoundary(settings, f.binding("parent"))).toThrow("CAPABILITY");
  } finally { f.close(); }
});

test("KV-only refuses mixed channel pairs and does not release when its peer never arrives", async () => {
  const f = boundaryFixture(), signal = new AbortController();
  let pending: Promise<unknown> | undefined;
  try {
    const workers = configure(f);
    const expected = sha(readFileSync(f.path));
    let exchanges = 0;
    const hook = createProviderProjectBoundary(f.path, expected, f.binding("parent"), { exchange: async () => { exchanges++; } });
    let released = false;
    pending = hook({ identity: f.binding("parent").identity, toolNames: ["drift_sync"], signal: signal.signal }).then(() => { released = true; }, (error: Error) => error);
    for (let i = 0; i < 100 && !existsSync(f.root + "/parent-0-ready.json"); i++) await Bun.sleep(5);
    expect(existsSync(f.root + "/parent-0-ready.json")).toBe(true);
    signal.abort(); expect(await pending).toBeInstanceOf(Error); expect(released).toBe(false); expect(exchanges).toBe(0);
    workers[1].communication_mode = "text_and_artifacts";
    writeFileSync(process.env.DRIFT_WORKER_CONFIG!, JSON.stringify({ version: 1, experimental: true, workers }), { mode: 0o600 });
    process.env.DRIFT_WORKER_CONFIG_SHA256 = sha(readFileSync(process.env.DRIFT_WORKER_CONFIG!));
    expect(() => createProviderProjectBoundary(f.path, expected, f.binding("parent"), { exchange: async () => {} })).toThrow("CAPABILITY");
  } finally { signal.abort(); await pending; f.close(); }
});

test("KV-only pinned pair parks even when the parent arrives first, exchanges twice, and settles both", async () => {
  const f = boundaryFixture(5000), abort = new AbortController(), pending: Promise<unknown>[] = [];
  const calls: string[] = [], counts = new Map<string, number>();
  const socket = f.root + "/kv.sock";
  const server = createServer(peer => peer.on("data", bytes => {
    const frame = JSON.parse(String(bytes));
    expect(Object.keys(frame).sort()).toEqual(["op", "route"]);
    const role = frame.route, sequence = counts.get(role) ?? 0;
    for (const name of ["parent", "child"]) expect(existsSync(`${f.root}/${name}-${sequence}-ready.json`)).toBe(true);
    calls.push(role); counts.set(role, sequence + 1);
    peer.write(JSON.stringify({ op: "exchanged", route: role, published: {
      v: 1, session: "pair", sequence, mode: "drift", source_worker: role, target_worker: role === "parent" ? "child" : "parent",
      rows: 1, source_rows: 1, copies: 1, bytes: 64, source_sha256: "a".repeat(64), text_bytes: 0, applied_ranks: ["rank-0"], receipts: ["b".repeat(64)],
    } }) + "\n");
  }));
  await new Promise<void>(resolve => server.listen(socket, resolve)); chmodSync(socket, 0o600);
  const file = (role: string, epoch: number, kind: string) => `${f.root}/${role}-${epoch}-${kind}.json`;
  const waitFile = async (path: string) => {
    for (let i = 0; i < 100 && !existsSync(path); i++) await Bun.sleep(5);
    expect(existsSync(path)).toBe(true); return sha(readFileSync(path));
  };
  try {
    configure(f);
    const path = f.root + "/exchange.json";
    const workers = ["parent", "child"].map(role => ({ ...f.binding(role).identity,
      route: role, exchange_session: "pair", source_worker: role, target_worker: role === "parent" ? "child" : "parent", ranks: ["rank-0"],
    })).map(({ model_sha256, translator_sha256, ...entry }) => entry);
    writeFileSync(path, JSON.stringify({ v: 1, socket, timeout_ms: 500, workers }), { mode: 0o600 });
    const settings = workerBoundarySettings({ DRIFT_PROVIDER_BOUNDARY_CONFIG: f.path, DRIFT_PROVIDER_BOUNDARY_SHA256: sha(readFileSync(f.path)), DRIFT_EXCHANGE_CONFIG: path, DRIFT_EXCHANGE_SHA256: sha(readFileSync(path)) });
    const hooks = ["parent", "child"].map(role => {
      const registered = createWorkerBoundary(settings, f.binding(role));
      expect(registered.role).toBe(role);
      return kvBoundary(registered);
    });
    const boundary = (role: string, finish = false) => ({ identity: f.binding(role).identity, toolNames: finish ? [] : ["drift_sync"], toolCalls: finish ? [] : [{ name: "drift_sync", arguments: {} }], signal: abort.signal });
    for (let epoch = 0; epoch < 2; epoch++) {
      let released = 0;
      const start = pending.length;
      pending.push(hooks[0](boundary("parent")).then(() => released++, error => error));
      await waitFile(file("parent", epoch, "ready"));
      await Bun.sleep(15); expect(released).toBe(0); expect(calls.length).toBe(epoch * 2);
      pending.push(hooks[1](boundary("child")).then(() => released++, error => error));
      const ready = await Promise.all(["parent", "child"].map(role => waitFile(file(role, epoch, "ready"))));
      for (const [index, role] of ["parent", "child"].entries()) {
        const exchange = await waitFile(file(role, epoch, "exchange"));
        writeFileSync(file(role, epoch, "release"), JSON.stringify({ v: 2, epoch, action: "resume", nonce: f.config.nonce,
          ready_sha256: ready[index], peer_ready_sha256: ready[1-index], exchange_sha256: exchange }), { mode: 0o600 });
      }
      await Promise.all(pending.slice(start)); expect(released).toBe(2);
    }
    const finish = ["parent", "child"].map((role, index) => hooks[index](boundary(role, true)));
    pending.push(...finish); await Promise.all(finish);
    for (const role of ["parent", "child"]) expect(existsSync(`${f.root}/${role}-settled.json`)).toBe(true);
    expect(calls.sort()).toEqual(["child", "child", "parent", "parent"]);
  } finally { abort.abort(); await Promise.allSettled(pending); server.close(); f.close(); }
});
