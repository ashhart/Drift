import { expect, test } from "bun:test";
import { createServer } from "node:net";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { createExchangeBoundary, withExchange } from "../../../scripts/omp/provider_exchange_boundary.mjs";
import { validateExchanged } from "../../../scripts/omp/exchange_client.mjs";

const published = (over: Record<string, unknown> = {}) => ({
  v: 1, session: "live-1", sequence: 0, mode: "drift", source_worker: "qwen", target_worker: "glm",
  rows: 12, source_rows: 1, copies: 12, bytes: 4096, source_sha256: "a".repeat(64),
  text_bytes: 0, applied_ranks: ["spark-a.invalid", "spark-b.invalid"], receipts: ["b".repeat(64), "c".repeat(64)], ...over,
});

async function serve(reply: unknown | ((frame: any) => unknown)) {
  const dir = mkdtempSync("/tmp/dx-");
  const path = join(dir, "x.sock");
  const seen: any[] = [];
  const server = createServer(socket => {
    socket.on("data", chunk => {
      const frame = JSON.parse(String(chunk).trim());
      seen.push(frame);
      const value = typeof reply === "function" ? (reply as any)(frame) : reply;
      if (value !== undefined) socket.write(JSON.stringify(value) + "\n");
    });
  });
  await new Promise<void>(resolve => server.listen(path, resolve));
  return { path, seen, close: () => { server.close(); rmSync(dir, { recursive: true, force: true }); } };
}

const boundary = (signal = new AbortController().signal) => ({ identity: {}, toolNames: ["task"], signal });

test("a validated exchange is recorded and the boundary proceeds", async () => {
  const server = await serve({ op: "exchanged", route: "r", published: published(), applied: [], foreign_rows: 0 });
  try {
    const exchange = createExchangeBoundary({ socket: server.path, route: "r" });
    const result = await exchange(boundary());
    expect(result.rows).toBe(12);
    expect(exchange.records.length).toBe(1);
    expect(server.seen[0]).toEqual({ op: "exchange", route: "r" });
  } finally { server.close(); }
});

test("an error frame from the coordinator cancels the turn and poisons the hook", async () => {
  const server = await serve({ op: "error", code: "EXCHANGE_FOREIGN_CAP" });
  try {
    const exchange = createExchangeBoundary({ socket: server.path, route: "r" });
    await expect(exchange(boundary())).rejects.toThrow("DRIFT_WORKER_CANCELLED");
    expect(exchange.poisoned).toBe(true);
    await expect(exchange(boundary())).rejects.toThrow("DRIFT_WORKER_CANCELLED");
  } finally { server.close(); }
});

test("a coordinator that never answers is bounded rather than parking the turn forever", async () => {
  const server = await serve(() => undefined);
  try {
    const exchange = createExchangeBoundary({ socket: server.path, route: "r", timeoutMs: 60 });
    const started = performance.now();
    await expect(exchange(boundary())).rejects.toThrow("DRIFT_WORKER_CANCELLED");
    expect(performance.now() - started).toBeLessThan(1000);
    expect(exchange.poisoned).toBe(true);
  } finally { server.close(); }
});

test("a user abort while the exchange is in flight cancels it", async () => {
  const server = await serve(() => undefined);
  try {
    const controller = new AbortController();
    const exchange = createExchangeBoundary({ socket: server.path, route: "r", timeoutMs: 5000 });
    const pending = exchange(boundary(controller.signal));
    controller.abort();
    await expect(pending).rejects.toThrow("DRIFT_WORKER_CANCELLED");
    expect(exchange.poisoned).toBe(true);
  } finally { server.close(); }
});

test("withExchange runs the exchange before the existing gate decides", async () => {
  const server = await serve({ op: "exchanged", route: "r", published: published(), applied: [], foreign_rows: 0 });
  try {
    const order: string[] = [];
    const exchange = createExchangeBoundary({ socket: server.path, route: "r" });
    const composed = withExchange(async () => { order.push("gate"); }, async b => { order.push("exchange"); await exchange(b); });
    await composed(boundary());
    expect(order).toEqual(["exchange", "gate"]);
  } finally { server.close(); }
});

test("a reply that is not this route, not drift mode or carries text is refused", () => {
  expect(() => validateExchanged({ op: "exchanged", route: "other", published: published() }, "r")).toThrow();
  expect(() => validateExchanged({ op: "exchanged", route: "r", published: published({ mode: "text" }) }, "r")).toThrow("EXCHANGE_MODE");
  expect(() => validateExchanged({ op: "exchanged", route: "r", published: published({ text_bytes: 1 }) }, "r")).toThrow("EXCHANGE_TEXT_FALLBACK");
});

test("a reply missing a receipt per applied rank is not evidence", () => {
  expect(() => validateExchanged({ op: "exchanged", route: "r", published: published({ receipts: ["b".repeat(64)] }) }, "r")).toThrow("EXCHANGE_RECEIPTS");
  expect(() => validateExchanged({ op: "exchanged", route: "r", published: published({ receipts: ["nope", "b".repeat(64)] }) }, "r")).toThrow("EXCHANGE_RECEIPTS");
  expect(() => validateExchanged({ op: "exchanged", route: "r", published: published({ applied_ranks: [] }) }, "r")).toThrow("EXCHANGE_RANKS");
});
