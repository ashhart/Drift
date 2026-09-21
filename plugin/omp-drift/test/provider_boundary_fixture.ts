import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, realpathSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
export const sha = (raw: Uint8Array | string) => createHash("sha256").update(raw).digest("hex");
export function boundaryFixture(duration = 2000) {
  const root = realpathSync(mkdtempSync(tmpdir() + "/provider-boundary-")), task = root + "/task";
  mkdirSync(task, { mode: 0o700 });
  const previous = [process.env.DRIFT_WORKER_CONFIG, process.env.DRIFT_WORKER_CONFIG_SHA256];
  const artifact = root + "/fixture"; writeFileSync(artifact, "public fixture", { mode: 0o600 });
  const workers = ["parent", "child"].map(role => ({ experimental_multi_turn: true, memory_mode: "linked", communication_mode: "text_and_artifacts", session_binding: "fixed",
    identity: { worker: role, session: role + "-fixed", model_id: role, model_sha256: "a".repeat(64), translator_sha256: "b".repeat(64) },
    limits: { max_input_bytes: 8192, max_output_tokens: 128, max_session_tokens: 1024, max_turns: 2, deadline_ms: 2000 }, context_window: 8192,
    expected: { backend: "fixture", nativeStates: ["fixture"] }, command: { executable: "/bin/echo", sha256: sha(readFileSync("/bin/echo")), args: [], cwd: task, env: {} },
    artifacts: [{ path: artifact, sha256: sha(readFileSync(artifact)) }] }));
  const owner = root + "/owner.json"; writeFileSync(owner, JSON.stringify({ version: 1, experimental: true, workers }), { mode: 0o600 });
  process.env.DRIFT_WORKER_CONFIG = owner; process.env.DRIFT_WORKER_CONFIG_SHA256 = sha(readFileSync(owner));
  const actors = workers.map(worker => ({ role: worker.identity.worker, model: "drift-experimental/" + worker.identity.model_id, worker: worker.identity.worker, session: worker.identity.session }));
  const now = Date.now(), config = { v: 1, root, task_root: task, started_at_ms: now, deadline_ms: now + duration, nonce: "c".repeat(32), actors };
  const path = root + "/boundary.json"; writeFileSync(path, JSON.stringify(config), { mode: 0o600 });
  return { root, task, path, expected: sha(readFileSync(path)), config, workers,
    binding: (role: string) => ({ identity: workers.find(worker => worker.identity.worker === role)!.identity, ompSession: role + "-omp", cwd: task }),
    async ready(role: string) { for (let i = 0; i < 100 && !existsSync(root + "/" + role + "-ready.json"); i++) await Bun.sleep(5); return sha(readFileSync(root + "/" + role + "-ready.json")); },
    release(role: string, own: string, peer: string) { writeFileSync(root + "/" + role + "-release.json", JSON.stringify({ v: 1, action: "resume", nonce: config.nonce, ready_sha256: own, peer_ready_sha256: peer, exchange_sha256: "d".repeat(64) }), { mode: 0o600 }); },
    close() { ["DRIFT_WORKER_CONFIG", "DRIFT_WORKER_CONFIG_SHA256"].forEach((key, i) => { if (previous[i] === undefined) delete process.env[key]; else process.env[key] = previous[i]; }); rmSync(root, { recursive: true, force: true }); } };
}
