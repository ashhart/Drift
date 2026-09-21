import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { StringDecoder } from "node:string_decoder";
import type { Envelope, WorkerTransport } from "./worker_protocol";

export class WorkerProcess implements WorkerTransport {
  private child: ChildProcessWithoutNullStreams;
  private decoder = new StringDecoder("utf8");
  private buffer = "";
  private stopped = false;
  private message?: (value: unknown) => void;
  private failure?: (error: Error) => void;
  constructor(command: string, args: string[], options: { cwd: string; env: NodeJS.ProcessEnv; maxBytes: number }) {
    this.child = spawn(command, args, { cwd: options.cwd, env: options.env, stdio: "pipe", shell: false });
    this.child.stdout.on("data", (chunk: Buffer) => {
      if (this.stopped) return;
      this.buffer += this.decoder.write(chunk);
      for (;;) {
        const end = this.buffer.indexOf("\n");
        if (end < 0) break;
        const line = this.buffer.slice(0, end); this.buffer = this.buffer.slice(end + 1);
        if (Buffer.byteLength(line) > options.maxBytes) { this.fail(); return; }
        try { this.message?.(JSON.parse(line)); } catch { this.fail(); return; }
      }
      if (Buffer.byteLength(this.buffer) > options.maxBytes) this.fail();
    });
    this.child.stderr.resume();
    this.child.on("error", () => this.fail());
    this.child.stdin.on("error", () => this.fail());
    this.child.on("close", () => { if (!this.stopped) this.fail(); });
  }
  subscribe(message: (value: unknown) => void, failure: (error: Error) => void): void { this.message = message; this.failure = failure; }
  send(message: Envelope): void {
    if (this.stopped) throw new Error("DRIFT_WORKER_CLOSED");
    this.child.stdin.write(JSON.stringify(message) + "\n");
  }
  private fail(): void { this.failure?.(new Error("DRIFT_WORKER_TRANSPORT")); this.close(); }
  close(): void {
    if (this.stopped) return;
    this.stopped = true; this.child.stdin.end(); this.child.kill("SIGTERM");
    const timer = setTimeout(() => { if (this.child.exitCode === null) this.child.kill("SIGKILL"); }, 500);
    timer.unref(); this.child.once("exit", () => clearTimeout(timer));
  }
}
