import { createHash } from "node:crypto";
import { createServer, type Server, type Socket } from "node:net";
import { ERROR_NAMES, isEnumName, isHexDigest, Op, signMessage, verifyAuth, type Phase, type ServiceErrorName } from "../src/protocol";
import { isNonNegativeInteger, isRecord } from "../src/util";

export interface FakeServiceOptions {
	secret: string;
	/** Sign responses with a wrong digest so the plugin must reject them. */
	corruptResponseAuth?: boolean;
	/** Answer every request with a line that is not JSON. */
	malformedResponse?: boolean;
	/** Never answer; lets the plugin's timeout fire. */
	silent?: boolean;
}

const manifestSha256 = createHash("sha256").update("manifest").digest("hex");

/**
 * In-test worker service: validates HMAC on every line, tracks the phase
 * machine from docs/reference/service/SERVICE_PROTOCOL.md and answers each op with a typed
 * result. Records every authenticated request for assertions.
 */
export class FakeService {
	phase: Phase = "SETUP";
	epoch = 0;
	poisoned = false;
	paused = false;
	checkpoints = 0;
	lastCheckpointSha256: string | undefined;
	mailSent = 0;
	assignments: Record<string, string> = {};
	task = "";
	manifestPath = "";
	readonly requests: Array<Record<string, unknown>> = [];
	readonly rejectedLines: string[] = [];
	connectionsOpened = 0;
	private readonly sockets = new Set<Socket>();
	private closeWaiters: Array<() => void> = [];

	private constructor(
		private readonly server: Server,
		private readonly options: FakeServiceOptions,
	) {}

	static start(options: FakeServiceOptions): Promise<FakeService> {
		return new Promise((resolve, reject) => {
			const server = createServer();
			const service = new FakeService(server, options);
			server.on("connection", socket => service.accept(socket));
			server.once("error", reject);
			server.listen(0, "127.0.0.1", () => resolve(service));
		});
	}

	get port(): number {
		const address = this.server.address();
		if (!address || typeof address === "string") throw new Error("fake service is not listening");
		return address.port;
	}

	get openConnections(): number {
		return this.sockets.size;
	}

	/** Resolves once every client socket has closed. */
	waitForAllClosed(): Promise<void> {
		if (this.sockets.size === 0) return Promise.resolve();
		return new Promise(resolve => this.closeWaiters.push(resolve));
	}

	requestsWithOp(op: Op): Array<Record<string, unknown>> {
		return this.requests.filter(request => request.op === op);
	}

	close(): Promise<void> {
		for (const socket of this.sockets) socket.destroy();
		return new Promise(resolve => this.server.close(() => resolve()));
	}

	status(): Record<string, unknown> {
		return {
			phase: this.phase,
			epoch: this.epoch,
			poisoned: this.poisoned,
			members: [0, 1].map(index => ({
				index,
				writer: 1 - index,
				epoch: Math.max(0, this.epoch - 1),
				sequence: this.epoch,
				source_position: this.epoch * 4,
				local_tokens: this.epoch * 4,
				mail_sent: index === 0 ? this.mailSent : 0,
				foreign_tokens: this.epoch * 2,
				mail_foreign_tokens: this.mailSent,
				gate: { "3": this.epoch > 0 ? "open" : "closed", "7": "open" },
				mass_mean: { "3": 0.12 },
			})),
			mailbox: {
				counts: {
					pending: 0,
					visible: this.mailSent,
					attended_not_proven: 0,
					causally_incorporated: 0,
					expired_without_incorporation: 0,
					rejected: 0,
					cancelled: 0,
				},
			},
			detectors: { nonfinite: false, norm_drift: { "3": 0.01 }, mass_oscillation: { "3": 0.2 } },
			manifest_sha256: manifestSha256,
			last_checkpoint_sha256: this.lastCheckpointSha256 ?? null,
		};
	}

	private accept(socket: Socket): void {
		this.connectionsOpened += 1;
		this.sockets.add(socket);
		socket.setEncoding("utf8");
		let buffer = "";
		socket.on("data", chunk => {
			buffer += String(chunk);
			let newline = buffer.indexOf("\n");
			while (newline >= 0) {
				const line = buffer.slice(0, newline);
				buffer = buffer.slice(newline + 1);
				if (!this.handleLine(socket, line)) return;
				newline = buffer.indexOf("\n");
			}
		});
		socket.on("error", () => undefined);
		socket.on("close", () => {
			this.sockets.delete(socket);
			if (this.sockets.size === 0) {
				const waiters = this.closeWaiters;
				this.closeWaiters = [];
				for (const waiter of waiters) waiter();
			}
		});
	}

	private send(socket: Socket, response: Record<string, unknown>): void {
		if (this.options.silent) return;
		if (this.options.malformedResponse) {
			socket.write("this is not json\n");
			return;
		}
		const signed = signMessage(response, this.options.secret);
		if (this.options.corruptResponseAuth) signed.auth = "0".repeat(64);
		socket.write(JSON.stringify(signed) + "\n");
	}

	/** Returns false when the line closed the connection. */
	private handleLine(socket: Socket, line: string): boolean {
		let parsed: unknown;
		try {
			parsed = JSON.parse(line);
		} catch {
			this.rejectedLines.push(line);
			socket.destroy();
			return false;
		}
		if (!isRecord(parsed) || !verifyAuth(parsed, this.options.secret)) {
			this.rejectedLines.push(line);
			const id = isRecord(parsed) && typeof parsed.id === "number" ? parsed.id : 0;
			this.send(socket, { id, ok: false, error: "AUTH" });
			socket.end();
			return false;
		}
		this.requests.push(parsed);
		const id = parsed.id;
		const outcome = this.handle(parsed);
		if ("error" in outcome) this.send(socket, { id, ok: false, error: outcome.error });
		else this.send(socket, { id, ok: true, result: outcome.result });
		return true;
	}

	private handle(request: Record<string, unknown>): { result: Record<string, unknown> } | { error: ServiceErrorName } {
		const op = request.op;
		if (!isNonNegativeInteger(op) || op < 1 || op > 9) return { error: "UNKNOWN_OP" };
		if (this.phase === "STRICT") {
			for (const [key, value] of Object.entries(request)) {
				if (key === "auth") continue;
				if (typeof value === "string" && !isEnumName(value) && !isHexDigest(value)) return { error: "MALFORMED" };
			}
		}
		const strictOnly = (fn: () => { result: Record<string, unknown> } | { error: ServiceErrorName }) => {
			if (this.phase !== "STRICT") return { error: "PHASE" as const };
			if (this.poisoned) return { error: "POISONED" as const };
			return fn();
		};
		switch (op) {
			case Op.SETUP:
				if (this.phase !== "SETUP") return { error: "PHASE" };
				if (typeof request.manifest !== "string" || typeof request.task !== "string" || !isRecord(request.assignments)) {
					return { error: "MALFORMED" };
				}
				this.manifestPath = request.manifest;
				this.task = request.task;
				this.assignments = Object.fromEntries(
					Object.entries(request.assignments).filter((entry): entry is [string, string] => typeof entry[1] === "string"),
				);
				return { result: { manifest_sha256: manifestSha256, members: [0, 1] } };
			case Op.START:
				if (this.phase !== "SETUP") return { error: "PHASE" };
				if (!isNonNegativeInteger(request.checkpoint_every) || request.checkpoint_every < 1) return { error: "MALFORMED" };
				this.phase = "STRICT";
				this.epoch = 0;
				return { result: { phase: this.phase, epoch: this.epoch } };
			case Op.TICK:
				return strictOnly(() => {
					if (!isNonNegativeInteger(request.epochs) || request.epochs < 1) return { error: "MALFORMED" };
					this.epoch += request.epochs;
					this.paused = false;
					return { result: { epoch: this.epoch } };
				});
			case Op.STATUS:
				return { result: this.status() };
			case Op.PAUSE:
				return strictOnly(() => {
					this.paused = true;
					return { result: { epoch: this.epoch } };
				});
			case Op.CHECKPOINT:
				return strictOnly(() => {
					this.checkpoints += 1;
					this.lastCheckpointSha256 = createHash("sha256").update("checkpoint-" + this.checkpoints).digest("hex");
					return { result: { sha256: this.lastCheckpointSha256 } };
				});
			case Op.ABORT:
				this.poisoned = true;
				this.phase = "CLOSED";
				return { result: { phase: this.phase, poisoned: true } };
			case Op.COMPLETE:
				return strictOnly(() => {
					this.phase = "CLOSED";
					return { result: { phase: this.phase, epoch: this.epoch } };
				});
			case Op.MAIL:
				return strictOnly(() => {
					if (!isNonNegativeInteger(request.sender) || request.sender > 1) return { error: "MALFORMED" };
					if (!isNonNegativeInteger(request.slots) || request.slots < 1) return { error: "MALFORMED" };
					this.mailSent += 1;
					return { result: { sender: request.sender, slots: request.slots } };
				});
			default:
				return { error: ERROR_NAMES[4] };
		}
	}
}

export { manifestSha256 as fakeManifestSha256 };
