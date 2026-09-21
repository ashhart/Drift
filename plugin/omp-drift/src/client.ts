import { createConnection, type Socket } from "node:net";
import { isServiceErrorName, signMessage, verifyAuth, type Op, type ServiceErrorName } from "./protocol";
import { isRecord } from "./util";

/** Service error names plus the client's own transport-level codes. */
export type ServiceErrorCode = ServiceErrorName | "TRANSPORT" | "TIMEOUT" | "BUSY" | "CLOSED";

export class ServiceError extends Error {
	constructor(
		readonly code: ServiceErrorCode,
		message?: string,
	) {
		super(message ?? code);
		this.name = "ServiceError";
	}

	/** Codes after which the socket is gone (or was closed on purpose). */
	get closesConnection(): boolean {
		return this.code === "AUTH" || this.code === "MALFORMED" || this.code === "TRANSPORT" ||
			this.code === "TIMEOUT" || this.code === "CLOSED";
	}
}

export interface ServiceClientOptions {
	host: string;
	port: number;
	secret: string;
	/** Per-request deadline; a timeout desynchronises the stream, so it closes the socket. */
	timeoutMs?: number;
	/** Called once when the socket closes, with the failure that caused it when there was one. */
	onClose?: (reason: ServiceError | undefined) => void;
}

interface InFlight {
	id: number;
	resolve: (result: Record<string, unknown>) => void;
	reject: (error: ServiceError) => void;
	timer: ReturnType<typeof setTimeout>;
}

export const defaultTimeoutMs = 60_000;

// Serialize requests and close the socket on authentication, framing or deadline failure.
export class ServiceClient {
	private socket: Socket | undefined;
	private buffer = "";
	private nextId = 1;
	private inFlight: InFlight | undefined;
	private open = false;
	private closedByUs = false;
	private closeNotified = false;

	constructor(private readonly options: ServiceClientOptions) {}

	get isOpen(): boolean {
		return this.open;
	}

	connect(): Promise<void> {
		return new Promise((resolve, reject) => {
			const socket = createConnection({ host: this.options.host, port: this.options.port });
			this.socket = socket;
			socket.setEncoding("utf8");
			socket.once("connect", () => {
				this.open = true;
				resolve();
			});
			socket.on("data", chunk => this.onData(String(chunk)));
			socket.on("error", error => {
				const failure = new ServiceError("TRANSPORT", "service connection failed: " + error.message);
				if (!this.open) reject(failure);
				this.fail(failure);
			});
			socket.on("close", () => {
				const wasOpen = this.open;
				this.open = false;
				if (!wasOpen && !this.closedByUs) reject(new ServiceError("TRANSPORT", "service connection closed"));
				this.settleClose(this.closedByUs ? undefined : new ServiceError("CLOSED", "service closed the connection"));
			});
		});
	}

	/** Send one typed request and resolve with its `result` object. */
	request(op: Op, fields: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
		if (!this.open || !this.socket) return Promise.reject(new ServiceError("CLOSED", "not connected to the service"));
		if (this.inFlight) return Promise.reject(new ServiceError("BUSY", "a service request is already in flight"));
		const id = this.nextId++;
		const message = signMessage({ ...fields, id, op }, this.options.secret);
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.fail(new ServiceError("TIMEOUT", "service request " + id + " timed out"));
			}, this.options.timeoutMs ?? defaultTimeoutMs);
			this.inFlight = { id, resolve, reject, timer };
			this.socket?.write(JSON.stringify(message) + "\n");
		});
	}

	/** Close the socket deliberately; a pending request rejects with CLOSED. */
	close(): void {
		this.closedByUs = true;
		this.open = false;
		this.rejectInFlight(new ServiceError("CLOSED", "connection closed"));
		this.socket?.destroy();
		this.settleClose(undefined);
	}

	private onData(chunk: string): void {
		this.buffer += chunk;
		let newline = this.buffer.indexOf("\n");
		while (newline >= 0) {
			const line = this.buffer.slice(0, newline).trim();
			this.buffer = this.buffer.slice(newline + 1);
			if (line.length > 0 && !this.handleLine(line)) return;
			newline = this.buffer.indexOf("\n");
		}
	}

	/** Returns false when the line closed the connection. */
	private handleLine(line: string): boolean {
		let parsed: unknown;
		try {
			parsed = JSON.parse(line);
		} catch {
			this.fail(new ServiceError("MALFORMED", "service sent a line that is not JSON"));
			return false;
		}
		if (!isRecord(parsed)) {
			this.fail(new ServiceError("MALFORMED", "service sent a non-object line"));
			return false;
		}
		if (!verifyAuth(parsed, this.options.secret)) {
			this.fail(new ServiceError("AUTH", "service response failed authentication"));
			return false;
		}
		const pending = this.inFlight;
		if (!pending || parsed.id !== pending.id || typeof parsed.ok !== "boolean") {
			this.fail(new ServiceError("MALFORMED", "service response did not match the request in flight"));
			return false;
		}
		if (parsed.ok) {
			if (!isRecord(parsed.result)) {
				this.fail(new ServiceError("MALFORMED", "service response carries no result object"));
				return false;
			}
			this.settle(pending);
			pending.resolve(parsed.result);
			return true;
		}
		if (!isServiceErrorName(parsed.error)) {
			this.fail(new ServiceError("MALFORMED", "service refused with an unknown error name"));
			return false;
		}
		this.settle(pending);
		if (parsed.error === "AUTH") {
			// The service drops unauthenticated peers; do not wait for it.
			this.fail(new ServiceError("AUTH", "service rejected the request as unauthenticated"));
			pending.reject(new ServiceError("AUTH", "service rejected the request as unauthenticated"));
			return false;
		}
		pending.reject(new ServiceError(parsed.error, "service refused: " + parsed.error));
		return true;
	}

	private settle(pending: InFlight): void {
		clearTimeout(pending.timer);
		if (this.inFlight === pending) this.inFlight = undefined;
	}

	private rejectInFlight(error: ServiceError): void {
		const pending = this.inFlight;
		if (!pending) return;
		this.settle(pending);
		pending.reject(error);
	}

	/** Tear the connection down after a protocol failure. */
	private fail(error: ServiceError): void {
		if (!this.open && !this.socket) return;
		this.open = false;
		this.rejectInFlight(error);
		this.socket?.destroy();
		this.settleClose(error);
	}

	private settleClose(reason: ServiceError | undefined): void {
		if (this.closeNotified) return;
		this.closeNotified = true;
		this.open = false;
		this.socket = undefined;
		this.options.onClose?.(reason);
	}
}
