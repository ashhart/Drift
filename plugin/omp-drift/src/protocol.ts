import { createHmac, timingSafeEqual } from "node:crypto";
import { isRecord } from "./util";

/** Fixed integer op enum from docs/reference/service/SERVICE_PROTOCOL.md. */
export const Op = {
	SETUP: 1,
	START: 2,
	TICK: 3,
	STATUS: 4,
	PAUSE: 5,
	CHECKPOINT: 6,
	ABORT: 7,
	COMPLETE: 8,
	MAIL: 9,
} as const;
export type Op = (typeof Op)[keyof typeof Op];

export const PHASES = ["SETUP", "STRICT", "CLOSED"] as const;
export type Phase = (typeof PHASES)[number];

export const ERROR_NAMES = ["AUTH", "PHASE", "MALFORMED", "POISONED", "BUDGET", "UNKNOWN_OP", "INTERNAL"] as const;
export type ServiceErrorName = (typeof ERROR_NAMES)[number];

export const AUTH_FIELD = "auth";

export function isPhase(value: unknown): value is Phase {
	return typeof value === "string" && (PHASES as readonly string[]).includes(value);
}

export function isServiceErrorName(value: unknown): value is ServiceErrorName {
	return typeof value === "string" && (ERROR_NAMES as readonly string[]).includes(value);
}

/** Enum names are bare identifiers; this is the only kind of free string the plugin renders. */
export function isEnumName(value: unknown): value is string {
	return typeof value === "string" && /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(value);
}

/** sha256 digests are 64 hex characters. */
export function isHexDigest(value: unknown): value is string {
	return typeof value === "string" && /^[0-9a-f]{64}$/i.test(value);
}

function quoteAscii(text: string): string {
	// JSON.stringify already escapes quotes, backslashes and C0 controls the
	// same way Python does; Python's ensure_ascii additionally escapes every
	// code unit outside printable ASCII, so match that for byte-identical
	// canonical forms on both sides.
	return JSON.stringify(text).replace(/[^\x20-\x7e]/g, char => {
		return "\\u" + char.charCodeAt(0).toString(16).padStart(4, "0");
	});
}

// Canonical JSON matches the documented Python service encoding.
export function canonicalJson(value: unknown): string {
	if (value === null) return "null";
	if (typeof value === "boolean") return value ? "true" : "false";
	if (typeof value === "number") {
		if (!Number.isFinite(value)) throw new Error("non-finite number in canonical JSON");
		return Number.isInteger(value) ? String(value) : JSON.stringify(value);
	}
	if (typeof value === "string") return quoteAscii(value);
	if (Array.isArray(value)) return "[" + value.map(item => canonicalJson(item)).join(",") + "]";
	if (isRecord(value)) {
		const keys = Object.keys(value)
			.filter(key => value[key] !== undefined)
			.sort();
		return "{" + keys.map(key => quoteAscii(key) + ":" + canonicalJson(value[key])).join(",") + "}";
	}
	throw new Error("unsupported value in canonical JSON");
}

function withoutAuth(message: Record<string, unknown>): Record<string, unknown> {
	const { [AUTH_FIELD]: _auth, ...rest } = message;
	return rest;
}

export function computeAuth(message: Record<string, unknown>, secret: string): string {
	return createHmac("sha256", secret).update(canonicalJson(withoutAuth(message))).digest("hex");
}

/** Return a copy of the message carrying its `auth` field. */
export function signMessage(message: Record<string, unknown>, secret: string): Record<string, unknown> {
	return { ...withoutAuth(message), [AUTH_FIELD]: computeAuth(message, secret) };
}

/** True when the message carries a hex `auth` that matches its canonical body. */
export function verifyAuth(message: Record<string, unknown>, secret: string): boolean {
	const provided = message[AUTH_FIELD];
	if (!isHexDigest(provided)) return false;
	let expected: string;
	try {
		expected = computeAuth(message, secret);
	} catch {
		return false;
	}
	const a = Buffer.from(provided.toLowerCase(), "hex");
	const b = Buffer.from(expected, "hex");
	return a.length === b.length && timingSafeEqual(a, b);
}
