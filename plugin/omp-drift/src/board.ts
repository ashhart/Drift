import { isEnumName, isHexDigest } from "./protocol";
import type { MemberStatus, ServiceStatus } from "./status";
import type { RunState } from "./types";

// Render only validated numbers, enum names and digests.
export function renderNumber(value: unknown): string {
	if (typeof value !== "number" || !Number.isFinite(value)) return "?";
	return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

export function renderEnum(value: unknown): string {
	return isEnumName(value) ? value : "?";
}

export function renderHex(value: unknown, length = 12): string {
	return isHexDigest(value) ? value.slice(0, length) : "—";
}

function renderLayerMap(map: Record<string, unknown>, format: (value: unknown) => string): string {
	const keys = Object.keys(map).filter(key => /^\d{1,4}$/.test(key));
	if (keys.length === 0) return "—";
	return keys.map(key => key + ":" + format(map[key])).join(" ");
}

function renderMember(member: MemberStatus): string {
	return (
		"m" + renderNumber(member.index) +
		" ▸ e" + renderNumber(member.epoch) +
		" seq " + renderNumber(member.sequence) +
		" pos " + renderNumber(member.sourcePosition) +
		" writer " + renderNumber(member.writer) +
		" · local " + renderNumber(member.localTokens) +
		" foreign " + renderNumber(member.foreignTokens) +
		" · mail sent " + renderNumber(member.mailSent) +
		" mail-foreign " + renderNumber(member.mailForeignTokens) +
		" · gates " + renderLayerMap(member.gate, renderEnum) +
		" · mass " + renderLayerMap(member.massMean, renderNumber)
	);
}

function renderMailbox(counts: Record<string, number>): string {
	const keys = Object.keys(counts).filter(key => isEnumName(key)).sort();
	if (keys.length === 0) return "mailbox ▸ —";
	return "mailbox ▸ " + keys.map(key => key + " " + renderNumber(counts[key])).join(" · ");
}

function renderDetectors(detectors: ServiceStatus["detectors"]): string {
	return (
		"detectors ▸ nonfinite " + (detectors.nonfinite ? "yes" : "no") +
		" · norm drift " + renderLayerMap(detectors.normDrift, renderNumber) +
		" · mass oscillation " + renderLayerMap(detectors.massOscillation, renderNumber)
	);
}

export function renderStatusLine(state: RunState): string {
	const line = "drift: " + renderEnum(state.phase) + " e" + renderNumber(state.epoch);
	return state.resumed ? line + " (resumed)" : line;
}

export function renderBoard(state: RunState): string[] {
	const header =
		"drift · " + renderEnum(state.phase) +
		" · epoch " + renderNumber(state.epoch) +
		" · manifest " + renderHex(state.manifestSha256) +
		(state.resumed ? " · resumed, not connected" : state.connected ? "" : " · disconnected");
	const status = state.lastStatus;
	if (!status) return [header, "backend reference · live workers BLOCKED", "no status received yet"];
	const lines = [
		header,
		"backend reference · live workers BLOCKED",
		"poisoned " + (status.poisoned ? "yes" : "no") + " · checkpoint " + renderHex(status.lastCheckpointSha256),
		...status.members.map(renderMember),
		renderMailbox(status.mailboxCounts),
		renderDetectors(status.detectors),
	];
	return lines;
}
