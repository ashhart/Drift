import { renderBoard, renderHex, renderStatusLine } from "./board";
import type { ServiceError } from "./client";
import { boardKey, roomMessageType, runStateEntryType, statusKey } from "./identity";
import { isHexDigest, isPhase, Op, type Phase } from "./protocol";
import { parseStatus, type ServiceStatus } from "./status";
import type { CommandContext, OmpExtensionApi, RunState } from "./types";
import { isNonNegativeInteger, isRecord } from "./util";

export const runStates = new WeakMap<object, RunState>();
/** Assignments staged with `/drift assign` before `/drift start`; member index → text. */
export const stagedAssignments = new WeakMap<object, Map<number, string>>();

export function runStateOf(context: CommandContext): RunState | undefined {
	return runStates.get(context.sessionManager);
}

export function assignmentsOf(context: CommandContext): Map<number, string> {
	let staged = stagedAssignments.get(context.sessionManager);
	if (!staged) {
		staged = new Map<number, string>();
		stagedAssignments.set(context.sessionManager, staged);
	}
	return staged;
}

export function roomMessage(api: OmpExtensionApi, content: string, details: Record<string, unknown>): void {
	api.sendMessage({
		customType: roomMessageType,
		content,
		display: true,
		details,
		attribution: "agent",
	});
}

export function renderStatus(context: CommandContext, state: RunState | undefined): void {
	context.ui.setStatus(statusKey, state ? renderStatusLine(state) : undefined);
}

export function renderRunBoard(context: CommandContext, state: RunState | undefined): void {
	context.ui.setWidget?.(boardKey, state ? renderBoard(state) : undefined);
}

export function renderAll(context: CommandContext, state: RunState | undefined): void {
	renderStatus(context, state);
	renderRunBoard(context, state);
}

/** Persist the typed run summary so a resumed session can show the last known phase. */
export function persistRunState(api: OmpExtensionApi, state: RunState): void {
	api.appendEntry(runStateEntryType, {
		manifestSha256: state.manifestSha256,
		phase: state.phase,
		epoch: state.epoch,
	});
}

/** Latest persisted run for this branch, or undefined when none. */
export function persistedRunFromBranch(entries: unknown[]): RunState | undefined {
	let persisted: RunState | undefined;
	for (const entry of entries) {
		if (!isRecord(entry)) continue;
		if ((entry.type !== "custom" && entry.type !== "custom_message") || entry.customType !== runStateEntryType) {
			continue;
		}
		const data = isRecord(entry.data) ? entry.data : isRecord(entry.details) ? entry.details : undefined;
		if (!data || !isPhase(data.phase)) continue;
		persisted = {
			phase: data.phase,
			epoch: isNonNegativeInteger(data.epoch) ? data.epoch : 0,
			manifestSha256: isHexDigest(data.manifestSha256) ? data.manifestSha256 : undefined,
			connected: false,
			resumed: true,
		};
	}
	return persisted;
}

// Apply typed status and announce phase, poison and checkpoint changes.
export function applyStatus(api: OmpExtensionApi, context: CommandContext, state: RunState, status: ServiceStatus): void {
	const previousPhase: Phase | undefined = state.lastStatus?.phase;
	const previousPoisoned = state.lastStatus?.poisoned === true;
	const previousCheckpoint = state.lastStatus?.lastCheckpointSha256;
	state.lastStatus = status;
	state.phase = status.phase;
	state.epoch = status.epoch;
	state.manifestSha256 = status.manifestSha256;
	renderAll(context, state);
	if (previousPhase !== undefined && previousPhase !== status.phase) {
		roomMessage(api, "[Drift] Phase " + previousPhase + " → " + status.phase + " at epoch " + status.epoch + ".", {
			kind: "phase",
			from: previousPhase,
			to: status.phase,
			epoch: status.epoch,
		});
	}
	if (!previousPoisoned && status.poisoned) {
		roomMessage(api, "[Drift] The run is poisoned at epoch " + status.epoch + "; no completion is possible.", {
			kind: "poisoned",
			epoch: status.epoch,
		});
	}
	if (status.lastCheckpointSha256 && status.lastCheckpointSha256 !== previousCheckpoint) {
		roomMessage(
			api,
			"[Drift] Checkpoint " + renderHex(status.lastCheckpointSha256) + " at epoch " + status.epoch + ".",
			{ kind: "checkpoint", sha256: status.lastCheckpointSha256, epoch: status.epoch },
		);
	}
}

/** Ask the service for its status and fold it in. Throws ServiceError on failure. */
export async function refreshStatus(api: OmpExtensionApi, context: CommandContext, state: RunState): Promise<ServiceStatus> {
	if (!state.client) throw new Error("not connected");
	const result = await state.client.request(Op.STATUS);
	const status = parseStatus(result);
	if (!status) {
		// Off-schema status is a protocol violation: drop the connection like any malformed line.
		state.client.close();
		throw new MalformedStatusError();
	}
	applyStatus(api, context, state, status);
	return status;
}

export class MalformedStatusError extends Error {
	constructor() {
		super("service sent a status object that is off schema");
		this.name = "MalformedStatusError";
	}
}

/** Mark the run disconnected after the socket went away; keeps the last known picture on screen. */
export function noteDisconnect(api: OmpExtensionApi, context: CommandContext, state: RunState, reason: ServiceError | undefined): void {
	state.connected = false;
	state.client = undefined;
	renderAll(context, state);
	if (reason) {
		roomMessage(api, "[Drift] Connection to the service closed (" + reason.code + ").", {
			kind: "disconnected",
			code: reason.code,
		});
	}
}

export function teardownRun(context: CommandContext): void {
	const state = runStateOf(context);
	if (state?.client) state.client.close();
	runStates.delete(context.sessionManager);
	stagedAssignments.delete(context.sessionManager);
	renderAll(context, undefined);
}
