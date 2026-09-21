import { renderHex } from "./board";
import { requireLiveRun, typedIntegers } from "./command_arguments";
import { notifyServiceError } from "./errors";
import { isHexDigest, Op } from "./protocol";
import { refreshStatus, renderAll, roomMessage, runStateOf } from "./room";
import type { CommandContext, OmpExtensionApi } from "./types";

export async function showStatus(api: OmpExtensionApi, context: CommandContext, rest: string): Promise<void> {
	if (typedIntegers(context, "status", rest, 0) === undefined) return;
	const state = runStateOf(context);
	if (!state) {
		context.ui.notify("No Drift run in this session.", "warning");
		return;
	}
	if (!state.client || !state.connected) {
		renderAll(context, state);
		context.ui.notify("Drift last known: " + state.phase + " at epoch " + state.epoch + " (not connected).", "info");
		return;
	}
	try {
		await refreshStatus(api, context, state);
		context.ui.notify("Drift: " + state.phase + " at epoch " + state.epoch + (state.lastStatus?.poisoned ? " (poisoned)" : "") + ".", "info");
	} catch (error) {
		notifyServiceError(context, "status", error);
	}
}

export async function tick(api: OmpExtensionApi, context: CommandContext, rest: string): Promise<void> {
	const values = typedIntegers(context, "tick", rest, 1);
	if (!values) return;
	const epochs = values[0] ?? 1;
	if (epochs < 1) {
		context.ui.notify("Drift tick needs at least one epoch.", "error");
		return;
	}
	const state = requireLiveRun(context, "tick");
	if (!state) return;
	try {
		await state.client!.request(Op.TICK, { epochs });
		await refreshStatus(api, context, state);
	} catch (error) {
		notifyServiceError(context, "tick", error);
	}
}

export async function simpleStrictOp(api: OmpExtensionApi, context: CommandContext, action: string, op: Op, rest: string): Promise<void> {
	if (typedIntegers(context, action, rest, 0) === undefined) return;
	const state = requireLiveRun(context, action);
	if (!state) return;
	try {
		await state.client!.request(op);
		await refreshStatus(api, context, state);
		context.ui.notify("Drift " + action + " done: " + state.phase + " at epoch " + state.epoch + ".", "info");
	} catch (error) {
		notifyServiceError(context, action, error);
	}
}

export async function checkpoint(api: OmpExtensionApi, context: CommandContext, rest: string): Promise<void> {
	if (typedIntegers(context, "checkpoint", rest, 0) === undefined) return;
	const state = requireLiveRun(context, "checkpoint");
	if (!state) return;
	try {
		const result = await state.client!.request(Op.CHECKPOINT);
		const digest = isHexDigest(result.sha256) ? result.sha256.toLowerCase() : undefined;
		const status = await refreshStatus(api, context, state);
		// applyStatus announces the digest when the status reports it; cover a service that only returns it here.
		if (digest && status.lastCheckpointSha256 !== digest) {
			roomMessage(api, "[Drift] Checkpoint " + renderHex(digest) + " at epoch " + status.epoch + ".", {
				kind: "checkpoint",
				sha256: digest,
				epoch: status.epoch,
			});
		}
		context.ui.notify("Drift checkpoint " + renderHex(digest ?? status.lastCheckpointSha256) + " written.", "info");
	} catch (error) {
		notifyServiceError(context, "checkpoint", error);
	}
}

export async function mail(api: OmpExtensionApi, context: CommandContext, rest: string): Promise<void> {
	const values = typedIntegers(context, "mail", rest, 2);
	if (!values) return;
	if (values.length === 0) {
		context.ui.notify("Usage: /drift mail <member-index> [slots]", "error");
		return;
	}
	const sender = values[0];
	const slots = values[1] ?? 1;
	if (slots < 1) {
		context.ui.notify("Drift mail needs at least one slot.", "error");
		return;
	}
	const state = requireLiveRun(context, "mail");
	if (!state) return;
	try {
		await state.client!.request(Op.MAIL, { sender, slots });
		await refreshStatus(api, context, state);
		context.ui.notify("Drift member " + sender + " wrote mail (" + slots + " slot" + (slots === 1 ? "" : "s") + ").", "info");
	} catch (error) {
		notifyServiceError(context, "mail", error);
	}
}
