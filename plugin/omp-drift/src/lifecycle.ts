import { requireLiveRun, typedIntegers } from "./command_arguments";
import { notifyServiceError } from "./errors";
import { Op } from "./protocol";
import { persistRunState, refreshStatus, roomMessage, runStateOf, teardownRun } from "./room";
import type { CommandContext, OmpExtensionApi, RunState } from "./types";

export async function complete(api: OmpExtensionApi, context: CommandContext, rest: string): Promise<void> {
	if (typedIntegers(context, "complete", rest, 0) === undefined) return;
	const state = requireLiveRun(context, "complete");
	if (!state) return;
	try {
		await state.client!.request(Op.COMPLETE);
		await refreshStatus(api, context, state);
	} catch (error) {
		notifyServiceError(context, "complete", error);
		return;
	}
	persistRunState(api, state);
	roomMessage(
		api,
		"[Drift] Run completed at epoch " + state.epoch + "; final outputs are in the audit store, not shown here.",
		{ kind: "completed", phase: state.phase, epoch: state.epoch, manifestSha256: state.manifestSha256 },
	);
	context.ui.notify("Drift run completed (" + state.phase + ").", "info");
}

export async function abortRun(api: OmpExtensionApi, context: CommandContext, state: RunState): Promise<boolean> {
	try {
		await state.client!.request(Op.ABORT);
		await refreshStatus(api, context, state).catch(() => undefined);
	} catch (error) {
		notifyServiceError(context, "abort", error);
		return false;
	}
	persistRunState(api, state);
	roomMessage(api, "[Drift] Run aborted at epoch " + state.epoch + "; the run is poisoned and cannot complete.", {
		kind: "aborted",
		phase: state.phase,
		epoch: state.epoch,
		manifestSha256: state.manifestSha256,
	});
	return true;
}

export async function abort(api: OmpExtensionApi, context: CommandContext, rest: string): Promise<void> {
	if (typedIntegers(context, "abort", rest, 0) === undefined) return;
	const state = runStateOf(context);
	if (!state?.client || !state.connected) {
		context.ui.notify("No Drift run is connected.", "error");
		return;
	}
	if (await abortRun(api, context, state)) context.ui.notify("Drift run aborted.", "warning");
}

export async function stop(api: OmpExtensionApi, context: CommandContext, rest: string): Promise<void> {
	if (typedIntegers(context, "stop", rest, 0) === undefined) return;
	const state = runStateOf(context);
	if (!state) {
		context.ui.notify("No Drift run in this session.", "info");
		return;
	}
	if (state.client && state.connected && state.phase === "STRICT") {
		await abortRun(api, context, state);
	}
	teardownRun(context);
	roomMessage(api, "[Drift] Stopped; the service connection is closed.", { kind: "stopped" });
	context.ui.notify("Drift stopped for this session.", "info");
}
