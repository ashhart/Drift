import { registerDriftCommand } from "./command";
import { persistedRunFromBranch, renderAll, roomMessage, runStateOf, runStates, teardownRun } from "./room";
import { renderHex } from "./board";
import type { CommandContext, OmpExtensionApi } from "./types";

// Restore a historical summary without reconnecting.
function showPersistedRun(api: OmpExtensionApi, context: CommandContext): void {
	if (runStateOf(context)) return;
	const entries = context.sessionManager.getBranch?.() ?? [];
	const state = persistedRunFromBranch(entries);
	if (!state) return;
	runStates.set(context.sessionManager, state);
	renderAll(context, state);
	roomMessage(
		api,
		"[Drift] Last known run: phase " + state.phase + ", epoch " + state.epoch + ", manifest " +
			renderHex(state.manifestSha256) + ". Not reconnected; use /drift start to begin a new run.",
		{ kind: "resumed", phase: state.phase, epoch: state.epoch, manifestSha256: state.manifestSha256 },
	);
}

export default async function driftExtension(api: OmpExtensionApi): Promise<void> {
	let subagent;
	if (process.env.DRIFT_WORKER_CONFIG) {
		const { installSubagentCommand } = await import("../../../scripts/omp/subagent_command.mjs");
		subagent = installSubagentCommand(api);
	}
	api.on("session_start", async (_event, context) => {
		// Start clean and restore only the persisted summary.
		if (runStateOf(context)) teardownRun(context);
		showPersistedRun(api, context);
	});

	api.on("session_shutdown", async (_event, context) => {
		// Close the socket without aborting; the service keeps the run's state.
		if (runStateOf(context)) teardownRun(context);
	});

	api.on("before_agent_start", async (event, context) => {
		const state = runStateOf(context);
		if (!state) return undefined;
		const base = Array.isArray(event.systemPrompt) ? event.systemPrompt : event.systemPrompt ? [event.systemPrompt] : [];
		// Expose only fixed reference-mode wording, phase and epoch.
		return {
			systemPrompt: [
				...base,
				(state.connected ? "A Drift reference run is active in this session" : "A Drift reference run was active in this session and is not connected now") +
					": phase " + state.phase + ", epoch " + state.epoch +
					". Live GLM/Qwen integration is BLOCKED; it is controlled only through /drift commands; you cannot read or write its contents.",
			],
		};
	});

	registerDriftCommand(api, subagent);
}
