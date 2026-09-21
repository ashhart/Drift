import { ServiceClient, ServiceError } from "./client";
import { renderHex } from "./board";
import { connectionSettings } from "./connection";
import { notifyServiceError } from "./errors";
import { isHexDigest, Op } from "./protocol";
import { assignmentsOf, noteDisconnect, persistRunState, refreshStatus, renderAll, roomMessage, runStateOf, runStates } from "./room";
import { parseStartArguments } from "./start_arguments";
import type { CommandContext, OmpExtensionApi, RunState } from "./types";
import { describeError } from "./util";

export async function startRun(api: OmpExtensionApi, context: CommandContext, rest: string): Promise<void> {
	const existing = runStateOf(context);
	if (existing?.connected) {
		context.ui.notify("A Drift run is already connected (" + existing.phase + "); run /drift stop first.", "error");
		return;
	}
	const parsed = parseStartArguments(rest);
	if (typeof parsed === "string") {
		context.ui.notify(parsed, "error");
		return;
	}
	const settings = connectionSettings();
	if (typeof settings === "string") {
		context.ui.notify(settings, "error");
		return;
	}

	const state: RunState = { phase: "SETUP", epoch: 0, connected: false };
	const client = new ServiceClient({
		host: settings.host,
		port: settings.port,
		secret: settings.secret,
		onClose: reason => {
			if (runStates.get(context.sessionManager) !== state) return;
			noteDisconnect(api, context, state, reason);
		},
	});
	try {
		await client.connect();
	} catch (error) {
		context.ui.notify("Drift could not connect to the service on " + settings.host + ":" + settings.port + ": " + describeError(error), "error");
		return;
	}
	state.client = client;
	state.connected = true;
	runStates.set(context.sessionManager, state);
	renderAll(context, state);

	const assignments: Record<string, string> = {};
	for (const [member, text] of assignmentsOf(context)) assignments[String(member)] = text;
	try {
		const setup = await client.request(Op.SETUP, { manifest: parsed.manifest, task: parsed.task, assignments });
		if (!isHexDigest(setup.manifest_sha256)) {
			client.close();
			throw new ServiceError("MALFORMED", "SETUP result carries no manifest_sha256 digest");
		}
		state.manifestSha256 = setup.manifest_sha256.toLowerCase();
		renderAll(context, state);
		await client.request(Op.START, { checkpoint_every: parsed.checkpointEvery });
		await refreshStatus(api, context, state);
	} catch (error) {
		notifyServiceError(context, "start", error);
		if (!(error instanceof ServiceError) || error.closesConnection) return;
		// A protocol refusal (PHASE, BUDGET, ...) leaves the socket open; give the owner the picture.
		await refreshStatus(api, context, state).catch(() => undefined);
		return;
	}
	assignmentsOf(context).clear();
	persistRunState(api, state);
	roomMessage(
		api,
		"[Drift] Run started in reference mode; live GLM/Qwen integration is BLOCKED: phase " + state.phase + ", epoch " + state.epoch + ", manifest " + renderHex(state.manifestSha256) + ".",
		{ kind: "started", phase: state.phase, epoch: state.epoch, manifestSha256: state.manifestSha256 },
	);
	context.ui.notify("Drift reference run is " + state.phase + " at epoch " + state.epoch + ".", "info");
}
