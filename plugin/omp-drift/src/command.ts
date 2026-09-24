import { Op } from "./protocol";
import { assignmentsOf, runStateOf } from "./room";
import { abort, complete, stop } from "./lifecycle";
import { checkpoint, mail, showStatus, simpleStrictOp, tick } from "./run_commands";
import { startRun } from "./start";
import type { CommandContext, OmpExtensionApi } from "./types";
import { parseIntegerArgument } from "./util";

export { connectionSettings } from "./connection";

const ACTIONS = ["start", "assign", "tick", "pause", "status", "checkpoint", "mail", "complete", "abort", "stop", "subagent"] as const;
type Action = (typeof ACTIONS)[number];
const FREE_TEXT_ACTIONS: ReadonlySet<Action> = new Set<Action>(["start", "assign"]);

export function registerDriftCommand(api: OmpExtensionApi, subagent?: (args: string, context: CommandContext) => Promise<void>): void {
	api.registerCommand("drift", {
		description: "Drive a Drift run over the local worker service: start, assign, tick, pause, status, checkpoint, mail, complete, abort, stop",
		getArgumentCompletions: prefix => {
			const query = prefix.trim().toLowerCase();
			if (query.includes(" ")) return null;
			const matches = ACTIONS.filter(action => action.startsWith(query));
			return matches.length > 0 ? matches.map(action => ({ value: action, label: action })) : null;
		},
		handler: async (args, context) => {
			const trimmed = args.trim();
			const space = trimmed.search(/\s/);
			const word = (space < 0 ? trimmed : trimmed.slice(0, space)).toLowerCase();
			const rest = space < 0 ? "" : trimmed.slice(space).trim();
			const action = ACTIONS.find(candidate => candidate === word);
			if (!action) {
				context.ui.notify("Usage: /drift <" + ACTIONS.join("|") + "> …", word.length === 0 ? "info" : "error");
				return;
			}

			const live = runStateOf(context);
			if (live?.phase === "STRICT" && live.connected && FREE_TEXT_ACTIONS.has(action)) {
				context.ui.notify("Drift refuses " + action + " during STRICT: free text is only accepted in SETUP.", "error");
				return;
			}

			switch (action) {
				case "subagent":
					if (subagent) await subagent(rest, context);
					else context.ui.notify("Native Drift needs pinned worker, boundary and exchange profiles; see docs/guides/DRIFT_SUBAGENTS.md.", "error");
					return;
				case "start":
					await startRun(api, context, rest);
					return;
				case "assign":
					stageAssignment(context, rest);
					return;
				case "status":
					await showStatus(api, context, rest);
					return;
				case "tick":
					await tick(api, context, rest);
					return;
				case "pause":
					await simpleStrictOp(api, context, "pause", Op.PAUSE, rest);
					return;
				case "checkpoint":
					await checkpoint(api, context, rest);
					return;
				case "mail":
					await mail(api, context, rest);
					return;
				case "complete":
					await complete(api, context, rest);
					return;
				case "abort":
					await abort(api, context, rest);
					return;
				case "stop":
					await stop(api, context, rest);
					return;
			}
		},
	});
}

export function stageAssignment(context: CommandContext, rest: string): void {
	const live = runStateOf(context);
	if (live?.connected) {
		context.ui.notify("Drift assignments are staged before /drift start; a run is already connected.", "error");
		return;
	}
	const space = rest.search(/\s/);
	const memberText = space < 0 ? rest : rest.slice(0, space);
	const text = space < 0 ? "" : rest.slice(space).trim();
	const member = parseIntegerArgument(memberText);
	if (member === undefined || text.length === 0) {
		context.ui.notify("Usage: /drift assign <member-index> <text> (before start).", "error");
		return;
	}
	assignmentsOf(context).set(member, text);
	context.ui.notify("Drift staged an assignment for member " + member + " (" + assignmentsOf(context).size + " staged).", "info");
}
