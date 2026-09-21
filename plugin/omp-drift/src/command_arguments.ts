import { runStateOf } from "./room";
import type { CommandContext, RunState } from "./types";
import { parseIntegerArgument } from "./util";

export function requireLiveRun(context: CommandContext, action: string): RunState | undefined {
	const state = runStateOf(context);
	if (!state || !state.client || !state.connected) {
		context.ui.notify(
			state?.resumed
				? "Drift is showing a resumed run but is not connected; run /drift stop, then start a new run."
				: "No Drift run is connected; run /drift start <manifest-path> first.",
			"error",
		);
		return undefined;
	}
	if (state.phase !== "STRICT") {
		context.ui.notify("Drift " + action + " needs a STRICT run; the run is " + state.phase + ".", "error");
		return undefined;
	}
	return state;
}


export function typedIntegers(context: CommandContext, action: string, rest: string, max: number): number[] | undefined {
	const tokens = rest.split(/\s+/).filter(token => token.length > 0);
	if (tokens.length > max) {
		context.ui.notify("Drift " + action + " takes at most " + max + " integer argument(s).", "error");
		return undefined;
	}
	const values: number[] = [];
	for (const token of tokens) {
		const value = parseIntegerArgument(token);
		if (value === undefined) {
			context.ui.notify("Drift " + action + " accepts integers only; free text is refused.", "error");
			return undefined;
		}
		values.push(value);
	}
	return values;
}
