import { ServiceError } from "./client";
import { MalformedStatusError } from "./room";
import type { CommandContext } from "./types";
import { describeError } from "./util";

export function notifyServiceError(context: CommandContext, action: string, error: unknown): void {
	if (error instanceof ServiceError) {
		context.ui.notify("Drift " + action + " failed: " + error.message + " [" + error.code + "]", "error");
		return;
	}
	if (error instanceof MalformedStatusError) {
		context.ui.notify("Drift " + action + " failed: " + error.message + " [MALFORMED]", "error");
		return;
	}
	context.ui.notify("Drift " + action + " failed: " + describeError(error), "error");
}
