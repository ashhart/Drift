import { parseIntegerArgument } from "./util";

const defaultCheckpointEvery = 10;

export function parseStartArguments(rest: string): { manifest: string; task: string; checkpointEvery: number } | string {
	const tokens = rest.split(/\s+/).filter(token => token.length > 0);
	if (tokens.shift() !== "--reference") {
		return "BLOCKED: OMP is not connected to the live GLM/Qwen workers; /drift start --reference <manifest-path> opts into the reference service only.";
	}
	let manifest = "";
	const task: string[] = [];
	let checkpointEvery = defaultCheckpointEvery;
	let mode: "manifest" | "task" | "checkpoint" = "manifest";
	for (const token of tokens) {
		if (token === "--task") {
			mode = "task";
			continue;
		}
		if (token === "--checkpoint-every") {
			mode = "checkpoint";
			continue;
		}
		if (mode === "manifest") {
			if (manifest.length > 0) return "Drift start takes one manifest path; put the task after --task.";
			manifest = token;
		} else if (mode === "task") {
			task.push(token);
		} else {
			const parsed = parseIntegerArgument(token);
			if (parsed === undefined || parsed < 1) return "Drift: --checkpoint-every takes a positive integer.";
			checkpointEvery = parsed;
			mode = "task";
		}
	}
	if (manifest.length === 0) return "Usage: /drift start --reference <manifest-path> [--task <text>] [--checkpoint-every <n>]";
	return { manifest, task: task.join(" "), checkpointEvery };
}
