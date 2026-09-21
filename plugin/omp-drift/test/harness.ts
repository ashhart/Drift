import drift from "../src/omp";

export const secret = "test-secret-do-not-display";
export const manifestPath = "/tmp/drift/run.yaml";

type SentMessage = {
	message: {
		customType: string;
		content: string;
		display: boolean;
		details?: Record<string, unknown>;
		attribution?: string;
	};
	options?: { triggerTurn?: boolean; deliverAs?: string };
};

export function createHarness() {
	const handlers = new Map<string, Array<(event: never, context: never) => unknown>>();
	const commands: Record<
		string,
		{
			handler: (args: string, context: unknown) => Promise<void>;
			getArgumentCompletions: (prefix: string) => Array<{ value: string; label: string }> | null;
		}
	> = {};
	const sent: SentMessage[] = [];
	const notifications: Array<{ message: string; level?: string }> = [];
	const status = new Map<string, string | undefined>();
	const widgets = new Map<string, string[] | undefined>();
	const sessionManager = {
		branch: [] as unknown[],
		getBranch: () => sessionManager.branch,
	};

	const api = {
		on: (event: string, handler: (event: never, context: never) => unknown) => {
			const list = handlers.get(event) ?? [];
			list.push(handler);
			handlers.set(event, list);
		},
		registerCommand: (name: string, definition: (typeof commands)[string]) => {
			commands[name] = definition;
		},
		sendMessage: (message: SentMessage["message"], options?: SentMessage["options"]) => {
			sent.push({ message, options });
		},
		appendEntry: (customType: string, data?: unknown) => {
			sessionManager.branch.push({ type: "custom", customType, data });
		},
	};

	const makeContext = () => ({
		sessionManager,
		hasUI: true,
		ui: {
			select: async () => undefined,
			notify: (message: string, level?: string) => {
				notifications.push({ message, level });
			},
			setStatus: (key: string, value: string | undefined) => {
				if (value === undefined) status.delete(key);
				else status.set(key, value);
			},
			setWidget: (key: string, content: string[] | undefined) => {
				if (content === undefined) widgets.delete(key);
				else widgets.set(key, content);
			},
		},
	});

	drift(api as never);
	const context = makeContext();

	const emitBeforeAgentStart = async (systemPrompt: string[] = []) => {
		let result: { systemPrompt: string[] } | undefined;
		for (const handler of handlers.get("before_agent_start") ?? []) {
			const next = await (handler as unknown as (event: unknown, ctx: unknown) => Promise<typeof result>)(
				{ systemPrompt },
				context,
			);
			if (next !== undefined && next !== null) result = next;
		}
		return result;
	};

	const fireSessionEvent = async (event: string, target: unknown = context) => {
		for (const handler of handlers.get(event) ?? []) {
			await (handler as unknown as (event: unknown, ctx: unknown) => Promise<void> | void)({}, target);
		}
	};

	const runCommand = (args: string) => commands.drift.handler(args, context);
	const completions = (prefix: string) => commands.drift.getArgumentCompletions(prefix);
	const roomMessages = (kind: string) =>
		sent.filter(entry => entry.message.customType === "drift-room" && entry.message.details?.kind === kind);
	const errors = () => notifications.filter(entry => entry.level === "error");
	const board = () => widgets.get("drift-board")?.join("\n") ?? "";
	const runStateEntries = () =>
		sessionManager.branch.filter(
			(entry): entry is { type: string; customType: string; data: Record<string, unknown> } =>
				typeof entry === "object" && entry !== null && (entry as { customType?: string }).customType === "drift-run-state",
		);

	return {
		api,
		context,
		sent,
		notifications,
		status,
		widgets,
		sessionManager,
		emitBeforeAgentStart,
		fireSessionEvent,
		runCommand,
		completions,
		roomMessages,
		errors,
		board,
		runStateEntries,
	};
}

export function setEnv(port: number | undefined, value: string | null = secret) {
	if (port === undefined) delete process.env.DRIFT_SERVICE_PORT;
	else process.env.DRIFT_SERVICE_PORT = String(port);
	// null deletes the secret; an explicit undefined would fall back to the default.
	if (value === null) delete process.env.DRIFT_SERVICE_SECRET;
	else process.env.DRIFT_SERVICE_SECRET = value;
}
