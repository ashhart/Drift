import type { ServiceClient } from "./client";
import type { Phase } from "./protocol";
import type { ServiceStatus } from "./status";

export interface SessionManagerLike {
	getBranch?(): unknown[];
}

export interface CommandContext {
	sessionManager: SessionManagerLike;
	/** False in print/RPC mode, where a triggered turn would collide with the next queued prompt. */
	hasUI?: boolean;
	ui: {
		select(
			title: string,
			options: Array<string | { label: string; description?: string }>,
		): Promise<string | undefined>;
		notify(message: string, level?: "info" | "warning" | "error"): void;
		setStatus(key: string, value: string | undefined): void;
		setWidget?(key: string, content: string[] | undefined, options?: unknown): void;
	};
}

export interface PromptEvent {
	systemPrompt?: string | string[];
}

export interface ToolCallEvent {
	toolName: string;
	toolCallId: string;
	input: Record<string, unknown>;
}

export interface OmpExtensionApi {
	on(
		event: "before_agent_start",
		handler: (
			event: PromptEvent,
			context: CommandContext,
		) => { systemPrompt: string[] } | undefined | Promise<{ systemPrompt: string[] } | undefined>,
	): void;
	on(
		event: "session_start" | "session_shutdown",
		handler: (event: unknown, context: CommandContext) => void | Promise<void>,
	): void;
	on(
		event: "tool_call",
		handler: (
			event: ToolCallEvent,
			context: CommandContext,
		) => { block?: boolean; reason?: string; input?: Record<string, unknown> } | undefined,
	): void;
	registerCommand(
		name: string,
		options: {
			description: string;
			getArgumentCompletions(prefix: string): Array<{ value: string; label: string }> | null;
			handler(args: string, context: CommandContext): Promise<void>;
		},
	): void;
	sendMessage<T = unknown>(
		message: {
			customType: string;
			content: string;
			display: boolean;
			details?: T;
			attribution?: "agent" | "user";
		},
		options?: { triggerTurn?: boolean; deliverAs?: "steer" | "followUp" | "nextTurn" },
	): void;
	/** Persist a custom entry in the session file; not sent to the LLM. */
	appendEntry<T = unknown>(customType: string, data?: T): void;
}

/** Live per-session picture of the run this session controls. */
export interface RunState {
	/** Open socket to the worker service; absent when the run is only known from the session file. */
	client?: ServiceClient;
	phase: Phase;
	epoch: number;
	manifestSha256?: string;
	lastStatus?: ServiceStatus;
	/** True while the socket is open. */
	connected: boolean;
	/** True when the run was rebuilt from a persisted entry and never reconnected. */
	resumed?: boolean;
}
