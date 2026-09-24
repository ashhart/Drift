import { kvTools } from "./worker_kv_policy";
import { kvTask, kvYield } from "./worker_kv_lifecycle";
import { canonical, fail } from "./worker_protocol";
import type { OwnerConfig } from "./worker_registration";

interface KvContext {
  model?: { provider: string; id: string };
  models?: { current(): { provider: string; id: string } | undefined };
  sessionManager?: { getSessionId(): string };
  invokeTool?(args: Record<string, unknown>, options: { signal?: AbortSignal }): Promise<{ isError?: boolean }>;
}
interface KvApi {
  registerTool(tool: ReturnType<typeof kvTools>[number] & { label: string; execute(id: string, args: unknown, signal?: AbortSignal, update?: unknown, context?: KvContext): Promise<unknown> }): void;
  setActiveTools(names: string[]): unknown;
  getActiveTools(): string[];
  on(event: "session_start" | "before_agent_start", handler: (event: unknown, context: KvContext) => Promise<void>): void;
}

export function installKvTools(api: KvApi, config: OwnerConfig): void {
  const entries = new Map(config.workers.map(entry => [entry.identity.model_id, entry]));
  const selected = (context?: KvContext) => {
    const model = context?.models?.current() ?? context?.model;
    return model?.provider === "drift-experimental" ? entries.get(model.id) : undefined;
  };
  const admitted = new Set<string>();
  const tools = new Map(config.workers.flatMap(entry => kvTools(entry.subagent_role)).map(tool => [tool.name, tool]));
  for (const tool of tools.values()) api.registerTool({ ...tool, label: "Drift " + tool.name, async execute(_id, args, signal, _update, context) {
    if (canonical(args) !== "{}" || signal?.aborted) fail("CAPABILITY");
    if (tool.name !== "drift_sync") {
      const role = selected(context)?.subagent_role, id = context?.sessionManager?.getSessionId();
      if (!id || !context?.invokeTool || (tool.name === "task" ? role !== "parent" || admitted.has(id) : role !== "child")) fail("CAPABILITY");
      admitted.add(id);
      const result = await context.invokeTool(structuredClone(tool.name === "task" ? kvTask : kvYield), { signal });
      if (result.isError || signal?.aborted) fail("CAPABILITY");
    }
    return { content: [{ type: "text", text: "ready" }], details: {} };
  } });
  const activate = async (_event: unknown, context: KvContext) => {
    const entry = selected(context);
    if (!entry) return;
    const names = kvTools(entry.subagent_role).map(tool => tool.name);
    await api.setActiveTools(names);
    if (canonical([...api.getActiveTools()].sort()) !== canonical(names.sort())) fail("CAPABILITY");
  };
  api.on("session_start", activate);
  api.on("before_agent_start", activate);
}
