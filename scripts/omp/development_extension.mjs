import { readFileSync, writeFileSync } from "node:fs";
import { installDevelopmentGuard } from "./development_guard.mjs";
import { registerDevelopmentTools } from "./development_tools.mjs";
import duo from "./stock_duo/src/omp.ts";
const scope = JSON.parse(readFileSync(process.env.DRIFT_ROOM_SCOPE, "utf8"));
const key = Symbol.for("drift.development.api." + scope.run);
const facts = globalThis[key] ??= { opened: false, workers: {}, task_calls: 0, hub_calls: 0, todo_calls: 0, blocked: 0, errors: [] };
const save = () => writeFileSync(process.env.DRIFT_ROOM_REPORT, JSON.stringify(facts));
const selector = context => { const model = context.models.current() ?? context.model; return `${model?.provider}/${model?.id}`; };
export default function developmentRoom(api) {
  let command;
  duo(new Proxy(api, { get(target, name) { if (name === "registerCommand") return (id, definition) => { if (id === "duo") command = definition; api.registerCommand(id, definition); }; return Reflect.get(target, name); } }));
  registerDevelopmentTools(api, (passed, hash) => { facts.public_verify_runs = (facts.public_verify_runs ?? 0) + 1; facts.public_verify_passes = (facts.public_verify_passes ?? 0) + Number(passed); if (passed) facts.last_verified_api_sha256 = hash; save(); });
  installDevelopmentGuard(api, scope, event => {
    if (event.kind === "blocked") {
      facts.blocked++; facts.guard_reasons ??= {}; facts.guard_tool_categories ??= {};
      facts.guard_reasons[event.reason] = (facts.guard_reasons[event.reason] ?? 0) + 1;
      facts.guard_tool_categories[event.tool_category] = (facts.guard_tool_categories[event.tool_category] ?? 0) + 1;
    }
    save();
  });
  api.on("session_start", async (_event, context) => {
    const id = selector(context);
    if (![scope.parent, scope.child].includes(id)) throw new Error("ROOM_SCOPE");
    facts.workers[id] ??= { turns: 0, input_tokens: 0, output_tokens: 0, shutdown: false };
    if (!facts.opened && id === scope.parent) { await command.handler(scope.child, context); facts.opened = true; }
    save();
  });
  api.on("tool_execution_start", event => { if (event.toolName === "task") facts.task_calls++; if (event.toolName === "hub") facts.hub_calls++; if (event.toolName === "todo") facts.todo_calls++; save(); });
  api.on("turn_end", (event, context) => {
    const own = facts.workers[selector(context)];
    if (!own || event.message?.role !== "assistant") return;
    own.turns++; own.input_tokens += event.message.usage?.input ?? 0; own.output_tokens += event.message.usage?.output ?? 0;
    if (event.message.stopReason === "error" || event.message.errorMessage) facts.errors.push(/\[(context|stream|close|unknown):([A-Z_]+)\]$/.exec(event.message.errorMessage ?? "")?.slice(1) ?? ["unknown", "UNKNOWN"]);
    save();
  });
  api.on("session_shutdown", (_event, context) => { const own = facts.workers[selector(context)]; if (own) own.shutdown = true; save(); });
}
