import { writeFileSync } from "node:fs";
import { installRoomGuard } from "./room_guard.mjs";
import duo from "./stock_duo/src/omp.ts";
const facts = { opened: false, tool_calls: {}, errors: [] };
const save = () => writeFileSync(process.env.DRIFT_CONTRACT_REPORT, JSON.stringify(facts));
export default function duoFixture(api) {
  let command;
  duo(new Proxy(api, { get(target, key) { if (key === "registerCommand") return (name, definition) => { if (name === "duo") command = definition; api.registerCommand(name, definition); }; return Reflect.get(target, key); } }));
  installRoomGuard(api, { root: process.cwd(), parent: "drift-experimental/glm-fixture", child: "drift-experimental/qwen-fixture" }, event => { facts.guard ??= []; facts.guard.push(event); save(); });
  api.on("before_agent_start", async (_event, context) => {
    if (facts.opened || (context.models.current()?.id ?? context.model?.id) !== "glm-fixture") return;
    facts.opened = true; save();
    await command.handler("drift-experimental/qwen-fixture", context);
  });
  api.on("tool_call", event => { facts.tool_calls[event.toolName] = (facts.tool_calls[event.toolName] ?? 0) + 1; save(); });
  api.on("turn_end", event => {
    const match = /\[(context|stream|close|unknown):([A-Z_]+)\]$/.exec(event.message?.errorMessage ?? "");
    if (match) facts.errors.push({ phase: match[1], code: match[2] }); save();
  });
}
