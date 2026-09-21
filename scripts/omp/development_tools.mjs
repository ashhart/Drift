import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { loadTaskScope, taskRead, taskEdit } from "./task_scope.mjs";
import { developmentVerify } from "./development_verify.mjs";
export function registerDevelopmentTools(api, record = () => {}) {
  const config = loadTaskScope(process.env.DRIFT_TASK_CONFIG, process.env.DRIFT_TASK_CONFIG_SHA256);
  const tools = [
    { name: "drift_task_read", description: "Read public api.py or verify.py inside the assigned API task", fields: { path: { type: "string", enum: ["api.py", "verify.py"] } }, execute: args => taskRead(config, args.path) },
    { name: "drift_task_write", description: "Replace the WHOLE api.py file with the complete supplied text; include every line that must remain", fields: { path: { type: "string", enum: ["api.py"] }, text: { type: "string" } }, execute: args => taskEdit(config, args.path, args.text) },
    { name: "drift_task_verify", description: "Run the fixed public HTTP checks in the task sandbox", fields: {}, execute: () => { const result = developmentVerify(config); record(JSON.parse(result).exit_code === 0, createHash("sha256").update(readFileSync(config.root + "/api.py")).digest("hex")); return result; } },
  ];
  for (const tool of tools) api.registerTool({ name: tool.name, label: tool.name, description: tool.description, parameters: { type: "object", properties: tool.fields, required: Object.keys(tool.fields), additionalProperties: false }, async execute(_id, args) { return { content: [{ type: "text", text: tool.execute(args) }], details: {} }; } });
}
