import { loadTaskScope, taskRead, taskEdit, taskRun } from "./task_scope.mjs";
export default function taskTools(api) {
  const config = loadTaskScope(process.env.DRIFT_TASK_CONFIG, process.env.DRIFT_TASK_CONFIG_SHA256);
  const tools = [
    { name: "drift_task_read", description: "Read one file inside the assigned task directory", fields: { path: { type: "string" } }, execute: args => taskRead(config, args.path) },
    { name: "drift_task_edit", description: "Write one file inside the assigned task directory", fields: { path: { type: "string" }, text: { type: "string" } }, execute: args => taskEdit(config, args.path, args.text) },
    { name: "drift_task_run", description: "Run one owner-allowlisted task verification command", fields: { name: { type: "string", enum: config.runs.map(run => run.name) } }, execute: args => taskRun(config, args.name) },
  ];
  for (const tool of tools) api.registerTool({ name: tool.name, label: tool.name, description: tool.description,
    parameters: { type: "object", properties: tool.fields, required: Object.keys(tool.fields), additionalProperties: false },
    async execute(_id, args) { return { content: [{ type: "text", text: tool.execute(args) }], details: {} }; },
  });
  api.on("session_start", async () => api.setActiveTools(tools.map(tool => tool.name)));
}
