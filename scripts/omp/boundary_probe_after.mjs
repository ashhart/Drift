import { appendFileSync } from 'node:fs';
export default function observe(api) {
  api.on('tool_call', (event, context) => {
    const model = context.models?.current() ?? context.model;
    appendFileSync(process.env.DRIFT_BOUNDARY_EVENTS, JSON.stringify({ kind: 'admitted', model: model?.id, pid: process.pid, at: Date.now(), tool: event.toolName }) + '\n');
  });
}
