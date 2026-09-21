import { appendFileSync, readFileSync } from 'node:fs';
export default function observe(api) {
  const record = (kind, event, context) => {
    const model = context.models?.current() ?? context.model;
    let worker = {};
    try { worker = JSON.parse(readFileSync(`${process.cwd()}/${model?.id}-worker.json`, 'utf8')); } catch {}
    appendFileSync(process.env.DRIFT_BOUNDARY_EVENTS, JSON.stringify({ kind, event_keys: Object.keys(event).sort(), signal_types: { event: typeof event.signal, context: typeof context.signal, abort: typeof context.abortSignal }, terminals: worker.terminal ?? 0, streams: worker.stream ?? 0, model: model?.id, pid: process.pid, at: Date.now(), tool: event.toolName ?? null, role: event.message?.role ?? null }) + '\n');
  };
  for (const kind of ['message_end', 'tool_call', 'tool_execution_start', 'tool_execution_end', 'session_shutdown']) api.on(kind, (event, context) => record(kind, event, context));
}
