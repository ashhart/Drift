import { createProjectBoundary } from './project_boundary_control.mjs';

export default function projectBoundary(api) {
  const expected = process.env.DRIFT_PROJECT_BOUNDARY_SHA256;
  const key = Symbol.for('drift.project.boundary.' + expected);
  const sessions = globalThis[key] ??= new Map();
  const selected = context => { const model = context.models?.current() ?? context.model; return `${model?.provider}/${model?.id}`; };
  const gate = context => {
    const model = selected(context), id = context.sessionManager.getSessionId();
    if (typeof id !== 'string' || !id) throw new Error('PROJECT_BOUNDARY');
    if (!sessions.has(model)) sessions.set(model, { id, gate: createProjectBoundary(process.env.DRIFT_PROJECT_BOUNDARY_CONFIG, expected) });
    const value = sessions.get(model); if (value.id !== id) throw new Error('PROJECT_BOUNDARY');
    return value.gate;
  };
  api.on('session_start', (_event, context) => gate(context).check(context));
  api.on('tool_call', async (event, context) => { await gate(context).wait(event.toolName, context); });
  api.on('session_shutdown', (_event, context) => { const value = sessions.get(selected(context)); if (value?.id === context.sessionManager.getSessionId()) value.gate.abort(); });
}
