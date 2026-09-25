import experimentalWorkers from './experimental_extension.mjs';
import { realpathSync } from 'node:fs';
import { createSubagentRelease } from './subagent_release.mjs';
import { createWorkerBoundary } from './worker_boundary_registration.mjs';
import { exchangeRequest } from './exchange_client.mjs';
import { digest, readPrivate, requireControl as require } from './paused_echo_files.mjs';

// Admits a fresh pair: the pinned exchange profile, the parent's boundary and both coordinator routes idle.
export async function admitPair(settings, parent, context) {
  const exchangeRaw = readPrivate(settings.exchangePath); require(digest(exchangeRaw) === settings.exchangeHash);
  const exchange = JSON.parse(exchangeRaw); require(exchange.v === 2);
  createWorkerBoundary(settings, { identity: parent.identity, ompSession: context.sessionManager.getSessionId(), cwd: realpathSync(context.cwd) });
  for (const route of exchange.workers) {
    const value = await exchangeRequest(exchange.socket, { op: 'status', route: route.route }, { timeoutMs: 1000 });
    require(value.op === 'status' && value.route === route.route && value.mode === 'drift'
      && value.sequence === 0 && value.foreign_rows === 0 && value.poisoned === false);
  }
  return createSubagentRelease(settings.boundaryPath, settings.boundaryHash, settings.exchangePath, settings.exchangeHash);
}

export function installSubagentCommand(api, registration = experimentalWorkers, admit = admitPair) {
  const runtime = registration(api, { manual: true }), { config, settings } = runtime;
  require(config.workers.length === 2 && config.workers.every(entry => entry.communication_mode === 'kv_only'));
  const parent = config.workers.find(entry => entry.subagent_role === 'parent');
  require(parent && config.workers.some(entry => entry.subagent_role === 'child'));
  let state = 'disabled', owner, timer, release, previousModel, previousTools, closing;
  const close = async () => {
    clearInterval(timer); release?.stop();
    if (!closing) closing = runtime.close();
    await closing;
  };
  const refuse = context => context.ui.notify('Drift refused this operation: use a fresh session, pinned KV-only profiles and a healthy coordinator; no text fallback was enabled.', 'error');
  // Disable returns the session to the model and tools it had before enable, once, however the workers closed.
  const restore = async () => {
    const model = previousModel, tools = previousTools; previousModel = previousTools = undefined;
    let restored = true;
    try { if (model && !(await api.setModel(model))) restored = false; } catch { restored = false; }
    try { if (tools) await api.setActiveTools(tools); } catch { restored = false; }
    return restored;
  };
  api.on('context', (_event, context) => {
    if (owner !== context.sessionManager.getSessionId()) return;
    try {
      if (state === 'enabled' && release.poll() === 'settled') { state = 'settled'; clearInterval(timer); }
      if (state === 'settled') context.abort();
    } catch { state = 'failed'; context.abort(); void close().catch(() => {}); }
  });
  api.on('session_shutdown', async (_event, context) => {
    if (owner === context.sessionManager.getSessionId()) { state = 'closed'; await close(); }
  });
  return async (args, context) => {
    // A refused request changes nothing; only an enable that fails after admission starts closes the pair.
    if (!['enable', 'status', 'disable'].includes(args)) { context.ui.notify('Usage: /drift subagent <enable|status|disable>', 'error'); return; }
    if (args === 'status') { context.ui.notify(`Drift subagent: ${state}; ${release?.epochs ?? 0} staged paired epochs.`, 'info'); return; }
    const id = context.sessionManager.getSessionId();
    if (args === 'disable') {
      if (owner && owner !== id) return refuse(context);
      context.abort(); state = 'closed';
      const closed = await close().then(() => true, () => false), restored = await restore();
      context.ui.notify((closed ? 'Drift workers closed' : 'Drift workers stopped without an ordered close')
        + (restored ? '' : '; the previous model or tools were not restored')
        + '; native resources require their independent cleanup receipts.', closed && restored ? 'info' : 'error');
      return;
    }
    if (state !== 'disabled' || !context.isIdle() || context.hasPendingMessages()
      || (context.sessionManager.getBranch() ?? []).some(entry => entry.type === 'message')) return refuse(context);
    try {
      release = await admit(settings, parent, context);
      const model = context.models.resolve('drift-experimental/' + parent.identity.model_id); require(model);
      previousModel = context.models.current(); previousTools = api.getActiveTools();
      require(await api.setModel(model));
      owner = id; runtime.enable(); state = 'enabled';
      timer = setInterval(() => {
        try { if (release.poll() === 'settled') { state = 'settled'; clearInterval(timer); } }
        catch { state = 'failed'; context.abort(); void close().catch(() => {}); }
      }, 10);
      context.ui.notify('Drift KV-only pair enabled; send the parent its own task, with child instructions supplied separately by the owner.', 'info');
    } catch {
      state = 'failed'; await close().catch(() => {});
      refuse(context);
    }
  };
}
