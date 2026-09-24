import experimentalWorkers from './experimental_extension.mjs';
import { realpathSync } from 'node:fs';
import { createSubagentRelease } from './subagent_release.mjs';
import { createWorkerBoundary } from './worker_boundary_registration.mjs';
import { exchangeRequest } from './exchange_client.mjs';
import { digest, readPrivate, requireControl as require } from './paused_echo_files.mjs';

export function installSubagentCommand(api, registration = experimentalWorkers) {
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
    try {
      require(['enable', 'status', 'disable'].includes(args));
      if (args === 'status') { context.ui.notify(`Drift subagent: ${state}; ${release?.epochs ?? 0} staged paired epochs.`, 'info'); return; }
      if (args === 'disable') {
        require(!owner || owner === context.sessionManager.getSessionId());
        context.abort(); state = 'closed'; await close();
        if (previousModel) await api.setModel(previousModel);
        if (previousTools) await api.setActiveTools(previousTools);
        context.ui.notify('Drift workers closed; native resources require their independent cleanup receipts.', 'info'); return;
      }
      require(state === 'disabled' && context.isIdle() && !context.hasPendingMessages());
      require(!(context.sessionManager.getBranch() ?? []).some(entry => entry.type === 'message'));
      const exchangeRaw = readPrivate(settings.exchangePath); require(digest(exchangeRaw) === settings.exchangeHash);
      const exchange = JSON.parse(exchangeRaw); require(exchange.v === 2);
      const id = context.sessionManager.getSessionId();
      createWorkerBoundary(settings, { identity: parent.identity, ompSession: id, cwd: realpathSync(context.cwd) });
      for (const route of exchange.workers) {
        const value = await exchangeRequest(exchange.socket, { op: 'status', route: route.route }, { timeoutMs: 1000 });
        require(value.op === 'status' && value.route === route.route && value.mode === 'drift'
          && value.sequence === 0 && value.foreign_rows === 0 && value.poisoned === false);
      }
      release = createSubagentRelease(settings.boundaryPath, settings.boundaryHash, settings.exchangePath, settings.exchangeHash);
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
      context.ui.notify('Drift refused this operation: use a fresh session, pinned KV-only profiles and a healthy coordinator; no text fallback was enabled.', 'error');
    }
  };
}
