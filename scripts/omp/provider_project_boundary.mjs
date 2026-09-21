import { join } from 'node:path';
import { renameSync } from 'node:fs';
import { createProjectBoundary } from './project_boundary_control.mjs';
import { createProjectRouteCheck } from './project_boundary_routes.mjs';
import { digest, fresh, readPrivate, requireControl as require } from './paused_echo_files.mjs';
import { writeBoundary } from './project_boundary_files.mjs';

const optional = path => { try { return readPrivate(path); } catch (error) { if (error.code !== 'ENOENT') throw error; } };
const failure = () => new Error('DRIFT_WORKER_CANCELLED');

export function createProviderProjectBoundary(path, expected, binding, { exchange } = {}) {
  try {
    const gate = createProjectBoundary(path, expected), raw = readPrivate(path); require(digest(raw) === expected);
    const config = JSON.parse(raw), identity = Object.freeze({ ...binding.identity });
    const actor = config.actors.find(value => value.worker === identity.worker && value.session === identity.session && value.model === 'drift-experimental/' + identity.model_id);
    require(actor); const peer = config.actors.find(value => value.role !== actor.role);
    const route = createProjectRouteCheck(config.actors), context = { cwd: binding.cwd, model: { provider: 'drift-experimental', id: identity.model_id } };
    route(actor, binding.ompSession, binding.cwd); gate.check(context);
    for (const role of [actor.role, peer.role]) for (const kind of ['failed', 'invalidated']) fresh(config.root, role + '-' + kind + '.json');
    const file = (role, kind) => join(config.root, role + '-' + kind + '.json');
    let failed = false, active = false;
    const poison = () => {
      if (failed) return;
      failed = true; gate.abort();
      try { writeBoundary(file(actor.role, 'failed'), { v: 1, status: 'FAILED', phase: 'provider_boundary', config_sha256: expected, nonce: config.nonce, ...actor }); } catch {}
      try { if (optional(file(actor.role, 'ready'))) renameSync(file(actor.role, 'ready'), file(actor.role, 'invalidated')); } catch {}
    };
    const check = () => {
      require(!failed && !optional(file(actor.role, 'failed')) && !optional(file(peer.role, 'failed')));
      route(actor, binding.ompSession, binding.cwd); gate.check(context);
    };
    return async boundary => {
      let timer, abort;
      try {
        check(); require(!active);
        require(boundary?.identity && Object.keys(boundary.identity).length === Object.keys(identity).length && Object.entries(identity).every(([key, value]) => boundary.identity[key] === value));
        require(Array.isArray(boundary.toolNames) && boundary.toolNames.every(name => typeof name === 'string' && name.length > 0));
        require(boundary.signal instanceof AbortSignal && !boundary.signal.aborted);
        if (!boundary.toolNames.length) return;
        active = true; abort = () => poison(); boundary.signal.addEventListener('abort', abort, { once: true });
        timer = setInterval(() => { try { require(!optional(file(peer.role, 'failed'))); } catch { poison(); } }, 10);
        const tool = boundary.toolNames.includes('task') ? 'task' : boundary.toolNames[0];
        await gate.wait(tool, context, exchange ? async () => {
          check(); require(!boundary.signal.aborted);
          const receipt = await exchange(boundary);
          require(!boundary.signal.aborted); check();
          return receipt;
        } : undefined);
        require(!boundary.signal.aborted); check();
      } catch { poison(); throw failure(); }
      finally { clearInterval(timer); if (abort) boundary.signal.removeEventListener('abort', abort); active = false; }
    };
  } catch { throw new Error('DRIFT_WORKER_CAPABILITY'); }
}
