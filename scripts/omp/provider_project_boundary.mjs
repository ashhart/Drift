import { join } from 'node:path';
import { renameSync } from 'node:fs';
import { createProjectBoundary } from './project_boundary_control.mjs';
import { createProjectRouteCheck } from './project_boundary_routes.mjs';
import { digest, fresh, readPrivate, requireControl as require } from './paused_echo_files.mjs';
import { writeBoundary } from './project_boundary_files.mjs';
import { terminalAction } from './project_boundary_terminal.mjs';

const optional = path => { try { return readPrivate(path); } catch (error) { if (error.code !== 'ENOENT') throw error; } };
const failure = () => new Error('DRIFT_WORKER_CANCELLED');

export function createProviderProjectBoundary(path, expected, binding, { exchange } = {}) {
  try {
    const raw = readPrivate(path); require(digest(raw) === expected);
    const config = JSON.parse(raw), identity = Object.freeze({ ...binding.identity });
    const actor = config.actors.find(value => value.worker === identity.worker && value.session === identity.session && value.model === 'drift-experimental/' + identity.model_id);
    require(actor); const peer = config.actors.find(value => value.role !== actor.role);
    const route = createProjectRouteCheck(config.actors), context = { cwd: binding.cwd, model: { provider: 'drift-experimental', id: identity.model_id } };
    const kvOnly = route.communicationMode === 'kv_only';
    if (kvOnly) require(config.v === 2 && typeof exchange === 'function');
    const gate = createProjectBoundary(path, expected, { peerRequired: kvOnly });
    route(actor, binding.ompSession, binding.cwd); gate.check(context);
    for (const role of [actor.role, peer.role]) for (const kind of ['failed', 'invalidated']) fresh(config.root, role + '-' + kind + '.json');
    const file = (role, kind) => join(config.root, role + '-' + kind + '.json');
    let failed = false, active = false, admitted = false;
    const poison = () => {
      if (failed) return;
      failed = true; gate.abort();
      try { writeBoundary(file(actor.role, 'failed'), { v: 1, status: 'FAILED', phase: 'provider_boundary', config_sha256: expected, nonce: config.nonce, ...actor }); } catch {}
      try { const paths = gate.paths(); if (paths && optional(paths.ready)) renameSync(paths.ready, paths.invalidated); } catch {}
    };
    const check = () => {
      require(!failed && !optional(file(actor.role, 'failed')) && !optional(file(peer.role, 'failed')));
      route(actor, binding.ompSession, binding.cwd); gate.check(context);
    };
    const run = async boundary => {
      let timer, abort;
      try {
        check(); require(!active);
        require(boundary?.identity && Object.keys(boundary.identity).length === Object.keys(identity).length && Object.entries(identity).every(([key, value]) => boundary.identity[key] === value));
        require(Array.isArray(boundary.toolNames) && boundary.toolNames.every(name => typeof name === 'string' && name.length > 0));
        require(boundary.signal instanceof AbortSignal && !boundary.signal.aborted);
        active = true; abort = () => poison(); boundary.signal.addEventListener('abort', abort, { once: true });
        timer = setInterval(() => { try { require(!optional(file(peer.role, 'failed'))); } catch { poison(); } }, 10);
        if (route.subagents && actor.role === 'parent' && boundary.toolNames[0] === 'task') {
          require(!admitted && boundary.toolNames.length === 1 && boundary.toolCalls?.length === 1);
          require(boundary.toolCalls[0].name === 'task' && Object.keys(boundary.toolCalls[0].arguments).length === 0);
          admitted = true; return;
        }
        const finishing = !boundary.toolNames.length || (route.subagents && actor.role === 'child' && boundary.toolNames[0] === 'yield');
        const action = kvOnly ? (finishing ? 'finish' : 'exchange') : terminalAction(config.v, actor.role, boundary);
        if (action === 'continue') return;
        if (action === 'finish') {
          await gate.finish(context, exchange?.finish ? () => exchange.finish(boundary) : undefined);
          check(); require(!boundary.signal.aborted);
          return;
        }
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
    return Object.assign(run, { abort: poison, role: actor.role });
  } catch { throw new Error('DRIFT_WORKER_CAPABILITY'); }
}
