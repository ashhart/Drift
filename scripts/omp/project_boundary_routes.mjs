import { realpathSync, statSync } from 'node:fs';
import { isAbsolute, sep } from 'node:path';
import { validateWorkerConfig } from '../../plugin/omp-drift/src/worker_registration.ts';
import { WorkerRoutes } from '../../plugin/omp-drift/src/worker_routes.ts';
import { digest, readPrivate } from './paused_echo_files.mjs';

const fail = () => { throw new Error('PROJECT_BOUNDARY_ROUTE'); };
const require = value => { if (!value) fail(); };
const exact = (value, keys) => value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).sort().join(',') === [...keys].sort().join(',');
const token = value => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/.test(value);
const actorFields = ['role', 'model', 'worker', 'session'];

export function createProjectRouteCheck(actors) {
  try {
    const path = process.env.DRIFT_WORKER_CONFIG, expected = process.env.DRIFT_WORKER_CONFIG_SHA256;
    require(typeof expected === 'string' && /^[0-9a-f]{64}$/.test(expected));
    require(Array.isArray(actors) && actors.length === 2);
    for (const actor of actors) require(exact(actor, actorFields) && ['parent', 'child'].includes(actor.role) && token(actor.worker) && token(actor.session) && typeof actor.model === 'string' && /^drift-experimental\/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/.test(actor.model));
    for (const field of actorFields) require(new Set(actors.map(actor => actor[field])).size === 2);
    const bound = actors.map(actor => Object.freeze({ ...actor })), routes = new WorkerRoutes();
    const owner = () => {
      require(process.env.DRIFT_WORKER_CONFIG === path && process.env.DRIFT_WORKER_CONFIG_SHA256 === expected);
      const raw = readPrivate(path, 262144); require(digest(raw) === expected);
      const config = validateWorkerConfig(JSON.parse(raw));
      require(exact(config, ['version', 'experimental', 'workers']) && config.workers.length === 2);
      for (const actor of bound) {
        const entries = config.workers.filter(entry => 'drift-experimental/' + entry.identity.model_id === actor.model);
        require(entries.length === 1); const entry = entries[0];
        require(entry.identity.worker === actor.worker && entry.identity.session === actor.session);
        if (entry.subagent_role !== undefined) require(entry.subagent_role === actor.role);
        require(entry.experimental_multi_turn === true && entry.memory_mode === 'linked' && ['text_and_artifacts', 'kv_only'].includes(entry.communication_mode) && entry.session_binding === 'fixed');
      }
      return config;
    };
    const initial = owner();
    const modes = new Set(initial.workers.map(entry => entry.communication_mode)); require(modes.size === 1);
    const communicationMode = initial.workers[0].communication_mode;
    const subagents = initial.workers.some(entry => entry.subagent_role !== undefined);
    if (subagents) require(initial.workers.map(entry => entry.subagent_role).sort().join() === 'child,parent');
    let failed = false;
    const check = (actor, ompSession, taskRoot) => {
      try {
        require(!failed && exact(actor, actorFields));
        const match = bound.find(value => actorFields.every(field => actor[field] === value[field])); require(match);
        require(typeof taskRoot === 'string' && isAbsolute(taskRoot) && realpathSync(taskRoot) === taskRoot && statSync(taskRoot).isDirectory());
        require(path !== taskRoot && !path.startsWith(taskRoot + sep));
        const entry = owner().workers.find(value => value.identity.worker === match.worker);
        const route = routes.bind(entry, ompSession);
        require(route.identity.worker === match.worker && route.identity.session === match.session && 'drift-experimental/' + route.identity.model_id === match.model);
      } catch { failed = true; fail(); }
    };
    return Object.assign(check, { communicationMode, subagents });
  } catch { fail(); }
}
