import { dirname, join } from 'node:path';
import { realpathSync as canonical } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { digest, fresh, privateRoot, readPrivate, requireControl as require } from './paused_echo_files.mjs';
import { writeBoundary } from './project_boundary_files.mjs';

const exact = (value, keys) => value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).sort().join(',') === [...keys].sort().join(',');
const token = value => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/.test(value);
const sha = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
const optional = path => { try { return readPrivate(path); } catch (error) { if (error.code !== 'ENOENT') throw error; } };

export function createProjectBoundary(path, expected, { peerRequired = false } = {}) {
  try {
    require(sha(expected)); const raw = readPrivate(path); require(digest(raw) === expected);
    const config = JSON.parse(raw);
    const repeated = config.v === 2;
    require(exact(config, ['v', 'root', 'task_root', 'started_at_ms', 'deadline_ms', 'nonce', 'actors', ...(repeated ? ['max_epochs'] : [])]) && [1,2].includes(config.v));
    if (repeated) require(Number.isSafeInteger(config.max_epochs) && config.max_epochs >= 1 && config.max_epochs <= 256);
    privateRoot(config.root); require(dirname(path) === config.root);
    require(canonical(config.task_root) === config.task_root && config.root !== config.task_root && !config.root.startsWith(config.task_root + '/'));
    require(typeof config.nonce === 'string' && /^[0-9a-f]{32}$/.test(config.nonce));
    require(Number.isSafeInteger(config.started_at_ms) && Number.isSafeInteger(config.deadline_ms));
    const now = Date.now(); require(config.started_at_ms <= now && now < config.deadline_ms && config.deadline_ms - config.started_at_ms <= 170000);
    const end = performance.now() + config.deadline_ms - now;
    require(Array.isArray(config.actors) && config.actors.length === 2);
    for (const actor of config.actors) require(exact(actor, ['role', 'model', 'worker', 'session']) && ['parent', 'child'].includes(actor.role) && token(actor.worker) && token(actor.session) && typeof actor.model === 'string' && /^drift-experimental\/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/.test(actor.model));
    for (const key of ['role', 'model', 'worker', 'session']) require(new Set(config.actors.map(actor => actor[key])).size === 2);
    let actor, pending, released = false, failed = false, epoch = 0, launched = false, settled = false;
    const name = (role, kind) => `${role}${repeated ? '-' + epoch : ''}-${kind}.json`;
    const filename = (role, kind) => join(config.root, name(role, kind));
    const record = current => ({ v: config.v, ...(repeated ? { epoch } : {}), phase: 'post_generation_pre_tool', nonce: config.nonce, config_sha256: expected, ...current });
    const peer = () => config.actors.find(value => value.role !== actor.role);
    const check = context => {
      require(!failed && Date.now() < config.deadline_ms && performance.now() < end);
      require(digest(readPrivate(path)) === expected && canonical(context.cwd) === config.task_root);
      const model = context.models?.current() ?? context.model;
      const current = config.actors.find(value => value.model === `${model?.provider}/${model?.id}`); require(current);
      if (!actor) { actor = current; fresh(config.root, name(actor.role, 'ready')); fresh(config.root, name(actor.role, 'release')); }
      require(actor === current);
    };
    const peerReady = () => {
      const raw = optional(filename(peer().role, 'ready')); if (!raw) return;
      const value = JSON.parse(raw), wanted = record(peer());
      require(exact(value, Object.keys(wanted)) && Object.entries(wanted).every(([key, field]) => value[key] === field));
      return digest(raw);
    };
    const park = async (context, exchange) => {
      try {
        const readyPath = fresh(config.root, name(actor.role, 'ready'));
        const releasePath = fresh(config.root, name(actor.role, 'release')), readyHash = writeBoundary(readyPath, record(actor));
        let exchangeHash;
        while (true) {
          check(context);
          if (exchange && !exchangeHash && peerReady()) {
            const published = await exchange(); check(context);
            exchangeHash = writeBoundary(fresh(config.root, name(actor.role, 'exchange')), published);
          }
          const bytes = optional(releasePath);
          if (bytes) {
            const value = JSON.parse(bytes);
            require(exact(value, ['v', 'action', 'nonce', 'ready_sha256', 'peer_ready_sha256', 'exchange_sha256', ...(repeated ? ['epoch'] : [])]) && value.v === config.v && value.action === 'resume' && value.nonce === config.nonce);
            if (repeated) require(value.epoch === epoch);
            require(value.ready_sha256 === readyHash && sha(value.exchange_sha256) && value.peer_ready_sha256 === peerReady());
            if (exchange) require(exchangeHash && value.exchange_sha256 === exchangeHash);
            require(digest(readPrivate(readyPath)) === readyHash); check(context);
            if (repeated) epoch++; else released = true;
            return true;
          }
          await new Promise(resolve => setTimeout(resolve, Math.max(1, Math.min(10, end - performance.now()))));
        }
      } catch { failed = true; throw new Error('PROJECT_BOUNDARY'); }
    };
    return {
      check(context) { try { check(context); } catch { failed = true; throw new Error('PROJECT_BOUNDARY'); } },
      abort() { failed = true; },
      paths() { return actor ? { ready: filename(actor.role, 'ready'), invalidated: filename(actor.role, 'invalidated') } : undefined; },
      async finish(context, confirm) {
        if (!repeated) { if (confirm) await confirm(); return; }
        try {
          check(context); require(epoch > 0 && !pending);
          const marker = (role, kind) => join(config.root, `${role}-${kind}.json`);
          const wanted = current => ({ ...record(current), phase: 'terminal', epoch });
          const verified = (current, kind) => {
            const raw = optional(marker(current.role, kind)); if (!raw) return false;
            const value = JSON.parse(raw), expected = wanted(current);
            require(exact(value, Object.keys(expected)) && Object.entries(expected).every(([key, field]) => value[key] === field));
            return true;
          };
          if (settled) {
            for (const current of config.actors) for (const kind of ['finished', 'settled']) require(verified(current, kind));
            return;
          }
          writeBoundary(fresh(config.root, `${actor.role}-finished.json`), wanted(actor));
          const waitPeer = async kind => {
            while (true) {
              check(context); const raw = optional(marker(peer().role, kind));
              if (raw) {
                const value = JSON.parse(raw), expected = wanted(peer());
                require(exact(value, Object.keys(expected)) && Object.entries(expected).every(([key, field]) => value[key] === field));
                return;
              }
              await new Promise(resolve => setTimeout(resolve, 10));
            }
          };
          await waitPeer('finished');
          if (confirm) await confirm();
          check(context); writeBoundary(fresh(config.root, `${actor.role}-settled.json`), wanted(actor));
          await waitPeer('settled');
          settled = true;
        } catch { failed = true; throw new Error('PROJECT_BOUNDARY'); }
      },
      async wait(tool, context, exchange) {
        try {
          check(context); require(typeof tool === 'string' && !settled);
          if (released) return false;
          if (pending) return pending;
          if (repeated) require(epoch < config.max_epochs);
          if (!peerRequired && actor.role === 'parent' && tool === 'task' && (!repeated || !launched)) { launched = true; return false; }
          if (!peerRequired && actor.role === 'parent' && (!repeated || !launched) && !peerReady()) return false;
          if (repeated) require(typeof exchange === 'function');
          pending = park(context, exchange);
          if (!repeated) return pending;
          try { return await pending; } finally { pending = undefined; }
        } catch { failed = true; throw new Error('PROJECT_BOUNDARY'); }
      },
    };
  } catch { throw new Error('PROJECT_BOUNDARY'); }
}
