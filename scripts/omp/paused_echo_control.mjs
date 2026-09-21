import { dirname } from 'node:path';
import { realpathSync as canonical } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { digest, fresh, privateRoot, readPrivate, requireControl, writeReady } from './paused_echo_files.mjs';
const exact = (value, keys) => value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).sort().join(',') === [...keys].sort().join(',');
export function createPausedEcho(path, expected) {
  try {
    requireControl(typeof expected === 'string' && /^[0-9a-f]{64}$/.test(expected));
    const raw = readPrivate(path); requireControl(digest(raw) === expected);
    const config = JSON.parse(raw);
    requireControl(exact(config, ['v', 'root', 'task_root', 'ready', 'release', 'started_at_ms', 'deadline_ms', 'model']) && config.v === 1);
    privateRoot(config.root); requireControl(dirname(path) === config.root);
    requireControl(typeof config.task_root === 'string' && canonical(config.task_root) === config.task_root);
    requireControl(config.root !== config.task_root && !config.root.startsWith(config.task_root + '/'));
    requireControl(typeof config.model === 'string' && /^drift-experimental\/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/.test(config.model));
    requireControl(Number.isSafeInteger(config.started_at_ms) && Number.isSafeInteger(config.deadline_ms));
    const now = Date.now();
    requireControl(config.started_at_ms <= now && now < config.deadline_ms && config.deadline_ms - config.started_at_ms <= 60000);
    const end = performance.now() + config.deadline_ms - now;
    const ready = fresh(config.root, config.ready), release = fresh(config.root, config.release);
    requireControl(ready !== release); let used = false, failed = false;
    const check = context => {
      requireControl(!failed && Date.now() < config.deadline_ms && performance.now() < end);
      privateRoot(config.root);
      const model = context?.models?.current() ?? context?.model;
      requireControl(`${model?.provider}/${model?.id}` === config.model && canonical(context.cwd) === config.task_root);
    };
    return {
      check(context) { try { check(context); } catch { failed = true; throw new Error('PAUSED_ECHO_CONTROL'); } },
      async wait(context, signal) {
        try {
          check(context); requireControl(!signal?.aborted && !used); used = true;
          fresh(config.root, config.ready); fresh(config.root, config.release);
          const readyHash = writeReady(ready);
          while (true) {
            check(context); requireControl(!signal?.aborted); let bytes;
            try { bytes = readPrivate(release); } catch (error) { if (error.code !== 'ENOENT') throw error; }
            if (bytes) {
              const value = JSON.parse(bytes);
              requireControl(exact(value, ['v', 'action', 'ready_sha256']) && value.v === 1 && value.action === 'resume' && value.ready_sha256 === readyHash);
              requireControl(digest(readPrivate(ready)) === readyHash); check(context); requireControl(!signal?.aborted); return;
            }
            await new Promise(resolve => setTimeout(resolve, Math.max(1, Math.min(10, end - performance.now()))));
          }
        } catch { failed = true; throw new Error('PAUSED_ECHO_CONTROL'); }
      },
    };
  } catch { throw new Error('PAUSED_ECHO_CONTROL'); }
}
