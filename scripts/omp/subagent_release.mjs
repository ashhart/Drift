import { join } from 'node:path';
import { digest, readPrivate, requireControl as require } from './paused_echo_files.mjs';
import { writeBoundary } from './project_boundary_files.mjs';

const optional = path => { try { return readPrivate(path); } catch (error) { if (error.code !== 'ENOENT') throw error; } };
const exact = (value, wanted) => value && Object.keys(value).sort().join() === Object.keys(wanted).sort().join()
  && Object.entries(wanted).every(([key, item]) => value[key] === item);

export function createSubagentRelease(path, expected, exchangePath, exchangeHash) {
  const load = (file, pin) => { const raw = readPrivate(file); require(digest(raw) === pin); return JSON.parse(raw); };
  const config = load(path, expected), exchange = load(exchangePath, exchangeHash);
  require(config.v === 2 && exchange.v === 2 && config.actors.length === 2 && exchange.workers.length === 2);
  const actors = ['parent', 'child'].map(role => config.actors.find(actor => actor.role === role));
  require(actors.every(Boolean));
  let epoch = 0, stopped = false;
  const frame = (actor, phase) => ({ v: 2, epoch, phase, nonce: config.nonce, config_sha256: expected, ...actor });
  const read = (actor, kind, terminal = false) => optional(join(config.root, `${actor.role}${terminal ? '' : '-' + epoch}-${kind}.json`));
  const check = () => {
    require(!stopped && Date.now() < config.deadline_ms);
    load(path, expected); load(exchangePath, exchangeHash);
    for (const actor of actors) require(!read(actor, 'failed', true) && !read(actor, 'invalidated', true));
  };
  return {
    stop() { stopped = true; },
    get epochs() { return epoch; },
    poll() {
      try {
        check();
        const settled = actors.map(actor => read(actor, 'settled', true));
        if (settled.every(Boolean)) {
          require(epoch > 0);
          settled.forEach((raw, index) => require(exact(JSON.parse(raw), frame(actors[index], 'terminal'))));
          stopped = true; return 'settled';
        }
        const ready = actors.map(actor => read(actor, 'ready')), receipts = actors.map(actor => read(actor, 'exchange'));
        if (![...ready, ...receipts].every(Boolean)) return 'waiting';
        require(epoch < config.max_epochs);
        actors.forEach((actor, index) => {
          require(exact(JSON.parse(ready[index]), frame(actor, 'post_generation_pre_tool')));
          const value = JSON.parse(receipts[index]), route = exchange.workers.find(item => item.worker === actor.worker && item.session === actor.session);
          require(route && route.delivery === 'next_turn_snapshot' && value.v === 1 && value.sequence === epoch
            && value.state === 'STAGED_NOT_APPLIED' && value.mode === 'drift' && value.text_bytes === 0
            && value.session === route.exchange_session && value.source_worker === route.source_worker && value.target_worker === route.target_worker
            && Number.isSafeInteger(value.rows) && value.rows > 0 && value.rows === value.source_rows * value.copies
            && Number.isSafeInteger(value.bytes) && value.bytes > 0 && /^[a-f0-9]{64}$/.test(value.source_sha256));
        });
        check();
        actors.forEach((actor, index) => writeBoundary(join(config.root, `${actor.role}-${epoch}-release.json`), {
          v: 2, epoch, action: 'resume', nonce: config.nonce, ready_sha256: digest(ready[index]),
          peer_ready_sha256: digest(ready[1 - index]), exchange_sha256: digest(receipts[index]),
        }));
        epoch++; return 'released';
      } catch { stopped = true; throw new Error('DRIFT_SUBAGENT_RELEASE'); }
    },
  };
}
