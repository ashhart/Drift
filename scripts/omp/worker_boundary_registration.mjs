import { isAbsolute, dirname } from 'node:path';
import { lstatSync } from 'node:fs';
import { createProviderProjectBoundary } from './provider_project_boundary.mjs';
import { createExchangeBoundary } from './provider_exchange_boundary.mjs';
import { digest, privateRoot, readPrivate, requireControl as require } from './paused_echo_files.mjs';

const exact = (value, fields) => value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).sort().join(',') === [...fields].sort().join(',');
const token = value => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/.test(value);

export function workerBoundarySettings(env = process.env) {
  const result = {
    boundaryPath: env.DRIFT_PROVIDER_BOUNDARY_CONFIG, boundaryHash: env.DRIFT_PROVIDER_BOUNDARY_SHA256,
    exchangePath: env.DRIFT_EXCHANGE_CONFIG, exchangeHash: env.DRIFT_EXCHANGE_SHA256,
  };
  if (Boolean(result.boundaryPath) !== Boolean(result.boundaryHash) || Boolean(result.exchangePath) !== Boolean(result.exchangeHash)
      || result.exchangePath && !result.boundaryPath) throw new Error('DRIFT_WORKER_CAPABILITY');
  return result;
}

function exchangeFor(settings, binding) {
  const { exchangePath: path, exchangeHash: expected } = settings;
  const raw = readPrivate(path); require(digest(raw) === expected);
  const config = JSON.parse(raw);
  require(exact(config, ['v', 'socket', 'timeout_ms', 'workers']) && config.v === 1);
  require(typeof config.socket === 'string' && isAbsolute(config.socket));
  privateRoot(dirname(config.socket));
  const info = lstatSync(config.socket);
  require(info.isSocket() && info.uid === process.getuid() && (info.mode & 0o777) === 0o600);
  require(Number.isSafeInteger(config.timeout_ms) && config.timeout_ms > 0 && config.timeout_ms <= 30000);
  require(Array.isArray(config.workers) && config.workers.length === 2);
  for (const item of config.workers) {
    require(exact(item, ['worker', 'session', 'model_id', 'route', 'exchange_session', 'source_worker', 'target_worker', 'ranks']));
    require(Object.entries(item).every(([key, value]) => key === 'ranks' || token(value)));
    require(item.source_worker !== item.target_worker);
    require(Array.isArray(item.ranks) && item.ranks.length > 0 && item.ranks.length <= 16 && item.ranks.every(token));
    require(new Set(item.ranks).size === item.ranks.length);
  }
  for (const field of ['worker', 'route']) require(new Set(config.workers.map(item => item[field])).size === 2);
  const entry = config.workers.find(item => ['worker', 'session', 'model_id'].every(key => item[key] === binding.identity[key]));
  require(entry);
  const exchange = createExchangeBoundary({ socket: config.socket, route: entry.route, timeoutMs: config.timeout_ms, expected: entry });
  return async boundary => {
    require(digest(readPrivate(path)) === expected);
    const current = lstatSync(config.socket);
    require(current.ino === info.ino && current.dev === info.dev && current.isSocket() && (current.mode & 0o777) === 0o600);
    return exchange(boundary);
  };
}

export function createWorkerBoundary(settings, binding) {
  try {
    if (!settings.boundaryPath) return undefined;
    const exchange = settings.exchangePath ? exchangeFor(settings, binding) : undefined;
    return createProviderProjectBoundary(settings.boundaryPath, settings.boundaryHash, binding, { exchange });
  } catch { throw new Error('DRIFT_WORKER_CAPABILITY'); }
}
