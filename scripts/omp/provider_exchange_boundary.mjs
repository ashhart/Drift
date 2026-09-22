import { exchangeRequest, validateExchanged } from './exchange_client.mjs';
import { createStagedExchange } from './exchange_staged.mjs';

const failure = () => new Error('DRIFT_WORKER_CANCELLED');

export function createExchangeBoundary({ socket, route, timeoutMs, expected, delivery = 'immediate' }) {
  if (delivery === 'next_turn_snapshot') return createStagedExchange({ socket, route, timeoutMs, expected });
  if (delivery !== 'immediate') throw failure();
  // Production registration calls this only after the project gate has parked both workers.
  let poisoned = false;
  const records = [];
  const run = async boundary => {
    if (poisoned) throw failure();
    try {
      if (!(boundary?.signal instanceof AbortSignal) || boundary.signal.aborted) throw failure();
      const reply = await exchangeRequest(socket, { op: 'exchange', route }, { signal: boundary.signal, timeoutMs });
      const published = validateExchanged(reply, route, expected, records.length);
      if (boundary.signal.aborted) throw failure();
      records.push(published);
      return published;
    } catch (error) {
      poisoned = true;
      throw failure();
    }
  };
  // Object.assign would copy the getter's value once, so the flag has to be defined on the function itself.
  Object.defineProperty(run, 'poisoned', { get: () => poisoned, enumerable: true });
  return Object.assign(run, { records });
}

export function withExchange(inner, exchange) {
  // This generic composition alone does not establish that either worker is parked.
  return async boundary => {
    await exchange(boundary);
    if (inner) await inner(boundary);
  };
}
