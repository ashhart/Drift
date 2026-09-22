import { requireControl as require } from './paused_echo_files.mjs';

const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);

export function terminalAction(version, role, boundary) {
  if (version !== 2) return boundary.toolNames.length ? 'exchange' : 'finish';
  if (!boundary.toolNames.length) return role === 'child' ? 'continue' : 'finish';
  if (!boundary.toolNames.includes('yield')) return 'exchange';
  require(role === 'child' && boundary.toolNames.length === 1);
  require(Array.isArray(boundary.toolCalls) && boundary.toolCalls.length === 1);
  const call = boundary.toolCalls[0];
  require(call.name === 'yield' && record(call.arguments));
  const { type } = call.arguments;
  let result = call.arguments.result;
  if (typeof result === 'string') result = JSON.parse(result);
  if (result === undefined || result === null) result = call.arguments;
  require(record(result) && Object.hasOwn(result, 'data') && result.data != null && !Object.hasOwn(result, 'error'));
  if (Array.isArray(type)) {
    require(type.length > 0 && type.every(value => typeof value === 'string'));
    return 'exchange';
  }
  require(type === undefined || type === null || typeof type === 'string');
  return 'finish';
}
