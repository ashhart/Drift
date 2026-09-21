import { constants, openSync, closeSync, fstatSync, readSync, writeSync, fsyncSync, lstatSync, realpathSync } from 'node:fs';
import { dirname, isAbsolute, join } from 'node:path';
import { createHash } from 'node:crypto';
export const requireControl = value => { if (!value) throw new Error('PAUSED_ECHO_CONTROL'); };
export const digest = bytes => createHash('sha256').update(bytes).digest('hex');
export function privateRoot(path) {
  requireControl(typeof path === 'string' && isAbsolute(path) && realpathSync(path) === path);
  const info = lstatSync(path);
  requireControl(info.isDirectory() && info.uid === process.getuid() && (info.mode & 0o777) === 0o700);
  return path;
}
export function readPrivate(path, maximum = 4096) {
  privateRoot(dirname(path));
  requireControl(!lstatSync(path).isSymbolicLink());
  requireControl(isAbsolute(path) && realpathSync(path) === path);
  const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const info = fstatSync(fd);
    requireControl(info.isFile() && info.uid === process.getuid() && !(info.mode & 0o077) && info.size > 0 && info.size <= maximum);
    const buffer = Buffer.alloc(maximum + 1); let size = 0, count;
    while ((count = readSync(fd, buffer, size, buffer.length - size, null)) > 0) { size += count; requireControl(size <= maximum); }
    requireControl(size === info.size); return buffer.subarray(0, size);
  } finally { closeSync(fd); }
}
export function fresh(root, leaf) {
  privateRoot(root); requireControl(typeof leaf === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\.json$/.test(leaf));
  const path = join(root, leaf);
  try { lstatSync(path); throw new Error('PAUSED_ECHO_CONTROL'); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  return path;
}
export function writeReady(path) {
  privateRoot(dirname(path));
  const raw = Buffer.from(JSON.stringify({ v: 1, phase: 'tool_boundary' }) + '\n');
  const fd = openSync(path, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW | constants.O_NONBLOCK, 0o600);
  try { requireControl(fstatSync(fd).isFile()); requireControl(writeSync(fd, raw) === raw.length); fsyncSync(fd); } finally { closeSync(fd); }
  return digest(raw);
}
