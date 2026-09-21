import { constants, openSync, closeSync, fstatSync, writeSync, fsyncSync, linkSync, unlinkSync } from 'node:fs';
import { dirname } from 'node:path';
import { randomUUID } from 'node:crypto';
import { digest, privateRoot, requireControl } from './paused_echo_files.mjs';

export function writeBoundary(path, value) {
  privateRoot(dirname(path));
  const raw = Buffer.from(JSON.stringify(value) + '\n'); requireControl(raw.length <= 4096);
  const temporary = path + '.' + randomUUID() + '.tmp';
  const fd = openSync(temporary, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
  try {
    requireControl(fstatSync(fd).isFile()); requireControl(writeSync(fd, raw) === raw.length); fsyncSync(fd);
    privateRoot(dirname(path)); linkSync(temporary, path);
  } finally { try { closeSync(fd); } finally { unlinkSync(temporary); } }
  return digest(raw);
}
