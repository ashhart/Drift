import { expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { realpathSync, mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, chmodSync, symlinkSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { createPausedEcho } from '../../../scripts/omp/paused_echo_control.mjs';
import pausedEcho from '../../../scripts/omp/paused_echo.mjs';
const hash = (bytes: string | Buffer) => createHash('sha256').update(bytes).digest('hex');
function fixture(milliseconds = 1000) {
  const parent = realpathSync(mkdtempSync(join(resolve(tmpdir()), 'pause-')));
  const root = join(parent, 'private'), task = join(parent, 'task');
  mkdirSync(root, { mode: 0o700 }); mkdirSync(task);
  const now = Date.now();
  const config = { v: 1, root, task_root: task, ready: 'ready.json', release: 'release.json', started_at_ms: now, deadline_ms: now + milliseconds, model: 'drift-experimental/qwen' };
  const path = join(root, 'config.json'); writeFileSync(path, JSON.stringify(config), { mode: 0o600 });
  return { root, task, config, path, pin: hash(readFileSync(path)), cleanup: () => rmSync(parent, { recursive: true, force: true }) };
}
const context = (task: string, id = 'qwen') => ({ cwd: task, models: { current: () => ({ provider: 'drift-experimental', id }) } });
function release(f: ReturnType<typeof fixture>, extra = {}) {
  writeFileSync(join(f.root, 'release.json'), JSON.stringify({ v: 1, action: 'resume', ready_sha256: hash(readFileSync(join(f.root, 'ready.json'))), ...extra }), { mode: 0o600 });
}
test('fixed echo pauses exactly once and resumes only for the pinned ready hash', async () => {
  const f = fixture();
  try {
    const gate = createPausedEcho(f.path, f.pin); const pending = gate.wait(context(f.task));
    expect(JSON.parse(readFileSync(join(f.root, 'ready.json'), 'utf8'))).toEqual({ v: 1, phase: 'tool_boundary' });
    release(f); await pending;
    await expect(gate.wait(context(f.task))).rejects.toThrow('PAUSED_ECHO_CONTROL');
  } finally { f.cleanup(); }
});
for (const fault of ['hash', 'extra', 'oversize', 'symlink', 'malformed']) test(`bad release ${fault} cannot resume`, async () => {
  const f = fixture();
  try {
    const gate = createPausedEcho(f.path, f.pin); const pending = gate.wait(context(f.task));
    if (fault === 'hash') release(f, { ready_sha256: '0'.repeat(64) });
    if (fault === 'extra') release(f, { text: 'forged' });
    if (fault === 'oversize') writeFileSync(join(f.root, 'release.json'), 'x'.repeat(4097));
    if (fault === 'symlink') symlinkSync(f.path, join(f.root, 'release.json'));
    if (fault === 'malformed') writeFileSync(join(f.root, 'release.json'), '{');
    await expect(pending).rejects.toThrow('PAUSED_ECHO_CONTROL');
  } finally { f.cleanup(); }
});
for (const fault of ['model', 'cwd', 'ready', 'release', 'timeout']) test(`boundary ${fault} fails closed`, async () => {
  const f = fixture(fault === 'timeout' ? 30 : 1000);
  try {
    const gate = createPausedEcho(f.path, f.pin);
    if (fault === 'ready') writeFileSync(join(f.root, 'ready.json'), '{}');
    if (fault === 'release') writeFileSync(join(f.root, 'release.json'), '{}');
    await expect(gate.wait(context(fault === 'cwd' ? f.root : f.task, fault === 'model' ? 'glm' : 'qwen'))).rejects.toThrow('PAUSED_ECHO_CONTROL');
  } finally { f.cleanup(); }
});
for (const fault of ['pin', 'public', 'inside', 'deadline', 'started', 'symlink']) test(`config ${fault} rejected before ready`, () => {
  const f = fixture();
  try {
    let path = f.path, pin = f.pin;
    if (fault === 'pin') pin = '0'.repeat(64);
    if (fault === 'public') chmodSync(f.root, 0o755);
    if (fault === 'inside') f.config.task_root = f.root;
    if (fault === 'deadline') f.config.deadline_ms = f.config.started_at_ms + 60001;
    if (fault === 'started') f.config.started_at_ms = Date.now() + 60000;
    if (['inside', 'deadline', 'started'].includes(fault)) { writeFileSync(path, JSON.stringify(f.config)); pin = hash(readFileSync(path)); }
    if (fault === 'symlink') { path = join(f.root, 'alias.json'); symlinkSync(f.path, path); }
    expect(() => createPausedEcho(path, pin)).toThrow('PAUSED_ECHO_CONTROL');
    expect(existsSync(join(f.root, 'ready.json'))).toBe(false);
  } finally { f.cleanup(); }
});
test('actual nativeEcho wrapper preserves result and records each usage event once', async () => {
  const f = fixture(); const previous = { ...process.env };
  try {
    process.env.DRIFT_PAUSED_ECHO_CONFIG = f.path; process.env.DRIFT_PAUSED_ECHO_SHA256 = f.pin;
    process.env.DRIFT_NATIVE_REPORT = join(f.root, 'report.json');
    let tool: any; const handlers = new Map<string, any[]>(); const active: string[][] = [];
    const api = { registerTool: (value: any) => { tool = value; }, on: (name: string, handler: any) => handlers.set(name, [...(handlers.get(name) ?? []), handler]), setActiveTools: (names: string[]) => active.push(names) };
    pausedEcho(api);
    for (const fn of handlers.get('session_start') ?? []) await fn({}, context(f.task));
    const pending = tool.execute('private-id', {}, undefined, undefined, context(f.task));
    expect(JSON.parse(readFileSync(process.env.DRIFT_NATIVE_REPORT, 'utf8')).tool_calls).toBe(0);
    release(f); expect(await pending).toEqual({ content: [{ type: 'text', text: 'echo-ok' }], details: {} });
    for (const fn of handlers.get('turn_end') ?? []) await fn({ message: { role: 'assistant', usage: { input: 5, output: 3 } } });
    const report = JSON.parse(readFileSync(process.env.DRIFT_NATIVE_REPORT, 'utf8'));
    expect([report.tool_calls, report.assistant_turns, report.input_tokens, report.output_tokens]).toEqual([1, 1, 5, 3]);
    expect(handlers.get('turn_end')?.length).toBe(1); expect(active).toEqual([['drift_native_echo']]);
    await expect(tool.execute('second', {}, undefined, undefined, context(f.task))).rejects.toThrow('PAUSED_ECHO_CONTROL');
  } finally { process.env = previous; f.cleanup(); }
});
test('abort and changed ready bytes cannot resume a parked call', async () => {
  for (const fault of ['abort', 'ready', 'dangling']) {
    const f = fixture();
    try {
      const abort = new AbortController(), gate = createPausedEcho(f.path, f.pin), started = Date.now();
      const pending = gate.wait(context(f.task), abort.signal);
      if (fault === 'abort') abort.abort();
      if (fault === 'ready') { release(f); writeFileSync(join(f.root, 'ready.json'), '{}'); }
      if (fault === 'dangling') symlinkSync(join(f.root, 'absent'), join(f.root, 'release.json'));
      await expect(pending).rejects.toThrow('PAUSED_ECHO_CONTROL');
      expect(Date.now() - started).toBeLessThan(200);
    } finally { f.cleanup(); }
  }
});
test('FIFO release is rejected without blocking and startup consumes the deadline', async () => {
  const f = fixture();
  try {
    const gate = createPausedEcho(f.path, f.pin), started = Date.now();
    const pending = gate.wait(context(f.task));
    expect(Bun.spawnSync(['/usr/bin/mkfifo', join(f.root, 'release.json')]).exitCode).toBe(0);
    await expect(pending).rejects.toThrow('PAUSED_ECHO_CONTROL');
    expect(Date.now() - started).toBeLessThan(200);
  } finally { f.cleanup(); }
  const expired = fixture(30);
  try {
    await new Promise(resolve => setTimeout(resolve, 40));
    expect(() => createPausedEcho(expired.path, expired.pin)).toThrow('PAUSED_ECHO_CONTROL');
  } finally { expired.cleanup(); }
});
