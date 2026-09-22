import { test, expect } from 'bun:test';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { createProjectBoundary } from '../../../scripts/omp/project_boundary_control.mjs';

const hash = (value: Uint8Array | string) => createHash('sha256').update(value).digest('hex');
function fixture() {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'boundary-epochs-'))), task = join(root, 'task');
  mkdirSync(task, { mode: 0o700 });
  const actors = ['parent', 'child'].map(role => ({ role, model: `drift-experimental/${role}`, worker: `${role}-worker`, session: `${role}-session` }));
  const config = { v: 2, max_epochs: 3, root, task_root: task, started_at_ms: Date.now(), deadline_ms: Date.now()+3000, nonce: 'a'.repeat(32), actors };
  const path = join(root, 'config.json'); writeFileSync(path, JSON.stringify(config), { mode: 0o600 });
  return { root, config, gate: () => createProjectBoundary(path, hash(readFileSync(path))),
    context: (role: string) => ({ cwd: task, model: { provider: 'drift-experimental', id: role } }),
    file: (role: string, epoch: number, kind: string) => join(root, `${role}-${epoch}-${kind}.json`),
    close: () => rmSync(root, { recursive: true, force: true }) };
}
type Fixture = ReturnType<typeof fixture>;
async function waitFile(f: Fixture, role: string, epoch: number, kind: string) {
  const path = f.file(role, epoch, kind);
  for (let i=0; i<200 && !existsSync(path); i++) await Bun.sleep(5);
  expect(existsSync(path)).toBe(true); return hash(readFileSync(path));
}
async function release(f: Fixture, epoch: number) {
  const ready = await Promise.all(['parent','child'].map(role => waitFile(f, role, epoch, 'ready')));
  for (const [index, role] of ['parent','child'].entries()) {
    const exchange = await waitFile(f, role, epoch, 'exchange');
    writeFileSync(f.file(role, epoch, 'release'), JSON.stringify({ v: 2, epoch, action: 'resume', nonce: f.config.nonce,
      ready_sha256: ready[index], peer_ready_sha256: ready[1-index], exchange_sha256: exchange }), { mode: 0o600 });
  }
}

test('three epochs require three fresh exchanges and never fall through after the first release', async () => {
  const f = fixture();
  try {
    const gates = [f.gate(),f.gate()], calls = [0,0];
    expect(await gates[0].wait('task',f.context('parent'))).toBe(false);
    for (let epoch=0; epoch<3; epoch++) {
      let settled = 0;
      const pending = ['parent','child'].map((role,index) => gates[index].wait(epoch === 1 && role === 'parent' ? 'task' : 'hub',f.context(role),async () => ({ seq: ++calls[index] })).then(value => { settled++; return value; }));
      await waitFile(f,'parent',epoch,'ready'); await waitFile(f,'child',epoch,'ready');
      await Bun.sleep(15); expect(settled).toBe(0);
      await release(f,epoch); expect(await Promise.all(pending)).toEqual([true,true]);
    }
    expect(calls).toEqual([3,3]);
    const confirmed: string[] = [];
    const first = gates[0].finish(f.context('parent'), async () => { confirmed.push('parent'); });
    await Bun.sleep(15); expect(confirmed).toEqual([]);
    const second = gates[1].finish(f.context('child'), async () => { confirmed.push('child'); });
    await Promise.all([first, second]); expect(confirmed.sort()).toEqual(['child', 'parent']);
    await expect(gates[0].wait('hub',f.context('parent'))).rejects.toThrow('PROJECT_BOUNDARY');
  } finally { f.close(); }
});

test('previous-epoch releases and cancellations cannot release a later tool', async () => {
  for (const fault of ['replay','abort']) {
    const f = fixture();
    try {
      const parent = f.gate(), child = f.gate();
      await parent.wait('task',f.context('parent'));
      const first = [parent.wait('hub',f.context('parent'),async()=>({seq:1})),child.wait('hub',f.context('child'),async()=>({seq:1}))];
      await release(f,0); await Promise.all(first);
      const pending = child.wait('hub',f.context('child'),async()=>({seq:2})).then(()=>null,error=>error);
      await waitFile(f,'child',1,'ready');
      if (fault==='replay') writeFileSync(f.file('child',1,'release'),readFileSync(f.file('child',0,'release')),{mode:0o600});
      else child.abort();
      expect((await pending)?.message).toBe('PROJECT_BOUNDARY');
      await expect(child.wait('hub',f.context('child'))).rejects.toThrow('PROJECT_BOUNDARY');
    } finally { f.close(); }
  }
});
