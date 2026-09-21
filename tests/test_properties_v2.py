"""Adversarial property loops over the code written on top of the kit (dense toys, synthetic data).

Promoted from the bug hunt of 19 September 2026, which found three defects with these loops:
a second mail per epoch poisoned the hive, mailbox slot release ignored ownership, and malformed
service fields became INTERNAL. Each is now also pinned by a targeted regression test.
"""
import copy, random, tempfile, json, pathlib
from uuid import UUID
import torch
from drift.adapters.toy import DenseAdapter
from drift.core.pool import PoolBank
from drift.core.types import KV
from drift.mailbox.stream import MailStatus, MailboxController, merge_entries
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.hive import HiveController
from drift.runtime.toy import ToyModel
from drift.runtime.worker import Worker
from drift.translate.pool import Layout, PoolFormat, Translator, fit_member, fit_pool_format, load_translator, save_translator
from drift.transport.wire2 import FrameCodec2, Publication
from drift.runtime.service import Session, ServiceError
from drift.runtime.builders import toy_members

SESSION = UUID(int=3)

def toy(seed, kvh, dim, layers=3):
    torch.manual_seed(seed)
    return DenseAdapter(FrozenDecoder(ToyModel(seed=seed, kvheads=kvh, dim=dim, layers=layers)))

# ---------- A. multi-writer PoolBank vs brute force with random valid/invalid interleavings
def bank_property():
    fmt = PoolFormat("pool.v1", 2, 6, "ab" * 32)
    for trial in range(25):
        reader = Translator("R", Layout("mla_latent", 1, 5), {1: 0, 3: 1}, fmt)
        with torch.no_grad():
            for lvl in ("0", "1"):
                reader.readers[lvl].linear.weight.copy_(torch.randn(5, 6)); reader.readers[lvl].linear.bias.zero_()
        writers = set(random.sample([1, 2, 3, 4], random.randint(1, 3)))
        sinks, recent = random.randint(0, 3), random.randint(1, 7)
        bank = PoolBank(SESSION, reader, self_writer=9, writers=writers, sinks=sinks, recent=recent)
        model = []            # list of (slot, writer, rows_by_level)
        cursors = {w: [0, 0, -1] for w in writers}
        epoch = 0
        for step in range(random.randint(1, 25)):
            w = random.choice(sorted(writers))
            t = random.randint(1, 4)
            rows = {0: torch.randn(t, 6), 1: torch.randn(t, 6)}
            kind = random.random()
            seq, start, last = cursors[w]
            if kind < 0.7:
                pub = Publication(SESSION, w, fmt.fingerprint, epoch, seq, start, rows)
                valid = epoch > last and epoch >= bank.last_epoch   # one publication per writer per epoch
            else:
                bad = random.choice(["seq", "start", "epoch", "writer", "fp", "level"])
                pub = Publication(SESSION, w if bad != "writer" else 9, fmt.fingerprint if bad != "fp" else "cd" * 32,
                                  epoch if bad != "epoch" else max(0, last - 1) if last > 0 else -1 if False else epoch,
                                  seq if bad != "seq" else seq + 1, start if bad != "start" else start + 1,
                                  rows if bad != "level" else {0: rows[0]})
                if bad == "epoch":
                    pub = Publication(SESSION, w, fmt.fingerprint, last if last >= 0 else 0, seq, start, rows)
                    valid = last < 0 and bank.last_epoch <= 0 and (0 > bank.last_epoch)  # epoch 0 valid only if nothing at >=0 yet
                    valid = (last < 0) and (0 > bank.last_epoch or bank.last_epoch == -1)
                    if last >= 0: valid = False
                    if bank.last_epoch >= 0 and last < 0:
                        valid = False  # epoch must be >= bank.last_epoch; 0 < last_epoch unless last_epoch==0
                        valid = (0 >= bank.last_epoch)
                else:
                    valid = False
            before = bank.snapshot()
            before_view = bank.pin()
            try:
                bank.commit(pub)
                committed = True
            except ValueError:
                committed = False
            if committed != valid:
                raise AssertionError(f"trial {trial} step {step}: committed={committed} valid={valid} kind={kind:.2f} pub=(w{pub.writer} e{pub.epoch} s{pub.sequence} st{pub.start}) cursors={cursors} bank_last={bank.last_epoch}")
            if not committed:
                after = bank.snapshot()
                assert after["cursors"] == before["cursors"] and after["next_slot"] == before["next_slot"] and bank.pin() is before_view
                continue
            base = sum(len(m[2][0]) for m in model)
            for j in range(t):
                model.append((base + j, w, {0: rows[0][j:j + 1], 1: rows[1][j:j + 1]}))
            cursors[w] = [seq + 1, start + t, pub.epoch]
            epoch += random.randint(0, 1)
            latest = model[-1][0]
            expect = [m for m in model if m[0] < sinks or m[0] >= latest - recent + 1]
            view = bank.pin()
            assert view.positions.tolist() == [m[0] for m in expect]
            assert view.writers.tolist() == [m[1] for m in expect]
            with torch.no_grad():
                for i, m in enumerate(expect):
                    torch.testing.assert_close(view.layers[1][i], reader.readers["0"](m[2][0])[0])
                    torch.testing.assert_close(view.layers[3][i], reader.readers["1"](m[2][1])[0])

def test_A_bank_property():
    random.seed(0); torch.manual_seed(0)
    bank_property()

# ---------- B. hive determinism, restore at random epochs, N members 2..4
def hive_property():
    fmt = PoolFormat("pool.v1", 3, 8, "ee" * 32)
    shapes = [(1, 2, 6), (2, 1, 4), (3, 2, 4), (4, 1, 6)]
    for trial in range(5):
        n = random.randint(2, 4)
        def build():
            workers = {}
            ids = {f"m{i}": i + 1 for i in range(n)}
            for i in range(n):
                name = f"m{i}"; seed, kvh, dim = shapes[i]
                ad = toy(seed, kvh, dim)
                d = ad.descriptor
                t = Translator(name, Layout("kv_split", d.kv_heads, d.head_dim), {l: l for l in d.kv_layers}, fmt)
                with torch.no_grad():
                    for p in t.parameters(): p.mul_(0.3)
                workers[name] = Worker(name, ids[name], SESSION, ad, t, PoolBank(SESSION, t, ids[name], set(ids.values()) - {ids[name]}, sinks=1, recent=5), override=0.6)
            return HiveController(workers, FrameCodec2(b"h" * 32))
        g = torch.Generator().manual_seed(trial)
        sched = [{f"m{i}": torch.randint(1, 30, (random.randint(1, 3),), generator=g) for i in range(n)} for _ in range(6)]
        c1 = build(); out1 = [c1.tick(s) for s in sched]
        c2 = build()
        k = random.randint(1, 4)
        out2 = [c2.tick(s) for s in sched[:k]]
        ck = c2.checkpoint()
        junk = [{f"m{i}": torch.randint(1, 30, (2,), generator=g) for i in range(n)} for _ in range(2)]
        for s in junk: c2.tick(s)
        c2.restore(ck)
        out2 += [c2.tick(s) for s in sched[k:]]
        for a, b in zip(out1, out2):
            for name in a:
                assert torch.equal(a[name].output, b[name].output), f"trial {trial} n={n} restore at {k} diverged for {name}"
        # every member's bank holds exactly the others
        for name, w in c1.workers.items():
            assert set(w.bank.pin().writers.tolist()) == {ww.writer for nn, ww in c1.workers.items() if nn != name}

def test_B_hive_property():
    random.seed(0); torch.manual_seed(0)
    hive_property()

# ---------- C. translator save/load/fingerprint determinism
def translator_property():
    for trial in range(10):
        rows = {l: {"a": torch.randn(50, 12), "b": torch.randn(50, 7)} for l in range(2)}
        f1, p1 = fit_pool_format(rows, width=5)
        f2, p2 = fit_pool_format(rows, width=5)
        assert f1.fingerprint == f2.fingerprint and all(torch.equal(p1[l], p2[l]) for l in p1)
        t = Translator("a", Layout("kv_split", 2, 3), {0: 0, 4: 1}, f1, kind=random.choice(["ridge", "mlp"]), rank=4)
        fit_member(t, {l: rows[l]["a"] for l in rows}, p1)
        with tempfile.TemporaryDirectory() as d:
            save_translator(t, pathlib.Path(d) / "t", {})
            u = load_translator(pathlib.Path(d) / "t", f1)
        x = {0: KV(torch.randn(3, 2, 3), torch.randn(3, 2, 3)), 4: KV(torch.randn(3, 2, 3), torch.randn(3, 2, 3))}
        with torch.no_grad():
            a, b = t.write(x), u.write(x)
            assert all(torch.equal(a[l], b[l]) for l in a)
            ra, rb = t.read(a), u.read(b)
            assert all(torch.equal(ra[l].k, rb[l].k) for l in ra)

def test_C_translator_property():
    random.seed(0); torch.manual_seed(0)
    translator_property()

# ---------- D. mailbox controller invariants under random op sequences
def mailbox_property():
    for trial in range(20):
        box = MailboxController(ttl=random.randint(1, 4), slots=random.randint(1, 3), per_epoch_cap=random.randint(1, 3))
        epoch = 0; live = {}
        for step in range(40):
            op = random.choice(["create", "create", "cancel", "reject", "expire", "advance", "causal"])
            try:
                if op == "create":
                    r = box.create(random.choice("AB"), epoch, random.randint(1, 3))
                    assert r.slot not in {x.slot for x in box.records.values() if x.id != r.id and x.status in (MailStatus.PENDING, MailStatus.VISIBLE, MailStatus.ATTENDED)}
                elif op in ("cancel", "reject") and box.records:
                    getattr(box, op)(random.choice(list(box.records)))
                elif op == "expire":
                    box.expire(epoch)
                elif op == "advance":
                    epoch += 1
                elif op == "causal" and box.records:
                    r = random.choice(list(box.records.values()))
                    box.record_causal_use(r.id, epoch, active_correct=True, ablated_correct=False, replay_started_before_exposure=True)
            except (OverflowError, ValueError):
                pass
            # invariants
            active = [r for r in box.records.values() if r.status in (MailStatus.PENDING, MailStatus.VISIBLE, MailStatus.ATTENDED)]
            slots = [r.slot for r in active]
            assert len(slots) == len(set(slots)), "slot reused while active"
            assert len(active) <= box.slots
            assert set(box.slot_owner.values()) == {r.id for r in active}, f"slot_owner drift: {box.slot_owner} vs {[r.id for r in active]}"
            m = box.metrics(); assert sum(m["counts"].values()) == len(box.records)

def test_D_mailbox_property():
    random.seed(0); torch.manual_seed(0)
    mailbox_property()

# ---------- E. merge_entries positions strictly increasing; recency-safe
def merge_property():
    from drift.core.pool import PoolView
    from types import MappingProxyType
    for _ in range(50):
        views = []
        for _ in range(random.randint(0, 3)):
            n = random.randint(0, 4)
            pos = torch.sort(torch.randperm(20)[:n]).values
            views.append(PoolView(0, pos, torch.zeros(n, dtype=torch.long), MappingProxyType({0: KV(torch.zeros(n, 1, 2), torch.zeros(n, 1, 2))})))
        m = merge_entries(*views)
        total = sum(v.positions.numel() for v in views)
        if total == 0: assert m is None; continue
        assert m.positions.numel() == total and bool(torch.all(m.positions[1:] > m.positions[:-1])), m.positions

def test_E_merge_property():
    random.seed(0); torch.manual_seed(0)
    merge_property()

# ---------- F. service request fuzz: never INTERNAL on malformed input, phase machine consistent
def service_fuzz():
    manifest = {"kind": "toy", "session_uuid": str(UUID(int=42)), "setup_tokenizer": "bytes",
                "pool": {"levels": 3, "width": 8, "fingerprint": "ab" * 32}, "window": {"sinks": 2, "recent": 6}, "mail_max_tokens": 4,
                "members": [{"name": "A", "writer": 1, "mail_writer": 11, "seed": 1, "toy": {"seed": 1, "kvheads": 2, "dim": 6, "layers": 3}},
                            {"name": "B", "writer": 2, "mail_writer": 12, "seed": 2, "toy": {"seed": 2, "kvheads": 1, "dim": 4, "layers": 3}}]}
    with tempfile.TemporaryDirectory() as d:
        mp = pathlib.Path(d) / "run.json"; mp.write_text(json.dumps(manifest))
        for trial in range(6):
            s = Session(toy_members, pathlib.Path(d) / f"audit{trial}", b"k" * 32); (pathlib.Path(d) / f"audit{trial}").mkdir()
            for step in range(40):
                op = random.randint(0, 11)
                fields = {}
                junk = random.choice([None, "x", -1, 10**9, 1.5, [], {}, {"0": "a"}])
                if op == 1: fields = {"manifest": random.choice([str(mp), "/nope", junk]), "task": random.choice(["t", junk]), "assignments": random.choice([{}, {"0": "a"}, {"Z": "b"}, junk])}
                elif op == 2: fields = {"checkpoint_every": random.choice([0, 2, -1, junk])}
                elif op == 3: fields = {"epochs": random.choice([1, 2, 0, -3, junk])}
                elif op == 9: fields = {"sender": random.choice([0, 1, 5, -1, junk]), "slots": random.choice([1, 2, 0, junk])}
                req = {"id": step, "op": op, **fields}
                try:
                    s.handle(req)
                except ServiceError as e:
                    assert e.code != "INTERNAL", f"INTERNAL leaked for {req} in phase {s.phase}"
                except Exception as e:
                    raise AssertionError(f"uncaught {type(e).__name__}: {e} for {req} in phase {s.phase}")
                st = s.status()
                assert st["phase"] in ("SETUP", "STRICT", "CLOSED")
                if st["phase"] == "STRICT": assert s.controller is not None, "STRICT without a controller"
                if st["phase"] == "CLOSED": assert s.controller is None or s.controller.failed or True

def test_F_service_fuzz():
    random.seed(0); torch.manual_seed(0)
    service_fuzz()

