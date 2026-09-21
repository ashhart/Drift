"""Observe owned cancellation without launching requests or claiming allocator-memory release."""
import math
import time
from dataclasses import asdict

from drift.serving.live_metrics import MetricsSnapshot

FIELDS = ('started', 'active', 'abort_sent', 'cancelled', 'completed', 'concurrent')


class CancellationObserver:
    """Keep baseline, activity, explicit cancellation and recovery inside one deadline."""

    def __init__(self, *, timeout=180, baseline_samples=3, settled_samples=3,
                 clock=time.monotonic, max_samples=4096):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        if any(type(value) is not int or value < 2 for value in (baseline_samples, settled_samples, max_samples)):
            raise ValueError('sample bounds must be integers of at least two')
        self.clock, self.started_at = clock, clock()
        self.deadline = self.started_at + timeout
        self.last_snapshot = None
        self.baseline_required, self.settled_required = baseline_samples, settled_samples
        self.max_samples, self.samples = max_samples, 0
        self.baseline_count, self.settled_count = 0, 0
        self.baseline, self.last_time, self.confirmed_at = None, None, None
        self.status, self.phase, self.reason = 'RUNNING', 'BASELINE', None
        self.activity_seen = False
        self.flags = {name: False for name in FIELDS}

    def remaining(self):
        return max(0.0, self.deadline - self.clock())

    def _finish(self, status, reason):
        self.status, self.reason = status, reason
        return self.report()

    def report(self):
        action = None
        if self.status == 'RUNNING':
            if self.phase == 'READY' and not self.flags['started']: action = 'start'
            if self.phase == 'ACTIVE' and not self.flags['abort_sent']: action = 'abort'
        return {'status': self.status, 'phase': self.phase, 'reason': self.reason, 'action': action,
                'baseline': asdict(self.baseline) if self.baseline else None,
                'baseline_samples': self.baseline_count, 'settled_samples': self.settled_count,
                'samples': self.samples, 'activity_seen': self.activity_seen,
                'evidence': dict(self.flags), 'allocator_memory': 'NOT_MEASURED',
                'elapsed_seconds': max(0.0, self.clock() - self.started_at),
                'remaining_seconds': self.remaining(),
                'last_snapshot': asdict(self.last_snapshot) if self.last_snapshot else None}

    def poll(self, sample, request_state):
        """Both callbacks must finish within the remaining deadline; sampling receives its timeout."""
        if self.status != 'RUNNING':
            return self.report()
        if not self.remaining():
            return self._finish('FAILED', 'DEADLINE_EXCEEDED')
        if self.samples >= self.max_samples:
            return self._finish('FAILED', 'SAMPLE_LIMIT_EXCEEDED')
        try:
            snapshot = sample(timeout=self.remaining())
        except Exception:
            return self._finish('INVALID', 'METRICS_UNAVAILABLE')
        self.samples += 1
        try:
            evidence = request_state()
            flags = {name: getattr(evidence, name) for name in FIELDS}
        except Exception:
            return self._finish('INVALID', 'LOCAL_EVIDENCE_UNAVAILABLE')
        now = self.clock()
        if now >= self.deadline:
            return self._finish('FAILED', 'DEADLINE_EXCEEDED')
        if self.last_time is not None and now < self.last_time:
            return self._finish('INVALID', 'CLOCK_REVERSED')
        self.last_time = now
        if not isinstance(snapshot, MetricsSnapshot):
            return self._finish('INVALID', 'METRICS_UNAVAILABLE')
        self.last_snapshot = snapshot
        if any(type(value) is not bool for value in flags.values()):
            return self._finish('INVALID', 'LOCAL_EVIDENCE_UNAVAILABLE')
        if flags['concurrent'] or snapshot.running + snapshot.waiting > 1:
            return self._finish('INVALID', 'CONCURRENT_ACTIVITY')
        if flags['completed']:
            return self._finish('INVALID', 'NATURAL_COMPLETION')
        if any(self.flags[name] and not flags[name] for name in ('started', 'abort_sent', 'cancelled')):
            return self._finish('INVALID', 'LOCAL_EVIDENCE_REGRESSED')
        if (flags['active'] or flags['abort_sent'] or flags['cancelled']) and not flags['started']:
            return self._finish('INVALID', 'CONTRADICTORY_LOCAL_EVIDENCE')
        if flags['cancelled'] and (not flags['abort_sent'] or flags['active']):
            return self._finish('INVALID', 'MISSING_ABORT_EVIDENCE')
        self.flags = flags
        if self.phase == 'BASELINE':
            return self._baseline(snapshot, now)
        if (snapshot.engine, snapshot.model) != (self.baseline.engine, self.baseline.model):
            return self._finish('INVALID', 'TARGET_CHANGED')
        if flags['abort_sent'] and not self.activity_seen:
            return self._finish('INVALID', 'ABORT_BEFORE_ACTIVITY')
        if not flags['started'] and (snapshot.running or snapshot.waiting):
            return self._finish('INVALID', 'CONCURRENT_ACTIVITY')
        if not self.activity_seen and flags['active'] and snapshot.running == 1 and snapshot.waiting == 0:
            self.activity_seen, self.phase = True, 'ACTIVE'
        if self.activity_seen and flags['abort_sent']:
            self.phase = 'DRAINING'
            self._settled(snapshot, now)
        return self.report()

    def _baseline(self, snapshot, now):
        if self.flags['started']:
            return self._finish('INVALID', 'STARTED_BEFORE_BASELINE')
        if snapshot.running or snapshot.waiting:
            return self._finish('INVALID', 'BASELINE_NOT_IDLE')
        if self.baseline is not None:
            if (snapshot.model, snapshot.engine) != (self.baseline.model, self.baseline.engine):
                return self._finish('INVALID', 'TARGET_CHANGED')
            usage = min(snapshot.kv_usage, self.baseline.kv_usage)
            snapshot = MetricsSnapshot(0, 0, usage, snapshot.model, snapshot.engine)
        self.baseline = snapshot
        if self.confirmed_at is None or now > self.confirmed_at:
            self.baseline_count += 1
            self.confirmed_at = now
        if self.baseline_count >= self.baseline_required:
            self.phase = 'READY'
        return self.report()

    def _settled(self, snapshot, now):
        settled = (self.flags['cancelled'] and not self.flags['active'] and snapshot.running == 0
                   and snapshot.waiting == 0 and snapshot.kv_usage <= self.baseline.kv_usage)
        if not settled:
            self.settled_count = 0
            return
        if self.confirmed_at is None or now > self.confirmed_at:
            self.settled_count += 1
            self.confirmed_at = now
        if self.settled_count >= self.settled_required:
            self.status, self.phase, self.reason = 'PASSED', 'SETTLED', 'OBSERVED_CANCELLATION_RECOVERY'
