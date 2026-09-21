"""Build the explicitly unlinked native GLM tool backend from pinned owner configuration."""
import os
import uuid

from drift.serving.glm_guard import EngineGuard
from drift.serving.glm_http import GlmHttp
from drift.serving.glm_session import GlmSession
from drift.serving.glm_diagnostics import GlmDiagnostics
from drift.serving.live_metric_fetch import fetch_metrics
from drift.serving.live_metrics import parse_metrics
from drift.serving.worker_artifacts import verify_worker_manifests


class AdmittedHttp(GlmHttp):
    def __init__(self, base, guard, *, diagnostics=None):
        super().__init__(base)
        self.guard = guard
        self.diagnostics = diagnostics if diagnostics is not None else GlmDiagnostics()

    def _spawn(self, path, body, timeout, deadline):
        child = super()._spawn(path, body, timeout, deadline)
        if path == '/v1/chat/completions': self.diagnostics.mark('glm_dispatch', request_dispatched=1)
        return child

    def stream(self, body, timeout):
        self.diagnostics.mark('glm_admit')
        self.guard.admit(timeout)
        self.diagnostics.mark('glm_dispatch')
        yield from super().stream(body, self.guard._remaining())


def factory(configuration):
    verify_worker_manifests(configuration)
    if configuration.get('memory_mode') != 'no_link':
        raise ValueError('linked GLM factory requires a verified memory restoration adapter')
    mode = configuration.get('service_auth')
    if mode not in ('none', 'bearer') or bool(os.environ.get('DRIFT_GLM_KEY')) != (mode == 'bearer'):
        raise ValueError('service authentication differs from pinned configuration')
    base = configuration.get('base_url', 'http://127.0.0.1:8888')
    model = configuration['pins']['model_id']
    def sample(timeout):
        return parse_metrics(fetch_metrics(base + '/metrics', timeout=timeout), model=model, engine='0')
    guard = EngineGuard(sample)
    diagnostics = GlmDiagnostics(configuration.get('diagnostics_path'))
    transport = AdmittedHttp(base, guard, diagnostics=diagnostics)
    def prepare(body):
        return {'cache_salt': 'drift:' + uuid.uuid4().hex, 'vllm_xargs': {'skip_writing_prefix_cache': 1}}
    return GlmSession(transport, model=model, prepare_turn=prepare, recovery=guard.recover, diagnostics=diagnostics)
