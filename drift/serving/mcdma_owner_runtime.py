"""Own MCDMA handles for the GLM snapshot owner without an SSH payload path."""
import time

from drift.exchange.lifetime import request_scope
from drift.serving.glm_http import GlmHttp
from drift.serving.glm_owner_worker import serve_owner
from drift.serving.mcdma_connections import McdmaConnections, targets, require
from drift.serving.mcdma_restoration import McdmaRestorationFactory
from drift.serving.mcdma_turn_outbox import McdmaTurnOutbox
from drift.serving.worker_activation_files import private_root


class OwnerTransport:
    def __init__(self, configuration, transport, opener, memory_root, clock):
        self.clock, self.started = clock, clock()
        self.connections, self.collector, self.closed = None, None, False
        require(configuration.get('memory_mode') == 'linked_snapshot')
        require(configuration.get('restoration_transport') == 'mcdma' and 'outbox' in configuration)
        GlmHttp(configuration.get('base_url', 'http://127.0.0.1:8888')).close()
        milliseconds = configuration['limits']['deadline_ms']
        require(type(milliseconds) is int and 1000 <= milliseconds <= 180000)
        self.deadline = self.started + milliseconds / 1000
        self.root = private_root(memory_root)
        require(not any((path / '.git').exists() for path in (self.root, *self.root.parents)))
        recipe, spec = configuration['restoration'], configuration['outbox']
        layers = recipe['layers']
        require(type(layers) is list and 0 < len(layers) <= 256
                and all(type(layer) is int and layer >= 0 for layer in layers) and len(set(layers)) == len(layers))
        require(spec['source_worker'] == recipe['target_worker'] == configuration['worker'])
        require(spec['target_worker'] == recipe['source_worker'])
        require(spec['recipe_sha256'] == configuration['owner_bank']['recipe_sha256'])
        self.layouts = {f'l{layer}': (512,) for layer in layers}
        self.spec, self.turns = spec, configuration['limits']['max_turns']
        self.collector = McdmaTurnOutbox(spec, self.layouts, None, None, max_turns=self.turns)
        self.maximum = recipe['max_rows']
        McdmaRestorationFactory(None, layouts=self.layouts, max_rows=self.maximum)
        self.transport, self.opener = {'v': 1, 'targets': targets(transport)}, opener

    def check(self):
        if self.closed or self.clock() >= self.deadline:
            raise TimeoutError('MCDMA_OWNER_DEADLINE')

    def connected(self):
        self.check()
        if self.connections is None:
            self.connections = McdmaConnections(self.transport, self.opener, deadline=self.deadline)
        self.check()
        return self.connections

    def outbox(self, spec, layouts, *, max_turns):
        require(spec == self.spec and layouts == self.layouts and max_turns == self.turns)
        require(self.collector.reader is None)
        link = self.connected()
        self.collector.reader, self.collector.publisher = link.reader, link.publisher
        return self.collector

    def publication(self, *args):
        link = self.connected()
        return McdmaRestorationFactory(link.publisher, layouts=self.layouts, max_rows=self.maximum)(*args)

    def close(self):
        self.closed = True
        try:
            if self.collector is not None:
                self.collector.close()
        finally:
            if self.connections is not None:
                self.connections.close()


def run_owner(configuration, transport, opener, *, memory_root, serve=serve_owner,
              input_fd=0, sink=None, clock=time.monotonic):
    runtime = OwnerTransport(configuration, transport, opener, memory_root, clock)
    try:
        with request_scope(runtime.check):
            runtime.check()
            result = serve(configuration, publication_factory=runtime.publication, outbox_factory=runtime.outbox,
                           memory_root=runtime.root, input_fd=input_fd, sink=sink)
            runtime.check()
            return result
    finally:
        runtime.close()
