"""Compose the pinned native owner routes without loading or restarting either model."""
from pathlib import Path

from drift.exchange.coordinator import ExchangeCoordinator
from drift.exchange.owner_links import AppendOwnerLink, SnapshotOwnerLink, require
from drift.exchange.owner_peer import OwnerPeer
from drift.exchange.owner_sources import CollectedOwnerSource, SelectedOwnerSource
from drift.exchange.session import ExchangeSession
from drift.serving.bridge_recipe import load_bridge_recipe
from drift.serving.worker_activation_files import private_root


class LazyOwnerPeer:
    def __init__(self, path, deadline):
        self.path, self.deadline, self.peer = path, deadline, None

    @property
    def requests(self):
        return self.peer.requests if self.peer else 0

    def request(self, frame):
        if self.peer is None:
            self.peer = OwnerPeer(self.path, self.deadline)
        return self.peer.request(frame)

    def close(self):
        if self.peer is not None:
            self.peer.close()


class NativeOwnerCoordinator:
    def __init__(self, profile, deadline):
        require(type(profile) is dict and set(profile) == {'v','socket','scratch','glm','qwen','recipe'} and profile['v'] == 1)
        glm, qwen, recipe = profile['glm'], profile['qwen'], profile['recipe']
        require(set(glm) == {'control','memory','exports','session','worker','ranks'})
        require(set(qwen) == {'control','memory','session','worker','rank'})
        require(set(recipe) == {'path','sha256','root'})
        require(glm['worker'] != qwen['worker'])
        scratch = private_root(profile['scratch'])
        for endpoint in (glm, qwen):
            private_root(endpoint['memory']); private_root(Path(endpoint['control']).parent)
        private_root(glm['exports']); private_root(Path(profile['socket']).parent)
        self.peers = {name:LazyOwnerPeer(endpoint['control'], deadline) for name,endpoint in (('glm',glm),('qwen',qwen))}
        recipes = {direction:load_bridge_recipe(direction, recipe['path'], recipe['sha256'],
                    artifact_root=recipe['root'], scratch=scratch, max_bytes=700*1048576)
                   for direction in ('forward','reverse')}
        self.sources = {
            'qwen-to-glm':SelectedOwnerSource(self.peers['qwen'], qwen['memory'], scratch, recipes['reverse'],
                session=qwen['session'], source_worker=qwen['worker'], target_worker=glm['worker']),
            'glm-to-qwen':CollectedOwnerSource(glm['exports'], scratch, recipes['forward'],
                source_worker=glm['worker'], target_worker=qwen['worker']),
        }
        links = {
            'qwen-to-glm':SnapshotOwnerLink(self.peers['glm'], glm['memory'], session=glm['session'],
                source_worker=qwen['worker'], target_worker=glm['worker'], recipe=recipe['sha256'], ranks=glm['ranks']),
            'glm-to-qwen':AppendOwnerLink(self.peers['qwen'], qwen['memory'], rank=qwen['rank'], session=qwen['session'],
                source_worker=glm['worker'], target_worker=qwen['worker']),
        }
        sessions = {}
        for route, source, target, ranks in (('qwen-to-glm',qwen,glm,glm['ranks']), ('glm-to-qwen',glm,qwen,[qwen['rank']])):
            sessions[route] = ExchangeSession(session=target['session'], source_worker=source['worker'],
                target_worker=target['worker'], mode='drift', ranks=ranks, source=self.sources[route],
                link=links[route], sink=None, copies=1, row_cap=512, publish_row_cap=512)
        self.server = ExchangeCoordinator(Path(profile['socket']), sessions, deadline, max_requests=64)

    def start(self):
        self.server.start()

    def close(self):
        try: self.server.close()
        finally:
            for peer in self.peers.values(): peer.close()
