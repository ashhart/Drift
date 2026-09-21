"""Build an explicitly pinned native-chat snapshot-restoration backend."""
from pathlib import Path
from drift.serving.glm_factory import factory as unlinked_factory
from drift.serving.glm_restore_prefix import PrefixVerifier
from drift.serving.glm_restore_route import PinnedRoute, canonical_file
from drift.serving.glm_restore_turn import RestoringTransport, TurnRestorer
from drift.serving.glm_turn_outbox import TurnOutbox
from drift.serving.live_publication import load_publication
from drift.serving.live_publication_transport import PublicationTransport
from drift.serving.worker_artifacts import digest

ROOT = Path('/dev/shm/glm53-handoff')


def factory(configuration):
    mode = configuration.get('memory_mode')
    if mode not in ('linked_snapshot', 'no_link_reserved'): raise ValueError('explicit restoration mode required')
    recipe = configuration['restoration']
    source = canonical_file(recipe['snapshot_path'], 1048576)
    expected, rows, layers = recipe['snapshot_sha256'], recipe['rows'], recipe['layers']
    maximum = recipe['max_rows']
    if type(maximum) is not int or not 1 <= maximum <= 4096 or type(rows) is not int or not 1 <= rows <= maximum:
        raise ValueError('snapshot row budget mismatch')
    if not isinstance(layers, list) or not layers or len(layers) > 256 or any(type(v) is not int or v < 0 for v in layers) or len(set(layers)) != len(layers):
        raise ValueError('snapshot layer declaration mismatch')
    if digest(source, 1048576) != expected: raise ValueError('snapshot digest mismatch')
    arrays = load_publication(source, {f'l{layer}': (512,) for layer in layers}, max_rows=maximum)
    if any(value.shape[0] != rows for value in arrays.values()) or digest(source, 1048576) != expected:
        raise ValueError('snapshot row count or bytes changed')
    owners = [recipe.get(key) for key in ('source_worker', 'target_worker')]
    if any(type(value) is not str or not 1 <= len(value) <= 96 for value in owners) or owners[0] == owners[1]:
        raise ValueError('distinct source ownership labels required')
    outbox = None
    if 'outbox' in configuration:
        spec = configuration['outbox']
        if mode != 'linked_snapshot' or (spec['source_worker'], spec['target_worker']) != (owners[1], owners[0]):
            raise ValueError('outbound ownership or mode differs')
        outbox = TurnOutbox(spec, {f'l{layer}': (512,) for layer in layers}, ROOT / 'tp-live-out',
                            max_turns=configuration['limits']['max_turns'])
    route = PinnedRoute(recipe['route_path'], recipe['route_sha256']) if mode == 'linked_snapshot' else None
    session = unlinked_factory({**configuration, 'memory_mode': 'no_link'})
    verifier = PrefixVerifier(configuration.get('base_url', 'http://127.0.0.1:8888'))
    def deadline():
        return session.started + session.limits['deadline_ms'] / 1000
    def publication(name):
        canonical_file(str(source), 1048576)
        return route.bind(PublicationTransport(source, expected, name, route.spec['peer'], restorer.remaining))
    restorer = TurnRestorer(verifier, publication, rows=rows, digest=expected, root=ROOT, deadline=deadline,
                            max_input_bytes=1048576, linked=mode == 'linked_snapshot', outbox=outbox)
    def prepare(body):
        restorer.max_input_bytes = session.limits['max_input_bytes']
        return restorer.prepare(body)
    session.transport = RestoringTransport(session.transport, restorer)
    session.prepare_turn, session.restoration = prepare, restorer
    session.restoration_ownership = dict(zip(('source_worker', 'target_worker'), owners))
    return session
