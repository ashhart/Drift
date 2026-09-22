"""Keep transport sequences across completed native request identities."""
import json

import pytest

from drift.serving.mcdma_forward import Complete, ForwardMailbox
from drift.serving.mcdma_mailbox import Reader
from tests.test_mcdma_forward import Region
from tests.test_mcdma_mailbox import _bridge_module


def setup_bridge(tmp_path):
    module = _bridge_module()
    region = Region()
    reader = Reader(region, session=91)
    bridge = module.Bridge(None, None, str(tmp_path / 'in'), str(tmp_path), 0, True, region)
    return module, bridge, reader


def complete(tmp_path, bridge, reader, name):
    folder = tmp_path / name
    folder.mkdir()
    (folder / 'finished').write_text(json.dumps(dict(failed='', writes_scheduled=1,
                                                   tap_count=0, source_start=20, source_stop=20)))
    bridge.publish_taps()
    forward = ForwardMailbox(reader, name, (3,))
    message = forward.peek()
    assert isinstance(message, Complete)
    forward.acknowledge(message)
    return forward


def test_new_request_after_completion_keeps_writer_and_wire_sequence(tmp_path):
    module, bridge, reader = setup_bridge(tmp_path)
    bridge.handle(module.pack('watch', 'request-one', 0))
    complete(tmp_path, bridge, reader, 'request-one')
    writer = bridge.forward
    bridge.handle(module.pack('watch', 'request-two', 0))
    complete(tmp_path, bridge, reader, 'request-two')
    assert bridge.forward is writer
    assert writer.sequence == 2 and reader.expected == 3
    assert writer.acknowledged()


def test_new_request_is_rejected_before_terminal_ack(tmp_path):
    module, bridge, reader = setup_bridge(tmp_path)
    bridge.handle(module.pack('watch', 'request-one', 0))
    with pytest.raises(ValueError):
        bridge.handle(module.pack('watch', 'request-two', 0))
    assert bridge.watching == 'request-one'


def test_previous_request_identity_cannot_be_replayed(tmp_path):
    module, bridge, reader = setup_bridge(tmp_path)
    bridge.handle(module.pack('watch', 'request-one', 0))
    complete(tmp_path, bridge, reader, 'request-one')
    bridge.handle(module.pack('watch', 'request-two', 0))
    complete(tmp_path, bridge, reader, 'request-two')
    with pytest.raises(ValueError):
        bridge.handle(module.pack('watch', 'request-one', 0))


def test_failed_writer_cannot_be_reused_for_another_request(tmp_path):
    module, bridge, reader = setup_bridge(tmp_path)
    bridge.handle(module.pack('watch', 'request-one', 0))
    complete(tmp_path, bridge, reader, 'request-one')
    bridge.forward_failed = True
    with pytest.raises(ValueError):
        bridge.handle(module.pack('watch', 'request-two', 0))
