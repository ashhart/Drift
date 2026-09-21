"""Bind the exchange ports to the real MCDMA mailboxes and the oMLX cache, with no runtime import at module load."""
import hashlib
import io

import numpy as np

from drift.serving.mcdma_forward import Complete, Tap, validate_translation
from drift.serving.omlx_cache import append_entries, tap_slots


class LiveExchangeError(RuntimeError):
    pass


class MailboxLink:
    """Outbound through the reverse publisher, inbound through the forward mailbox; one link, two directions."""

    def __init__(self, publisher, forward):
        self.publisher, self.forward = publisher, forward

    def deliver(self, session, sequence, body, copies):
        return self.publisher.deliver(session, sequence, body, copies)

    def confirm_applied(self, delivered, rows):
        """Block until every rank's connector receipt binds, then report the ranks and their applied digests."""
        result = self.publisher.confirm_applied(delivered, rows)
        receipts = result['receipts']
        ranks = tuple(sorted(receipts))
        return dict(ranks=ranks, receipts=tuple(receipts[rank]['sha256'] for rank in ranks))

    def peek(self):
        """Normalize control fields while retaining the exact message for the sink and ACK."""
        tap = self.forward.peek()
        if tap is None:
            return None
        if isinstance(tap, Complete):
            return dict(op='complete', tap_count=tap.tap_count, source_start=tap.source_start,
                        source_stop=tap.source_stop, payload=tap)
        if not isinstance(tap, Tap):
            raise LiveExchangeError('EXCHANGE_TAP')
        return dict(start=tap.start, stop=tap.stop, rows=tap.stop - tap.start, payload=tap)

    def acknowledge(self, tap):
        self.forward.acknowledge(tap['payload'])


class OwnRowSource:
    """Tap this side's own slots, translate them for the partner and pack exactly one publication."""

    def __init__(self, cache, rope, layers, target_layers, reader, slots, positions, gain=1.0):
        self.cache, self.rope, self.layers, self.target_layers = cache, rope, tuple(layers), tuple(target_layers)
        self.reader, self.slots, self.positions, self.gain = reader, slots, positions, gain

    def snapshot(self, copies):
        slots, positions = self.slots(), self.positions()
        rows = len(slots)
        if rows <= 0:
            raise LiveExchangeError('EXCHANGE_NO_OWN_ROWS')
        own = tap_slots(self.cache, list(self.layers), self.rope, np.asarray(slots), np.asarray(positions))
        flat = {layer: np.concatenate((own[layer][0].reshape(rows, -1), own[layer][1].reshape(rows, -1)), axis=1).astype(np.float32)
                for layer in self.layers}
        latents = self.reader.read(flat, self.gain)
        sink = io.BytesIO()
        np.savez(sink, **{f'l{layer}': np.asarray(latents[layer], dtype=np.float16) for layer in self.target_layers})
        body = sink.getvalue()
        return dict(body=body, sha256=hashlib.sha256(body).hexdigest(), rows=rows)


class ForeignRowSink:
    """Translate and validate a partner tap before the cache is touched, then append it."""

    def __init__(self, cache, rope, layers, reader, bank, index_dim, own_position, dtype=None, evaluate=None):
        self.cache, self.rope, self.layers, self.reader = cache, rope, tuple(layers), reader
        self.bank, self.index_dim, self.own_position = bank, index_dim, own_position
        self.dtype, self.evaluate = dtype, evaluate

    def prepare(self, tap):
        """No cache mutation happens here: the caller checks its foreign row cap against the returned count."""
        if type(tap) is dict:
            tap = tap['payload']
        if isinstance(tap, Complete):
            return 0, None
        entries, keys, order, _ = self.reader.read(tap.latents)
        rows = validate_translation(entries, keys, order, tap.stop - tap.start, self.layers, self.index_dim)
        return rows, (entries, keys, order, tap.start, rows)

    def commit(self, prepared):
        if prepared is None:
            return 0
        entries, keys, order, start, rows = prepared
        sources = start + np.asarray(order, dtype=np.int64)
        positions = self.bank.positions(sources, self.own_position())
        first_slot = append_entries(self.cache, entries, self.rope, positions, self.index_dim,
                                    dtype=self.dtype, index_keys=keys)
        self.bank.remember(entries, sources, first_slot)
        if self.evaluate is not None:
            self.evaluate()
        return rows
