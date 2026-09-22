"""Retain one delivered publication until the receiver confirms cache application."""
from drift.exchange import validation


class OutboundPublication:
    def __init__(self, owner):
        self.owner, self.pending = owner, None

    def deliver(self, *, staged=False):
        owner = self.owner
        if self.pending is not None:
            owner._poison('EXCHANGE_PENDING')
        try:
            snapshot = owner._invoke('EXCHANGE_LINK_FAILED', owner.source.snapshot, owner.copies)
            rows = validation.snapshot(snapshot, owner.copies)
            if owner.published_rows + rows > owner.publish_row_cap:
                owner._poison('EXCHANGE_PUBLISH_CAP')
            record = owner._record(rows=rows, source_rows=snapshot['rows'], copies=owner.copies,
                                   size=len(snapshot['body']), sha256=snapshot['sha256'], text_bytes=0,
                                   applied=(), receipts=())
            transfer = owner.link.stage if staged else owner.link.deliver
            delivered = owner._invoke('EXCHANGE_LINK_FAILED', transfer,
                                      owner.session, owner.sequence, snapshot['body'], owner.copies, post=False)
            self.pending = (record, delivered)
            owner._checkpoint()
            return {key: value for key, value in record.items() if key not in ('applied_ranks', 'receipts')} | {
                'state': 'STAGED_NOT_APPLIED' if staged else 'DELIVERED_NOT_APPLIED'}
        except Exception as error:
            owner._poison('EXCHANGE_LINK_FAILED', error)

    def confirm(self, sequence):
        owner = self.owner
        if self.pending is None:
            owner._poison('EXCHANGE_NO_PENDING')
        record, delivered = self.pending
        if type(sequence) is not int or sequence != record['sequence'] or sequence != owner.sequence:
            owner._poison('EXCHANGE_SEQUENCE')
        try:
            applied = owner._invoke('EXCHANGE_LINK_FAILED', owner.link.confirm_applied, delivered, record['rows'])
            return {**record, 'applied_ranks': tuple(applied['ranks']), 'receipts': tuple(applied['receipts'])}
        except Exception as error:
            owner._poison('EXCHANGE_LINK_FAILED', error)
