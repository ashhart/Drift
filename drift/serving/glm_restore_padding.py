"""Verify local padding without changing native chat fields or exposing token IDs."""
from drift.serving.glm_restore_prompt import reserve_prompt


def padded_reservation(verifier, body, rows, remaining, enabled):
    reserved = reserve_prompt(body, rows)
    proof = verifier.verify(body, reserved, rows, remaining())
    if not enabled:
        return reserved, proof
    first = proof['reserve_start']
    own_start = ((first+rows+63)//64)*64+64
    inserted = own_start-first
    reserved = reserve_prompt(body, inserted)
    padded = verifier.verify(body, reserved, inserted, remaining())
    if padded['reserve_start'] != first or padded['own_tokens'] != proof['own_tokens']:
        raise ValueError('causal reservation changed native own input')
    return reserved, {**padded, 'reserve_tokens': rows, 'padding_tokens': inserted-rows,
                       'prefill_boundary': own_start}
