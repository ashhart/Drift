"""Track complete allocations, incremental extensions and invalidating preemption."""


def merge_blocks(previous, additions, replace=False):
    if additions is None:
        return () if replace else previous
    groups = tuple(tuple(int(block) for block in group) for group in additions)
    if replace or not previous:
        return groups
    return tuple((previous[i] if i < len(previous) else ()) +
                 (groups[i] if i < len(groups) else ())
                 for i in range(max(len(previous), len(groups))))


def invalidate_preempted(connector, output):
    for request_id in getattr(output, "preempted_req_ids", ()):
        pending = connector._tp_pending.get(request_id)
        if pending is not None:
            pending.block_ids = ()
        state = connector._live.get(request_id)
        if state is not None:
            state["blocks"] = ()
            state["failed"] = "live Drift request preempted; a fresh session is required"


def live_blocks(state, additions, replace):
    if replace and state.get("allocated", False) and not state.get("failed"):
        state["failed"] = "live Drift request resumed; a fresh session is required"
    state["blocks"] = merge_blocks(state["blocks"], additions, replace)
    state["allocated"] = True
