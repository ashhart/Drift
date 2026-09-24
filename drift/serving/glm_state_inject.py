"""Write an exported recurrent state into a live GLM request at the end of its reserve.

The owner's handoff connector exports, per rank, the page bytes of that rank's recurrent-layer state blocks
(recurrent state plus short-convolution window) for the end of an export prompt, into a file or into the
handoff daemon's arena under a time-limited lease, as the rank's ready file records. When a live request's first
prefill step ends exactly at its reserve, each rank copies those bytes unchanged into the block holding the
request's running state, so the own tokens after the reserve continue from the exported state. A byte copy is
only valid on the same cache layout, so every page's registered shape and dtype must match the live cache.
"""
import json
import os
import time

try:
    from glm53_handoff import read_header_bytes
except ImportError:
    from drift.serving.glm53_handoff import read_header_bytes


def load_blob(connector, name, rank):
    """This rank's export as bytes, from its file or from the handoff daemon's arena, as its ready file says."""
    folder = connector._output_path/name
    ready = json.loads((folder/f'rank{rank}.ready').read_text())
    if ready.get('mode') != 'arena':
        return (folder/f'rank{rank}.bin').read_bytes()
    arena = connector._map_arena()
    if not arena or arena[3] != ready.get('arena_inode'):
        raise ValueError('the export arena is not mapped in this worker')
    if time.time()-float(ready.get('created', 0)) >= float(connector._arena_ttl):
        raise ValueError('the export arena lease has expired')
    start, length = int(ready['offset']), int(ready['length'])
    return bytes(arena[1][start:start+length])


def read_pages(blob, name, rank, world):
    """{(layer, part): (entry, page bytes)} for the prompt-end state this rank exported under `name`."""
    header, data_start = read_header_bytes(blob)
    if (header.get('format') != 'glm53-handoff-raw-v1' or header.get('handoff_id') != name
            or (header.get('tp_rank'), header.get('tp_size')) != (rank, world)):
        raise ValueError("state blob is not this rank's export")
    end, view, pages = int(header['n_tokens']), memoryview(blob), {}
    for entry in header['tensors']:
        if entry.get('kind') != 'state':
            continue
        tokens = [int(t) for t in entry['state_tokens']]
        size = int(entry['page_bytes'])
        if end not in tokens or int(entry.get('pages_per_block', 1)) != 1 or int(entry['nbytes']) != size*len(tokens):
            raise ValueError('state entry has no single prompt-end page')
        key = (entry['layer'], int(entry['part']))
        if key in pages:
            raise ValueError('duplicate state entry')
        start = data_start+int(entry['offset'])+tokens.index(end)*size
        if start+size > len(view):
            raise ValueError('truncated state blob')
        pages[key] = (entry, bytes(view[start:start+size]))
    if not pages:
        raise ValueError('state blob holds no recurrent state')
    return pages


def stage(connector, step, pages):
    """Destination page views and source bytes for every state part, checked before anything is written."""
    import torch
    targets = {name for name, group in connector._layer_to_group.items()
               if group in connector._state_groups and name in connector._kv_caches}
    if {layer for layer, _ in pages} != targets:
        raise ValueError("state layers do not match this server's recurrent layers")
    staged = []
    for (layer, index), (entry, data) in sorted(pages.items()):
        cache = connector._kv_caches[layer]
        parts = list(cache) if isinstance(cache, (list, tuple)) else [cache]
        if not 0 <= index < len(parts):
            raise ValueError('state part outside the layer cache')
        part = parts[index]
        if list(part.shape) != list(entry['registered_shape']) or str(part.dtype).removeprefix('torch.') != entry['dtype']:
            raise ValueError('exported state layout differs from the live cache')
        group = connector._layer_to_group[layer]
        blocks = step.blocks[group] if group < len(step.blocks) else ()
        slot = (step.after-1)//connector._state_groups[group]
        if not 0 <= slot < len(blocks):
            raise ValueError('running state block outside the request')
        rows, ratio = connector._page_rows(part, (blocks[slot],))
        page = part[rows[0]].reshape(-1).view(torch.uint8) if ratio == 1 else None
        if page is None or page.data_ptr() != part[rows[0]].data_ptr() or page.numel() < len(data):
            raise ValueError('running state page is not one contiguous page')
        staged.append((page, torch.frombuffer(bytearray(data), dtype=torch.uint8).to(page.device)))
    return staged


def write_state(connector, step, then=None):
    """Copy this rank's exported state into the running state block, then run `then`; the step must end at the reserve."""
    import torch
    if step.after != step.reserve_start+step.reserve:
        raise RuntimeError('a recurrent state is written only when the step ends at the reserve')
    started = time.perf_counter()
    rank, world = connector._tp()
    staged = stage(connector, step, read_pages(load_blob(connector, step.state_blob, rank), step.state_blob, rank, world))
    on_gpu = any(page.is_cuda for page, _ in staged)
    if on_gpu:
        torch.cuda.synchronize()                                     # the step that ran over the placeholders is done
    for page, source in staged:
        page[:source.numel()].copy_(source)
    checks = [(page[:source.numel()] == source).all() for page, source in staged]
    if on_gpu:
        torch.cuda.synchronize()
    if not all(check.item() for check in checks):
        raise RuntimeError('recurrent state readback mismatch')
    report = {'rank': int(rank), 'parts': len(staged), 'bytes': int(sum(s.numel() for _, s in staged)), 'position': int(step.after)}
    if then is not None:
        report.update(then())                                         # the memory's own contribution, from the written base
    report['seconds'] = round(time.perf_counter()-started, 4)
    folder = connector._live_out/step.name
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder/f'.state.rank{rank}.tmp'
    temporary.write_text(json.dumps(report))
    os.replace(temporary, folder/f'state.rank{rank}.json')
    return report
