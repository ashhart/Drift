"""Materialize native cache arrays on their producing thread before releasing ownership."""


def settle_native_caches(caches, evaluate, array_type):
    if type(caches) not in (list, tuple) or len(caches) > 256:
        raise ValueError('UNSUPPORTED_NATIVE_CACHE_CONTAINER')
    arrays, seen = [], set()
    visited = 0

    def collect(value, depth=0):
        nonlocal visited
        visited += 1
        if visited > 16384 or depth > 16:
            raise ValueError('NATIVE_CACHE_STRUCTURE_LIMIT')
        if id(value) in seen:
            return
        if isinstance(value, array_type):
            seen.add(id(value)); arrays.append(value)
        elif type(value) in (list, tuple, dict):
            seen.add(id(value))
            for child in value.values() if type(value) is dict else value:
                collect(child, depth + 1)

    for cache in caches:
        collect(cache.state)
        for value in vars(cache).values():
            collect(value)
    if arrays:
        evaluate(*arrays)
    return len(arrays)
