from __future__ import annotations


def shared_causal_endpoints(text: str, source_offsets: list[tuple[int, int]],
                            target_offsets: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Align rows ending at the SAME original-text byte boundary.

    Input offsets must already be validated as original-string CHARACTER offsets
    (e.g. a checked fast tokenizer). Zero-length special tokens are excluded.
    If several tokens end at one boundary, select the final token there. No
    nearest-neighbor or future-looking overlap averaging is done. This only
    supplies offline supervision; inference receives no shared text/offset map.
    """
    byte_ends = [0]
    for char in text:
        byte_ends.append(byte_ends[-1] + len(char.encode("utf-8")))

    def index(offsets):
        result, previous_end = {}, 0
        for token, (start, end) in enumerate(offsets):
            if not 0 <= start <= end <= len(text):
                raise ValueError("offset is not in the original character coordinate system")
            if start == end:
                continue
            if end < previous_end:
                raise ValueError("nonmonotonic tokenizer offsets")
            previous_end = end
            result[byte_ends[end]] = token
        return result

    source, target = index(source_offsets), index(target_offsets)
    return [(source[end], target[end], end) for end in sorted(source.keys() & target.keys())]
