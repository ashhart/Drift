"""Wait for typed peer completion and all forward taps within one deadline."""
import queue
import threading
import time


def wait_for_completion(stream, drain, forward_done, timeout_s=40):
    if not 0 < timeout_s <= 600:
        raise ValueError("invalid peer completion deadline")
    control = queue.Queue(maxsize=1)

    def read_control():
        try:
            control.put(stream.readline(32) == "peer_done\n")
        except Exception:
            control.put(False)

    threading.Thread(target=read_control, daemon=True).start()
    deadline, peer_done, taps = time.monotonic() + timeout_s, False, []
    while time.monotonic() < deadline:
        taps.extend(drain())
        if not peer_done:
            try:
                peer_done = control.get_nowait()
                if not peer_done:
                    raise RuntimeError("peer completion control missing or invalid")
            except queue.Empty:
                pass
        if peer_done and forward_done():
            return taps
        time.sleep(min(0.01, max(0, deadline - time.monotonic())))
    raise TimeoutError("peer completion or forward terminal marker missing")
