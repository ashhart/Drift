"""Launch the pinned GLM MCDMA owner under the existing process supervisor."""
import argparse

from drift.serving.mcdma_native_loader import load_opener
from drift.serving.mcdma_owner_runtime import run_owner
from drift.serving.worker_owner_stdio import pinned_json


def launch(configuration, runtime):
    if (type(runtime) is not dict or set(runtime) != {'v', 'binding', 'library', 'transport', 'memory_root'}
            or type(runtime['v']) is not int or runtime['v'] != 1):
        raise ValueError('MCDMA_RUNTIME_CONFIG')
    opener = None
    def connect(peer, *, src):
        nonlocal opener
        if opener is None:
            opener = load_opener(binding=runtime['binding'], library=runtime['library'])
        return opener(peer, src=src)
    return run_owner(configuration, runtime['transport'], connect, memory_root=runtime['memory_root'])


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('config', 'config-sha256', 'runtime', 'runtime-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    try:
        return launch(pinned_json(args.config, args.config_sha256), pinned_json(args.runtime, args.runtime_sha256))
    except Exception:
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
