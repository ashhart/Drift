"""Read one public note, tap without decoding, then convert its final canonical row privately."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from drift.serving.qwen_fixture import PUBLIC_NOTE, REVERSE_SHA256, build_fixture, sha256_file


def main():
    from tokenizers import Tokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','translator','tap-script','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if any(args.out.iterdir()): raise ValueError('fixture child requires an empty private directory')
    if sha256_file(args.translator) != REVERSE_SHA256: raise ValueError('translator hash differs')
    tokenizer = args.checkpoint/'tokenizer.json'
    ids = Tokenizer.from_file(str(tokenizer)).encode(PUBLIC_NOTE,add_special_tokens=False).ids
    if not 1 <= len(ids) <= 32: raise ValueError('source token limit exceeded')
    ids_path = args.out/'ids.json'
    with ids_path.open('x') as stream:
        json.dump({'records':[{'id':'source','ids_qwen':ids}]},stream)
    fingerprint = {'config_sha256':sha256_file(args.checkpoint/'config.json'),
                   'tokenizer_sha256':sha256_file(tokenizer),'tap_script_sha256':sha256_file(args.tap_script)}
    index = args.checkpoint/'model.safetensors.index.json'
    fingerprint['weights_index_sha256'] = sha256_file(index) if index.is_file() else None
    fingerprint['full_weight_identity'] = 'NOT_MEASURED'
    tapped_at = time.monotonic()
    completed = subprocess.run([sys.executable,str(args.tap_script),'--ids',str(ids_path),
        '--out',str(args.out/'taps'),'--checkpoint',str(args.checkpoint)],check=True,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=max(.1,270-(time.monotonic()-started)))
    if {path.name for path in (args.out/'taps').iterdir()} != {'source.npz'}:
        raise ValueError('tap outputs differ from the single controlled input')
    tap_seconds = time.monotonic()-tapped_at
    converted_at = time.monotonic()
    result = build_fixture(args.out/'taps/source.npz',args.translator,args.out/'publication.npz',len(ids))
    result.update(source_fingerprint=fingerprint,tap_process_seconds=tap_seconds,
                  translation_seconds=time.monotonic()-converted_at,child_seconds=time.monotonic()-started,
                  ids_file_sha256=sha256_file(ids_path),model_requests=1,warmup_requests=0)
    with (args.out/'fixture.json').open('x') as stream: json.dump(result,stream,sort_keys=True)
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception: raise SystemExit(2) from None
