"""Keep fixture evidence private and read only lock availability and aggregate host memory."""
import fcntl
import os
from pathlib import Path
import re
import stat
import subprocess
import uuid


def claim_fixture(root):
    root = Path(root).expanduser()
    if root.is_symlink(): raise ValueError('fixture root must not be a symlink')
    root = root.resolve()
    if any((parent/'.git').exists() for parent in (root,*root.parents)):
        raise ValueError('raw fixture evidence must stay outside repositories')
    root.mkdir(parents=True,mode=0o700,exist_ok=True)
    if stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise ValueError('fixture root must be private')
    claim = root/('qwen-fixture-'+uuid.uuid4().hex)
    claim.mkdir(mode=0o700)
    (claim/'raw').mkdir(mode=0o700)
    return claim


def host_snapshot(timeout=2):
    lock = Path('~/drift/local/studio/.model.lock').expanduser()
    result = {'lock_exists':lock.exists(),'lock_available':None,'reclaimable_gib':None}
    if lock.is_file():
        try:
            with lock.open('r') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                result['lock_available']=True
                fcntl.flock(stream,fcntl.LOCK_UN)
        except BlockingIOError: result['lock_available']=False
        except OSError: result['lock_available']=None
    else: result['lock_available']=not lock.exists()
    try:
        completed = subprocess.run(['vm_stat'],check=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                                   text=True,timeout=timeout)
        match = re.search(r'page size of (\d+) bytes',completed.stdout)
        if not match: raise ValueError
        pages = {key:int(value) for key,value in re.findall(r'^(Pages [^:]+):\s*(\d+)\.',completed.stdout,re.M)}
        keys = ('Pages free','Pages inactive','Pages speculative','Pages purgeable')
        if any(key not in pages for key in keys): raise ValueError
        result['reclaimable_gib'] = sum(pages[key] for key in keys)*int(match[1])/2**30
    except (OSError,ValueError,subprocess.SubprocessError): pass
    return result
