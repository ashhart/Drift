"""Park two owned fixed-echo actors while hashing and discarding their output."""
import hashlib
import json
import os
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path


def remaining(deadline):
    value=deadline-time.monotonic()
    if value<=0:raise TimeoutError('ROUND_DEADLINE')
    return value


def read_json(path,maximum=65536):
    path=Path(path)
    if path.resolve()!=path or path.is_symlink():raise ValueError('ROUND_PATH')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise ValueError('ROUND_FILE')
        raw=stream.read(maximum+1)
    if len(raw)>maximum:raise ValueError('ROUND_SIZE')
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


class Actor:
    def __init__(self,command,cwd,env):
        self.process=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        self.counts={};self.excess=False;self.terminated=False
        def consume(name,stream):
            size=0;digest=hashlib.sha256()
            for chunk in iter(lambda:stream.read(8192),b''):
                size+=len(chunk);digest.update(chunk)
                if size>2097152:self.excess=True;self.signal(signal.SIGTERM)
            self.counts[name]={'bytes':size,'sha256':digest.hexdigest()};stream.close()
        self.threads=[threading.Thread(target=consume,args=(name,stream),daemon=True) for name,stream in (('stdout',self.process.stdout),('stderr',self.process.stderr))]
        for thread in self.threads:thread.start()

    def signal(self,value):
        self.terminated=True
        try:os.killpg(self.process.pid,value)
        except ProcessLookupError:pass

    def close(self,deadline):
        if self.process.poll() is None:
            self.signal(signal.SIGTERM)
            try:self.process.wait(timeout=min(1,remaining(deadline)))
            except (subprocess.TimeoutExpired,TimeoutError):
                self.signal(signal.SIGKILL);self.process.wait(timeout=max(.05,deadline-time.monotonic()))
        try:os.killpg(self.process.pid,signal.SIGKILL);self.terminated=True
        except ProcessLookupError:pass
        for thread in self.threads:thread.join(timeout=max(0,deadline-time.monotonic()))
        return dict(exit_code=self.process.returncode,reaped=self.process.poll() is not None,terminated=self.terminated,output_limit=self.excess,joined=all(not t.is_alive() for t in self.threads),**self.counts)


def rendezvous(actors,exchange,deadline):
    while True:
        remaining(deadline)
        if any(actor.process.poll() is not None or actor.excess for actor,root in actors.values()):raise ValueError('ROUND_ACTOR_EARLY_EXIT')
        if all((root/'ready.json').exists() for actor,root in actors.values()):break
        time.sleep(min(.01,remaining(deadline)))
    hashes={}
    for name,(actor,root) in actors.items():
        value,digest=read_json(root/'ready.json',4096)
        if value!={'v':1,'phase':'tool_boundary'}:raise ValueError('ROUND_READY')
        hashes[name]=digest
    exchange();remaining(deadline)
    if any(actor.process.poll() is not None for actor,root in actors.values()):raise ValueError('ROUND_ACTOR_EARLY_EXIT')
    for name,(actor,root) in actors.items():
        if read_json(root/'ready.json',4096)[1]!=hashes[name]:raise ValueError('ROUND_READY_CHANGED')
        raw=json.dumps({'v':1,'action':'resume','ready_sha256':hashes[name]}).encode()
        path=root/'release.tmp';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.link(path,root/'release.json');path.unlink()
    while any(actor.process.poll() is None for actor,root in actors.values()):time.sleep(min(.01,remaining(deadline)))
    return {name:actor.close(deadline) for name,(actor,root) in actors.items()}
