"""Bound owner-selected SSH file copies and one persistent Unix-socket forward."""
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import socket
import subprocess
import stat
import time
from linked_round_process import remaining

READ="""import os,pathlib,stat,sys
p=pathlib.Path(sys.argv[1]);maximum=int(sys.argv[2]);assert p.resolve()==p
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
with os.fdopen(fd,'rb') as f:
 s=os.fstat(f.fileno());assert stat.S_ISREG(s.st_mode) and 0<s.st_size<=maximum
 data=f.read(maximum+1);assert len(data)<=maximum
sys.stdout.buffer.write(data)
"""
WRITE="""import hashlib,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);maximum=int(sys.argv[2]);expected=sys.argv[3];assert p.parent.resolve()==p.parent and p.parent.is_dir() and not p.parent.stat().st_mode&0o077
raw=sys.stdin.buffer.read(maximum+1);assert len(raw)<=maximum and hashlib.sha256(raw).hexdigest()==expected
fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
"""
LOCATE="""import hashlib,json,pathlib,re,sys
root=pathlib.Path(sys.argv[1]);assert root.resolve()==root
folders=list(root.iterdir());assert len(folders)==1
p=folders[0];assert p.is_dir() and p.resolve()==p and re.fullmatch('restore-[A-Za-z0-9_.-]+',p.name)
f=p/'manifest.json';assert f.resolve()==f and f.stat().st_size<=1048576
raw=f.read_bytes();print(json.dumps({'directory':p.name,'sha256':hashlib.sha256(raw).hexdigest()}))
"""


class Remote:
    def __init__(self,command,deadline):
        self.command,self.deadline=command,deadline
        self.base=[command['executable'],*command['args'][:-1]]

    def call(self,program,args,target,maximum,source=None):
        command='python3 -I -S -c '+shlex.quote(program)+' '+shlex.join([str(v) for v in args])
        incoming=open(source,'rb') if source else subprocess.DEVNULL
        child=subprocess.Popen([*self.base,command],cwd=self.command['cwd'],env=self.command['env'],stdin=incoming,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,start_new_session=True)
        total=0;digest=hashlib.sha256()
        try:
            with Path(target).open('xb') as stream:
                os.chmod(target,0o600)
                while True:
                    if not select.select([child.stdout],[],[],remaining(self.deadline))[0]:raise TimeoutError('ROUND_SSH_TIMEOUT')
                    block=os.read(child.stdout.fileno(),65536)
                    if not block:break
                    total+=len(block)
                    if total>maximum:raise ValueError('ROUND_SSH_SIZE')
                    digest.update(block);stream.write(block)
            if child.wait(timeout=remaining(self.deadline))!=0:raise ValueError('ROUND_SSH_FAILED')
            return digest.hexdigest()
        finally:
            if source:incoming.close()
            if child.poll() is None:child.kill();child.wait(timeout=1)
            child.stdout.close()

    def download(self,remote,local,maximum,expected=None):
        digest=self.call(READ,[remote,maximum],local,maximum)
        if expected is not None and digest!=expected:raise ValueError('ROUND_SOURCE_HASH')
        return digest

    def upload(self,local,remote,maximum,expected,scratch):
        if Path(local).stat().st_size>maximum or hashlib.sha256(Path(local).read_bytes()).hexdigest()!=expected:raise ValueError('ROUND_UPLOAD_HASH')
        self.call(WRITE,[remote,maximum,expected],scratch,0,source=local)


class Forward:
    def __init__(self,remote,path,local):
        self.remote=remote;self.local=Path(local);self.socket=None
        spec=remote.command
        argv=[spec['executable'],*spec['args'][:-2],'-N','-o','ExitOnForwardFailure=yes','-o','StreamLocalBindMask=0177','-o','StreamLocalBindUnlink=no','-L',str(local)+':'+path,spec['args'][-2]]
        self.process=subprocess.Popen(argv,cwd=spec['cwd'],env=spec['env'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            while not self.local.exists():
                if self.process.poll() is not None:raise ValueError('ROUND_FORWARD_EXIT')
                time.sleep(min(.01,remaining(remote.deadline)))
            info=self.local.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_mode&0o777!=0o600 or info.st_uid!=os.getuid():raise ValueError('ROUND_FORWARD_MODE')
            self.inode=info.st_ino
            self.socket=socket.socket(socket.AF_UNIX);self.socket.settimeout(remaining(remote.deadline));self.socket.connect(str(local))
        except Exception:self.close();raise

    def request(self,value):
        raw=json.dumps(value,allow_nan=False).encode()+b'\n'
        if len(raw)>4096:raise ValueError('ROUND_OWNER_SIZE')
        self.socket.settimeout(remaining(self.remote.deadline));self.socket.sendall(raw);result=b''
        while not result.endswith(b'\n'):
            self.socket.settimeout(remaining(self.remote.deadline));block=self.socket.recv(4097-len(result))
            if not block:raise ValueError('ROUND_OWNER_EOF')
            result+=block
            if len(result)>4096:raise ValueError('ROUND_OWNER_SIZE')
        if result.count(b'\n')!=1:raise ValueError('ROUND_OWNER_FRAME')
        from drift.serving.worker_owner_socket import _object,_constant
        return json.loads(result,object_pairs_hook=_object,parse_constant=_constant)

    def close(self):
        if self.socket:self.socket.close();self.socket=None
        if self.process.poll() is None:self.process.terminate()
        try:self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=1)
        if self.local.exists() and self.local.lstat().st_ino==getattr(self,'inode',None):self.local.unlink()
        return {'reaped':self.process.poll() is not None,'exit_code':self.process.returncode}
