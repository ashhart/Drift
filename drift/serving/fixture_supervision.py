"""Invoke the shared on-host supervisor through parent-owned control and evidence pipes."""
import json
import os
import subprocess
import sys
import time


def run_fixture_child(command, *, deadline, work_seconds=280):
    if not 0 < work_seconds <= 280 or deadline-time.monotonic() <= work_seconds:
        raise ValueError('fixture supervision budget unavailable')
    control_read,control_write = os.pipe()
    evidence_read,evidence_write = os.pipe()
    process = None
    try:
        args = [sys.executable,'-m','drift.serving.worker_supervisor','--control-fd',str(control_read),
                '--evidence-fd',str(evidence_write),'--wall-seconds',str(work_seconds),'--',*command]
        process = subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                   pass_fds=(control_read,evidence_write))
        os.close(control_read); control_read=None
        os.close(evidence_write); evidence_write=None
        try:
            process.wait(timeout=min(work_seconds+2,max(0,deadline-time.monotonic()-5)))
        except subprocess.TimeoutExpired:
            os.close(control_write); control_write=None
            process.wait(timeout=max(0,deadline-time.monotonic()))
        os.set_blocking(evidence_read,False)
        raw = os.read(evidence_read,16385)
        if len(raw)>16384: raise ValueError('oversized fixture supervision evidence')
        result = json.loads(raw)
        keys = {'event','reason','child_pid','returncode','child_reaped','process_group_alive',
                'term_sent','kill_sent','wall_seconds','input_bytes','output_bytes'}
        if not isinstance(result,dict) or set(result)!=keys:
            raise ValueError('invalid fixture supervision evidence')
        result['supervisor_returncode']=process.returncode
        result['supervisor_pid']=process.pid
        if time.monotonic()>=deadline: raise TimeoutError('fixture supervision exceeded deadline')
        return result
    finally:
        for fd in (control_read,control_write,evidence_read,evidence_write):
            if fd is not None: os.close(fd)
