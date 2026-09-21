"""Start only the pinned fixed echo actor with a fresh private pause envelope."""
import json
import os
from pathlib import Path
import time
from linked_round_process import Actor
from native_room_profile import digest

PROMPT='Call drift_native_echo exactly once, then answer OK; do not call any other tools.'


def launch(profile,entry,prepared,root,started_ms):
    task=root/'task';code=Path(prepared['code'])/'scripts/omp';model='drift-experimental/'+entry['identity']['model_id']
    pause=dict(v=1,root=str(root),task_root=str(task),ready='ready.json',release='release.json',started_at_ms=started_ms,deadline_ms=started_ms+50000,model=model)
    config=root/'pause.json'
    with config.open('x') as stream:json.dump(pause,stream)
    config.chmod(0o600)
    env={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','PI_CODING_AGENT_DIR':str(task/'agent'),'PI_CONFIG_DIR':os.path.relpath(task,Path.home()),'PI_NO_PTY':'1','TERM':'dumb','NO_COLOR':'1','TMPDIR':str(task),'DRIFT_WORKER_CONFIG':profile['owner']['path'],'DRIFT_WORKER_CONFIG_SHA256':profile['owner']['sha256'],'DRIFT_NATIVE_REPORT':str(root/'facts.json'),'DRIFT_PAUSED_ECHO_CONFIG':str(config),'DRIFT_PAUSED_ECHO_SHA256':digest(config)}
    command=[profile['omp']['path'],'--cwd',str(task),'--no-session','--no-tools','--no-lsp','--no-pty','--no-extensions','--no-skills','--no-rules','--no-title','--no-prewalk','--extension',str(code/'experimental_extension.mjs'),'--extension',str(code/'paused_echo.mjs'),'--model',model,'--thinking','off','--print','--mode','json','--max-time','50',PROMPT]
    (root/'launch.json').write_text(json.dumps({'command':command,'env':env},sort_keys=True,indent=2))
    return Actor(command,task,env)
