"""Pin the two staged native actors and prepare a dispatch-free private bundle."""
import ast
import json
import os
from pathlib import Path
import shutil
from native_room_profile import command,digest,pinned
from linked_round_evidence import require

LIMITS=dict(max_input_bytes=262144,max_output_tokens=64,max_session_tokens=8192,max_turns=2,deadline_ms=50000)


def load(path,expected):
    profile=json.loads(pinned({'path':str(path),'sha256':expected},65536).read_bytes())
    require(set(profile)=={'version','purpose','owner','omp','cleanup','endpoints','endpoint_pins','recipes','socket_root'} and profile['version']==1 and profile['purpose']=='bounded-linked-echo-round')
    owner=json.loads(pinned(profile['owner'],262144).read_bytes());pinned(profile['omp'])
    require(owner.get('version')==1 and owner.get('experimental') is True and len(owner['workers'])==2)
    entries={}
    for entry in owner['workers']:
        require(entry['limits']==LIMITS and all(type(v) is int for v in entry['limits'].values()))
        require(entry['memory_mode']=='linked' and entry.get('experimental_multi_turn',False) is False and entry['expected']['backend']=='live')
        name='qwen' if entry['expected']['nativeStates']==['retained'] else 'glm' if entry['expected']['nativeStates']==['reconstructed'] else None
        require(name is not None and name not in entries);entries[name]=entry;command(entry['command'])
        require(entry['artifacts'])
        for item in entry['artifacts']:pinned(item)
    require(set(entries)==set(profile['endpoints'])=={'glm','qwen'})
    for name,entry in entries.items():
        endpoint=profile['endpoints'][name];identity=entry['identity']
        require(endpoint['session']==identity['session'] and endpoint['worker']==identity['worker'])
        require(all(type(endpoint[k]) is str and endpoint[k].startswith('/') and '..' not in Path(endpoint[k]).parts for k in ('path','memory_root')))
    for item in profile['endpoint_pins']:pinned(item)
    require(len(profile['cleanup'])==2)
    for item in profile['cleanup']:
        command(item['command']);require(item['command'].get('artifacts'))
        require(any(item['worker']==e['identity']['worker'] and item['session']==e['identity']['session'] for e in entries.values()))
    require({item['worker'] for item in profile['cleanup']}=={e['identity']['worker'] for e in entries.values()})
    require(set(profile['recipes'])=={'forward','reverse'})
    for recipe in profile['recipes'].values():pinned(recipe['manifest'],1048576);require(type(recipe['max_bytes']) is int and 0<recipe['max_bytes']<=700000000)
    socket_root=Path(profile['socket_root']);require(socket_root.is_absolute() and len(str(socket_root))<75)
    return profile,entries


def prepare(profile,evidence):
    evidence=Path(evidence).resolve();require(not any((p/'.git').exists() for p in (evidence,*evidence.parents)))
    evidence.mkdir(mode=0o700);code=evidence/'code';code.mkdir(mode=0o700)
    here=Path(__file__).resolve().parent;repo=here.parents[1]
    scripts={p.relative_to(repo) for p in here.glob('linked_round*.py')}
    scripts|={Path('scripts/omp')/name for name in ('native_room_profile.py','native_room_evidence.py','native_echo.mjs','paused_echo.mjs','paused_echo_control.mjs','paused_echo_files.mjs','experimental_extension.mjs')}
    from provider_boundary_setup import FILES as BOUNDARY_FILES
    scripts|={Path('scripts/omp')/name for name in BOUNDARY_FILES}
    scripts|={p.relative_to(repo) for p in (repo/'plugin/omp-drift/src').glob('worker_*.ts')}
    scripts|={Path('drift/__init__.py'),Path('drift/serving/__init__.py'),Path('drift/translate/__init__.py'),Path('drift/serving/translation_bridge.py'),Path('drift/serving/worker_owner_socket.py')}
    queue=list(scripts)
    while queue:
        relative=queue.pop();target=code/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/relative,target)
        if target.suffix!='.py':continue
        for node in ast.walk(ast.parse(target.read_bytes())):
            names=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [v.name for v in node.names] if isinstance(node,ast.Import) else []
            for name in names:
                candidate=Path(name.replace('.','/')+'.py')
                if name.startswith('drift.') and candidate not in scripts:scripts.add(candidate);queue.append(candidate)
    for name in ('glm','qwen'):
        root=evidence/name;root.mkdir(mode=0o700);(root/'task').mkdir(mode=0o700);(root/'task/agent').mkdir()
    socket_root=Path(profile['socket_root']);socket_root.mkdir(mode=0o700)
    return {'version':1,'code':str(code),'pins':{str(p.relative_to(code)):digest(p) for p in code.rglob('*') if p.is_file()},'socket_root':str(socket_root)}


def verify(prepared):
    root=Path(prepared['code'])
    require(all(digest(root/path)==sha for path,sha in prepared['pins'].items()))
    socket_root=Path(prepared['socket_root']);require(socket_root.resolve()==socket_root and socket_root.stat().st_mode&0o777==0o700 and not list(socket_root.iterdir()))
