"""Collect only verified generated prefixes from native GLM turn outboxes."""
from pathlib import Path
import re
import time
from drift.serving.glm_outbox_files import finish_marker, read_tap, require, write_manifest, write_rows
from drift.serving.worker_activation_files import private_root
from drift.serving.glm_terminal_evidence import VERIFIED, terminal_fields, verify_terminal


def bounded(value, maximum):
    require(type(value) is int and 0 < value <= maximum)
    return value


class TurnOutbox:
    def __init__(self, config, layouts, native_root, *, max_turns=2, clock=time.monotonic, pause=time.sleep):
        self.root, self.native = private_root(config['memory_root']), Path(native_root)
        self.source, self.target, self.recipe = config['source_worker'], config['target_worker'], config['recipe_sha256']
        require(all(type(v) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',v) for v in (self.source,self.target)))
        require(self.source != self.target and re.fullmatch('[0-9a-f]{64}',self.recipe))
        require(type(layouts) is dict and 0 < len(layouts) <= 256 and all(re.fullmatch(r'l[0-9]+',k) and tuple(v)==(512,) for k,v in layouts.items()))
        self.layouts={key:tuple(value) for key,value in layouts.items()}
        self.raw_rows,self.raw_bytes=bounded(config['max_raw_rows'],65536),bounded(config['max_raw_bytes'],128*1048576)
        self.rows,self.bytes=bounded(config['max_rows'],4096),bounded(config['max_bytes'],128*1048576)
        self.publications,self.turns=bounded(config['max_publications'],1024),bounded(max_turns,1024)
        self.clock,self.pause,self.current,self.used,self.failed=clock,pause,None,set(),False

    def begin(self,name,proof,maximum):
        require(not self.failed and self.current is None and name not in self.used and len(self.used)<self.turns)
        require(type(name) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',name))
        require(proof.get('verified') is True)
        require(all(type(proof.get(k)) is int and proof[k]>=0 for k in ('prompt_tokens','reserve_start','reserve_tokens')))
        prompt,first=proof['prompt_tokens'],proof['reserve_start']+proof['reserve_tokens']
        require(0 <= first < prompt and 0 < bounded(maximum,self.rows) and prompt+maximum<=self.raw_rows)
        require((prompt+maximum)*sum(shape[0]*2 for shape in self.layouts.values())+4096*len(self.layouts)<=self.raw_bytes)
        output=self.root/name;output.mkdir(mode=0o700)
        self.used.add(name);self.current={'name':name,'prompt':prompt,'first':first,'maximum':maximum,'output':output}

    def _remaining(self,deadline):
        if self.failed or self.clock()>=deadline: raise TimeoutError('native outbox deadline exceeded')
        return deadline-self.clock()

    def finish(self,name,maximum,deadline):
        try:
            self._remaining(deadline)
            state=self.current
            require(state is not None and name==state['name'] and type(maximum) is int and 0<maximum<=state['maximum'])
            folder=self.native/name
            while not (folder/'finished').exists():
                require(not any(folder.glob('error.rank*')))
                self.pause(min(.01,self._remaining(deadline)))
            marker=finish_marker(folder)
            files=sorted(folder.glob('*.npz'))
            require(len(files)<=self.publications and [p.name for p in files]==[f'{i:06d}.npz' for i in range(len(files))])
            cursor,raw_bytes,selected_bytes,selected_rows=state['first'],0,0,0
            raw,selected=[],[]
            for seq,path in enumerate(files):
                self._remaining(deadline)
                arrays,receipt=read_tap(path,self.layouts,self.raw_bytes-raw_bytes,self.raw_rows,state['output'])
                self._remaining(deadline)
                require(receipt['start']==cursor and receipt['stop']<=state['prompt']+maximum)
                cursor=receipt['stop'];raw_bytes+=receipt['bytes'];raw.append({'seq':seq,**receipt})
                start=max(receipt['start'],state['prompt'])
                if cursor<=start: continue
                rows=cursor-start;require(selected_rows+rows<=self.rows)
                output=write_rows(state['output'],len(selected),{key:value[start-receipt['start']:] for key,value in arrays.items()},self.bytes-selected_bytes)
                selected.append({**output,'source_seq':seq,'source_start':start,'source_stop':cursor,'rows':rows})
                selected_rows+=rows;selected_bytes+=output['bytes']
            self._remaining(deadline)
            terminal=terminal_fields(marker)
            if terminal is not None:
                verify_terminal(terminal,raw,state['first'],cursor,raw_bytes,selected)
            report={'session':name,'source_worker':self.source,'target_worker':self.target,'recipe_sha256':self.recipe,
                    'evidence':'EXPORTED_PREFIX' if selected_rows else 'EMPTY_EXPORTED_PREFIX','full_completion':False,
                    'final_tail':'UNKNOWN','scope':'newly_generated_only','new_own_inputs':'NOT_EXPORTED',
                    'translation':'NOT_PERFORMED','prompt_tokens':state['prompt'],'verified_stop':cursor,
                    'selected_rows':selected_rows,'selected_bytes':selected_bytes,'raw_bytes':raw_bytes,
                    'native_finished_marker':True,'writes_scheduled':marker['writes_scheduled']}
            if terminal is not None:
                report.update(final_tail=VERIFIED,native_terminal=terminal)
            artifact=write_manifest(state['output'],{**report,'raw_publications':raw,'publications':selected})
            self._remaining(deadline);self.current=None
            return {**report,**artifact}
        except TimeoutError:
            self.failed=True
            raise
        except Exception:
            self.failed=True
            raise ValueError('NATIVE_OUTBOX_FAILED') from None

    def close(self):
        self.failed=True
