"""Perform exactly one declared last-row and generated-prefix exchange."""
import json
from pathlib import Path
from drift.serving.translation_bridge import translate_tap,translate_prefix
from linked_round_remote import Remote,Forward,LOCATE
from linked_round_process import read_json,remaining


def require(value):
    if not value:raise ValueError('ROUND_EXCHANGE')


class Exchange:
    def __init__(self,profile,entries,recipes,root,deadline):
        self.profile,self.recipes,self.root,self.deadline=profile,recipes,root,deadline
        self.remote={name:Remote(entry['command'],deadline) for name,entry in entries.items()}
        self.forwards={};self.receipt={}

    def __call__(self):
        p=self.profile;g,q=p['endpoints']['glm'],p['endpoints']['qwen']
        for name,endpoint in (('glm',g),('qwen',q)):
            self.forwards[name]=Forward(self.remote[name],endpoint['path'],Path(p['socket_root'])/(name+'.sock'))
        tap=self.forwards['qwen'].request(dict(op='tap',path='round-own.npz',first=0,max_rows=4096))
        own=self.root/'qwen';own.mkdir(mode=0o700)
        self.remote['qwen'].download(q['memory_root']+'/round-own.npz',own/'tap.npz',134217728,tap['sha256'])
        reverse=translate_tap(self.recipes['reverse'],own/'tap.npz',tap,session=q['session'],source_worker=q['worker'],target_worker=g['worker'],seq=1,first=0,output=self.root/'reverse.npz',max_source_rows=4096,max_source_bytes=134217728,max_output_rows=12,max_output_bytes=1048576)
        self.remote['glm'].upload(reverse['path'],g['memory_root']+'/round-v2.npz',1048576,reverse['sha256'],self.root/'upload-reverse.empty')
        frame=dict(v=1,session=g['session'],version=2,path='round-v2.npz',sha256=reverse['sha256'],rows=12,source_worker=q['worker'],target_worker=g['worker'],recipe_sha256=self.recipes['reverse'].sha256)
        published=self.forwards['glm'].request({'op':'publish','snapshot':frame})
        require(published=={'op':'published',**{key:value for key,value in frame.items() if key not in ('v','path')},'bytes':reverse['bytes']})
        remote=self.remote['glm'];remote.call(LOCATE,[g['outbox_root']],self.root/'outbox-location.json',4096)
        location,_=read_json(self.root/'outbox-location.json',4096)
        require(set(location)=={'directory','sha256'} and location['directory'].startswith('restore-') and '/' not in location['directory'])
        source=self.root/'glm';source.mkdir(mode=0o700)
        manifest_remote=g['outbox_root']+'/'+location['directory']+'/manifest.json'
        remote.download(manifest_remote,source/'manifest.json',1048576,location['sha256'])
        manifest,_=read_json(source/'manifest.json',1048576)
        require(manifest['selected_rows']>0 and manifest['publications'] and manifest['session']==location['directory'])
        selected=manifest['publications'][0]
        require(selected['file']=='own-000000.npz' and type(selected['bytes']) is int and 0<selected['bytes']<=1048576)
        remote.download(g['outbox_root']+'/'+location['directory']+'/'+selected['file'],source/selected['file'],1048576,selected['sha256'])
        forward=translate_prefix(self.recipes['forward'],source/'manifest.json',location['sha256'],publication_seq=0,session=location['directory'],source_worker=g['worker'],target_worker=q['worker'],output=self.root/'forward.npz',max_source_rows=64,max_source_bytes=1048576,max_output_rows=128,max_output_bytes=4194304,remaining=128)
        self.remote['qwen'].upload(forward['path'],q['memory_root']+'/round-forward.npz',4194304,forward['sha256'],self.root/'upload-forward.empty')
        appended=self.forwards['qwen'].request(dict(op='append',path='round-forward.npz',sha256=forward['sha256'],rows=forward['rows']))
        require(appended==dict(v=1,session=q['session'],seq=2,source_worker=g['worker'],target_worker=q['worker'],op='appended',rows=forward['rows'],sha256=forward['sha256'],foreign_total=forward['rows']))
        remaining(self.deadline)
        self.receipt=dict(tap=tap,reverse={k:v for k,v in reverse.items() if k!='path'},published=published,forward={k:v for k,v in forward.items() if k!='path'},appended=appended)
        (self.root/'exchange.json').write_text(json.dumps(self.receipt,sort_keys=True,indent=2))
        return self.receipt

    def close(self):
        result={}
        for name,forward in self.forwards.items():
            try:result[name]=forward.close()
            except Exception:result[name]={'reaped':False,'error':'FORWARD_CLEANUP_FAILED'}
        return result
