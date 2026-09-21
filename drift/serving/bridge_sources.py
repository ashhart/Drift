"""Validate exact native tap receipts and incomplete generated-prefix manifests."""
import json
from pathlib import Path
import re
from drift.serving.bridge_files import GLM_LAYOUT, QWEN_LAYOUT, archive_bound, frozen, integer, pin, require
from drift.serving.live_publication import load_publication
from drift.serving.worker_activation_files import bounded_path, private_root
from drift.serving.glm_terminal_evidence import VERIFIED, verify_terminal


def actors(session, source, target):
    require(source!=target)
    require(all(type(value) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',value) for value in (session,source,target)))


def read_arrays(path, digest, layouts, scratch, rows, maximum, max_rows):
    path=Path(path);private_root(path.parent);bounded_path(path.parent,path.name)
    integer(maximum,128*1048576);integer(max_rows,4096);integer(rows,max_rows)
    with frozen(path,digest,scratch,maximum) as fixed:
        archive_bound(fixed,maximum)
        arrays=load_publication(fixed,layouts,max_rows=max_rows)
        require(all(value.shape[0]==rows for value in arrays.values()))
    return arrays


def tap_source(path, receipt, *, session, source_worker, target_worker, seq, first, scratch, max_rows, max_bytes):
    actors(session,source_worker,target_worker);integer(seq,1024);integer(first,1000000,0)
    require(type(receipt) is dict and set(receipt)=={'v','session','seq','op','source_worker','target_worker','rows','next_first','sha256'})
    expected=dict(v=1,session=session,seq=seq,op='tapped',source_worker=source_worker,target_worker=target_worker)
    require(all(type(receipt[key]) is type(value) and receipt[key]==value for key,value in expected.items()))
    rows=integer(receipt['rows'],max_rows);require(type(receipt['next_first']) is int and receipt['next_first']==first+rows)
    arrays=read_arrays(path,pin(receipt['sha256']),QWEN_LAYOUT,scratch,rows,max_bytes,max_rows)
    return arrays,dict(source_session=session,source_seq=seq,source_sha256=receipt['sha256'],source_start=first+rows-1,source_stop=first+rows,source_rows=1,available_source_rows=rows)


def prefix_source(path, digest, *, session, source_worker, target_worker, recipe_sha256, publication_seq, scratch, max_rows, max_bytes):
    actors(session,source_worker,target_worker);integer(publication_seq,1023,0)
    path=Path(path);private_root(path.parent);require(path.name=='manifest.json')
    with frozen(path,pin(digest),scratch,1048576) as fixed: report=json.loads(fixed.read_bytes())
    expected=dict(session=session,source_worker=source_worker,target_worker=target_worker,recipe_sha256=recipe_sha256,evidence='EXPORTED_PREFIX',full_completion=False,final_tail='UNKNOWN',scope='newly_generated_only',new_own_inputs='NOT_EXPORTED',translation='NOT_PERFORMED',native_finished_marker=True)
    verified=type(report) is dict and report.get('final_tail')==VERIFIED
    if verified: expected['final_tail']=VERIFIED
    require(type(report) is dict and set(report)==set(expected)|{'prompt_tokens','verified_stop','selected_rows','selected_bytes','raw_bytes','writes_scheduled','raw_publications','publications'}|({'native_terminal'} if verified else set()))
    require(all(type(report[key]) is type(value) and report[key]==value for key,value in expected.items()))
    prompt=integer(report['prompt_tokens'],65536);stop=integer(report['verified_stop'],65536)
    integer(report['raw_bytes'],128*1048576);integer(report['writes_scheduled'],1000000,0)
    require(type(report['raw_publications']) is list and len(report['raw_publications'])<=1024)
    items=report['publications'];require(type(items) is list and 0<len(items)<=1024 and publication_seq<len(items))
    total_rows=total_bytes=0;cursor=prompt;native_seq=-1
    for seq,item in enumerate(items):
        require(type(item) is dict and set(item)=={'file','sha256','bytes','source_seq','source_start','source_stop','rows'})
        require(item['file']==f'own-{seq:06d}.npz');pin(item['sha256']);integer(item['source_seq'],1023,0)
        require(item['source_seq']>native_seq);native_seq=item['source_seq']
        rows=integer(item['rows'],4096);size=integer(item['bytes'],128*1048576)
        require(type(item['source_start']) is int and item['source_start']==cursor)
        require(type(item['source_stop']) is int and item['source_stop']==cursor+rows)
        cursor+=rows;total_rows+=rows;total_bytes+=size
    require(type(report['selected_rows']) is int and report['selected_rows']==total_rows<=4096)
    require(type(report['selected_bytes']) is int and report['selected_bytes']==total_bytes<=128*1048576 and cursor==stop)
    if verified:
        terminal=report['native_terminal']
        require(type(terminal) is dict and type(terminal.get('source_start')) is int and 0<=terminal['source_start']<=prompt)
        verify_terminal(terminal,report['raw_publications'],terminal['source_start'],stop,report['raw_bytes'],items)
    item=items[publication_seq];source=bounded_path(path.parent,item['file'])
    arrays=read_arrays(source,item['sha256'],GLM_LAYOUT,scratch,item['rows'],max_bytes,max_rows)
    require(source.stat().st_size==item['bytes'])
    evidence=dict(final_tail=report['final_tail'])
    if verified: evidence['native_terminal']=terminal
    return arrays,dict(source_session=session,source_seq=publication_seq,source_native_seq=item['source_seq'],source_sha256=item['sha256'],source_manifest_sha256=digest,source_start=item['source_start'],source_stop=item['source_stop'],source_rows=item['rows'],scope='newly_generated_only',new_own_inputs='NOT_EXPORTED',**evidence)
