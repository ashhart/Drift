"""Require native tool completion, captured versions and every rank receipt."""
from native_room_evidence import read_cleanup


def require(value):
    if not value:raise ValueError('ROUND_EVIDENCE')


def check_bank(value,initial,updated):
    require(value.get('status')=='PASSED' and value.get('control_failed') is False and value.get('owner_thread_joined') is True)
    restoration=value['restoration'];turns=restoration['turns']
    require(restoration['completed_turns']==2 and len(turns)==2 and len({t['session'] for t in turns})==2)
    for version,(turn,digest) in enumerate(zip(turns,(initial,updated)),1):
        snapshot=turn['snapshot']
        require(turn['linked'] is True and snapshot['version']==version and snapshot['rows']==12 and snapshot['sha256']==digest==turn['snapshot_sha256'])
        require(turn['receipts']==[dict(rank=rank,world_size=2,sequence=0,rows=12,sha256=digest) for rank in (0,1)])
    return {'versions':[1,2],'rank_receipts':4,'completed_turns':2,'close_failed_flag':restoration.get('failed'),'first_token_causality':'NOT_ESTABLISHED'}


def check_facts(value):
    require(value.get('tool_calls')==1 and value.get('other_tools')==0 and value.get('assistant_turns')==2 and value.get('diagnostics')==[])
    require(all(type(value.get(k)) is int and value[k]>=0 for k in ('input_tokens','output_tokens')))
    require(0<value['output_tokens']<=128 and 0<value['input_tokens']+value['output_tokens']<=8192)
    return value
