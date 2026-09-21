"""Prepare without dispatch, then run one exclusive bounded native linked echo round."""
import argparse
import json
from pathlib import Path
import time
import sys
from drift.serving.bridge_recipe import load_bridge_recipe
from linked_round_actor import launch
from linked_round_evidence import check_bank,check_facts,read_cleanup,require
from linked_round_exchange import Exchange
from linked_round_process import rendezvous,read_json
from linked_round_profile import load,prepare,verify
from native_room_profile import digest


def run(path,expected,evidence,*,dispatch=False,prepared_sha256=None):
    profile,entries=load(path,expected);evidence=Path(evidence).resolve();ready=evidence/'prepared.json'
    if not dispatch:
        prepared=prepare(profile,evidence);prepared['profile_sha256']=expected
        ready.write_text(json.dumps(prepared,sort_keys=True,indent=2))
        return {'verdict':'PREPARED','evidence':str(evidence),'prepared_sha256':digest(ready),'profile_sha256':expected}
    require(prepared_sha256 is not None and digest(ready)==prepared_sha256)
    prepared,_=read_json(ready,1048576);require(prepared['profile_sha256']==expected);verify(prepared)
    require(Path(__file__).resolve()==Path(prepared['code'])/'scripts/omp/linked_round.py')
    require(Path(sys.modules['drift.serving.bridge_recipe'].__file__).resolve()==Path(prepared['code'])/'drift/serving/bridge_recipe.py')
    (evidence/'run.started').open('x').close()
    recipes={};preload=time.monotonic();scratch=evidence/'translator-scratch';scratch.mkdir(mode=0o700)
    for direction,spec in profile['recipes'].items():
        recipes[direction]=load_bridge_recipe(direction,spec['manifest']['path'],spec['manifest']['sha256'],artifact_root=spec['artifact_root'],scratch=scratch,max_bytes=spec['max_bytes'])
    preload=time.monotonic()-preload
    for item in profile['cleanup']:read_cleanup(item,fresh=True)
    started=time.monotonic();started_ms=int(time.time()*1000);actors={};processes={};cleanup=[];errors=[];facts={};forwarders={};bank={}
    transfer=evidence/'transfer';transfer.mkdir(mode=0o700)
    exchange=Exchange(profile,entries,recipes,transfer,started+50)
    try:
        for name,entry in entries.items():actors[name]=(launch(profile,entry,prepared,evidence/name,started_ms),evidence/name)
        processes=rendezvous(actors,exchange,started+50)
        require(all(v['exit_code']==0 and v['reaped'] and v['joined'] and not v['terminated'] and not v['output_limit'] for v in processes.values()))
        for name in entries:facts[name]=check_facts(read_json(evidence/name/'facts.json')[0])
        remote=exchange.remote['glm'];remote.deadline=started+60
        remote.download(profile['endpoints']['glm']['bank_receipt'],transfer/'bank.json',1048576)
        raw,_=read_json(transfer/'bank.json',1048576)
        bank=check_bank(raw,profile['endpoints']['glm']['initial_sha256'],exchange.receipt['reverse']['sha256'])
        require(raw['restoration']['turns'][0]['session']==exchange.receipt['forward']['source_session'])
    except Exception as error:errors.append('ROUND_TIMEOUT' if isinstance(error,TimeoutError) else 'ROUND_FAILED')
    finally:
        forwarders=exchange.close()
        for name,(actor,root) in actors.items():
            try:processes[name]=actor.close(started+60)
            except Exception:errors.append('ACTOR_CLEANUP_UNVERIFIED')
        for item in profile['cleanup']:
            try:cleanup.append(read_cleanup(item,fresh=False))
            except Exception:errors.append('CLEANUP_UNVERIFIED')
        socket_root=Path(profile['socket_root'])
        if socket_root.exists() and not list(socket_root.iterdir()):socket_root.rmdir()
    passed=not errors and len(cleanup)==2 and all(v['passed'] for v in cleanup) and len(facts)==2 and bool(bank) and all(v['reaped'] for v in forwarders.values())
    report=dict(verdict='PASSED' if passed else 'FAILED',qualification='one-subset-linked-native-echo-round',profile_sha256=expected,prepared_sha256=prepared_sha256,preload_seconds=preload,seconds=time.monotonic()-started,actors=processes,facts=facts,exchange=exchange.receipt,bank=bank,cleanup=cleanup,forwarders=forwarders,errors=errors,full_knowledge=False,final_tail='UNKNOWN',new_own_inputs='NOT_EXPORTED',semantic_use='NOT_EVALUATED',duo_project='NOT_RUN')
    (evidence/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--profile-sha256',required=True);parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--run',action='store_true');parser.add_argument('--prepared-sha256')
    args=parser.parse_args();result=run(args.profile,args.profile_sha256,args.evidence,dispatch=args.run,prepared_sha256=args.prepared_sha256)
    print(json.dumps(result,sort_keys=True));raise SystemExit(0 if result['verdict'] in ('PREPARED','PASSED') else 2)
