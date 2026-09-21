import json
import pytest
from test_glm_session import Transport, turn
from drift.serving.glm_session import GlmSession
from drift.serving.worker_contract import WorkerError


def backend(transport, path=None):
    from drift.serving.glm_diagnostics import GlmDiagnostics
    value=GlmSession(transport,model='fixture',diagnostics=GlmDiagnostics(path))
    value.open({'limits':{'max_input_bytes':262144,'max_session_tokens':20000,'deadline_ms':50000},
                'system_prompt':['NEVER_EXPORT_SYSTEM'],'tools':[]})
    value.own_prompt({'text':'NEVER_EXPORT_PROMPT'})
    return value


def private_path(tmp_path):
    root=tmp_path/'private';root.mkdir(mode=0o700)
    return root/'diagnostic.json'


def test_second_count_over_cumulative_budget_is_limit_before_second_completion(tmp_path):
    path=private_path(tmp_path)
    transport=Transport([turn({'content':'NEVER_EXPORT_ANSWER'},input_tokens=9212,output_tokens=131),
                         turn({'content':'MUST_NOT_DISPATCH'},input_tokens=10658,output_tokens=1)])
    session=backend(transport,path)
    list(session.stream(512));session.own_prompt({'text':'NEVER_EXPORT_TOOL_RESULT'})
    with pytest.raises(WorkerError,match='^LIMIT$'):list(session.stream(512))
    assert len(transport.bodies)==1 and session.used==9343
    value=json.loads(path.read_text())
    assert value['phase']=='glm_budget'
    assert value['counts']=={'counted_tokens':10658,'admitted_tokens':0,'used_tokens':9343,'budget_tokens':20000,'request_dispatched':0}
    assert 'NEVER_EXPORT' not in path.read_text() and 'MUST_NOT_DISPATCH' not in path.read_text()


@pytest.mark.parametrize('failure',['count','admit','parse'])
def test_failure_phase_and_sanitized_class_preserve_no_payload(tmp_path,failure):
    from drift.serving.glm_factory import AdmittedHttp
    from drift.serving.glm_diagnostics import GlmDiagnostics
    path=private_path(tmp_path);observer=GlmDiagnostics(path)
    if failure=='admit':
        class Guard:
            def admit(self,timeout):raise ValueError('NEVER_EXPORT_ADMIT')
        transport=AdmittedHttp('http://127.0.0.1:8888',Guard(),diagnostics=observer)
        transport.count_tokens=lambda *args:20
    else:
        class Fake(Transport):
            def count_tokens(self,*args):
                if failure=='count':raise ValueError('NEVER_EXPORT_COUNT')
                return 20
        transport=Fake([turn({'tool_calls':[{'index':0,'id':'NEVER_EXPORT_ID','function':{'name':'NEVER_EXPORT_UNKNOWN','arguments':'{}'}}]},reason='tool_calls')])
    session=GlmSession(transport,model='fixture',diagnostics=observer)
    session.open({'limits':{'max_input_bytes':4096,'max_session_tokens':20000,'deadline_ms':50000},'system_prompt':['NEVER_EXPORT'],'tools':[]})
    session.own_prompt({'text':'NEVER_EXPORT'})
    with pytest.raises(ValueError):list(session.stream(32))
    value=json.loads(path.read_text())
    assert value['phase']=='glm_'+failure
    assert value['errors'][0]['exception']=='ValueError'
    assert 'NEVER_EXPORT' not in path.read_text()


def test_protocol_exports_limit_only_without_numeric_private_diagnostics(tmp_path):
    from drift.serving.worker_session import WorkerSession
    from drift.serving.glm_diagnostics import GlmDiagnostics
    path=private_path(tmp_path)
    transport=Transport([turn({'content':'unused'},input_tokens=20000)])
    model=GlmSession(transport,model='fixture',diagnostics=GlmDiagnostics(path))
    limits=dict(max_input_bytes=262144,max_output_tokens=64,max_session_tokens=20000,max_turns=6,deadline_ms=50000)
    pins=dict(model_id='fixture',model_sha256='a'*64,translator_sha256='b'*64)
    worker=WorkerSession('glm',pins,limits,model)
    def frame(seq,op,payload):return dict(v=1,session='owned',worker='glm',seq=seq,op=op,payload=payload)
    assert list(worker.handle(frame(1,'open',{**pins,'limits':limits,'system_prompt':['NEVER_EXPORT'],'tools':[]})))[0]['op']=='opened'
    assert list(worker.handle(frame(2,'own_prompt',{'text':'NEVER_EXPORT'})))[0]['op']=='own_prompt_ack'
    responses=list(worker.handle(frame(3,'stream',{'max_tokens':64})))
    assert len(responses)==1 and responses[0]['op']=='error' and responses[0]['payload']=={'code':'LIMIT'}
    assert '20000' not in json.dumps(responses) and 'NEVER_EXPORT' not in json.dumps(responses)
    assert json.loads(path.read_text())['counts']['request_dispatched']==0


def test_factory_opts_in_shared_private_observer_without_model_dispatch(tmp_path,monkeypatch):
    import drift.serving.glm_factory as module
    path=private_path(tmp_path)
    monkeypatch.setattr(module,'verify_worker_manifests',lambda config:None)
    monkeypatch.delenv('DRIFT_GLM_KEY',raising=False)
    session=module.factory(dict(memory_mode='no_link',service_auth='none',pins={'model_id':'fixture'},diagnostics_path=str(path)))
    assert session.diagnostics is session.transport.diagnostics
    assert session.transport.child is None
    session.close()
    assert path.read_bytes()==b'' and path.stat().st_mode & 0o777==0o600


def test_malformed_tool_preserves_usage_and_numeric_parse_evidence(tmp_path):
    from drift.serving.glm_diagnostics import GlmDiagnostics
    path=private_path(tmp_path)
    arguments='{"private":"NEVER_EXPORT'
    transport=Transport([turn({'tool_calls':[{'index':0,'id':'private-id','function':{
        'name':'read','arguments':arguments}}]},reason='tool_calls',input_tokens=9585,output_tokens=512)])
    session=GlmSession(transport,model='fixture',diagnostics=GlmDiagnostics(path))
    session.open({'limits':{'max_input_bytes':262144,'max_session_tokens':65536,'deadline_ms':50000},
                  'system_prompt':['NEVER_EXPORT'],'tools':[{'name':'read'}]})
    session.own_prompt({'text':'NEVER_EXPORT'})
    with pytest.raises(json.JSONDecodeError):list(session.stream(512))
    counts=json.loads(path.read_text())['counts']
    assert counts['input_tokens']==9585 and counts['generated_tokens']==512
    assert counts['requested_output_tokens']==512 and counts['tool_calls']==1
    assert counts['argument_bytes']==len(arguments.encode()) and counts['finish_reason_code']==2
    assert counts['json_error_position']==11 and counts['json_document_chars']==len(arguments)
    assert session.poisoned and transport.closed and session.pending==set()
    assert 'NEVER_EXPORT' not in path.read_text() and 'private-id' not in path.read_text()
