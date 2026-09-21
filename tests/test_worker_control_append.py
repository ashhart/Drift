from copy import deepcopy
import pytest
from drift.serving.glm_session import GlmSession
from drift.serving.worker_studio_backend import StudioBackend
from drift.serving.worker_studio_control import QWEN_CONTROL
from test_worker_session import Backend, LIMITS, PINS, command
from test_worker_own_control import setup
from test_glm_session import Transport, turn


class AdditiveBackend(Backend):
    def own_control_append(self, payload): self.calls.append(deepcopy(payload))


def test_additive_protocol_preserves_opt_in_bounds_and_counts():
    rejected=setup(AdditiveBackend(),False)
    assert list(rejected.handle(command(1,'own_control_append',{'system_prompt':['fresh']})))[0]['payload']['code']=='CAPABILITY'
    backend=AdditiveBackend();session=setup(backend,True)
    for index in range(LIMITS['max_turns']):
        result=list(session.handle(command(index+1,'own_control_append',{'system_prompt':['fresh']})))
        assert result[0]['op']=='own_control_append_ack'
    assert session.tokens==0 and session.turns==0 and len(backend.calls)==LIMITS['max_turns']+1
    assert list(session.handle(command(4,'own_control_append',{'system_prompt':['too many']})))[0]['payload']['code']=='LIMIT'


@pytest.mark.parametrize('payload',[{'system_prompt':[]},{'system_prompt':'bad'},{'system_prompt':['x'],'replace':True}])
def test_additive_rejects_malformed_or_empty_updates(payload):
    session=setup(AdditiveBackend(),True)
    assert list(session.handle(command(1,'own_control_append',payload)))[0]['payload']['code']=='PROTOCOL'


def test_additive_cannot_change_prefilled_turn():
    session=setup(AdditiveBackend(),True)
    list(session.handle(command(1,'own_prompt',{'text':'own'})))
    assert list(session.handle(command(2,'own_control_append',{'system_prompt':['late']})))[0]['payload']['code']=='PROTOCOL'


def test_glm_additive_updates_preserve_history_order_and_usage():
    transport=Transport([turn({'content':'done'}),turn({'content':'finished'},input_tokens=55)])
    backend=GlmSession(transport,model='fixture')
    backend.open({'limits':{**LIMITS,'max_session_tokens':200},'system_prompt':['large static base'],'tools':[]})
    backend.own_prompt({'text':'first'});list(backend.stream(8));past=deepcopy(backend.messages)
    backend.own_control_append({'system_prompt':['new developer']})
    backend.own_control_append({'system_prompt':['new receipt']})
    assert backend.messages==past
    backend.own_prompt({'text':'next'});events=list(backend.stream(8))
    assert backend.messages[:len(past)]==past
    assert [m['role'] for m in transport.bodies[-1]['messages'][len(past):]]==['system','system','user']
    content='\n'.join(m['content'] for m in transport.bodies[-1]['messages'][len(past):-1])
    assert content.count('new developer')==content.count('new receipt')==1
    assert 'large static base' not in content and 'supersedes' not in content
    assert events[-1]['payload']['usage']['input_tokens']==55 and backend.pending_control==[]


@pytest.mark.parametrize('kind',['glm','qwen'])
@pytest.mark.parametrize('first_snapshot',[False,True])
def test_snapshot_rejects_unflushed_updates_before_losing_them(kind,first_snapshot):
    backend=(GlmSession(Transport([]),model='fixture') if kind=='glm' else
             StudioBackend(lambda value:None,{'cache':object()},object(),lambda *args:('',[])))
    backend.open({'limits':LIMITS,'system_prompt':['initial'],'tools':[]})
    if first_snapshot:backend.own_control({'system_prompt':['first snapshot']})
    backend.own_control_append({'system_prompt':['old addition']})
    pending=deepcopy(backend.pending_control)
    with pytest.raises(RuntimeError,match='PROTOCOL'):
        backend.own_control({'system_prompt':['replacement']})
    assert backend.pending_control==pending


def test_qwen_additive_waits_for_tools_preserves_cache_prefix_and_user_authority():
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):return [1,2,7,9,4,5]
    cache=object();state={'cache':cache};operations=[]
    backend=StudioBackend(operations.append,state,Tokenizer(),lambda *args:('',[]),control_format=QWEN_CONTROL)
    backend.open({**PINS,'limits':LIMITS,'system_prompt':['static'],'tools':[]})
    backend.codec.messages=[{'role':'assistant','content':'two calls'}];backend.codec.consumed=[1,2,7,9]
    backend.awaiting_tools=2;backend.own_control_append({'system_prompt':['fresh peer context']})
    backend.tool_result({'call_id':'a','text':'first','is_error':False})
    assert len(backend.codec.messages)==1
    backend.tool_result({'call_id':'b','text':'second','is_error':False})
    assert [m['role'] for m in backend.codec.messages]==['assistant','tool','tool','user']
    assert 'does not replace or outrank' in backend.codec.messages[-1]['content']
    assert 'supersedes' not in backend.codec.messages[-1]['content']
    assert state['cache'] is cache and operations[-1]=={'op':'continue','ids':[4,5]}
    assert backend.input_tokens==2 and backend.pending_control==[]
