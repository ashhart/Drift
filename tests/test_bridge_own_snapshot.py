from copy import deepcopy
import hashlib
import numpy as np
import pytest
from drift.serving.bridge_files import QWEN_LAYOUT
from drift.serving.bridge_recipe import PinnedRecipe
from test_translation_bridge import Reverse, folders


def selected(source):
    path=source/'selected.npz';np.savez(path,**{key:np.full((1,2,256),3,np.float16) for key in QWEN_LAYOUT})
    return dict(v=1,session='qwen',seq=4,op='own_snapshot',source_worker='qwen',target_worker='glm',selection='last_own_row',scope='own',rows=1,after=5000,source_start=7000,source_stop=7001,available_own_rows=7001,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def translate(source,output,receipt):
    from drift.serving.translation_bridge import translate_own_snapshot
    return translate_own_snapshot(PinnedRecipe('reverse',Reverse(),'a'*64),source/'selected.npz',receipt,
        session='qwen',source_worker='qwen',target_worker='glm',seq=4,after=5000,
        output=output/'reverse.npz',max_source_bytes=1048576,max_output_rows=12,max_output_bytes=1048576)


def test_selected_bridge_preserves_absolute_interval_and_only_translates_one_row(tmp_path):
    source,output=folders(tmp_path);receipt=selected(source);result=translate(source,output,receipt)
    assert (result['source_start'],result['source_stop'],result['source_rows'],result['rows'])==(7000,7001,1,12)
    assert result['available_source_rows']==7001 and result['transmitted_source_rows']==1
    assert result['source_after']==5000 and result['source_selection']=='last_own_row'
    assert result['scope']=='own' and result['full_completion'] is False and result['copies']==12


@pytest.mark.parametrize('change',[{'scope':'foreign'},{'source_worker':'glm'},{'target_worker':'qwen'},
    {'session':'stale'},{'seq':3},{'after':7001},{'source_start':6999},{'source_stop':7000},
    {'rows':True},{'rows':2},{'available_own_rows':1000001},{'sha256':'0'*64},{'text':'forbidden'},
    {'selection':'all_own_rows'},{'after':True}])
def test_selected_bridge_rejects_forged_stale_replayed_or_oversized_receipts(tmp_path,change):
    source,output=folders(tmp_path);receipt=selected(source);receipt.update(change)
    with pytest.raises((ValueError,RuntimeError)):translate(source,output,receipt)
    assert not (output/'reverse.npz').exists()


@pytest.mark.parametrize('fault',['extra_row','foreign_layer','nonfinite'])
def test_selected_bridge_validates_single_row_and_all_layers(tmp_path,fault):
    source,output=folders(tmp_path);receipt=selected(source)
    arrays={key:np.full((2 if fault=='extra_row' else 1,2,256),3,np.float16) for key in QWEN_LAYOUT}
    if fault=='foreign_layer':arrays['k0']=arrays.pop('k3')
    if fault=='nonfinite':arrays['k3'][0,0,0]=np.nan
    np.savez(source/'selected.npz',**arrays);receipt['sha256']=hashlib.sha256((source/'selected.npz').read_bytes()).hexdigest()
    with pytest.raises((ValueError,RuntimeError)):translate(source,output,receipt)
    assert not (output/'reverse.npz').exists()
