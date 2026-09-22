"""A pinned full-own policy preserves earlier rows instead of silently cropping them."""
import numpy as np
import pytest

from drift.serving.bridge_files import GLM_LAYOUT
from drift.serving.bridge_recipe import PinnedRecipe
from drift.serving.translation_bridge import translate_tap
from tests.test_translation_bridge import folders, tap_input


class AllRows:
    def read(self, arrays, gain):
        assert gain == 1 and all(value.shape == (3,1024) for value in arrays.values())
        assert all(np.array_equal(value[:,0],[1,2,3]) for value in arrays.values())
        return {int(key[1:]):np.repeat(np.arange(1,4,dtype=np.float32)[:,None],512,axis=1)
                for key in GLM_LAYOUT}


@pytest.mark.parametrize('maximum', [2,3])
def test_full_own_policy_preserves_cursor_and_enforces_output_bound(tmp_path, maximum):
    source, output = folders(tmp_path); receipt = tap_input(source)
    recipe = PinnedRecipe('reverse',AllRows(),'a'*64,reverse_source_policy='all_new_own_rows',reverse_copies=1)
    def run():
        return translate_tap(recipe,source/'tap.npz',receipt,session='qwen-turn',
                             source_worker='qwen',target_worker='glm',seq=2,first=10,
                             output=output/'reverse.npz',max_source_rows=4,max_source_bytes=1048576,
                             max_output_rows=maximum,max_output_bytes=1048576)
    if maximum==2:
        with pytest.raises(ValueError): run()
        assert not (output/'reverse.npz').exists()
    else:
        result = run()
        assert (result['source_start'],result['source_stop'],result['source_rows'],result['rows'])==(10,13,3,3)
        assert result['copies']==1 and result['subset']=='all_new_own_rows'
