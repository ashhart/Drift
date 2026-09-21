"""Validate the aggregate-only fixture receipt before it leaves private staging."""
import math
import hashlib
import re


def validate_receipt(value):
    from drift.serving.qwen_fixture import PUBLIC_NOTE, REVERSE_SHA256
    hashes=('source_text_sha256','source_tap_sha256','translator_sha256','fixture_sha256','ids_file_sha256')
    numbers=('tap_process_seconds','translation_seconds','child_seconds')
    exact={'generated_tokens':0,'source_layers':12,'fixture_layers':11,'fixture_shape_per_array':[12,512],
           'copies':12,'gain_power':1.0,'model_requests':1,'warmup_requests':0}
    keys=set(hashes)|set(numbers)|set(exact)|{'source_tokens','source_row','source_shape_per_array','source_fingerprint'}
    if not isinstance(value,dict) or set(value)!=keys: raise ValueError('fixture receipt fields differ')
    if any(not isinstance(value[key],str) or not re.fullmatch('[0-9a-f]{64}',value[key]) for key in hashes):
        raise ValueError('fixture receipt hash is invalid')
    if value['translator_sha256']!=REVERSE_SHA256 or value['source_text_sha256']!=hashlib.sha256(PUBLIC_NOTE.encode()).hexdigest():
        raise ValueError('fixture source or translator differs')
    for key,expected in exact.items():
        if type(value[key]) is not type(expected) or value[key]!=expected: raise ValueError('fixture receipt scope differs')
    tokens=value['source_tokens']
    if type(tokens) is not int or not 1<=tokens<=32 or type(value['source_row']) is not int or value['source_row']!=tokens-1:
        raise ValueError('fixture source cursor differs')
    if value['source_shape_per_array']!=[tokens,2,256]: raise ValueError('fixture source shape differs')
    if any(type(value[key]) not in (int,float) or not math.isfinite(value[key]) or not 0<=value[key]<=300 for key in numbers):
        raise ValueError('fixture timings differ')
    fingerprint=value['source_fingerprint']
    expected={'config_sha256','tokenizer_sha256','tap_script_sha256','weights_index_sha256','full_weight_identity'}
    if not isinstance(fingerprint,dict) or set(fingerprint)!=expected or fingerprint['full_weight_identity']!='NOT_MEASURED':
        raise ValueError('fixture fingerprint differs')
    for key in expected-{'full_weight_identity'}:
        entry=fingerprint[key]
        if key=='weights_index_sha256' and entry is None: continue
        if not isinstance(entry,str) or not re.fullmatch('[0-9a-f]{64}',entry): raise ValueError('invalid fingerprint hash')
    return value
