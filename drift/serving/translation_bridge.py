"""Translate one declared private source subset without changing either own-input channel."""
from pathlib import Path
import numpy as np
from drift.serving.bridge_files import GLM_LAYERS, QWEN_LAYERS, GLM_LAYOUT, QWEN_LAYOUT, integer, output_bound, pin, publish, require
from drift.serving.bridge_recipe import PinnedRecipe
from drift.serving.bridge_sources import prefix_source, tap_source
from drift.serving.live_forward import emitted_rows, forward_publication
from drift.serving.live_publication import wire_arrays
from drift.serving.worker_activation_files import bounded_path, private_root


def output_root(recipe, direction, output, max_output_rows, max_output_bytes):
    require(isinstance(recipe,PinnedRecipe) and recipe.direction==direction);pin(recipe.sha256)
    integer(max_output_rows,4096);integer(max_output_bytes,128*1048576)
    output=Path(output);root=private_root(output.parent);bounded_path(root,output.name);require(not output.exists())
    return root


def translate_tap(recipe, path, receipt, *, session, source_worker, target_worker, seq, first, output, max_source_rows, max_source_bytes, max_output_rows, max_output_bytes):
    root=output_root(recipe,'reverse',output,max_output_rows,max_output_bytes)
    require(max_output_rows>=12 and max_output_bytes<=1048576)
    output_bound(GLM_LAYOUT,12,max_output_bytes)
    arrays,source=tap_source(path,dict(receipt),session=session,source_worker=source_worker,target_worker=target_worker,seq=seq,first=first,scratch=root,max_rows=max_source_rows,max_bytes=max_source_bytes)
    return reverse_selected(recipe, arrays, source, source_worker, target_worker, output, max_output_bytes)


def translate_own_snapshot(recipe, path, receipt, *, session, source_worker, target_worker, seq, after, output, max_source_bytes, max_output_rows, max_output_bytes):
    from drift.serving.bridge_snapshot import own_snapshot_source
    root=output_root(recipe,'reverse',output,max_output_rows,max_output_bytes)
    require(max_output_rows>=12 and max_output_bytes<=1048576)
    output_bound(GLM_LAYOUT,12,max_output_bytes)
    arrays,source=own_snapshot_source(path,dict(receipt),session=session,source_worker=source_worker,target_worker=target_worker,seq=seq,after=after,scratch=root,max_bytes=max_source_bytes)
    return reverse_selected(recipe, arrays, source, source_worker, target_worker, output, max_output_bytes)


def reverse_selected(recipe, arrays, source, source_worker, target_worker, output, max_output_bytes):
    own={layer:np.concatenate((arrays[f'k{layer}'][-1:].reshape(1,-1),arrays[f'v{layer}'][-1:].reshape(1,-1)),axis=1) for layer in QWEN_LAYERS}
    translated=recipe.reader.read(own,1.0)
    one=wire_arrays({f'l{layer}':value for layer,value in translated.items()},GLM_LAYOUT,1)
    payload=wire_arrays({key:np.repeat(value,12,axis=0) for key,value in one.items()},GLM_LAYOUT,12)
    result=publish(output,payload,max_output_bytes)
    return {**result,**source,'source_worker':source_worker,'target_worker':target_worker,'recipe_sha256':recipe.sha256,'rows':12,'copies':12,'gain_power':1.0,'subset':'last_one_own_row','full_completion':False}


def translate_prefix(recipe, path, manifest_sha256, *, publication_seq, session, source_worker, target_worker, output, max_source_rows, max_source_bytes, max_output_rows, max_output_bytes, remaining):
    root=output_root(recipe,'forward',output,max_output_rows,max_output_bytes);integer(remaining,1000000,0)
    arrays,source=prefix_source(path,manifest_sha256,session=session,source_worker=source_worker,target_worker=target_worker,recipe_sha256=recipe.sha256,publication_seq=publication_seq,scratch=root,max_rows=max_source_rows,max_bytes=max_source_bytes)
    latents={layer:arrays[f'l{layer}'] for layer in GLM_LAYERS}
    output_bound(QWEN_LAYOUT,emitted_rows(recipe.reader,latents,source['source_rows']),max_output_bytes)
    payload,rows=forward_publication(recipe.reader,latents,source['source_rows'],QWEN_LAYOUT,1.5,max_rows=max_output_rows,remaining=remaining)
    result=publish(output,payload,max_output_bytes)
    return {**result,**source,'source_worker':source_worker,'target_worker':target_worker,'recipe_sha256':recipe.sha256,'rows':rows,'gain_power':1.5,'subset':'one_generated_prefix_publication','full_completion':False}
