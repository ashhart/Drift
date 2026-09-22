"""Reuse the oMLX GLM-to-Qwen translator and cache entry points for a fresh reader."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from drift.serving.bridge_files import GLM_LAYERS, QWEN_LAYERS
from drift.serving.live_recipe import load_recipe
from drift.serving.mcdma_forward import validate_translation
from drift.serving.omlx_cache import Rope, append_entries
from drift.transcript.files import read


class QwenReader:
    def __init__(self, *, checkpoint, manifest, manifest_sha256, index, index_sha256, need_gb):
        if type(need_gb) not in (int, float) or not math.isfinite(need_gb) or need_gb <= 0:
            raise ValueError('TRANSCRIPT_MEMORY_BUDGET')
        from drift.translate.index_keys import IndexKeyReader
        if hashlib.sha256(read(manifest, 1048576)).hexdigest() != manifest_sha256:
            raise ValueError('TRANSCRIPT_RECIPE_PIN')
        recipe = load_recipe(manifest, 'v4', GLM_LAYERS, QWEN_LAYERS, 2, 256)
        if recipe.metadata['manifest_sha256'] != manifest_sha256:
            raise ValueError('TRANSCRIPT_RECIPE_CHANGED')
        if hashlib.sha256(read(index, 128 * 1048576)).hexdigest() != index_sha256:
            raise ValueError('TRANSCRIPT_INDEX_PIN')
        selector = IndexKeyReader.load(Path(index), QWEN_LAYERS, recipe.base.sha256)
        if selector.sha256 != index_sha256:
            raise ValueError('TRANSCRIPT_INDEX_CHANGED')
        from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
        apply_mlx_vlm_qwen4_exp_compat_patch()
        import mlx.core as mx
        from mlx_vlm.utils import load_model
        from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
        from tokenizers import Tokenizer
        from drift.translate.mlx_reader import MlxForwardReader
        from drift.serving.studio_guard import acquire
        acquire('transcript-memory', need_gb=need_gb)
        self.reader = MlxForwardReader(recipe.translator, selector, recipe.gain_power)
        self.mx, self.index_dim = mx, selector.index_dim
        self.pins = {**recipe.metadata, 'index_sha256': index_sha256}
        checkpoint = Path(checkpoint)
        config = json.loads(read(checkpoint / 'config.json', 1048576))
        cfg = config.get('text_config', config)
        if cfg.get('model_type') != 'qwen4_exp_text':
            raise ValueError('TRANSCRIPT_QWEN_FAMILY')
        rotary = cfg.get('rope_parameters') or {}
        self.rope = Rope(float(rotary.get('rope_theta', 1e7)),
                         int(cfg['head_dim'] * float(rotary.get('partial_rotary_factor', 1.0))))
        self.context_limit = int(cfg['max_position_embeddings'])
        self.tokenizer = Tokenizer.from_file(str(checkpoint / 'tokenizer.json'))
        self.stop = {self.tokenizer.token_to_id(name) for name in ('<|im_end|>', '<|endoftext|>')}
        with _force_qwen4_exp_sanitize_on_load(checkpoint):
            self.model = load_model(checkpoint)
        self.lm = self.model.language_model

    def answer(self, latents, question, *, max_new, max_rows, deadline):
        mx = self.mx
        started = time.monotonic()
        if started >= deadline:
            raise TimeoutError('TRANSCRIPT_READER_TIMEOUT')
        entries, keys, order, _ = self.reader.read(latents)
        rows = validate_translation(entries, keys, order, len(next(iter(latents.values()))),
                                    QWEN_LAYERS, self.index_dim)
        prompt = f'<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if rows > max_rows or not ids or rows + len(ids) + max_new > self.context_limit:
            raise ValueError('TRANSCRIPT_READER_CAPACITY')
        if time.monotonic() >= deadline:
            raise TimeoutError('TRANSCRIPT_READER_TIMEOUT')
        cache = self.lm.make_cache()
        append_entries(cache, entries, self.rope, np.arange(rows), self.index_dim,
                       dtype=mx.bfloat16, index_keys=keys)
        mx.eval([cache[layer].keys for layer in QWEN_LAYERS])
        translation_seconds = time.monotonic() - started

        def step(tokens, start):
            if time.monotonic() >= deadline:
                raise TimeoutError('TRANSCRIPT_READER_TIMEOUT')
            positions = mx.arange(start, start + len(tokens), dtype=mx.int32)[None]
            logits = self.lm(mx.array(tokens, dtype=mx.int32)[None], cache=cache, position_ids=positions).logits[0, -1]
            mx.eval(logits)
            return logits

        logits = step(ids, rows)
        generated, position = [], rows + len(ids)
        stopped = False
        for index in range(max_new):
            token = int(mx.argmax(logits).item())
            if token in self.stop:
                stopped = True
                break
            generated.append(token)
            if index + 1 < max_new:
                logits = step([token], position)
                position += 1
        if time.monotonic() >= deadline:
            raise TimeoutError('TRANSCRIPT_READER_TIMEOUT')
        return {'answer': self.tokenizer.decode(generated), 'memory_rows': rows,
                'question_tokens': len(ids), 'generated_tokens': len(generated),
                'finish_reason': 'stop' if stopped else 'length', 'cache_applied': True,
                'translation_and_append_seconds': translation_seconds, 'recipe': self.pins}
