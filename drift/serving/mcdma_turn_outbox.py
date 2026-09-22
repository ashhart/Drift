"""Capture a native turn's forward mailbox without claiming receiver application."""
import math
import time

from drift.exchange.lifetime import checkpoint, request_scope
from drift.serving.glm_outbox_files import require, write_manifest, write_rows
from drift.serving.glm_turn_outbox import TurnOutbox
from drift.serving.mcdma_forward import Complete, ForwardMailbox


class McdmaTurnOutbox(TurnOutbox):
    def __init__(self, config, layouts, reader, publisher, *, max_turns=2,
                 clock=time.monotonic, pause=time.sleep):
        super().__init__(config, layouts, config['memory_root'], max_turns=max_turns, clock=clock, pause=pause)
        self.reader, self.publisher, self.forward = reader, publisher, None

    def begin(self, name, proof, maximum):
        try:
            checkpoint()
            require(self.forward is None or self.forward.finished)
            super().begin(name, proof, maximum)
            self.forward = ForwardMailbox(self.reader, name, tuple(int(key[1:]) for key in self.layouts))
            self.publisher.watch(name)
            checkpoint()
        except BaseException:
            self.close()
            raise

    def finish(self, name, maximum, deadline):
        try:
            require(type(deadline) in (int, float) and math.isfinite(deadline))
            with request_scope(lambda: self._remaining(deadline)):
                checkpoint()
                state = self.current
                require(state is not None and name == state['name'] and type(maximum) is int
                        and 0 < maximum <= state['maximum'])
                cursor, raw_bytes, selected_bytes, selected_rows = state['first'], 0, 0, 0
                raw, selected = [], []
                while True:
                    checkpoint()
                    tap = self.forward.peek()
                    if tap is None:
                        self.pause(min(.01, self._remaining(deadline)))
                        continue
                    if isinstance(tap, Complete):
                        require(tap.source_start == state['first'] and tap.source_stop == cursor)
                        report = self._report(name, state, cursor, raw_bytes, selected_bytes, selected_rows)
                        report['native_terminal'] = dict(tap_count=tap.tap_count,
                                                        source_start=tap.source_start, source_stop=tap.source_stop)
                        artifact = write_manifest(state['output'], {**report, 'raw_publications': raw,
                                                                    'publications': selected})
                        checkpoint()
                        self.forward.acknowledge(tap)
                        self.current = None
                        return {**report, **artifact}
                    require(len(raw) < self.publications and tap.start == cursor
                            and tap.stop <= state['prompt'] + maximum)
                    require(raw_bytes + tap.meta['bytes'] <= self.raw_bytes)
                    start = max(tap.start, state['prompt'])
                    if tap.stop > start:
                        rows = tap.stop - start
                        require(selected_rows + rows <= self.rows)
                        arrays = {f'l{key}': value[start-tap.start:] for key, value in tap.latents.items()}
                        output = write_rows(state['output'], len(selected), arrays, self.bytes-selected_bytes)
                        selected.append({**output, 'source_seq': tap.meta['tap'], 'source_start': start,
                                         'source_stop': tap.stop, 'rows': rows})
                        selected_rows += rows
                        selected_bytes += output['bytes']
                    raw.append({'seq': tap.meta['tap'], 'start': tap.start, 'stop': tap.stop,
                                'bytes': tap.meta['bytes'], 'sha256': tap.sha256})
                    raw_bytes += tap.meta['bytes']
                    cursor = tap.stop
                    checkpoint()
                    self.forward.acknowledge(tap)
        except BaseException as error:
            self.close()
            if isinstance(error, (TimeoutError, KeyboardInterrupt, SystemExit)):
                raise
            raise ValueError('MCDMA_OUTBOX_FAILED') from error

    def _report(self, name, state, cursor, raw_bytes, selected_bytes, selected_rows):
        return {'session': name, 'source_worker': self.source, 'target_worker': self.target,
                'recipe_sha256': self.recipe, 'transport': 'mcdma', 'cache_applied': False,
                'evidence': 'EXPORTED_PREFIX' if selected_rows else 'EMPTY_EXPORTED_PREFIX',
                'full_completion': False, 'final_tail': 'UNKNOWN', 'scope': 'newly_generated_only',
                'new_own_inputs': 'NOT_EXPORTED', 'translation': 'NOT_PERFORMED',
                'prompt_tokens': state['prompt'], 'verified_stop': cursor, 'selected_rows': selected_rows,
                'selected_bytes': selected_bytes, 'raw_bytes': raw_bytes, 'forward_complete': True}

    def close(self):
        super().close()
        if self.forward is not None:
            self.forward.poisoned = True
