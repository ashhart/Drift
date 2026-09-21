"""Explicitly own one capped linked request with a publication readiness gate."""
import math

from drift.serving.live_owned_process import OwnedProcess
from drift.serving.live_qualification_events import EventCounts, QualificationError


class LinkedEvents(EventCounts):
    def __init__(self, limits, reserve):
        super().__init__(limits, reserve)
        self.ready = self.released = False

    def accept(self, event):
        if isinstance(event, dict) and event.get('publication_ready') is True:
            if event != {'publication_ready': True} or self.ready or self.started or self.terminal:
                raise QualificationError('READINESS_PROTOCOL')
            self.ready = True
            self.events += 1
            if self.events > self.limits.max_events: raise QualificationError('EVENT_LIMIT')
            return
        if not self.released and isinstance(event, dict) and not event.get('cancelled'):
            raise QualificationError('OUTPUT_BEFORE_RELEASE')
        super().accept(event)


class LinkedSession(OwnedProcess):
    def __init__(self, command, claim, limits):
        command = list(command)
        if limits.max_new > 256 or limits.max_prompt_tokens > 256 or limits.max_seconds > 180:
            raise QualificationError('LINKED_LIMIT_CONTRACT')
        flags = ('--session', '--max-new', '--reserve', '--max-prompt-tokens', '--wait-publication',
                 '--ready-timeout', '--no-tap', '--control-stdin', '--no-link')
        if any(arg.startswith(flag + '=') for arg in command for flag in flags) or '--no-link' in command:
            raise QualificationError('LINKED_COMMAND_CONTRACT')
        def value(flag):
            if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
                raise QualificationError('LINKED_COMMAND_CONTRACT')
            return command[command.index(flag) + 1]
        if value('--session') != claim.name or value('--max-new') != str(limits.max_new):
            raise QualificationError('LINKED_COMMAND_CONTRACT')
        if value('--max-prompt-tokens') != str(limits.max_prompt_tokens):
            raise QualificationError('LINKED_COMMAND_CONTRACT')
        if any(command.count(flag) != 1 for flag in ('--wait-publication', '--no-tap', '--control-stdin')):
            raise QualificationError('LINKED_COMMAND_CONTRACT')
        try:
            reserve, wait = int(value('--reserve')), float(value('--ready-timeout'))
            if not 1 <= reserve <= min(12, limits.max_prompt_tokens): raise ValueError
            if not math.isfinite(wait) or not 0 < wait <= min(180, limits.max_seconds): raise ValueError
        except ValueError:
            raise QualificationError('LINKED_COMMAND_CONTRACT') from None
        super().__init__(command, claim, limits, reserve, LinkedEvents(limits, reserve))

    @property
    def publication_ready(self):
        return self.counts.ready

    def release(self):
        self.poll(0)
        if not self.counts.ready or self.counts.released or self.counts.terminal or self.failure:
            self._reject('RELEASE_PROTOCOL')
        try:
            self.child.stdin.write(b'{"op":"release"}\n')
            self.child.stdin.flush()
        except (OSError, ValueError):
            self._reject('RELEASE_SEND_FAILURE')
        self.counts.released = True

    def report(self):
        return dict(super().report(), linked=True, publication_ready=self.counts.ready,
                    release_sent=self.counts.released)
