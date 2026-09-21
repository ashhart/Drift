"""Own one no-link runner and retain only aggregate cancellation evidence."""
from drift.serving.live_owned_process import OwnedProcess
from drift.serving.live_qualification_claim import claim_session
from drift.serving.live_qualification_events import Limits, QualificationError


class OwnedSession(OwnedProcess):
    def __init__(self, command, claim, limits):
        self.command, self.claim, self.limits = list(command), claim, limits
        protected = ('--session', '--max-new', '--reserve', '--no-link', '--control-stdin')
        if any(arg.startswith(flag + '=') for arg in self.command for flag in protected):
            raise QualificationError('COMMAND_CONTRACT')
        for flag, expected in (('--session', claim.name), ('--max-new', str(limits.max_new))):
            if self.command.count(flag) != 1:
                raise QualificationError('COMMAND_CONTRACT')
            index = self.command.index(flag) + 1
            if index >= len(self.command) or self.command[index] != expected:
                raise QualificationError('COMMAND_CONTRACT')
        if any(self.command.count(flag) != 1 for flag in ('--no-link', '--control-stdin')):
            raise QualificationError('COMMAND_CONTRACT')
        try:
            if self.command.count('--reserve') != 1:
                raise ValueError
            self.reserve = int(self.command[self.command.index('--reserve') + 1])
            if not 0 <= self.reserve <= limits.max_prompt_tokens:
                raise ValueError
        except (ValueError, IndexError):
            raise QualificationError('COMMAND_CONTRACT') from None
        super().__init__(command, claim, limits, self.reserve)
