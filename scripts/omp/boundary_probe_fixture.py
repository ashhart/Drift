"""Public worker events for boundary ordering and lifetime qualification."""
import json
from pathlib import Path
import time
from scripts.omp.duo_backend_fixture import DuoFixture


class BoundaryFixture(DuoFixture):
    def __init__(self, config):
        super().__init__(config)
        self.control_root = Path(config['boundary_root'])

    def stream(self, maximum):
        if self.worker == 'glm-fixture' and self.facts['stream'] == 2:
            end = time.monotonic() + 2
            while not (self.control_root / 'child-ready.json').exists() and time.monotonic() < end:
                time.sleep(0.01)
        for event in super().stream(maximum):
            if event['op'] == 'terminal':
                time.sleep(0.2)
                self.record('terminal') if 'terminal' in self.facts else self.facts.update(terminal=1)
                self.path.write_text(json.dumps(self.facts))
            yield event
