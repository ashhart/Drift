"""The Stockledger task's hidden checks pass a correct reader and fail readers that skip rules or name no line."""
import textwrap
from scripts.live.stockledger_task import CHECKS, grade

REFERENCE = textwrap.dedent('''
    import json, re
    from datetime import datetime

    FIELDS = {"event_id", "occurred_at", "warehouse", "sku", "delta"}


    def _pairs(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(pairs)


    def _check(event):
        if not isinstance(event, dict) or set(event) != FIELDS:
            raise ValueError("fields")
        if not isinstance(event["event_id"], str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", event["event_id"]):
            raise ValueError("event_id")
        stamp = event["occurred_at"]
        if not isinstance(stamp, str) or not re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z", stamp):
            raise ValueError("occurred_at")
        datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
        if not isinstance(event["warehouse"], str) or not re.fullmatch(r"[a-z0-9-]{1,32}", event["warehouse"]):
            raise ValueError("warehouse")
        if not isinstance(event["sku"], str) or not re.fullmatch(r"[A-Z0-9_-]{1,64}", event["sku"]):
            raise ValueError("sku")
        delta = event["delta"]
        if type(delta) is not int or not -1000000000 <= delta <= 1000000000:
            raise ValueError("delta")


    def read_events(stream):
        data = stream.read()
        if data.startswith(b"\\xef\\xbb\\xbf"):
            raise ValueError("line 1: byte order mark")
        events = []
        for number, raw in enumerate(data.split(b"\\n"), 1):
            try:
                text = raw.decode("utf-8")
                if not text.strip(" \\t\\r"):
                    continue
                event = json.loads(text, object_pairs_hook=_pairs)
                _check(event)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"line {number}: {error}") from None
            events.append(event)
        return events
    ''')


def test_a_correct_reader_passes_every_check():
    report = grade(f"```python\n{REFERENCE}```")
    assert report["passed"] == report["total"] == len(CHECKS), [n for n, ok in report["results"].items() if not ok]


def test_a_lenient_reader_or_a_message_without_the_line_fails_those_checks():
    lenient = REFERENCE.replace('raise ValueError(f"line {number}: {error}")', 'raise ValueError(f"invalid event 1 to 64: {error}")')
    report = grade(f"```python\n{lenient}```")
    assert not report["results"]["missing field"] and not report["results"]["error names the physical line"]
    assert report["results"]["one valid event"] and grade("no code here")["passed"] == 0
