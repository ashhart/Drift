"""Whether an answer carries a question's key: the rule the drop-in, held-out and dev gates score by.

- A def line is parsed and must be callable as the source defines it (drift.eval.signatures); the opening of one,
  such as a training key "def pull(", is matched as words.
- A question for a value (what value is assigned to NAME; the default value of a parameter) whose key is a number
  matches when the value the answer gives has that value, in any Python spelling or with thousands commas:
  0xFF000000, 64_000, 64,000 or 1e3. The value given is the one the answer states: the first number after "is" or
  "equals" following a mention of the constant or parameter. Failing a statement, it is the number the answer assigns
  ("head_dim=256"), then the first number after the first mention, then the first number. So "65536 (or 64_000 in
  the code)" gives 65536 even when the code is quoted later, and a quoted "heads=2, head_dim=256" gives 256. Digits
  inside a name, such as the 5 in EF_ARM_ABI_VER5, are not numbers.
- A value question whose key is a string matches that string quoted in the answer ('...', "..." or `...`), or an answer
  that is the string alone, so words the answer repeats from the question do not count.
- A question for the function that fails with a message matches the function's name as a whole identifier, after the
  answer's copies of the question's quoted message and file path are removed.
- Any other question matches the key's words as whole words of the answer, ignoring case and punctuation; the last
  word may carry a plural s or es.
An empty key matches nothing.
"""
from __future__ import annotations
import ast
import re
from drift.eval import signatures

_NUMBER = re.compile(r"(?<![A-Za-z0-9_.,])[-+]?(?:0[xX][0-9a-fA-F_]+|0[oO][0-7_]+|0[bB][01_]+|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
                     r"|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][-+]?\d+)?|\.\d[\d_]*(?:[eE][-+]?\d+)?)(?![A-Za-z0-9_])")
_DECIMAL = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_QUOTED = re.compile(r"`([^`\n]+)`|\"([^\"\n]*)\"|'([^'\n]*)'")
_VALUE = re.compile(r"what value is assigned to|what is the default value of", re.I)
_TARGET = re.compile(r"what value is assigned to ([A-Za-z_][A-Za-z0-9_]*)|default value of the ([A-Za-z_][A-Za-z0-9_]*) parameter", re.I)
_RAISES = re.compile(r"fails with the message", re.I)


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def _has_words(key: str, text: str) -> bool:
    wanted = words(key)
    if not wanted:
        return False
    pattern = r"(?<![a-z0-9])" + r" ".join(map(re.escape, wanted)) + r"(?:e?s)?(?![a-z0-9])"
    return re.search(pattern, " ".join(words(text))) is not None


def numbers(text: str) -> list[int | float]:
    """Every number the text writes as a Python literal, by value."""
    found = []
    for match in _NUMBER.finditer(text):
        value = _literal(match.group(0))
        if value is not None:
            found.append(value)
    return found


def _literal(written: str) -> int | float | None:
    try:
        value = ast.literal_eval(written.replace(",", ""))
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _given(text: str, question: str) -> int | float | None:
    """The number the answer gives for the question's constant or parameter."""
    target = _TARGET.search(question)
    name = next((group for group in target.groups() if group), None) if target else None
    if name:
        for mention in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
            verb = re.search(r"\b(?:is|equals)\b", text[mention.end():mention.end() + 150])
            start = mention.end() + verb.end() if verb else None
            stated = _NUMBER.search(text, start, start + 80) if verb else None
            if stated and _literal(stated.group(0)) is not None:
                return _literal(stated.group(0))
        for assignment in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*(?::[^=\n,)]*)?=(?!=)[\s`*]*", text):
            value = numbers(text[assignment.end():])
            if value and _NUMBER.match(text, assignment.end()):
                return value[0]
    mention = re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) if name else None
    written = numbers(text[mention.end():] if mention else text)
    return written[0] if written else None


def _quoted(text: str) -> list[str]:
    spans = []
    for match in _QUOTED.finditer(text):
        span = next(group for group in match.groups() if group is not None).strip()
        if len(span) >= 2 and span[0] == span[-1] and span[0] in "'\"":      # `"param"` inside backticks
            span = span[1:-1]
        spans.append(span)
    return spans


def _without_echo(text: str, question: str) -> str:
    for quoted in re.findall(r"\"([^\"]+)\"", question):
        text = text.replace(quoted, " ")
    for path in re.findall(r"[\w./-]+\.(?:py|md)\b", question):
        text = text.replace(path, " ")
    return text


def matches(key: str, question: str, text: str) -> bool:
    """Whether the answer text carries the key of the question, by the rules above."""
    if not key.strip():
        return False
    if key.lstrip().startswith(("def ", "async def ")) and signatures.parse(key) is not None:
        return signatures.score(text, key)["callable"]
    if _VALUE.search(question):
        if _DECIMAL.fullmatch(key.strip()):
            value = ast.literal_eval(key.strip())
            return _given(text, question) == value
        return any(words(span) == words(key) for span in _quoted(text)) or words(text) == words(key)
    if _RAISES.search(question):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(key.strip())}(?![A-Za-z0-9_])"
        return re.search(pattern, _without_echo(text, question), re.I) is not None
    return _has_words(key, text)
