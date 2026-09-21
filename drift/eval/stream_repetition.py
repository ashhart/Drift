"""Bounded output-text repetition proxy, independent of model tokenizers and runtimes.

This is an engineering abort signal, not evidence of semantic echo or calibrated
degeneration detection. Policy must be fixed on development data before trials.
No text, word strings, or token IDs appear in exported diagnostics.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RepetitionPolicy:
    window_words: int = 128
    ngram: int = 4
    min_words: int = 32
    threshold: float = 0.6
    max_word_chars: int = 256

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in
               (self.window_words, self.ngram, self.min_words, self.max_word_chars)):
            raise ValueError("repetition sizes must be positive integers")
        if not self.ngram < self.min_words <= self.window_words:
            raise ValueError("require ngram < min_words <= window_words")
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("repetition threshold must be finite and in (0, 1]")


class RepetitionDetected(RuntimeError):
    """The configured output safety policy tripped; the session must abort."""


class StreamingRepetition:
    def __init__(self, policy: RepetitionPolicy):
        self.policy = policy
        self._words = deque(maxlen=policy.window_words)
        self._pending = ""
        self._finished = False
        self.words_seen = 0
        self.peak_rate = self.rate = 0.0
        self.reason = None

    def _finish_word(self):
        if not self._pending:
            return
        self._words.append(self._pending)
        self._pending = ""
        self.words_seen += 1
        words = tuple(self._words)
        total = len(words) - self.policy.ngram + 1
        if total > 0:
            unique = len({words[i:i + self.policy.ngram] for i in range(total)})
            self.rate = (total - unique) / total
            self.peak_rate = max(self.peak_rate, self.rate)
            if len(words) >= self.policy.min_words and self.rate >= self.policy.threshold:
                self.reason = "repeated_word_ngrams"

    def feed(self, text: str, *, final: bool = False) -> bool:
        if self.reason:
            return True  # Latched: later novel output cannot erase a detected loop.
        if self._finished:
            raise ValueError("output arrived after stream completion")
        for char in text:
            if char.isspace():
                self._finish_word()
            elif len(self._pending) == self.policy.max_word_chars:
                self.reason = "word_size_limit"
            else:
                self._pending += char
            if self.reason:
                break
        if final and not self.reason:
            self._finish_word()
            self._finished = True
        return self.reason is not None

    def report(self) -> dict:
        return {"metric": "whitespace_word_ngram_repetition", "policy": asdict(self.policy),
                "words_seen": self.words_seen, "retained_words": len(self._words),
                "pending_chars": len(self._pending), "rate": self.rate,
                "peak_rate": self.peak_rate, "triggered": self.reason is not None, "reason": self.reason}
