"""Typed service operations and canonical authentication."""
from __future__ import annotations
import hmac
import json
from enum import IntEnum
from typing import Any, Mapping


class Op(IntEnum):
    SETUP = 1
    START = 2
    TICK = 3
    STATUS = 4
    PAUSE = 5
    CHECKPOINT = 6
    ABORT = 7
    COMPLETE = 8
    MAIL = 9



class ServiceError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code



def js_number(value: float) -> str:
    """ECMAScript Number::toString formatting, so both sides sign identical bytes."""
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("nonfinite numbers are not serializable")
    if value == int(value) and abs(value) < 1e21:
        return str(int(value))
    text = repr(value)                      # shortest round-trip digits, like JS
    mantissa, _, exponent = text.partition("e")
    if not exponent:
        return text
    digits = mantissa.replace("-", "").replace(".", "")
    sign = "-" if value < 0 else ""
    e = int(exponent)
    point = mantissa.replace("-", "").find(".")
    n = e + (point if point >= 0 else len(mantissa.replace("-", "")))   # decimal exponent
    if -6 < n <= 21:
        if n <= 0:
            return sign + "0." + "0" * (-n) + digits.rstrip("0")
        if n >= len(digits):
            return sign + digits + "0" * (n - len(digits))
        return sign + digits[:n] + "." + digits[n:]
    exp = n - 1
    body = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    return f"{sign}{body}e{'+' if exp >= 0 else '-'}{abs(exp)}"



def _canonical(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return js_number(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Mapping):
        return "{" + ",".join(json.dumps(str(k)) + ":" + _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    raise TypeError(f"unsupported type in protocol message: {type(value).__name__}")



def _int(request: Mapping[str, Any], key: str, default: int | None = None) -> int:
    value = request.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ServiceError("MALFORMED")
    return value



def canonical_json(message: Mapping[str, Any]) -> bytes:
    """Sorted keys, no whitespace, JavaScript number formatting (matches the plugin)."""
    return _canonical({k: v for k, v in message.items() if k != "auth"}).encode()



def sign(message: Mapping[str, Any], secret: bytes) -> str:
    return hmac.new(secret, canonical_json(message), "sha256").hexdigest()



def verify(message: Mapping[str, Any], secret: bytes) -> bool:
    auth = message.get("auth")
    return isinstance(auth, str) and hmac.compare_digest(auth, sign(message, secret))
